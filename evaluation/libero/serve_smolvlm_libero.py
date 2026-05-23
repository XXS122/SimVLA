#!/usr/bin/env python3
"""
SimVLA LIBERO Policy Server (WebSocket)

A WebSocket-based policy server for LIBERO evaluation:
- Uses msgpack_numpy serialization for efficient data transfer
- Sends server metadata on connection
- Receives: observation/image, observation/wrist_image, observation/state, prompt
- Returns: {"actions": [...]}

State format (8D): [ee_pos(3), axis_angle(3), gripper_qpos(2)]
Action format (7D): [delta_xyz(3), delta_axisangle(3), gripper_cmd(1)]
"""

import argparse
import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import websockets

try:
    import msgpack
    import msgpack_numpy
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack_numpy not installed, using JSON fallback")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# episode 级静态特征缓存（key: connection_id, value: static_context tensor）
_static_cache: dict = {}

# Configuration
CONFIG = {
    "state_dim": 8,
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
}


def load_model(checkpoint_path: str, norm_stats_path: str = None, smolvlm_model_path: str = None):
    """Load SimVLA model and processor."""
    global model, processor
    
    logger.info(f"Loading SimVLA from {checkpoint_path}...")
    
    model = SmolVLMVLA.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    smolvlm_path = smolvlm_model_path or "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)
    
    if norm_stats_path and os.path.exists(norm_stats_path):
        logger.info(f"Loading norm stats from: {norm_stats_path}")
        model.action_space.load_norm_stats(norm_stats_path)
        if hasattr(model.action_space, 'state_norm_stats') and model.action_space.state_norm_stats:
            logger.info(f"   State norm: mean={model.action_space.state_norm_stats.mean[:3].tolist()}")
        if hasattr(model.action_space, 'action_norm_stats') and model.action_space.action_norm_stats:
            logger.info(f"   Action norm: mean={model.action_space.action_norm_stats.mean[:3].tolist()}")
    else:
        logger.warning("No norm_stats loaded!")
    
    logger.info(f"Model loaded! Device: {device}, Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")


def preprocess_images(image0: np.ndarray, image1: np.ndarray):
    """Preprocess images to model input format."""
    image_size = CONFIG["image_size"]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img0 = Image.fromarray(image0.astype(np.uint8))
    img1 = Image.fromarray(image1.astype(np.uint8))
    
    img0_t = transform(img0)
    img1_t = transform(img1)
    
    # Pad to 3 views (model processes all views together)
    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([[True, True, False]])
    
    return images.unsqueeze(0), image_mask


def decode_numpy(obj):
    """Decode numpy array from msgpack_numpy dict format."""
    if isinstance(obj, dict):
        if b'__ndarray__' in obj or '__ndarray__' in obj:
            data_key = b'data' if b'data' in obj else 'data'
            dtype_key = b'dtype' if b'dtype' in obj else 'dtype'
            shape_key = b'shape' if b'shape' in obj else 'shape'
            
            data = obj[data_key]
            dtype_str = obj[dtype_key]
            shape = obj[shape_key]
            
            if isinstance(dtype_str, bytes):
                dtype_str = dtype_str.decode()
            
            if shape and isinstance(shape[0], bytes):
                shape = tuple(int(s) for s in shape)
            else:
                shape = tuple(shape)
            
            return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
    return obj


def infer(observation: Dict[str, Any], conn_id: int = None) -> Dict[str, Any]:
    """Run inference on a single observation."""
    global model, processor, _static_cache

    try:
        # Extract observation fields
        image0 = observation.get("observation/image")
        image1 = observation.get("observation/wrist_image")
        state = observation.get("observation/state", np.zeros(8))
        prompt = observation.get("prompt", "")

        # Decode msgpack_numpy format if needed
        image0 = decode_numpy(image0)
        image1 = decode_numpy(image1)
        state = decode_numpy(state)

        # Ensure numpy arrays
        if not isinstance(image0, np.ndarray):
            image0 = np.array(image0, dtype=np.uint8)
        if not isinstance(image1, np.ndarray):
            image1 = np.array(image1, dtype=np.uint8)
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        if len(state) < 8:
            state = np.pad(state, (0, 8 - len(state)))
        state = state[:8]

        # Preprocess images
        images, image_mask = preprocess_images(image0, image1)
        images = images.to(device)
        image_mask = image_mask.to(device)

        # Encode language instruction
        lang = processor.encode_language([prompt])
        lang = {k: v.to(device) for k, v in lang.items()}

        # Proprioception
        proprio_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # 静态-动态解耦推理
        adaptive = CONFIG.get("adaptive", True)
        cos_threshold = CONFIG.get("cos_threshold", 0.97)
        min_steps = CONFIG.get("min_steps", 2)
        use_static_cache = CONFIG.get("use_static_cache", True)

        if use_static_cache and conn_id is not None:
            # 第一步：计算并缓存静态特征（agentview + 语言）
            if _static_cache.get(conn_id) is None:
                static_context = model.encode_static_context(
                    input_ids=lang['input_ids'],
                    image_input=images,
                    image_mask=image_mask,
                    static_view_idx=0,
                )
                _static_cache[conn_id] = static_context
            else:
                static_context = _static_cache[conn_id]

            # 每步：只编码动态视角（eye_in_hand）
            dynamic_feats = model.encode_dynamic_view(
                image_input=images,
                image_mask=image_mask,
                dynamic_view_idx=1,
            )

            actions = model.generate_actions_with_cache(
                static_context=static_context,
                dynamic_feats=dynamic_feats,
                proprio=proprio_tensor,
                steps=CONFIG["action_horizon"],
                adaptive=adaptive,
                cos_threshold=cos_threshold,
                min_steps=min_steps,
            )
        else:
            # 回退：完整推理（无缓存）
            actions = model.generate_actions(
                input_ids=lang['input_ids'],
                image_input=images,
                image_mask=image_mask,
                proprio=proprio_tensor,
                steps=CONFIG["action_horizon"],
                adaptive=adaptive,
                cos_threshold=cos_threshold,
                min_steps=min_steps,
            )

        actual_steps = getattr(model, 'last_actual_steps', CONFIG["action_horizon"])
        logger.debug(f"Inference used {actual_steps} ODE steps")

        actions = actions.cpu().numpy()[0]

        return {"actions": actions}

    except Exception as e:
        logger.error(f"Inference error: {e}")
        traceback.print_exc()
        return {"actions": np.zeros((CONFIG["action_horizon"], CONFIG["action_dim"]))}


async def handle_connection(websocket, path=None):
    """Handle a WebSocket connection."""
    logger.info(f"Connection from {websocket.remote_address} opened")
    conn_id = id(websocket)
    _static_cache[conn_id] = None  # 新 episode，清空静态特征缓存

    use_static_cache = CONFIG.get("use_static_cache", True)
    adaptive = CONFIG.get("adaptive", True)
    logger.info(f"Episode started: static_cache={'enabled' if use_static_cache else 'disabled'}, "
                f"adaptive={'enabled' if adaptive else 'disabled'}")

    try:
        # Send server metadata on connection
        metadata = {
            "model": "SimVLA",
            "action_dim": CONFIG["action_dim"],
            "action_horizon": CONFIG["action_horizon"],
            "image_size": CONFIG["image_size"],
        }
        if HAS_MSGPACK:
            await websocket.send(msgpack_numpy.packb(metadata, use_bin_type=True))
        else:
            import json
            await websocket.send(json.dumps(metadata))

        # Process requests
        async for message in websocket:
            try:
                # Parse request
                if HAS_MSGPACK and isinstance(message, bytes):
                    request = msgpack_numpy.unpackb(message, raw=False)
                else:
                    import json
                    request = json.loads(message)

                # Run inference (pass conn_id only when static cache is enabled)
                effective_conn_id = conn_id if CONFIG.get("use_static_cache", True) else None
                result = infer(request, conn_id=effective_conn_id)

                # Send response (convert numpy to list for compatibility)
                actions = result["actions"]
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()

                response_data = {"actions": actions}

                if HAS_MSGPACK:
                    import msgpack
                    response = msgpack.packb(response_data, use_bin_type=True)
                else:
                    import json
                    response = json.dumps(response_data)

                await websocket.send(response)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
                error_msg = f"Error: {str(e)}"
                await websocket.send(error_msg)

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _static_cache.pop(conn_id, None)
        logger.info(f"Connection from {websocket.remote_address} closed")


async def serve(host: str, port: int):
    """Start the WebSocket server."""
    logger.info(f"Creating SimVLA server (host: {host}, port: {port})")
    
    async with websockets.serve(handle_connection, host, port, max_size=None, compression=None):
        logger.info(f"SimVLA server listening on {host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="SimVLA LIBERO Server (WebSocket)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SimVLA checkpoint")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="Path to normalization stats JSON")
    parser.add_argument("--smolvlm_model", type=str,
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="SmolVLM model path or HuggingFace repo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # ③ 自适应推理步数
    parser.add_argument("--adaptive", action="store_true", default=False,
                        help="Use adaptive ODE steps (cosine similarity early stopping)")
    parser.add_argument("--no_adaptive", dest="adaptive", action="store_false")
    parser.add_argument("--cos_threshold", type=float, default=0.97,
                        help="Cosine similarity threshold for adaptive stopping")
    parser.add_argument("--min_steps", type=int, default=4,
                        help="Minimum ODE steps before adaptive stopping")
    # ① 静态动态解耦
    parser.add_argument("--use_static_cache", action="store_true", default=False,
                        help="Cache static VLM features per episode (faster inference, may reduce accuracy)")
    parser.add_argument("--no_static_cache", dest="use_static_cache", action="store_false")

    args = parser.parse_args()

    # Store inference config globally
    CONFIG["adaptive"] = args.adaptive
    CONFIG["cos_threshold"] = args.cos_threshold
    CONFIG["min_steps"] = args.min_steps
    CONFIG["use_static_cache"] = args.use_static_cache

    if not HAS_MSGPACK:
        logger.warning("msgpack_numpy not installed! Install with: pip install msgpack-numpy")

    load_model(args.checkpoint, args.norm_stats, args.smolvlm_model)

    logger.info(f"Starting SimVLA server on {args.host}:{args.port}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Action horizon: {CONFIG['action_horizon']}")
    logger.info(f"  Adaptive steps: {args.adaptive} (threshold={args.cos_threshold}, min={args.min_steps})")
    logger.info(f"  Static cache: {args.use_static_cache}")

    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
