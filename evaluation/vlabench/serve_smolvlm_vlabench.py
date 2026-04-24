#!/usr/bin/env python3
"""
SimVLA VLABench 推理服务器（WebSocket）

与 VLABench 的 OpenPiPolicy 客户端协议完全兼容。

启动方式（simvla 环境）：
    conda activate simvla
    CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_vlabench.py \
        --checkpoint /root/SimVLA/simvla_output/simvla_vlabench_small/ckpt-10000 \
        --norm_stats ../../norm_stats/vlabench_norm.json \
        --smolvlm_model /root/model/smolvlm-500M \
        --port 8001
"""

import argparse
import asyncio
import collections
import functools
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import msgpack
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import websockets

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "state_dim": 7,
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
}

# ── msgpack numpy 编解码（与 openpi.py 协议一致）──────────────────────────────

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)

# ── 模型加载 ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, norm_stats_path: str = None, smolvlm_model_path: str = None):
    global model, processor

    logger.info(f"加载 SimVLA from {checkpoint_path} ...")

    # 若提供了本地 smolvlm_model_path，覆盖 config 中可能残留的旧服务器路径
    if smolvlm_model_path:
        from models.configuration_smolvlm_vla import SmolVLMVLAConfig
        cfg = SmolVLMVLAConfig.from_pretrained(checkpoint_path)
        logger.info(f"覆盖 smolvlm_model_path: {cfg.smolvlm_model_path} -> {smolvlm_model_path}")
        cfg.smolvlm_model_path = smolvlm_model_path
        model = SmolVLMVLA.from_pretrained(checkpoint_path, config=cfg)
    else:
        model = SmolVLMVLA.from_pretrained(checkpoint_path)

    model = model.to(device)
    model.eval()

    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model_path or model.config.smolvlm_model_path)

    if norm_stats_path and os.path.exists(norm_stats_path):
        logger.info(f"加载 norm stats: {norm_stats_path}")
        model.action_space.load_norm_stats(norm_stats_path)
    else:
        logger.warning("未加载 norm_stats，动作可能不准确！")

    logger.info(f"模型加载完成，device={device}")

# ── 图像预处理 ────────────────────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def preprocess_images(front: np.ndarray, wrist: np.ndarray):
    img_f = _transform(Image.fromarray(front.astype(np.uint8)))
    img_w = _transform(Image.fromarray(wrist.astype(np.uint8)))
    padding = torch.zeros_like(img_f)
    images = torch.stack([img_f, img_w, padding], dim=0).unsqueeze(0)
    image_mask = torch.tensor([[True, True, False]])
    return images, image_mask

# ── 推理 ──────────────────────────────────────────────────────────────────────

def infer(obs: Dict[str, Any]) -> Dict[str, Any]:
    front = obs["observation/image"]
    wrist = obs["observation/wrist_image"]
    state = np.array(obs.get("observation/state", np.zeros(7)), dtype=np.float32)
    prompt = obs.get("prompt", "")

    # VLABench 传来的 state 是 [pos(3), quat(4), gripper(1)] = 8维
    # 我们只取 pos(3) + axis_angle(3) + gripper(1) = 7维
    if len(state) >= 8:
        from scipy.spatial.transform import Rotation as R
        pos = state[:3]
        quat = state[3:7]  # xyzw
        # 转 axis-angle
        try:
            aa = R.from_quat(quat).as_rotvec().astype(np.float32)
        except Exception:
            aa = np.zeros(3, dtype=np.float32)
        gripper = state[7:8]
        state = np.concatenate([pos, aa, gripper])
    state = state[:7]

    images, image_mask = preprocess_images(front, wrist)
    images = images.to(device)
    image_mask = image_mask.to(device)

    lang = processor.encode_language([prompt])
    lang = {k: v.to(device) for k, v in lang.items()}

    proprio = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        actions = model.generate_actions(
            input_ids=lang["input_ids"],
            image_input=images,
            image_mask=image_mask,
            proprio=proprio,
            steps=CONFIG["action_horizon"],
        )

    return {"actions": actions.cpu().numpy()[0]}  # [T, 7]

# ── WebSocket 服务 ────────────────────────────────────────────────────────────

async def handle_connection(websocket, path=None):
    logger.info(f"连接来自 {websocket.remote_address}")
    try:
        metadata = {
            "model": "SimVLA-VLABench",
            "action_dim": CONFIG["action_dim"],
            "action_horizon": CONFIG["action_horizon"],
            "image_size": CONFIG["image_size"],
        }
        await websocket.send(packb(metadata))

        async for message in websocket:
            try:
                request = unpackb(message)
                result = infer(request)
                actions = result["actions"]
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()
                await websocket.send(packb({"actions": actions}))
            except Exception as e:
                logger.error(f"推理错误: {e}")
                traceback.print_exc()
                await websocket.send(str(e).encode())

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info(f"连接关闭 {websocket.remote_address}")


async def serve(host: str, port: int):
    async with websockets.serve(handle_connection, host, port, max_size=None, compression=None):
        logger.info(f"SimVLA-VLABench 服务器监听 {host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, default=None)
    parser.add_argument("--smolvlm_model", type=str,
                        default=os.environ.get("SIMVLA_SMOLVLM_MODEL", "/root/model/smolvlm-500M"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    load_model(args.checkpoint, args.norm_stats, args.smolvlm_model)
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
