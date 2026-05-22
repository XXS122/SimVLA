"""
SmolVLM-VLA Model

HuggingFace-compatible Vision-Language-Action policy using SmolVLM-500M-Instruct
as the visual-language backbone.

Key differences from FlorenceVLA:
  - Uses SmolVLM-500M-Instruct (efficient 500M parameter model)
  - 512x512 image input (SmolVLM-500M uses 512x512 patches)
  - All views processed together by SmolVLM, no aux_visual_inputs
  - Unified VLM output for multi-view inputs
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel, AutoProcessor, AutoModelForImageTextToText
from .transformer_smolvlm import SmolVLMActionTransformer, SmolVLMActionTransformerV2
from .action_hub import build_action_space
from .configuration_smolvlm_vla import SmolVLMVLAConfig


class SmolVLMVLA(PreTrainedModel):
    """
    SmolVLM-VLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • SmolVLM-500M-Instruct backbone (vision-language)
      • SmolVLMActionTransformer (flow matching action head)
      • Action space (pre/post-processing + loss)
      
    Key differences from FlorenceVLA:
      • All camera views are input to VLM together (no aux_visual_inputs)
      • 512x512 image resolution (SmolVLM-500M uses 512x512 patches)
      • Efficient 500M parameter model
    """
    config_class = SmolVLMVLAConfig
    base_model_prefix = "smolvlm_vla"
    supports_gradient_checkpointing = True

    def __init__(self, config: SmolVLMVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.image_size: int = config.image_size
        self.num_views: int = config.num_views
        
        # Action space
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # SmolVLM backbone
        logging.info(f"Loading SmolVLM from: {config.smolvlm_model_path}")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            config.smolvlm_model_path,
            torch_dtype=torch.float32,  # Use float32 for training stability
            trust_remote_code=True,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.smolvlm_model_path,
            trust_remote_code=True,
        )
        
        # Get SmolVLM hidden size from model config
        # SmolVLM-500M has hidden_size from text_config
        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        logging.info(f"SmolVLM hidden size: {vlm_hidden_size}")

        # DiT/AdaLN mode setting
        self.use_adaln = getattr(config, 'use_adaln', False)
        self.use_hypernet = getattr(config, 'use_hypernet', False)

        # Flow matching action head
        if self.use_hypernet:
            self.transformer = SmolVLMActionTransformerV2(
                hidden_size=config.hidden_size,
                vlm_hidden_size=vlm_hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dim_action=dim_action,
                dim_propio=dim_proprio,
                dim_time=config.dim_time,
                max_len_seq=config.max_len_seq,
                hypernet_rank=getattr(config, 'hypernet_rank', 16),
            )
            logging.info(f"✓ HyperNet mode enabled: rank={getattr(config, 'hypernet_rank', 16)}")
        else:
            self.transformer = SmolVLMActionTransformer(
                hidden_size=config.hidden_size,
                vlm_hidden_size=vlm_hidden_size,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dim_action=dim_action,
                dim_propio=dim_proprio,
                dim_time=config.dim_time,
                max_len_seq=config.max_len_seq,
                use_adaln=self.use_adaln,
            )
            if self.use_adaln:
                logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
            else:
                logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    # ============================= SmolVLM encoder =============================
    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - multi-view images
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
        language_instruction: list[str] | None = None,  # Optional text prompts
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-view images via SmolVLM2.
        
        All views are processed together by SmolVLM, producing unified features.
        No aux_visual_inputs needed - everything goes through VLM.

        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
            
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # Prepare images for SmolVLM - flatten views and filter by mask
        # SmolVLM can handle multiple images as part of multi-image inference
        batch_features = []
        
        for b in range(B):
            # Get valid images for this sample
            valid_mask = image_mask[b].bool()
            valid_images = pixel_values[b][valid_mask]  # [num_valid, C, H, W]
            
            if valid_images.shape[0] == 0:
                raise ValueError("At least one image view must be valid per batch.")
            
            # Convert to PIL images for SmolVLM processor
            pil_images = []
            for img_tensor in valid_images:
                # Denormalize and convert to PIL
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # Assuming normalized with ImageNet stats, denormalize
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Build message for SmolVLM with multiple images
            content = []
            for i, img in enumerate(pil_images):
                content.append({"type": "image", "image": img})
            
            # Add text prompt if provided
            if language_instruction is not None and b < len(language_instruction):
                content.append({"type": "text", "text": language_instruction[b]})
            else:
                content.append({"type": "text", "text": "Describe the robot's observation."})
            
            messages = [{"role": "user", "content": content}]
            
            # Process with SmolVLM
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            # Get encoder outputs (hidden states) instead of generating text
            with torch.no_grad():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Use the last hidden state as features
            # Shape: [1, seq_len, hidden_size]
            hidden_states = outputs.hidden_states[-1]
            batch_features.append(hidden_states.squeeze(0))  # [seq_len, hidden_size]
        
        # Pad to same length and stack
        max_len = max(f.shape[0] for f in batch_features)
        hidden_size = batch_features[0].shape[-1]
        
        padded_features = torch.zeros(B, max_len, hidden_size, device=device, dtype=batch_features[0].dtype)
        for b, feat in enumerate(batch_features):
            padded_features[b, :feat.shape[0]] = feat
        
        return {"vlm_features": padded_features}

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - Already preprocessed
        image_mask: torch.Tensor,           # [B, V]
        input_ids: torch.LongTensor | None = None,  # [B, L] - Pre-tokenized text
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient VLM forward for training - uses FULL VLM to fuse vision and language.
        
        Key improvement: Uses complete VLM forward (vision encoder + language model)
        to get features that fuse visual and linguistic information, rather than
        just using the vision encoder alone.
        
        Pipeline:
          pixel_values → vision_encoder → image_features
                                               ↓
          input_ids → text_embeddings ─────────┤
                                               ↓
                                 [image_feats, text_embeds] (concat)
                                               ↓
                                 language_model forward
                                               ↓
                                 fused VLM features → return
        
        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # ========== Step 1: Get vision features ==========
        # Flatten images: [B, V, C, H, W] -> [B*V, C, H, W]
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        
        # Get valid images
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]
        
        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")
        
        # Encode images through SmolVLM's vision encoder (SigLIP)
        vision_outputs = self.vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get image features and project to LM space
        image_features = vision_outputs.last_hidden_state  # [num_valid, num_patches, vision_hidden]
        
        # Project to language model space using the connector/projector
        if hasattr(self.vlm.model, 'connector'):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            image_features = self.vlm.model.multi_modal_projector(image_features)
        
        # ========== Step 2: Get text embeddings ==========
        # Idefics3 (SmolVLM) uses 'text_model' instead of 'language_model'
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)  # [B, L, D]
        
        # ========== Step 3: Build combined sequence per sample ==========
        # For each sample, concatenate: [image_features_view1, ..., image_features_viewN, text_embeds]
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        
        # Reconstruct image features with batch structure
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        
        # Count valid views per sample for proper concatenation
        valid_per_sample = image_mask.sum(dim=1).int()  # [B]
        
        batch_inputs_embeds = []
        max_seq_len = 0
        
        for b in range(B):
            # Get valid image features for this sample
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid]  # [num_valid, num_patches, D]
            sample_image_feats = sample_image_feats.reshape(-1, hidden_size)  # [num_valid*num_patches, D]
            
            # Get text embeddings for this sample
            sample_text_embeds = text_embeds[b]  # [L, D]
            
            # Concatenate: [image_features, text_embeds]
            combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)  # [T, D]
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])
        
        # ========== Step 4: Pad and stack ==========
        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        
        for b, embeds in enumerate(batch_inputs_embeds):
            seq_len = embeds.shape[0]
            padded_inputs_embeds[b, :seq_len] = embeds
            attention_mask[b, :seq_len] = 1
        
        # ========== Step 5: Forward through text model (Idefics3/SmolVLM) ==========
        # This fuses visual and linguistic information through the full transformer
        lm_outputs = self.vlm.model.text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use the last hidden state as VLM features
        # This now contains fused vision-language representations
        vlm_features = lm_outputs.last_hidden_state  # [B, max_seq_len, D]
        
        return {"vlm_features": vlm_features}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, L] - tokenized language instruction
        image_input: torch.FloatTensor,     # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V]
        proprio: torch.Tensor,              # [B, dim_proprio]
        action: torch.Tensor,               # [B, T=num_actions, D=dim_action]
        t_override: torch.Tensor | None = None,       # 外部传入的时间采样（课程学习）
        next_proprio: torch.Tensor | None = None,     # 辅助世界模型：下一时刻 proprio
        state_loss_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching training.

        1) Time sampling: t ~ Beta(alpha, 1) * 0.999 + 0.001（alpha 由课程学习控制）
        2) Interpolation: x_t = t * noise + (1-t) * actions
        3) Target: velocity u_t = noise - actions
        4) Model predicts v_t, compute MSE(v_t, u_t)
        5) 可选：辅助 state 预测损失（next_proprio 不为 None 时启用）
        """
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = input_ids.shape[0]
        device = input_ids.device

        # 时间采样：优先使用外部传入的 t_override（课程学习），否则用默认 Beta(1.5, 1)
        if t_override is not None:
            t = t_override.to(device)
        else:
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.5, device=device),
                torch.tensor(1.0, device=device)
            )
            t = beta_dist.sample((B,)) * 0.999 + 0.001
        # 对齐 dtype：t 默认 float32，混合精度训练下 action 是 bfloat16，
        # t_expanded * noise 会把 x_t 提升为 float32，导致 action_encoder dtype 不匹配
        t = t.to(dtype=action.dtype)

        # Normalize action and proprio
        if hasattr(self.action_space, 'normalize_action'):
            action_norm = self.action_space.normalize_action(action)
        elif hasattr(self.action_space, 'normalize'):
            action_norm = self.action_space.normalize(action)
        else:
            action_norm = action

        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        # Flow Matching
        noise = torch.randn_like(action_norm)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * action_norm
        u_t = noise - action_norm

        # 辅助世界模型损失：需要 action_feats
        use_state_loss = next_proprio is not None
        if use_state_loss:
            v_t, action_feats = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                t=t,
                proprio=proprio_norm,
                return_features=True,
            )
        else:
            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                t=t,
                proprio=proprio_norm,
            )

        velocity_loss = torch.mean(torch.square(v_t - u_t))
        loss_dict = {"velocity_loss": velocity_loss}

        if use_state_loss:
            state_pred = self.transformer.state_pred_head(action_feats[:, 0, :])  # [B, dim_proprio]
            if hasattr(self.action_space, 'normalize_state'):
                next_proprio_norm = self.action_space.normalize_state(next_proprio.to(device))
            else:
                next_proprio_norm = next_proprio.to(device)
            state_loss = torch.mean(torch.square(state_pred - next_proprio_norm))
            loss_dict["state_loss"] = state_loss_weight * state_loss

        return loss_dict

    # ================================= inference =================================
    @torch.no_grad()
    def encode_static_context(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        static_view_idx: int = 0,
    ) -> torch.Tensor:
        """
        只编码静态视角（agentview）+ 语言，返回缓存特征。
        episode 开始时调用一次，后续每步复用。
        """
        static_images = image_input[:, static_view_idx:static_view_idx + 1, ...]
        static_mask = image_mask[:, static_view_idx:static_view_idx + 1]
        enc = self.forward_vlm_efficient(static_images, static_mask, input_ids)
        return enc["vlm_features"]  # [B, T_static, D]

    @torch.no_grad()
    def encode_dynamic_view(
        self,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        dynamic_view_idx: int = 1,
    ) -> torch.Tensor:
        """
        只编码动态视角（eye_in_hand），每步调用。
        不经过 text_model，只跑 vision_encoder + connector。
        """
        if image_input.dim() == 6:
            if image_input.size(2) == 1:
                image_input = image_input.squeeze(2)
            else:
                image_input = image_input[:, :, 0]

        dyn_img = image_input[:, dynamic_view_idx, ...]  # [B, C, H, W]

        vision_outputs = self.vlm.model.vision_model(
            pixel_values=dyn_img,
            output_hidden_states=True,
            return_dict=True,
        )
        feats = vision_outputs.last_hidden_state  # [B, num_patches, vision_hidden]

        if hasattr(self.vlm.model, 'connector'):
            feats = self.vlm.model.connector(feats)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            feats = self.vlm.model.multi_modal_projector(feats)

        return feats  # [B, num_patches, D]

    @torch.no_grad()
    def generate_actions_with_cache(
        self,
        static_context: torch.Tensor,
        dynamic_feats: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
        adaptive: bool = True,
        cos_threshold: float = 0.97,
        min_steps: int = 2,
    ) -> torch.Tensor:
        """
        使用缓存的静态特征 + 当前动态特征推理，避免重复跑 text_model。
        """
        vlm_features = torch.cat([static_context, dynamic_feats], dim=1)

        B = static_context.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        steps = max(1, int(steps))
        dt = -1.0 / steps
        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        t = 1.0
        v_prev = None
        actual_steps = 0

        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)
            v_t = self.transformer(
                vlm_features=vlm_features,
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
            )
            x_t = x_t + dt * v_t
            t = t + dt
            actual_steps += 1

            if adaptive and v_prev is not None and actual_steps >= min_steps:
                v_flat = v_t.reshape(B, -1)
                v_prev_flat = v_prev.reshape(B, -1)
                cos_sim = torch.nn.functional.cosine_similarity(v_flat, v_prev_flat, dim=-1).mean()
                if cos_sim.item() > cos_threshold:
                    break
            v_prev = v_t

        self.last_actual_steps = actual_steps
        return self.action_space.postprocess(x_t)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
        adaptive: bool = False,
        cos_threshold: float = 0.97,
        min_steps: int = 2,
    ) -> torch.Tensor:
        """
        Flow Matching inference (Euler integration).

        adaptive=True 时：每步计算速度向量余弦相似度，收敛则提前停止。
        """
        self.eval()
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        steps = max(1, int(steps))
        dt = -1.0 / steps

        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        t = 1.0
        v_prev = None
        actual_steps = 0

        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)
            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
            )
            x_t = x_t + dt * v_t
            t = t + dt
            actual_steps += 1

            if adaptive and v_prev is not None and actual_steps >= min_steps:
                v_flat = v_t.reshape(B, -1)
                v_prev_flat = v_prev.reshape(B, -1)
                cos_sim = torch.nn.functional.cosine_similarity(v_flat, v_prev_flat, dim=-1).mean()
                if cos_sim.item() > cos_threshold:
                    break

            v_prev = v_t

        self.last_actual_steps = actual_steps
        return self.action_space.postprocess(x_t)

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """Build FastAPI app for SmolVLM-VLA inference."""
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # Decode images
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload:
                        continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                        
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # Process inputs
                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # Build proprio tensor
                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))

                # Align to model device/dtype
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs["proprio"] = to_model(proprio.unsqueeze(0))

                # Inference
                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """Launch the FastAPI service."""
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)
