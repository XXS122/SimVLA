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

import math
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel, AutoProcessor, AutoModelForImageTextToText
from .transformer_smolvlm import SmolVLMActionTransformer, HistoryEncoder, PhysicsPredicateDecoder
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
        self.use_cross_attn = getattr(config, 'use_cross_attn', True)

        # CVAE 子目标潜变量设置
        self.use_subgoal_vae = getattr(config, 'use_subgoal_vae', False)
        self.kl_weight = getattr(config, 'kl_weight', 0.001)

        # Latent Diffusion Model 设置
        self.use_latent_flow = getattr(config, 'use_latent_flow', True)
        self.latent_flow_steps = getattr(config, 'latent_flow_steps', 5)
        self.latent_fm_weight = getattr(config, 'latent_fm_weight', 1.0)

        # 损失函数设置
        self.use_huber_loss = getattr(config, 'use_huber_loss', False)
        self.huber_delta = getattr(config, 'huber_delta', 1.0)
        self.gripper_weight = getattr(config, 'gripper_weight', 1.0)

        # 时间步采样策略
        self.time_sampling = getattr(config, 'time_sampling', 'beta')

        # HistoryEncoder（GRU 历史感知）
        self.use_history_encoder = getattr(config, 'use_history_encoder', False)
        self.history_seq_len = getattr(config, 'history_seq_len', 4)
        self.switch_loss_weight = getattr(config, 'switch_loss_weight', 0.05)
        if self.use_history_encoder:
            self.history_encoder = HistoryEncoder(
                proprio_dim=dim_proprio,
                hidden_size=getattr(config, 'history_hidden', 128),
                adaln_hidden=config.hidden_size,
            )
            logging.info(f"✓ HistoryEncoder enabled: GRU hidden={getattr(config,'history_hidden',128)}, K={self.history_seq_len}")

        # PhysicsPredicateDecoder（物理谓词嵌入）
        self.use_physics_cot = getattr(config, 'use_physics_cot', False)
        self.physics_weight = getattr(config, 'physics_weight', 0.01)
        if self.use_physics_cot:
            self.physics_decoder = PhysicsPredicateDecoder(
                vlm_hidden=vlm_hidden_size,
                adaln_hidden=config.hidden_size,
            )
            logging.info(f"✓ PhysicsPredicateDecoder enabled: weight={self.physics_weight}")

        # Flow matching action head (SmolVLM version - no aux_visual)
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
            use_cross_attn=self.use_cross_attn,
            use_subgoal_vae=self.use_subgoal_vae,
            subgoal_latent_dim=getattr(config, 'subgoal_latent_dim', 64),
            num_actions=config.num_actions,
        )

        if self.use_adaln:
            logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
        else:
            logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        if self.use_subgoal_vae:
            mode = "LDM (LatentFlowNet)" if self.use_latent_flow else "CVAE prior sampling"
            logging.info(f"✓ SubgoalVAE enabled: z_goal via {mode}")

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
        input_ids: torch.LongTensor,              # [B, L] - tokenized language instruction
        image_input: torch.FloatTensor,           # [B, V, C, H, W]
        image_mask: torch.Tensor,                 # [B, V]
        proprio: torch.Tensor,                    # [B, dim_proprio]
        action: torch.Tensor,                     # [B, T=num_actions, D=dim_action]
        proprio_sequence: torch.Tensor | None = None,  # [B, K, dim_proprio] 历史序列
        physics_labels: torch.Tensor | None = None,    # [B, 5] 弱监督物理谓词标签
        switch_labels: torch.Tensor | None = None,     # [B, 1] gripper 切换剩余步比例
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching training.

        1) Time sampling: t ~ Beta(1.5, 1) * 0.999 + 0.001
        2) Interpolation: x_t = t * noise + (1-t) * actions
        3) Target: velocity u_t = noise - actions
        4) Model predicts v_t, compute MSE(v_t, u_t)
        5) (Optional) Auxiliary losses: physics predicates, GRU switch prediction
        """
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = input_ids.shape[0]
        device = input_ids.device

        # 时间步采样
        if self.time_sampling == 'logit_normal':
            # Logit-Normal(0, 1)：在 t=0.5 附近更均匀，低时间步样本更多
            u = torch.randn(B, device=device)
            t = torch.sigmoid(u) * 0.999 + 0.0005
        elif self.time_sampling == 'cosine':
            # Cosine schedule：偏向 t→0（精细去噪阶段）
            u = torch.rand(B, device=device)
            t = 1.0 - torch.cos(u * math.pi / 2) * 0.999
        else:
            # 默认：Beta(1.5, 1)
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.5, device=device),
                torch.tensor(1.0, device=device)
            )
            t = beta_dist.sample((B,)) * 0.999 + 0.001

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

        # CVAE 子目标潜变量（仅 AdaLN 模式 + use_subgoal_vae 时启用）
        z_goal = None
        kl_loss = None
        latent_fm_loss = None
        if self.use_subgoal_vae and self.use_adaln:
            # detach 防止 freeze_steps 内 VAE 梯度干扰 VLM
            vlm_pooled = enc["vlm_features"].mean(dim=1).detach()
            post_mu, post_log_var = self.transformer.subgoal_vae.encode_posterior(
                vlm_pooled, action_norm
            )
            prior_mu, prior_log_var = self.transformer.subgoal_vae.encode_prior(vlm_pooled)
            # z_0：后验采样，作为 LDM 的目标终点
            z_0 = self.transformer.subgoal_vae.reparameterize(post_mu, post_log_var)
            kl_loss = self.transformer.subgoal_vae.kl_loss(
                post_mu, post_log_var, prior_mu, prior_log_var
            )

            # z 空间 Flow Matching 损失（LDM）
            if self.use_latent_flow:
                # 复用已采样的 t（或单独采样 t_z）
                if self.time_sampling == 'logit_normal':
                    u_z = torch.randn(B, device=device)
                    t_z = torch.sigmoid(u_z) * 0.999 + 0.0005
                elif self.time_sampling == 'cosine':
                    u_z = torch.rand(B, device=device)
                    t_z = 1.0 - torch.cos(u_z * math.pi / 2) * 0.999
                else:
                    t_z = beta_dist.sample((B,)) * 0.999 + 0.001
                z_noise = torch.randn_like(z_0)
                t_z_exp = t_z.view(-1, 1)
                z_t_latent = t_z_exp * z_noise + (1 - t_z_exp) * z_0
                u_z = z_noise - z_0
                v_z = self.transformer.latent_flow_net(z_t_latent, t_z, vlm_pooled)
                latent_fm_loss = torch.mean(torch.square(v_z - u_z))

            # 训练时直接用后验 z_0 注入 AdaLN（不走推理积分路径）
            z_goal = z_0

        # HistoryEncoder：GRU 历史感知条件
        h_cond = None
        switch_loss = None
        if self.use_history_encoder and self.use_adaln:
            if proprio_sequence is not None:
                # 归一化历史序列
                if hasattr(self.action_space, 'normalize_state'):
                    B_seq, K_seq, D_seq = proprio_sequence.shape
                    seq_flat = proprio_sequence.reshape(B_seq * K_seq, D_seq)
                    seq_norm = self.action_space.normalize_state(seq_flat).reshape(B_seq, K_seq, D_seq)
                elif hasattr(self.action_space, 'normalize'):
                    B_seq, K_seq, D_seq = proprio_sequence.shape
                    seq_flat = proprio_sequence.reshape(B_seq * K_seq, D_seq)
                    seq_norm = self.action_space.normalize(seq_flat).reshape(B_seq, K_seq, D_seq)
                else:
                    seq_norm = proprio_sequence
                h_t, h_cond = self.history_encoder(seq_norm)
            else:
                # 无历史时用当前帧单步更新
                proprio_seq_single = proprio_norm.unsqueeze(1)  # [B, 1, D]
                h_t, h_cond = self.history_encoder(proprio_seq_single)

            # 辅助损失：预测 gripper 切换剩余步比例
            if switch_labels is not None:
                switch_pred = self.history_encoder.switch_pred(h_t)  # [B, 1]
                switch_loss = F.huber_loss(switch_pred, switch_labels, delta=0.2)

        # PhysicsPredicateDecoder：物理谓词条件
        physics_cond = None
        physics_loss = None
        if self.use_physics_cot and self.use_adaln:
            vlm_pooled = enc["vlm_features"].mean(dim=1)
            pred_logits, physics_cond = self.physics_decoder(vlm_pooled)
            # 辅助损失：对4个二分类谓词用 BCE，对第3维（height）用 MSE
            if physics_labels is not None:
                loss_binary = F.binary_cross_entropy_with_logits(
                    pred_logits[:, [0, 1, 3, 4]], physics_labels[:, [0, 1, 3, 4]]
                )
                loss_height = F.mse_loss(torch.sigmoid(pred_logits[:, 2]), physics_labels[:, 2])
                physics_loss = loss_binary + 0.1 * loss_height

        # Model prediction (no aux_visual_inputs for SmolVLM)
        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
            z_goal=z_goal,
            h_cond=h_cond,
            physics_cond=physics_cond,
        )

        # 损失函数：Huber 或 MSE，支持 gripper 维度加权
        if self.use_huber_loss:
            per_dim_loss = F.huber_loss(v_t, u_t, delta=self.huber_delta, reduction='none')
        else:
            per_dim_loss = torch.square(v_t - u_t)

        if self.gripper_weight != 1.0:
            weight = torch.ones_like(per_dim_loss)
            weight[..., -1] = self.gripper_weight  # gripper 是最后一维
            per_dim_loss = per_dim_loss * weight

        velocity_loss = per_dim_loss.mean()

        loss_dict = {"velocity_loss": velocity_loss}
        if kl_loss is not None:
            loss_dict["kl_loss"] = kl_loss * self.kl_weight
        if latent_fm_loss is not None:
            loss_dict["latent_fm_loss"] = latent_fm_loss * self.latent_fm_weight
        if switch_loss is not None:
            loss_dict["switch_loss"] = switch_loss * self.switch_loss_weight
        if physics_loss is not None:
            loss_dict["physics_loss"] = physics_loss * self.physics_weight

        return loss_dict

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
        h_state: torch.Tensor | None = None,       # [B, history_hidden] 持久化 GRU 状态
        z_goal_cache: torch.Tensor | None = None,  # [B, latent_dim] 持久化 SubgoalVAE 子目标
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Flow Matching inference (Euler integration).

        Returns
        -------
        actions    : [B, T, D]
        new_h      : [B, history_hidden] | None  — 更新后的 GRU 隐状态
        new_z_goal : [B, latent_dim] | None      — 当前子目标（供下步缓存复用）
        """
        self.eval()
        enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = input_ids.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        # Normalize proprio
        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        # HistoryEncoder：更新 GRU 隐状态
        new_h = None
        h_cond_infer = None
        if self.use_history_encoder and self.use_adaln:
            proprio_seq_single = proprio_norm.unsqueeze(1)  # [B, 1, D]
            new_h, h_cond_infer = self.history_encoder(proprio_seq_single, h_init=h_state)

        # PhysicsPredicateDecoder：物理谓词条件（推理时无监督，仅用预测值）
        physics_cond_infer = None
        if self.use_physics_cot and self.use_adaln:
            vlm_pooled = enc["vlm_features"].mean(dim=1)
            _, physics_cond_infer = self.physics_decoder(vlm_pooled)

        # Euler integration
        steps = max(1, int(steps))
        dt = -1.0 / steps

        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        t = 1.0

        # SubgoalVAE：生成或复用子目标 z_goal
        # 复用缓存可保证同一 episode 内子目标稳定，避免每步重采样导致目标跳变
        z_goal = None
        new_z_goal = None
        if self.use_subgoal_vae and self.use_adaln:
            if z_goal_cache is not None:
                # 复用 episode 首步生成的 z_goal，保持目标一致性
                z_goal = z_goal_cache.to(device=device, dtype=dtype)
            else:
                vlm_pooled = enc["vlm_features"].mean(dim=1)
                if self.use_latent_flow:
                    # LDM：从噪声 z_T 积分到 z_0（多样本平均降低方差）
                    latent_dim = self.transformer.subgoal_vae.latent_dim
                    n_samples = 4  # 多样本平均降低随机性
                    z_accum = torch.zeros(B, latent_dim, device=device, dtype=dtype)
                    dt_z = -1.0 / self.latent_flow_steps
                    for _ in range(n_samples):
                        z_t_latent = torch.randn(B, latent_dim, device=device, dtype=dtype)
                        t_z = 1.0
                        while t_z > -dt_z / 2:
                            t_z_tensor = torch.full((B,), t_z, device=device, dtype=dtype)
                            v_z = self.transformer.latent_flow_net(z_t_latent, t_z_tensor, vlm_pooled)
                            z_t_latent = z_t_latent + dt_z * v_z
                            t_z += dt_z
                        z_accum = z_accum + z_t_latent
                    z_goal = z_accum / n_samples
                else:
                    # 直接使用先验均值，不加噪声（推理时确定性更好）
                    prior_mu, _ = self.transformer.subgoal_vae.encode_prior(vlm_pooled)
                    z_goal = prior_mu
            new_z_goal = z_goal

        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)

            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
                z_goal=z_goal,
                h_cond=h_cond_infer,
                physics_cond=physics_cond_infer,
            )

            x_t = x_t + dt * v_t
            t = t + dt

        return self.action_space.postprocess(x_t), new_h, new_z_goal

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

                # Inference (generate_actions 返回三元组，取第一项)
                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps)[0].squeeze(0).float().cpu().numpy()
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
