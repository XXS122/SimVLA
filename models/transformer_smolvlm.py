"""
SmolVLM Action Transformer

Action Transformer specifically designed for SmolVLM-VLA.
Key difference from the original transformer:
  - No aux_visual_inputs: all views are processed together by SmolVLM
  - VLM outputs a single unified feature for all views
  - Simpler architecture: x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
"""

from __future__ import annotations

import math
from functools import partial
from typing import Final, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- Small utils ----------------------------------

def _to_2tuple(x) -> Tuple:
    """Minimal replacement for timm.layers.to_2tuple."""
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        t = tuple(x)
        return (t[0], t[1]) if len(t) >= 2 else (t[0], t[0])
    return (x, x)


def _has_sdp_attention() -> bool:
    """Check if we can use PyTorch fused scaled_dot_product_attention."""
    return hasattr(F, "scaled_dot_product_attention")


# ---------------------------------- MLP --------------------------------------

class Mlp(nn.Module):
    """MLP used in ViT-style blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# -------------------------------- Attention ----------------------------------

class Attention(nn.Module):
    """Multi-Head Self-Attention with optional fused SDPA fallback."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = _has_sdp_attention()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------------- Utilities -----------------------------------

def basic_init(module: nn.Module) -> None:
    """Apply basic initialization to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 100) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ------------------------------- Core Layers ----------------------------------

class TransformerBlock(nn.Module):
    """Standard Transformer block (pre-LN)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------- DiT Layers (AdaLN) ----------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """DiT Block with Adaptive Layer Normalization (AdaLN)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation_params.chunk(6, dim=-1)
        )
        
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm)
        
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    """DiT Final Layer with AdaLN."""

    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# --------------------------- SubgoalVAE (CVAE 子目标潜变量) ---------------------------------------

class SubgoalVAE(nn.Module):
    """
    条件 VAE，将 VLM 特征（+ 训练时的动作块）编码为子目标潜变量 z_goal。

    训练时：z ~ q(z | vlm_pooled, action_chunk)  — 后验，利用未来动作信息
    推理时：z ~ p(z | vlm_pooled)               — 先验，仅依赖当前视觉-语言状态

    z_goal 作为额外条件注入 AdaLN 动作 Transformer，使模型能够在潜在空间中
    表示子任务的多模态分布，对长程任务中的阶段切换更鲁棒。
    """

    def __init__(
        self,
        vlm_hidden_size: int,
        action_dim: int,
        num_actions: int,
        latent_dim: int = 64,
        action_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        action_flat_dim = action_dim * num_actions

        # 先验网络：p(z | vlm_pooled)
        self.prior_net = nn.Sequential(
            nn.Linear(vlm_hidden_size, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim * 2),
        )

        # 后验网络：q(z | vlm_pooled, action_chunk)
        self.posterior_net = nn.Sequential(
            nn.Linear(vlm_hidden_size + action_flat_dim, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim * 2),
        )

        self.action_dropout = nn.Dropout(action_dropout)

        # 初始化：输出层接近零，使训练初期 z 接近标准正态
        nn.init.constant_(self.prior_net[-1].weight, 0)
        nn.init.constant_(self.prior_net[-1].bias, 0)
        nn.init.constant_(self.posterior_net[-1].weight, 0)
        nn.init.constant_(self.posterior_net[-1].bias, 0)

    def encode_prior(self, vlm_pooled: torch.Tensor):
        """从先验分布编码：p(z | vlm_pooled)。返回 (mu, log_var)，各 [B, latent_dim]。"""
        params = self.prior_net(vlm_pooled)
        return params.chunk(2, dim=-1)

    def encode_posterior(self, vlm_pooled: torch.Tensor, action_chunk: torch.Tensor):
        """从后验分布编码：q(z | vlm_pooled, action_chunk)。返回 (mu, log_var)，各 [B, latent_dim]。"""
        B = vlm_pooled.shape[0]
        action_flat = self.action_dropout(action_chunk.reshape(B, -1))
        x = torch.cat([vlm_pooled, action_flat], dim=-1)
        params = self.posterior_net(x)
        return params.chunk(2, dim=-1)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """重参数化采样：z = mu + eps * std。"""
        std = torch.exp(0.5 * log_var.clamp(-10, 10))
        return mu + std * torch.randn_like(std)

    def kl_loss(
        self,
        post_mu: torch.Tensor,
        post_log_var: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_log_var: torch.Tensor,
        free_bits: float = 0.5,
    ) -> torch.Tensor:
        """
        KL(q || p) 散度，使用 Free Bits 防止 KL 崩溃。

        Free Bits：对每个潜变量维度，KL 低于 free_bits 时不计入梯度，
        避免后验过早退化为先验。
        """
        prior_var = prior_log_var.exp().clamp(min=1e-6)
        post_var = post_log_var.exp().clamp(min=1e-6)
        kl_per_dim = 0.5 * (
            prior_log_var - post_log_var
            + (post_var + (post_mu - prior_mu) ** 2) / prior_var
            - 1.0
        )
        # Free Bits：每维 KL 至少为 free_bits
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.mean()


# --------------------------- LatentFlowNet（z 空间 Flow Matching）-------------------------------

class LatentFlowNet(nn.Module):
    """
    轻量 MLP，在 latent_dim 维的 z 空间做 Flow Matching。

    以 vlm_pooled 为条件，从噪声 z_T 逐步积分到子目标潜变量 z_0。
    与主干动作 Flow Matching 形成层次化双层扩散结构：
      - z 空间（64维）：LatentFlowNet，5步 Euler 积分
      - 动作空间（7维）：SmolVLMActionTransformer，10步 Euler 积分

    训练时：z_0 来自 SubgoalVAE 后验，LatentFlowNet 学习从噪声到 z_0 的速度场
    推理时：从 z_T ~ N(0,I) 出发，Euler 积分得到 z_goal，注入 AdaLN 条件
    """

    def __init__(
        self,
        latent_dim: int = 64,
        vlm_hidden_size: int = 576,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # 将 vlm_pooled 压缩到 latent_dim，与 z_t 和 t_emb 维度对齐
        self.vlm_proj = nn.Linear(vlm_hidden_size, latent_dim)

        # 速度场网络：输入 [z_t, t_emb, vlm_cond] 拼接
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

        # 输出层初始化为零，训练初期不干扰 CVAE 收敛
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, vlm_pooled: torch.Tensor) -> torch.Tensor:
        """
        预测 z 空间速度场。

        Parameters
        ----------
        z_t       : [B, latent_dim]  — 当前时刻的 z
        t         : [B]              — 时间步（0~1）
        vlm_pooled: [B, vlm_hidden_size] — VLM 全局特征（条件）

        Returns
        -------
        v_z : [B, latent_dim]  — 预测速度场
        """
        t_emb = timestep_embedding(t, self.latent_dim)          # [B, latent_dim]
        vlm_cond = self.vlm_proj(vlm_pooled)                    # [B, latent_dim]
        x = torch.cat([z_t, t_emb, vlm_cond], dim=-1)          # [B, latent_dim*3]
        return self.net(x)                                       # [B, latent_dim]


class SmolVLMActionTransformer(nn.Module):
    """
    Flow Matching Transformer for action prediction - SmolVLM Version.

    Key difference from ActionTransformer:
      - No aux_visual_inputs: SmolVLM processes all views together
      - Simpler forward: x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
      - Only one visual input stream
    
    Supports two modes:
    - Concat mode (use_adaln=False): Original architecture
    - AdaLN mode (use_adaln=True): DiT style conditioning
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vlm_hidden_size: int = 576,  # Will be overridden by actual model config
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_action: int = 26,
        dim_propio: int = 21,
        dim_time: int = 32,
        max_len_seq: int = 1024,
        use_adaln: bool = False,
        use_subgoal_vae: bool = False,
        subgoal_latent_dim: int = 64,
        num_actions: int = 10,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.dim_propio = dim_propio
        self.use_adaln = use_adaln
        self.use_subgoal_vae = use_subgoal_vae

        if use_adaln:
            # ========== DiT Mode: AdaLN ==========
            self.blocks = nn.ModuleList(
                [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # Condition encoders
            self.time_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            # VLM pooling projection (no aux_visual needed)
            self.vlm_cond_proj = nn.Linear(vlm_hidden_size, hidden_size)
            # Proprio projection
            self.proprio_proj = nn.Linear(dim_propio, hidden_size)

            # Action encoder
            self.action_encoder = nn.Linear(dim_action, hidden_size)

            # Position encoding
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)

            # Final layer
            self.final_layer = FinalLayer(hidden_size, dim_action)

            # CVAE 子目标潜变量（仅 AdaLN 模式支持）
            if use_subgoal_vae:
                self.subgoal_vae = SubgoalVAE(
                    vlm_hidden_size=vlm_hidden_size,
                    action_dim=dim_action,
                    num_actions=num_actions,
                    latent_dim=subgoal_latent_dim,
                )
                self.subgoal_proj = nn.Linear(subgoal_latent_dim, hidden_size)
                # z 空间 Flow Matching 网络（LDM）
                self.latent_flow_net = LatentFlowNet(
                    latent_dim=subgoal_latent_dim,
                    vlm_hidden_size=vlm_hidden_size,
                )
        else:
            # ========== Concat Mode: Original architecture ==========
            self.blocks = nn.ModuleList(
                [TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )

            # VLM projection only (no aux_visual_proj needed for SmolVLM)
            self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)

            self.norm = nn.LayerNorm(hidden_size)
            
            # Action encoder/decoder
            action_input_dim = dim_action + dim_time + dim_propio
            self.action_encoder = nn.Linear(action_input_dim, hidden_size)
            self.action_decoder = nn.Linear(hidden_size, dim_action)

        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,  # [B, T_vlm, D] - unified features from SmolVLM
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        z_goal: torch.Tensor | None = None,  # [B, latent_dim] - 子目标潜变量（可选）
    ) -> torch.Tensor:
        """
        Forward pass for SmolVLM Action Transformer.

        Inputs
        ------
        vlm_features : [B, T_vlm, D] - Unified features from SmolVLM (all views processed together)
        action_with_noise : [B, T_action, dim_action]
        proprio : [B, dim_proprio]
        t : [B]
        z_goal : [B, latent_dim] - 子目标潜变量，仅 AdaLN 模式使用（可选）

        Returns
        -------
        Tensor: Predicted velocity, [B, T_action, dim_action]
        """
        if self.use_adaln:
            return self._forward_adaln(vlm_features, action_with_noise, proprio, t, z_goal=z_goal)
        else:
            return self._forward_concat(vlm_features, action_with_noise, proprio, t)
    
    def _forward_concat(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concat mode forward pass.
        
        Simplified: x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)
        No aux_visual_inputs needed.
        """
        B, num_actions = action_with_noise.shape[:2]

        # Encode (action + proprio + time) → tokens
        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens)  # [B, T_action, H]

        # Project VLM features and concatenate (no aux_visual needed)
        x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

        # Add positional embeddings
        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}."
            )
        x = x + self.pos_emb[:, :seq_len, :]

        # Transformer backbone
        for block in self.blocks:
            x = block(x)

        # Decode only the action segment
        return self.action_decoder(self.norm(x[:, :num_actions]))
    
    def _forward_adaln(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        z_goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        DiT/AdaLN mode forward pass.

        Conditions (time, vlm, proprio, z_goal) injected via AdaLN.
        No aux_visual needed for SmolVLM.
        """
        B, num_actions = action_with_noise.shape[:2]

        # ========== 1. Build global condition c ==========
        # Time embedding
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_proj(t_emb)  # [B, H]

        # VLM condition: Global Average Pooling
        vlm_cond = self.vlm_cond_proj(vlm_features.mean(dim=1))  # [B, H]

        # Proprio condition
        proprio_cond = self.proprio_proj(proprio)  # [B, H]

        # Fuse all conditions
        c = t_emb + vlm_cond + proprio_cond  # [B, H]

        # 子目标潜变量条件（若启用）
        if z_goal is not None and self.use_subgoal_vae:
            c = c + self.subgoal_proj(z_goal)  # [B, H]

        # ========== 2. Encode action sequence ==========
        x = self.action_encoder(action_with_noise)  # [B, T_action, H]

        # Add position encoding
        x = x + self.pos_emb[:, :num_actions, :]

        # ========== 3. DiT Blocks with AdaLN ==========
        for block in self.blocks:
            x = block(x, c)

        # ========== 4. Final Layer with AdaLN ==========
        return self.final_layer(x, c)


__all__ = [
    "SmolVLMActionTransformer",
    "SubgoalVAE",
    "LatentFlowNet",
    "TransformerBlock",
    "DiTBlock",
    "FinalLayer",
    "Attention",
    "Mlp",
    "timestep_embedding",
]
