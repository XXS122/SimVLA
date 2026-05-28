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


# --------------------------- Main Model (SmolVLM Version) ---------------------------------------

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.dim_propio = dim_propio
        self.use_adaln = use_adaln

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

        # 辅助世界模型头：预测执行动作后的下一个 proprio 状态
        self.state_pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, dim_propio),
        )

        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,  # [B, T_vlm, D] - unified features from SmolVLM
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for SmolVLM Action Transformer.

        return_features=True 时额外返回动作 token 特征（用于辅助世界模型损失）。
        """
        if self.use_adaln:
            return self._forward_adaln(vlm_features, action_with_noise, proprio, t, return_features)
        else:
            return self._forward_concat(vlm_features, action_with_noise, proprio, t, return_features)
    
    def _forward_concat(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Concat mode forward pass.
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

        action_feats = self.norm(x[:, :num_actions])  # [B, T_action, H]
        velocity = self.action_decoder(action_feats)   # [B, T_action, dim_action]

        if return_features:
            return velocity, action_feats
        return velocity
    
    def _forward_adaln(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        DiT/AdaLN mode forward pass.
        """
        B, num_actions = action_with_noise.shape[:2]

        # ========== 1. Build global condition c ==========
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_proj(t_emb)  # [B, H]

        vlm_cond = self.vlm_cond_proj(vlm_features.mean(dim=1))  # [B, H]
        proprio_cond = self.proprio_proj(proprio)  # [B, H]
        c = t_emb + vlm_cond + proprio_cond  # [B, H]

        # ========== 2. Encode action sequence ==========
        x = self.action_encoder(action_with_noise)  # [B, T_action, H]
        x = x + self.pos_emb[:, :num_actions, :]

        # ========== 3. DiT Blocks with AdaLN ==========
        for block in self.blocks:
            x = block(x, c)

        # ========== 4. Final Layer with AdaLN ==========
        velocity = self.final_layer(x, c)  # [B, T_action, dim_action]

        if return_features:
            return velocity, x
        return velocity


# ================================ HyperNet ===================================

class HyperNet(nn.Module):
    """
    Hypernetwork: maps task_vec (mean-pooled VLM features) to per-layer low-rank
    weight deltas for the action transformer MLP layers.

    Architecture: shared trunk compresses task_vec to a compact task_emb,
    then per-layer heads map task_emb to low-rank factors A and B.

    For each transformer block:
      ΔW_fc1 = B_fc1 @ A_fc1  [M, H]   (rank-r approximation)
      ΔW_fc2 = B_fc2 @ A_fc2  [H, M]

    Parameter budget (rank=4, task_emb_dim=32, H=768, M=3072):
      trunk:  ~0.47M
      heads:  ~11.8M (12 layers × ~0.98M per head)
      total:  ~12.3M new parameters
    """

    def __init__(
        self,
        vlm_hidden_size: int = 576,
        hidden_size: int = 768,
        num_layers: int = 12,
        rank: int = 4,
        mlp_ratio: float = 4.0,
        task_emb_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.hidden_size = hidden_size
        self.mlp_hidden = int(hidden_size * mlp_ratio)
        self.task_emb_dim = task_emb_dim

        H, M, r, D = hidden_size, self.mlp_hidden, rank, task_emb_dim
        # A factors: A_fc1[r,H] + A_fc2[r,M]
        # B factors: B_fc1[M,r] + B_fc2[H,r]
        a_dim = rank * (hidden_size + self.mlp_hidden)
        b_dim = rank * (hidden_size + self.mlp_hidden)

        # Shared trunk: compress task_vec → compact task embedding
        self.trunk = nn.Sequential(
            nn.Linear(vlm_hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, task_emb_dim),
        )

        # Per-layer A-heads: random init so gradients flow from the start
        self.heads_A = nn.ModuleList([
            nn.Linear(task_emb_dim, a_dim) for _ in range(num_layers)
        ])
        # Per-layer B-heads: zero init so delta=B@A=0 at init (stable start)
        self.heads_B = nn.ModuleList([
            nn.Linear(task_emb_dim, b_dim) for _ in range(num_layers)
        ])
        for head in self.heads_B:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, task_vec: torch.Tensor):
        """
        Args:
            task_vec: [B, vlm_hidden_size]
        Returns:
            list of (A_fc1, B_fc1, A_fc2, B_fc2) tuples, length = num_layers
              A_fc1: [B, r, hidden_size]
              B_fc1: [B, mlp_hidden, r]
              A_fc2: [B, r, mlp_hidden]
              B_fc2: [B, hidden_size, r]

            HyperNetMlp consumes these factors directly via LoRA-style
            decomposed matmul, avoiding the [B, M, H] delta materialization.
        """
        feat = self.trunk(task_vec)  # [B, task_emb_dim]
        r, H, M = self.rank, self.hidden_size, self.mlp_hidden
        factors = []
        for head_A, head_B in zip(self.heads_A, self.heads_B):
            raw_A = head_A(feat)  # [B, r*(H+M)]
            raw_B = head_B(feat)  # [B, r*(H+M)]
            A_fc1 = raw_A[:, :r * H].view(-1, r, H)
            A_fc2 = raw_A[:, r * H:].view(-1, r, M)
            B_fc1 = raw_B[:, :M * r].view(-1, M, r)
            B_fc2 = raw_B[:, M * r:].view(-1, H, r)
            factors.append((A_fc1, B_fc1, A_fc2, B_fc2))
        return factors

    @staticmethod
    def materialize_delta(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """诊断用：从因子展开成完整 delta 矩阵 B @ A。"""
        return B @ A


class HyperNetMlp(nn.Module):
    """MLP that accepts per-sample weight deltas from HyperNet."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        bias: bool | Tuple[bool, bool] = True,
        drop: float | Tuple[float, float] = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(
        self,
        x: torch.Tensor,
        A_fc1: torch.Tensor | None = None,
        B_fc1: torch.Tensor | None = None,
        A_fc2: torch.Tensor | None = None,
        B_fc2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        LoRA-style efficient forward:
          x @ (W + B@A)^T  =  x @ W^T  +  (x @ A^T) @ B^T

        基础路径用 F.linear（cuBLAS GEMM），LoRA 修正路径用两次小 einsum，
        避免拼出 [B, M, H] 的完整权重矩阵。

        Args:
            x:     [B, T, H]
            A_fc1: [B, r, H]    B_fc1: [B, M, r]
            A_fc2: [B, r, M]    B_fc2: [B, H, r]
        """
        # ---- fc1: x @ (W_fc1 + B_fc1@A_fc1)^T ----
        h = self.fc1(x)  # [B, T, M] 基础路径
        if A_fc1 is not None and B_fc1 is not None:
            # x [B,T,H] @ A_fc1^T [B,H,r] -> [B, T, r]
            lora = torch.einsum("bth,brh->btr", x, A_fc1)
            # [B,T,r] @ B_fc1^T [B,r,M] -> [B, T, M]
            lora = torch.einsum("btr,bmr->btm", lora, B_fc1)
            h = h + lora

        h = self.act(h)
        h = self.drop1(h)
        h = self.norm(h)

        # ---- fc2: h @ (W_fc2 + B_fc2@A_fc2)^T ----
        out = self.fc2(h)  # [B, T, H] 基础路径
        if A_fc2 is not None and B_fc2 is not None:
            # h [B,T,M] @ A_fc2^T [B,M,r] -> [B, T, r]
            lora = torch.einsum("btm,brm->btr", h, A_fc2)
            # [B,T,r] @ B_fc2^T [B,r,H] -> [B, T, H]
            lora = torch.einsum("btr,bhr->bth", lora, B_fc2)
            out = out + lora

        return self.drop2(out)


class HyperNetTransformerBlock(nn.Module):
    """TransformerBlock whose MLP accepts per-sample weight deltas."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = HyperNetMlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
        )

    def forward(
        self,
        x: torch.Tensor,
        A_fc1: torch.Tensor | None = None,
        B_fc1: torch.Tensor | None = None,
        A_fc2: torch.Tensor | None = None,
        B_fc2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x), A_fc1, B_fc1, A_fc2, B_fc2)
        return x


class SmolVLMActionTransformerV2(nn.Module):
    """
    HyperNet Policy: action transformer whose MLP weights are dynamically
    modulated by task-specific low-rank deltas generated from VLM features.

    Two parallel conditioning paths:
      1. Concat path  — VLM tokens concatenated to action sequence (token-level)
      2. HyperNet path — VLM mean-pooled → weight deltas (parameter-level)

    Interface is identical to SmolVLMActionTransformer (concat mode).
    """

    def __init__(
        self,
        hidden_size: int = 768,
        vlm_hidden_size: int = 576,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_action: int = 26,
        dim_propio: int = 21,
        dim_time: int = 32,
        max_len_seq: int = 1024,
        use_adaln: bool = False,  # ignored, kept for API compatibility
        hypernet_rank: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.dim_propio = dim_propio

        # HyperNet: task_vec → per-layer weight deltas
        self.hypernet = HyperNet(
            vlm_hidden_size=vlm_hidden_size,
            hidden_size=hidden_size,
            num_layers=depth,
            rank=hypernet_rank,
            mlp_ratio=mlp_ratio,
            task_emb_dim=32,
        )

        # Transformer blocks with HyperNet-modulated MLP
        self.blocks = nn.ModuleList([
            HyperNetTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Concat path: project VLM tokens into action transformer space
        self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm = nn.LayerNorm(hidden_size)

        action_input_dim = dim_action + dim_time + dim_propio
        self.action_encoder = nn.Linear(action_input_dim, hidden_size)
        self.action_decoder = nn.Linear(hidden_size, dim_action)

        # Auxiliary world model head (same as original)
        self.state_pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, dim_propio),
        )

        self.apply(basic_init)
        # Re-zero HyperNet B-heads after basic_init so delta=B@A=0 at init
        for head in self.hypernet.heads_B:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Diagnostic caches (detached, no grad)
        # last_factors: list of (A_fc1, B_fc1, A_fc2, B_fc2) tuples
        # 用 factors 比 deltas 省 ~1000x 显存（rank=4 时）
        self.last_task_vec: torch.Tensor | None = None
        self.last_factors: list | None = None

    def forward(
        self,
        vlm_features: torch.Tensor,   # [B, T_vlm, D_vlm]
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        B, num_actions = action_with_noise.shape[:2]

        # ---- HyperNet path: 生成 LoRA 因子 (A, B)，不展开成完整 delta ----
        task_vec = vlm_features.mean(dim=1)  # [B, D_vlm]
        factors = self.hypernet(task_vec)    # list of (A_fc1, B_fc1, A_fc2, B_fc2)

        # 诊断缓存（仅存因子，detached）
        self.last_task_vec = task_vec.detach()
        self.last_factors = [tuple(f.detach() for f in tup) for tup in factors]

        # ---- Encode action tokens ----
        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])
        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens)  # [B, T_action, H]

        # ---- Concat path: append projected VLM tokens ----
        x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}.")
        x = x + self.pos_emb[:, :seq_len, :]

        # ---- Transformer blocks with LoRA-style weight modulation ----
        for block, (A1, B1, A2, B2) in zip(self.blocks, factors):
            x = block(x, A1, B1, A2, B2)

        action_feats = self.norm(x[:, :num_actions])   # [B, T_action, H]
        velocity = self.action_decoder(action_feats)    # [B, T_action, dim_action]

        if return_features:
            return velocity, action_feats
        return velocity


__all__ = [
    "SmolVLMActionTransformer",
    "SmolVLMActionTransformerV2",
    "HyperNet",
    "HyperNetMlp",
    "HyperNetTransformerBlock",
    "TransformerBlock",
    "DiTBlock",
    "FinalLayer",
    "Attention",
    "Mlp",
    "timestep_embedding",
]
