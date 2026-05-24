"""
SmolVLM Action Transformer

Action Transformer specifically designed for SmolVLM-VLA.
Key difference from the original transformer:
  - No aux_visual_inputs: all views are processed together by SmolVLM
  - VLM outputs a single unified feature for all views
  - Simpler architecture: x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

Extensions:
  - CTAF (use_ctaf=True): output Fourier coefficients instead of discrete action tokens,
    guaranteeing C∞ smooth trajectories at zero extra parameter cost.
  - PSCA (use_psca=True): each MLP block gets LoRA adapters (B=0 init → delta=0 at start),
    enabling inference-time physical self-consistency adaptation via adapt_step().
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


# ========================= CTAF utility ======================================

def query_fourier(coeff: torch.Tensor, T: int, num_freqs: int) -> torch.Tensor:
    """
    Reconstruct a trajectory from Fourier coefficients at T evenly-spaced points.

    Layout of coeff along dim-1:
      index 0        : DC component  (c₀)
      index 2k-1     : cos coefficient for frequency k  (k = 1 … num_freqs-1)
      index 2k       : sin coefficient for frequency k

    a(τ) = c₀ + Σₖ₌₁^{M-1} [ c_cos_k · cos(2πkτ) + c_sin_k · sin(2πkτ) ]

    Args:
        coeff:     [B, 2*num_freqs-1, D]
        T:         number of output time steps
        num_freqs: M (total frequency count including DC)
    Returns:
        [B, T, D]
    """
    device, dtype = coeff.device, coeff.dtype
    tau = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)  # [T]
    n_coeff = 2 * num_freqs - 1

    # Build Fourier basis matrix [T, n_coeff]
    basis = torch.empty(T, n_coeff, device=device, dtype=dtype)
    basis[:, 0] = 1.0
    for k in range(1, num_freqs):
        basis[:, 2 * k - 1] = torch.cos(2.0 * math.pi * k * tau)
        basis[:, 2 * k]     = torch.sin(2.0 * math.pi * k * tau)

    return torch.einsum("tn,bnd->btd", basis, coeff)  # [B, T, D]


# ========================= PSCA building blocks ==============================

class PSCAMlp(nn.Module):
    """
    MLP with per-layer LoRA adapters for Physical Self-Consistency Adaptation.

    LoRA delta: ΔW = B @ A  (initialized to 0 because B = 0 at start).
    During training the adapters are optimized jointly with the base weights.
    During inference adapt_step() updates only A/B params via the physical
    consistency error, leaving the base weights (fc1/fc2) unchanged.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float | Tuple[float, float] = 0.0,
        psca_rank: int = 8,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = _to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

        # LoRA: A uses kaiming init, B is zero → delta = B@A = 0 at start
        self.lora_A1 = nn.Parameter(torch.empty(psca_rank, in_features))
        self.lora_B1 = nn.Parameter(torch.zeros(hidden_features, psca_rank))
        self.lora_A2 = nn.Parameter(torch.empty(psca_rank, hidden_features))
        self.lora_B2 = nn.Parameter(torch.zeros(out_features, psca_rank))
        nn.init.kaiming_uniform_(self.lora_A1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A2, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fc1 + LoRA delta
        h = self.fc1(x) + F.linear(F.linear(x, self.lora_A1), self.lora_B1)
        h = self.act(h)
        h = self.drop1(h)
        # fc2 + LoRA delta
        out = self.fc2(h) + F.linear(F.linear(h, self.lora_A2), self.lora_B2)
        return self.drop2(out)


class PSCATransformerBlock(nn.Module):
    """TransformerBlock (pre-LN) whose MLP has PSCA LoRA adapters."""

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, psca_rank: int = 8
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1)
        self.mlp = PSCAMlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            drop=0.1,
            psca_rank=psca_rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


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

    Supports four concat-mode variants (use_adaln=False):
      use_ctaf=False, use_psca=False : original architecture
      use_ctaf=True,  use_psca=False : CTAF — Fourier coefficient decoder, C∞ smooth output
      use_ctaf=False, use_psca=True  : PSCA — LoRA adapters in each MLP block
      use_ctaf=True,  use_psca=True  : CTAF + PSCA combined

    AdaLN mode (use_adaln=True) is unchanged — CTAF/PSCA do not apply there.
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
        use_adaln: bool = False,
        # --- CTAF ---
        use_ctaf: bool = False,
        num_fourier_freqs: int = 5,
        # --- PSCA ---
        use_psca: bool = False,
        psca_rank: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dim_action = dim_action
        self.dim_time = dim_time
        self.dim_propio = dim_propio
        self.use_adaln = use_adaln
        self.use_ctaf = use_ctaf
        self.num_fourier_freqs = num_fourier_freqs
        self.use_psca = use_psca

        if use_adaln:
            # ========== DiT Mode: AdaLN (CTAF/PSCA not applied) ==========
            self.blocks = nn.ModuleList(
                [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
            )
            self.time_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.vlm_cond_proj = nn.Linear(vlm_hidden_size, hidden_size)
            self.proprio_proj = nn.Linear(dim_propio, hidden_size)
            self.action_encoder = nn.Linear(dim_action, hidden_size)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)
            self.final_layer = FinalLayer(hidden_size, dim_action)
        else:
            # ========== Concat Mode ==========
            # Choose block type based on PSCA flag
            if use_psca:
                self.blocks = nn.ModuleList([
                    PSCATransformerBlock(hidden_size, num_heads, mlp_ratio, psca_rank)
                    for _ in range(depth)
                ])
            else:
                self.blocks = nn.ModuleList([
                    TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                    for _ in range(depth)
                ])

            self.vlm_proj = nn.Linear(vlm_hidden_size, hidden_size)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len_seq, hidden_size), requires_grad=True)
            nn.init.normal_(self.pos_emb, std=0.02)
            self.norm = nn.LayerNorm(hidden_size)

            action_input_dim = dim_action + dim_time + dim_propio
            self.action_encoder = nn.Linear(action_input_dim, hidden_size)

            # Output decoder: CTAF (Fourier coefficients) or standard per-token
            if use_ctaf:
                n_coeff = 2 * num_fourier_freqs - 1
                self.coeff_decoder = nn.Linear(hidden_size, n_coeff * dim_action)
            else:
                self.action_decoder = nn.Linear(hidden_size, dim_action)

        # Auxiliary world-model head: predicts next proprio state.
        # Used for training with state_loss and for PSCA's self-supervised signal.
        self.state_pred_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, dim_propio),
        )

        self.apply(basic_init)

    def forward(
        self,
        vlm_features: torch.Tensor,  # [B, T_vlm, D]
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        if self.use_adaln:
            return self._forward_adaln(vlm_features, action_with_noise, proprio, t)
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
        B, num_actions = action_with_noise.shape[:2]

        # Encode (action + proprio + time) → tokens
        time_emb = timestep_embedding(t, self.dim_time)
        time_tokens = time_emb.unsqueeze(1).expand(B, num_actions, self.dim_time)
        proprio_tokens = proprio.unsqueeze(1).expand(B, num_actions, proprio.shape[-1])

        action_tokens = torch.cat([action_with_noise, proprio_tokens, time_tokens], dim=-1)
        x = self.action_encoder(action_tokens)  # [B, T_action, H]

        # Concat VLM tokens
        x = torch.cat([x, self.vlm_proj(vlm_features)], dim=1)

        seq_len = x.shape[1]
        if seq_len > self.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len_seq={self.pos_emb.shape[1]}.")
        x = x + self.pos_emb[:, :seq_len, :]

        for block in self.blocks:
            x = block(x)

        action_feats = self.norm(x[:, :num_actions])  # [B, T_action, H]

        if self.use_ctaf:
            # Pool action features → Fourier coefficients → query at T time points
            feat = action_feats.mean(dim=1)  # [B, H]
            n_coeff = 2 * self.num_fourier_freqs - 1
            coeff = self.coeff_decoder(feat).view(B, n_coeff, self.dim_action)
            velocity = query_fourier(coeff, num_actions, self.num_fourier_freqs)
        else:
            velocity = self.action_decoder(action_feats)  # [B, T_action, D]

        if return_features:
            return velocity, action_feats
        return velocity

    def _forward_adaln(
        self,
        vlm_features: torch.Tensor,
        action_with_noise: torch.Tensor,
        proprio: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, num_actions = action_with_noise.shape[:2]

        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_proj(t_emb)
        vlm_cond = self.vlm_cond_proj(vlm_features.mean(dim=1))
        proprio_cond = self.proprio_proj(proprio)
        c = t_emb + vlm_cond + proprio_cond

        x = self.action_encoder(action_with_noise)
        x = x + self.pos_emb[:, :num_actions, :]

        for block in self.blocks:
            x = block(x, c)

        return self.final_layer(x, c)


__all__ = [
    "SmolVLMActionTransformer",
    "TransformerBlock",
    "PSCATransformerBlock",
    "PSCAMlp",
    "DiTBlock",
    "FinalLayer",
    "Attention",
    "Mlp",
    "timestep_embedding",
    "query_fourier",
]
