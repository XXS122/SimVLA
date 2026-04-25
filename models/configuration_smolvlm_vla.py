"""
SmolVLM-VLA Configuration

Configuration class for SmolVLM-500M-Instruct based VLA model.
Uses SmolVLM as the vision-language backbone instead of Florence2.
"""

import os
from transformers.configuration_utils import PretrainedConfig


class SmolVLMVLAConfig(PretrainedConfig):
    """
    Configuration class for the **SmolVLM-VLA (SmolVLM Vision-Language-Action)** model.

    This configuration defines all submodules of SmolVLM-VLA:
      - The visual-language backbone (SmolVLM-500M-Instruct)
      - The temporal/action transformer
      - The action/proprio setup
      
    Key differences from FlorenceVLA:
      - Uses SmolVLM (500M) instead of Florence2
      - Input image size: 512x512 (SmolVLM-500M uses 512x512 patches)
      - All views input to VLM directly, no aux_visual_inputs
      - Efficient model suitable for on-device applications
    """

    model_type = "smolvlm_vla"

    def __init__(
        self,
        # === SmolVLM backbone ===
        smolvlm_model_path: str = os.environ.get("SIMVLA_SMOLVLM_MODEL", "/root/model/smolvlm-500M"),
        
        # === Transformer head ===
        hidden_size: int = 768,  # Action transformer hidden size
        depth: int = 12,  # Number of transformer layers
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dim_time: int = 32,
        max_len_seq: int = 512,  

        # === Action & proprio ===
        num_actions: int = 30,
        action_mode: str = "galaxea_joint",
        use_proprio: bool = True,
        
        # === DiT/AdaLN Mode ===
        use_adaln: bool = False,
        use_cross_attn: bool = True,     # AdaLN 模式下是否启用 cross-attention to VLM 全序列

        # === Image settings ===
        image_size: int = 384,  # Can be 384 or 512
        num_views: int = 3,  # Number of camera views

        # === CVAE 子目标潜变量 ===
        use_subgoal_vae: bool = False,
        subgoal_latent_dim: int = 64,
        kl_weight: float = 0.001,

        # === Latent Diffusion Model（z 空间 Flow Matching）===
        use_latent_flow: bool = True,    # 是否用 LDM 替换先验采样（配合 use_subgoal_vae）
        latent_flow_steps: int = 5,      # 推理时 z 空间 Euler 积分步数
        latent_fm_weight: float = 1.0,   # latent FM 损失权重

        # === 损失函数 ===
        use_huber_loss: bool = False,    # True: Huber loss，False: MSE
        huber_delta: float = 1.0,        # Huber loss 的 delta 参数
        gripper_weight: float = 1.0,     # gripper 维度损失权重倍率

        # === 时间步采样策略 ===
        time_sampling: str = 'beta',     # 'beta' | 'logit_normal' | 'cosine'

        # === HistoryEncoder（GRU 历史感知）===
        use_history_encoder: bool = False,
        history_hidden: int = 128,       # GRU 隐状态维度
        history_seq_len: int = 4,        # 训练时使用的连续帧数（含当前帧）
        switch_loss_weight: float = 0.05, # 阶段切换辅助损失权重

        # === PhysicsPredicateDecoder（物理谓词嵌入）===
        use_physics_cot: bool = False,
        physics_weight: float = 0.01,    # 物理谓词辅助损失权重

        **kwargs,
    ):
        # SmolVLM backbone path
        self.smolvlm_model_path = smolvlm_model_path
        
        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        
        # DiT/AdaLN settings
        self.use_adaln = use_adaln
        self.use_cross_attn = use_cross_attn

        # Image settings
        self.image_size = image_size
        self.num_views = num_views

        # CVAE 子目标潜变量设置
        self.use_subgoal_vae = use_subgoal_vae
        self.subgoal_latent_dim = subgoal_latent_dim
        self.kl_weight = kl_weight

        # Latent Diffusion Model 设置
        self.use_latent_flow = use_latent_flow
        self.latent_flow_steps = latent_flow_steps
        self.latent_fm_weight = latent_fm_weight

        # 损失函数设置
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.gripper_weight = gripper_weight

        # 时间步采样策略
        self.time_sampling = time_sampling

        # HistoryEncoder
        self.use_history_encoder = use_history_encoder
        self.history_hidden = history_hidden
        self.history_seq_len = history_seq_len
        self.switch_loss_weight = switch_loss_weight

        # PhysicsPredicateDecoder
        self.use_physics_cot = use_physics_cot
        self.physics_weight = physics_weight

        # Initialize base HF config attributes
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Convert this configuration into a fully serializable dictionary.
        """
        output = super().to_dict()
        return output
