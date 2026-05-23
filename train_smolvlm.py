"""
SmolVLM-VLA Training Script

Training script for SmolVLM-VLA using SmolVLM-500M-Instruct as backbone.
Uses 512x512 image resolution and unified VLM features (no aux_visual_inputs).

Usage:
    python train_smolvlm.py \
        --output_dir ./runs/smolvlm_vla \
        --train_metas_path ./train_metas.json \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --action_mode galaxea_joint \
        --num_actions 10
"""

import os
import math
import time
import json
import random
import argparse
from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_smolvlm_dataloader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

import logging
import sys

# TensorBoard (optional, for histograms)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ============================================================
# Logger
# ============================================================
def get_logger(name="train_smolvlm", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train_smolvlm.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("SmolVLM-VLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, default=None, 
                        help="Path to pretrained SmolVLM-VLA checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="runnings_smolvlm", 
                        help="Directory to save checkpoints")

    # SmolVLM backbone
    parser.add_argument("--smolvlm_model_path", type=str, 
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="Path or HF repo for SmolVLM backbone")
    
    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, 
                        help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384, 
                        help="Image size for SmolVLM (default: 384, can be 384 or 512)")

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, 
                        help="LR multiplier for VLM backbone")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)
    
    # Action mode
    parser.add_argument("--action_mode", type=str, default="galaxea_joint",
                        help="Action mode: galaxea_joint, galaxea, libero_joint, etc.")
    
    # Data loading
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Normalization
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="Path to normalization statistics JSON file")
    
    # Action horizon
    parser.add_argument("--num_actions", type=int, default=10,
                        help="Action horizon (number of future actions to predict)")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    
    # Resume control
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint")
    
    # DiT/AdaLN mode
    parser.add_argument("--use_adaln", action="store_true", default=False,
                        help="Use DiT-style AdaLN conditioning")

    # HyperNet mode
    parser.add_argument("--use_hypernet", action="store_true", default=False,
                        help="Use HyperNet policy (VLM generates weight deltas)")
    parser.add_argument("--hypernet_rank", type=int, default=4,
                        help="Rank for HyperNet low-rank weight deltas")

    # ⑤ 课程学习时间采样
    parser.add_argument("--no_curriculum_time", action="store_true", default=False,
                        help="Disable curriculum time sampling (use fixed Beta(1.5,1))")

    # ⑥ 辅助世界模型损失
    parser.add_argument("--use_state_loss", action="store_true", default=False,
                        help="Enable auxiliary state prediction loss")
    parser.add_argument("--state_loss_weight", type=float, default=0.1,
                        help="Weight for state prediction loss")
    
    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size for action transformer")
    parser.add_argument("--depth", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def sample_time(B: int, global_step: int, total_steps: int, device) -> torch.Tensor:
    """根据训练进度动态调整 Beta 分布参数（课程学习时间采样）。"""
    progress = global_step / max(total_steps, 1)
    if progress < 0.3:
        alpha = 1.0   # 均匀采样，探索全局结构
    elif progress < 0.7:
        alpha = 1.5   # 原始设置
    else:
        alpha = 2.5   # 偏向小 t，强化精细控制
    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(1.0, device=device),
    )
    return beta_dist.sample((B,)) * 0.999 + 0.001


def build_optimizer(model: SmolVLMVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_vlm=1.0):
    """Build optimizer with separate param groups."""
    vlm_params = list(model.vlm.parameters())

    # Get action output params based on mode
    if hasattr(model.transformer, 'final_layer'):
        action_params = list(model.transformer.final_layer.parameters()) + list(model.transformer.action_encoder.parameters())
    else:
        action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())

    # HyperNet gets its own group (higher LR, not frozen during freeze_steps)
    hypernet_params = []
    if hasattr(model.transformer, 'hypernet'):
        hypernet_params = list(model.transformer.hypernet.parameters())

    exclude = set(map(id, vlm_params + action_params + hypernet_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]

    param_groups = [
        {"name": "vlm",              "params": vlm_params,              "lr": 0.0,       "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0,       "weight_decay": weight_decay},
        {"name": "action_heads",     "params": action_params,           "lr": lr,        "weight_decay": weight_decay},
    ]
    if hypernet_params:
        # HyperNet starts from random init → needs higher LR; not frozen during freeze_steps
        param_groups.append(
            {"name": "hypernet", "params": hypernet_params, "lr": lr * 3.0, "weight_decay": weight_decay}
        )
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups:
        if g["name"] == name:
            g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name:
            return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start:
        return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Update learning rates for all param groups."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "action_heads": args.learning_rate,
        "hypernet": args.learning_rate * 3.0,  # HyperNet always active, higher LR
    }

    def schedule(step, base_lr):
        return linear_warmup_cosine(
            step, args.freeze_steps, args.warmup_steps,
            args.iters, base_lr, args.min_lr_ratio
        )

    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "action_heads", base["action_heads"])
        set_group_lr(optim, "hypernet", base["hypernet"])  # not frozen
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    
    # WandB setup
    wandb_api_key = os.environ.get("WANDB_API_KEY") or args.wandb_api_key
    wandb_project = os.environ.get("WANDB_PROJECT") or args.wandb_project
    use_wandb = WANDB_AVAILABLE and wandb_api_key

    log_with = ["tensorboard"]
    if use_wandb:
        log_with.append("wandb")
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Accelerator setup
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs]
    )

    # Initialize trackers
    tracker_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "smolvlm_model_path": args.smolvlm_model_path,
        "freeze_steps": args.freeze_steps,
        "warmup_steps": args.warmup_steps,
        "save_interval": args.save_interval,
        "action_mode": args.action_mode,
        "num_actions": args.num_actions,
        "image_size": args.image_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "use_adaln": args.use_adaln,
    }
    
    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"smolvlm-{time.strftime('%Y%m%d-%H%M%S')}"}}
        )
    else:
        accelerator.init_trackers("SmolVLM-VLA-Training", config=tracker_config)

    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")
    logger.info(f"Using SmolVLM backbone: {args.smolvlm_model_path}")
    logger.info(f"Image size: {args.image_size}x{args.image_size}")

    # Load model
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.action_hub import build_action_space
    
    action_space_kwargs = {}
    if args.norm_stats_path:
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path
        logger.info(f"Using normalization stats from: {args.norm_stats_path}")
    
    load_path = args.models
    
    if load_path and os.path.isdir(load_path) and os.path.exists(os.path.join(load_path, "model.safetensors")):
        logger.info(f"Loading SmolVLM-VLA from checkpoint: {load_path}")
        model = SmolVLMVLA.from_pretrained(load_path)
        
        if args.action_mode != model.action_mode:
            logger.warning(f"Overriding model action_mode from '{model.action_mode}' to '{args.action_mode}'")
            model.action_mode = args.action_mode
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
        elif action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
            
        if args.num_actions != model.num_actions:
            logger.warning(f"Overriding model num_actions from {model.num_actions} to {args.num_actions}")
            model.config.num_actions = args.num_actions
            model.num_actions = args.num_actions
            
        model_use_adaln = getattr(model, 'use_adaln', False)
        if args.use_adaln != model_use_adaln:
            logger.warning(f"⚠️ Cannot change use_adaln when loading from checkpoint")
    else:
        logger.info(f"Initializing SmolVLM-VLA from config")
        logger.info(f"  smolvlm_model_path: {args.smolvlm_model_path}")
        logger.info(f"  action_mode: {args.action_mode}")
        logger.info(f"  num_actions: {args.num_actions}")
        logger.info(f"  use_adaln: {args.use_adaln}")
        
        config = SmolVLMVLAConfig(
            smolvlm_model_path=args.smolvlm_model_path,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            action_mode=args.action_mode,
            num_actions=args.num_actions,
            use_adaln=args.use_adaln,
            use_hypernet=args.use_hypernet,
            hypernet_rank=args.hypernet_rank,
            image_size=args.image_size,
        )
        model = SmolVLMVLA(config)
        
        if action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
    
    # Build processor
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)

    # Create SmolVLM dataloader (384x384 images)
    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Optimizer
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_vlm=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)

    # TensorBoard SummaryWriter for histograms (main process only)
    tb_writer = None
    if HAS_TENSORBOARD and accelerator.is_main_process:
        tb_writer = SummaryWriter(log_dir=str(output_dir))

    # Diagnostic state
    loss_history: deque = deque(maxlen=100)  # for rolling std
    best_velocity_loss = float("inf")
    last_delta_norm = 0.0

    # Training loop
    model.train()

    start_step = 0
    if args.resume and load_path and os.path.isdir(load_path):
        state_json = os.path.join(load_path, "state.json")
        if os.path.exists(state_json):
            try:
                with open(state_json, "r") as f:
                    state_data = json.load(f)
                    start_step = int(state_data.get("global_step", 0))
                    best_velocity_loss = float(state_data.get("best_velocity_loss", float("inf")))
                logger.info(f"Resuming from step: {start_step}, best_loss: {best_velocity_loss:.4f}")
            except Exception:
                pass

    def _save_ckpt(tag: str):
        save_dir = os.path.join(output_dir, tag)
        accelerator.print(f"💾 Saving model to {save_dir}")
        accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
        with open(os.path.join(save_dir, "state.json"), "w") as f:
            json.dump({
                "global_step": global_step,
                "best_velocity_loss": best_velocity_loss,
                "hypernet_mean_delta_norm": last_delta_norm,
            }, f)

    global_step, t0 = start_step, time.time()
    logger.info(f"🚀 Start SmolVLM-VLA training for {args.iters} iterations")
    logger.info(f"   world_size={accelerator.num_processes}")
    if args.use_hypernet:
        logger.info(f"   HyperNet mode: rank={args.hypernet_rank}")

    for batch in train_dataloader:
        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        # Update LR
        update_group_lrs(optim, global_step, args)

        # 时间采样（⑤ 课程学习 or 固定 Beta(1.5,1)）
        B = inputs["action"].shape[0]
        device = inputs["action"].device
        if args.no_curriculum_time:
            beta_dist = torch.distributions.Beta(
                torch.tensor(1.5, device=device),
                torch.tensor(1.0, device=device),
            )
            t_sample = beta_dist.sample((B,)) * 0.999 + 0.001
        else:
            t_sample = sample_time(B, global_step, args.iters, device)
        inputs["t_override"] = t_sample

        # ⑥ 辅助世界模型损失：控制 next_proprio 是否传入
        if not args.use_state_loss:
            inputs.pop("next_proprio", None)
        else:
            inputs["state_loss_weight"] = args.state_loss_weight

        # 过滤掉 forward 不接受的字段（如 attention_mask 等）
        _FORWARD_KEYS = {
            "input_ids", "image_input", "image_mask", "proprio", "action",
            "t_override", "next_proprio", "state_loss_weight",
        }
        inputs = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}

        # Forward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())

        # Track velocity loss for diagnostics
        velocity_loss_val = loss_dict.get("velocity_loss", loss).detach().float().item()
        loss_history.append(velocity_loss_val)

        # Backward
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # ---- Gradient norm diagnostics (before optimizer step) ----
        if global_step % args.log_interval == 0 and accelerator.is_main_process:
            grad_logs = {}
            for g in optim.param_groups:
                gn = sum(
                    p.grad.norm() ** 2 for p in g["params"] if p.grad is not None
                ) ** 0.5
                grad_logs[f"grad/{g['name']}_norm"] = gn.item()
        else:
            grad_logs = {}

        optim.step()
        optim.zero_grad()

        # ---- Logging ----
        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            # 课程学习 alpha
            progress = global_step / max(args.iters, 1)
            logs["time_sampling/alpha"] = 1.0 if progress < 0.3 else (1.5 if progress < 0.7 else 2.5)
            logs.update(grad_logs)

            # Loss rolling std (every 100 steps)
            if len(loss_history) >= 10:
                arr = list(loss_history)
                mean_l = sum(arr) / len(arr)
                std_l = (sum((v - mean_l) ** 2 for v in arr) / len(arr)) ** 0.5
                logs["loss/velocity_std100"] = std_l / (mean_l + 1e-8)

            # HyperNet diagnostics
            if accelerator.is_main_process and args.use_hypernet:
                unwrapped = accelerator.unwrap_model(model)
                hn = unwrapped.transformer
                if hasattr(hn, 'last_task_vec') and hn.last_task_vec is not None:
                    with torch.no_grad():
                        deltas = hn.hypernet(hn.last_task_vec[:1])
                        delta_norms = []
                        for i, (d1, d2) in enumerate(deltas):
                            n1 = d1[0].norm().item()
                            n2 = d2[0].norm().item()
                            logs[f"hypernet/layer{i}_fc1_norm"] = n1
                            logs[f"hypernet/layer{i}_fc2_norm"] = n2
                            delta_norms.extend([n1, n2])
                        mean_delta = sum(delta_norms) / len(delta_norms)
                        logs["hypernet/mean_delta_norm"] = mean_delta
                        last_delta_norm = mean_delta

                        base_norms = [
                            p.norm().item()
                            for name, p in hn.named_parameters()
                            if 'blocks' in name and ('fc1.weight' in name or 'fc2.weight' in name)
                        ]
                        base_norm = sum(base_norms) / len(base_norms) if base_norms else 1.0
                        logs["hypernet/delta_to_base_ratio"] = mean_delta / (base_norm + 1e-8)

            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr_core={logs.get('lr_transformer_core', 0):.2e} "
                    f"lr_action={logs.get('lr_action_heads', 0):.2e} "
                    f"lr_vlm={logs.get('lr_vlm', 0):.2e} ({dt:.2f}s/it)"
                )

        # task_vec diversity (every 1000 steps)
        if global_step % 1000 == 0 and accelerator.is_main_process and args.use_hypernet:
            unwrapped = accelerator.unwrap_model(model)
            hn = unwrapped.transformer
            if hasattr(hn, 'last_task_vec') and hn.last_task_vec is not None and hn.last_task_vec.shape[0] > 1:
                with torch.no_grad():
                    tv = F.normalize(hn.last_task_vec.float(), dim=-1)
                    cos_sim = (tv @ tv.T).mean().item()
                    accelerator.log({"hypernet/task_vec_diversity": 1.0 - cos_sim}, step=global_step)

        # TensorBoard histograms (every 5000 steps)
        if tb_writer is not None and global_step % 5000 == 0 and args.use_hypernet:
            unwrapped = accelerator.unwrap_model(model)
            hn = unwrapped.transformer
            for name, param in hn.hypernet.named_parameters():
                tb_writer.add_histogram(f"weights/hypernet/{name}", param.data, global_step)
            if hasattr(hn, 'last_deltas') and hn.last_deltas:
                tb_writer.add_histogram("delta/layer0_fc1", hn.last_deltas[0][0][0], global_step)
                tb_writer.add_histogram("delta/last_fc2", hn.last_deltas[-1][1][0], global_step)

        # ---- Checkpointing ----
        global_step += 1
        if accelerator.is_main_process:
            # Early frequent saves (first 5000 steps)
            if global_step < 5000 and global_step % 1000 == 0:
                _save_ckpt(f"ckpt-{global_step}")

            # Best checkpoint
            if velocity_loss_val < best_velocity_loss:
                best_velocity_loss = velocity_loss_val
                _save_ckpt("ckpt-best")

            # Regular interval + final
            if global_step == args.iters or global_step % args.save_interval == 0:
                _save_ckpt(f"ckpt-{global_step}")

        if global_step >= args.iters:
            break

    if tb_writer is not None:
        tb_writer.close()
    accelerator.end_training()


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("SmolVLM-VLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
