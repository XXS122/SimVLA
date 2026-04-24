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
import signal
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_smolvlm_dataloader
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

import logging
import sys

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
                        default=os.environ.get("SIMVLA_SMOLVLM_MODEL", "/root/model/smolvlm-500M"),
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
    parser.add_argument("--gpu_mem_threshold", type=float, default=80.0,
                        help="GPU 显存使用率阈值（%%），超过则保存 checkpoint 并退出（0 表示禁用）")
    
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
    parser.add_argument("--no_cross_attn", action="store_true", default=False,
                        help="AdaLN 模式下禁用 cross-attention to VLM（消融实验用）")

    # CVAE 子目标潜变量
    parser.add_argument("--use_subgoal_vae", action="store_true", default=False,
                        help="启用 CVAE 子目标潜变量（需配合 --use_adaln）")
    parser.add_argument("--subgoal_latent_dim", type=int, default=64,
                        help="子目标潜变量维度")
    parser.add_argument("--kl_weight", type=float, default=0.001,
                        help="KL 散度损失权重（最终值）")
    parser.add_argument("--kl_warmup_steps", type=int, default=10000,
                        help="KL annealing 步数：前 N 步线性增加 kl_weight")

    # Latent Diffusion Model（z 空间 Flow Matching）
    parser.add_argument("--use_latent_flow", action="store_true", default=False,
                        help="启用 z 空间 LDM（LatentFlowNet），替换 CVAE 先验采样（需配合 --use_subgoal_vae）")
    parser.add_argument("--latent_flow_steps", type=int, default=5,
                        help="推理时 z 空间 Euler 积分步数")
    parser.add_argument("--latent_fm_weight", type=float, default=1.0,
                        help="latent FM 损失权重")

    # 损失函数
    parser.add_argument("--use_huber_loss", action="store_true", default=False,
                        help="使用 Huber loss 替代 MSE（对噪声演示更鲁棒）")
    parser.add_argument("--huber_delta", type=float, default=1.0,
                        help="Huber loss 的 delta 参数")
    parser.add_argument("--gripper_weight", type=float, default=1.0,
                        help="gripper 维度损失权重倍率（建议 3.0~10.0）")

    # 时间步采样策略
    parser.add_argument("--time_sampling", type=str, default="beta",
                        choices=["beta", "logit_normal", "cosine"],
                        help="Flow Matching 时间步采样策略")
    
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
def get_gpu_memory_usage_pct() -> float:
    """返回当前进程所在 GPU 的显存使用率（0~100）。"""
    if not torch.cuda.is_available():
        return 0.0
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    return allocated / total * 100.0


def emergency_save(model, output_dir, global_step, accelerator, logger, reason="gpu_oom"):
    """紧急保存 checkpoint 并退出。"""
    if accelerator.is_main_process:
        save_dir = os.path.join(output_dir, f"ckpt-{global_step}-{reason}")
        logger.warning(f"⚠️  {reason.upper()} — 紧急保存 checkpoint 到 {save_dir}")
        try:
            accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
            with open(os.path.join(save_dir, "state.json"), "w") as f:
                json.dump({"global_step": global_step, "reason": reason}, f)
            logger.warning(f"✅ 紧急 checkpoint 已保存，正在退出...")
        except Exception as e:
            logger.error(f"紧急保存失败: {e}")
    accelerator.wait_for_everyone()
    os.kill(os.getpid(), signal.SIGTERM)



    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: SmolVLMVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_vlm=1.0):
    """Build optimizer with separate param groups."""
    vlm_params = list(model.vlm.parameters())
    
    # Get action output params based on mode
    if hasattr(model.transformer, 'final_layer'):
        action_params = list(model.transformer.final_layer.parameters()) + list(model.transformer.action_encoder.parameters())
    else:
        action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    
    exclude = set(map(id, vlm_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
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
            use_cross_attn=not args.no_cross_attn,
            image_size=args.image_size,
            use_subgoal_vae=args.use_subgoal_vae,
            subgoal_latent_dim=args.subgoal_latent_dim,
            kl_weight=args.kl_weight,
            use_latent_flow=args.use_latent_flow,
            latent_flow_steps=args.latent_flow_steps,
            latent_fm_weight=args.latent_fm_weight,
            use_huber_loss=args.use_huber_loss,
            huber_delta=args.huber_delta,
            gripper_weight=args.gripper_weight,
            time_sampling=args.time_sampling,
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

    # Training loop
    model.train()
    
    start_step = 0
    if args.resume and load_path and os.path.isdir(load_path):
        state_json = os.path.join(load_path, "state.json")
        if os.path.exists(state_json):
            try:
                with open(state_json, "r") as f:
                    start_step = int(json.load(f).get("global_step", 0))
                logger.info(f"Resuming from step: {start_step}")
            except Exception:
                pass
    
    global_step, t0 = start_step, time.time()
    logger.info(f"🚀 Start SmolVLM-VLA training for {args.iters} iterations")
    logger.info(f"   world_size={accelerator.num_processes}")

    for batch in train_dataloader:
        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        
        # Update LR
        update_group_lrs(optim, global_step, args)

        # Forward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)

        # KL annealing：前 kl_warmup_steps 步线性增加 kl_weight
        if "kl_loss" in loss_dict and args.kl_warmup_steps > 0:
            kl_scale = min(1.0, global_step / args.kl_warmup_steps)
            loss_dict["kl_loss"] = loss_dict["kl_loss"] * kl_scale

        loss = sum(loss_dict.values())
        
        # Backward
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        # Logging
        if global_step % args.log_interval == 0:
            # GPU 显存监控
            if args.gpu_mem_threshold > 0:
                gpu_pct = get_gpu_memory_usage_pct()
                if gpu_pct >= args.gpu_mem_threshold:
                    logger.warning(
                        f"GPU 显存使用率 {gpu_pct:.1f}% >= 阈值 {args.gpu_mem_threshold}%，触发紧急保存"
                    )
                    emergency_save(model, output_dir, global_step, accelerator, logger, reason="gpu_oom")

            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                # 基础日志
                log_str = (
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"v_loss={logs.get('velocity_loss', logs['loss_total']):.4f} "
                )
                # 有 kl_loss 时额外打印
                if "kl_loss" in logs:
                    log_str += f"kl_loss={logs['kl_loss']:.4f} "
                # 有 latent_fm_loss 时额外打印
                if "latent_fm_loss" in logs:
                    log_str += f"z_fm_loss={logs['latent_fm_loss']:.4f} "
                log_str += (
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_action={logs['lr_action_heads']:.2e} "
                    f"lr_vlm={logs['lr_vlm']:.2e} "
                    f"gpu={get_gpu_memory_usage_pct():.0f}% "
                    f"({dt:.2f}s/it)"
                )
                logger.info(log_str)
        
        # Checkpointing
        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)
                    
        if global_step >= args.iters:
            break

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
