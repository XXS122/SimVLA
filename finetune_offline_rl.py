"""
SimVLA Offline RL Fine-tuning via Advantage-Weighted Regression (AWR).

Quality proxy: trajectory efficiency (shorter = better).
  reward_i = max_traj_len / traj_len_i
  advantage_i = reward_i - mean(rewards)
  weight_i = clip(exp(advantage_i / temperature), 0.1, 10.0)

Loss: weighted BC = weight_i * MSE(v_t, u_t)

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  accelerate launch --num_processes=2 --mixed_precision bf16 \\
      finetune_offline_rl.py \\
      --checkpoint ./runs/simvla_hypernet/ckpt-50000 \\
      --train_metas_path ./datasets/metas/libero_train.json \\
      --norm_stats_path ./norm_stats/libero_norm.json \\
      --temperature 0.5 --iters 20000 --learning_rate 5e-5 \\
      --output_dir ./runs/awr_finetune
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import deque
from typing import Dict, Iterable

import h5py
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset_smolvlm import SmolVLMDataReader, _dataloader_worker_init
from datasets.domain_handler.registry import get_handler_cls
from datasets.utils import action_slice
from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor


# ─────────────────────────── AWR Dataset ────────────────────────────────────

class AWRDataReader(SmolVLMDataReader):
    """
    SmolVLMDataReader with per-trajectory AWR weights.

    Weights are pre-computed at init by reading HDF5 files to get trajectory
    lengths. Shorter trajectories (more efficient) get higher weights.
    """

    def __init__(self, *args, temperature: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self._traj_weights: Dict[str, Dict[int, float]] = {}
        self._precompute_weights()

    def _precompute_weights(self):
        """Read HDF5 files once to compute per-trajectory AWR weights."""
        print("[AWR] Pre-computing trajectory weights...")
        for name, meta in self.metas.items():
            datalist = meta.get("datalist", [])
            lengths = []
            for item in datalist:
                h5_path = item.get("path", "") if isinstance(item, dict) else ""
                if not h5_path or not os.path.exists(h5_path):
                    lengths.append(100)
                    continue
                try:
                    with h5py.File(h5_path, "r") as f:
                        if "data" not in f:
                            lengths.append(100)
                            continue
                        demo_lens = [
                            len(f["data"][k]["actions"])
                            for k in f["data"].keys()
                            if "actions" in f["data"][k]
                        ]
                        avg = sum(demo_lens) / len(demo_lens) if demo_lens else 100
                        lengths.append(avg)
                except Exception:
                    lengths.append(100)

            if not lengths:
                continue

            max_len = max(lengths)
            rewards = [max_len / max(l, 1) for l in lengths]
            r_mean = sum(rewards) / len(rewards)

            self._traj_weights[name] = {}
            for i, r in enumerate(rewards):
                adv = r - r_mean
                w = float(min(max(math.exp(adv / self.temperature), 0.1), 10.0))
                self._traj_weights[name][i] = w

            w_vals = list(self._traj_weights[name].values())
            print(f"[AWR] {name}: {len(w_vals)} trajs, "
                  f"weight range [{min(w_vals):.3f}, {max(w_vals):.3f}]")

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        """Iterate over one dataset, attaching AWR weight to each sample."""
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        if self.training:
            random.shuffle(traj_indices)

        Handler = get_handler_cls(dataset_name)
        handler = Handler(meta=meta, num_views=self.num_views)
        weights = self._traj_weights.get(dataset_name, {})

        for traj_idx in traj_indices:
            w = weights.get(traj_idx, 1.0)
            try:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),
                    action_mode=self.action_mode,
                ):
                    idx_for_delta = meta.get("idx_for_delta", [])
                    has_proprio = "proprio" in sample
                    slice_result = action_slice(sample["abs_trajectory"], idx_for_delta)

                    if has_proprio:
                        sample["action"] = slice_result["action"]
                    else:
                        sample.update(slice_result)
                    del sample["abs_trajectory"]

                    sample["traj_weight"] = torch.tensor(w, dtype=torch.float32)
                    yield sample
            except Exception:
                continue


# ─────────────────────────── Optimizer ──────────────────────────────────────

def build_awr_optimizer(
    model: SmolVLMVLA,
    lr: float,
    freeze_vlm: bool = True,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Fine-tuning optimizer. Optionally freeze VLM backbone."""
    if freeze_vlm:
        for p in model.vlm.parameters():
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[AWR] Trainable params: {n_params / 1e6:.1f}M "
          f"({'VLM frozen' if freeze_vlm else 'VLM unfrozen'})")
    return torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95),
                             weight_decay=weight_decay)


# ─────────────────────────── Main ───────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SimVLA Offline RL Fine-tuning (AWR)"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Pretrained SimVLA checkpoint path")
    parser.add_argument("--output_dir", default="./runs/awr_finetune")
    parser.add_argument("--train_metas_path", required=True)
    parser.add_argument("--norm_stats_path", default=None)
    parser.add_argument("--smolvlm_model_path", default=None,
                        help="Override SmolVLM path (default: from checkpoint config)")
    # AWR
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="AWR temperature (lower = sharper weighting)")
    # Training
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--freeze_vlm", action="store_true", default=True)
    parser.add_argument("--no_freeze_vlm", dest="freeze_vlm", action="store_false")
    # Logging
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    accelerator = Accelerator(mixed_precision="bf16")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load pretrained model ──────────────────────────────────────────────
    accelerator.print(f"[AWR] Loading checkpoint: {args.checkpoint}")
    model = SmolVLMVLA.from_pretrained(args.checkpoint)

    if args.norm_stats_path and os.path.exists(args.norm_stats_path):
        model.action_space.load_norm_stats(args.norm_stats_path)
        accelerator.print(f"[AWR] Loaded norm stats: {args.norm_stats_path}")

    smolvlm_path = args.smolvlm_model_path or model.config.smolvlm_model_path
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset = AWRDataReader(
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        training=True,
        action_mode=model.action_mode,
        temperature=args.temperature,
        image_size=getattr(model.config, "image_size", 384),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=_dataloader_worker_init,
        persistent_workers=args.num_workers > 0,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optim = build_awr_optimizer(
        model, args.learning_rate,
        freeze_vlm=args.freeze_vlm,
        weight_decay=args.weight_decay,
    )

    model, optim, dataloader = accelerator.prepare(model, optim, dataloader)

    # ── TensorBoard ────────────────────────────────────────────────────────
    writer = (
        SummaryWriter(args.output_dir)
        if accelerator.is_main_process else None
    )

    # ── Training loop ──────────────────────────────────────────────────────
    _FORWARD_KEYS = {
        "input_ids", "image_input", "image_mask", "proprio", "action",
        "t_override", "next_proprio", "state_loss_weight",
    }

    loss_history: deque = deque(maxlen=100)
    global_step = 0
    t0 = time.time()

    accelerator.print(
        f"[AWR] Starting fine-tuning for {args.iters} iterations\n"
        f"      temperature={args.temperature}, lr={args.learning_rate}, "
        f"freeze_vlm={args.freeze_vlm}"
    )

    for batch in dataloader:
        if global_step >= args.iters:
            break

        # Language encoding
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)

        # Extract AWR weight before building inputs
        traj_weight = batch.pop("traj_weight", None)  # [B]

        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        # Filter to forward-accepted keys only
        inputs = {k: v for k, v in inputs.items() if k in _FORWARD_KEYS}

        # Forward (no curriculum time for fine-tuning — use fixed Beta(1.5,1))
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())

        # AWR weighting: scale loss by mean batch weight
        if traj_weight is not None:
            w = traj_weight.cuda(non_blocking=True).mean()
            loss = loss * w

        # Backward
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        loss_val = loss.detach().float().item()
        loss_history.append(loss_val)
        global_step += 1

        # ── Logging ───────────────────────────────────────────────────────
        if global_step % args.log_interval == 0 and accelerator.is_main_process:
            elapsed = time.time() - t0
            logs = {
                "loss/total": loss_val,
                "lr": optim.param_groups[0]["lr"],
            }
            if len(loss_history) >= 10:
                arr = list(loss_history)
                mean_l = sum(arr) / len(arr)
                logs["loss/mean100"] = mean_l
            if traj_weight is not None:
                logs["awr/batch_weight"] = traj_weight.mean().item()

            if writer:
                for k, v in logs.items():
                    writer.add_scalar(k, v, global_step)

            print(
                f"[AWR] step={global_step}/{args.iters} "
                f"loss={loss_val:.4f} "
                f"elapsed={elapsed:.0f}s"
            )

        # ── Checkpoint ────────────────────────────────────────────────────
        if global_step % args.save_interval == 0 and accelerator.is_main_process:
            ckpt_dir = os.path.join(args.output_dir, f"ckpt-{global_step}")
            accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
            print(f"[AWR] Saved checkpoint: {ckpt_dir}")

    # ── Final checkpoint ───────────────────────────────────────────────────
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "ckpt-final")
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        print(f"[AWR] Fine-tuning complete. Final checkpoint: {final_dir}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
