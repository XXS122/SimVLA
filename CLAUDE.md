# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimVLA is a Vision-Language-Action (VLA) model for robotic manipulation. It combines a SmolVLM-500M-Instruct backbone (vision-language encoder) with a configurable Action Transformer head to predict robot actions from camera observations and language instructions. The primary evaluation benchmark is [LIBERO](https://libero-project.github.io/) (HDF5-based robot demonstration data).

## Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
# flash-attn requires CUDA; install separately if needed:
pip install flash-attn>=2.5.0
```

Key environment variables (set before training/evaluation):
- `SIMVLA_SMOLVLM_MODEL` — SmolVLM HF repo or local path (default: `HuggingFaceTB/SmolVLM-500M-Instruct`)
- `LIBERO_DATASETS` — Root directory of LIBERO HDF5 files
- `SIMVLA_TRAIN_METAS` — Path to training metadata JSON
- `SIMVLA_CHECKPOINTS` — Checkpoint to resume from (optional)
- `CUDA_VISIBLE_DEVICES` — GPU assignment

## Data Preparation

```bash
# 1. Generate training metadata from LIBERO HDF5 files
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_goal \
    --output ./datasets/metas/libero_goal_train.json

# 2. Compute action/state normalization statistics
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_goal \
    --output ./norm_stats/libero_goal_norm.json
```

## Training

```bash
# Small model (hidden=768, depth=12, heads=12)
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Large model (hidden=1024, depth=24, heads=16)
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Direct invocation (accelerate multi-GPU)
accelerate launch --mixed_precision bf16 train_smolvlm.py \
    --output_dir ./runs/my_run \
    --train_metas_path ./datasets/metas/libero_goal_train.json \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --batch_size 8 --hidden_size 768 --depth 12 --num_heads 12
```

Key training flags: `--freeze_steps` (steps before unfreezing VLM backbone, default 1000), `--learning_coef` (LR multiplier for VLM backbone, default 0.1), `--num_actions` (action horizon, default 10), `--image_size` (384 or 512), `--use_adaln` / `--use_ctaf` / `--use_psca` (optional architecture extensions).

## Evaluation (LIBERO)

```bash
# Step 1: Start the FastAPI model server
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/my_run/ckpt-150000 \
    --norm_stats ./norm_stats/libero_norm.json \
    --port 8102

# Step 2: Run parallel evaluation across all 4 LIBERO task suites
bash evaluation/libero/run_eval_all.sh 8102 50 "eval_run_name" "0 1 2 3"
# Args: <port> <num_trials> <eval_name> <gpu_ids>
```

## Architecture

### Model (`models/`)

**`modeling_smolvlm_vla.py`** — `SmolVLMVLA` is the top-level `PreTrainedModel`. It composes:
1. SmolVLM-500M backbone (vision encoder + language transformer)
2. `SmolVLMActionTransformer` — temporal/action decoder head
3. `NormStats` — action normalization/denormalization wrapper

**`transformer_smolvlm.py`** — `SmolVLMActionTransformer` takes VLM features + proprioceptive state and predicts an action sequence of length `num_actions`. Supports optional AdaLN (DiT-style), CTAF (Fourier decoder), and PSCA (LoRA adapters) modes.

**`configuration_smolvlm_vla.py`** — HuggingFace `PretrainedConfig` subclass. Key fields: `hidden_size`, `depth`, `num_heads`, `action_mode`, `image_size`, `use_adaln`, `use_ctaf`, `use_psca`.

**`processing_smolvlm_vla.py`** — `SmolVLMVLAProcessor` prepares multimodal inputs: 3 camera views (agentview, eye_in_hand, third_person) at 384×384 or 512×512, plus language tokenization.

**`action_hub.py`** — Action space registry. `libero_joint` mode = 7-dim actions (Δxyz, Δeuler, gripper).

### Datasets (`datasets/`)

**`dataset_smolvlm.py`** — `SmolVLMDataReader` (`IterableDataset`): reads episode windows from HDF5 files using metadata JSONs, applies normalization, returns batches with keys `image_input [B,V,C,H,W]`, `proprio [B,8]`, `action [B,T,7]`, `language_instruction`.

**`domain_handler/`** — Pluggable data handler pattern. `libero_hdf5.py` implements LIBERO-specific HDF5 loading. New datasets register via `registry.py`.

### Data Flow

```
LIBERO HDF5 → create_libero_meta.py → JSON metadata
                                           ↓
compute_libero_norm_stats.py → norm_stats JSON
                                           ↓
SmolVLMDataReader → SmolVLMVLAProcessor → SmolVLMVLA → MSE loss on actions
```

### Evaluation Flow

```
serve_smolvlm_libero.py (FastAPI + WebSocket server)
         ↑ WebSocket
libero_client.py × 4 parallel (one per LIBERO suite)
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `train_smolvlm.py` | Main training entry point |
| `models/modeling_smolvlm_vla.py` | Top-level model class |
| `models/transformer_smolvlm.py` | Action transformer head |
| `datasets/dataset_smolvlm.py` | Data loading pipeline |
| `evaluation/libero/serve_smolvlm_libero.py` | Inference server |
| `norm_stats/libero_norm.json` | Pre-computed normalization stats |
