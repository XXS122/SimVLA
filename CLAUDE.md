# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimVLA is a Vision-Language-Action (VLA) model for robotic manipulation. It combines a SmolVLM-500M-Instruct backbone (vision-language encoder) with a configurable Action Transformer head to predict robot actions from camera observations and language instructions. The primary evaluation benchmark is [LIBERO](https://libero-project.github.io/) (HDF5-based robot demonstration data).

## Branch

All development happens on **`feature/ctaf-psca`**. Do not create or push to other branches.

## Environment Setup

```bash
pip install -r requirements.txt
pip install flash-attn>=2.5.0   # requires CUDA
```

All path and credential configuration lives in `paths.env` (git-ignored). Load it before running anything:

```bash
source paths.env
```

Key variables defined in `paths.env`:

| Variable | Purpose |
|----------|---------|
| `SIMVLA_SMOLVLM_MODEL` | SmolVLM HF repo or local path |
| `LIBERO_DATASETS` | Root directory of LIBERO HDF5 files |
| `SIMVLA_CHECKPOINTS` | Pretrained SimVLA checkpoint (optional) |
| `SIMVLA_RESUME_CKPT` | Checkpoint to resume training from (optional) |
| `WANDB_API_KEY` | WandB API key — if set, training auto-enables WandB |
| `WANDB_PROJECT` | WandB project name |
| `CUDA_DEVICES` | GPU IDs (e.g. `"0"` or `"0,1,2,3"`) |
| `NUM_GPUS` | Number of GPUs to use |

Note: `LIBERO_DATASETS` points to the raw HDF5 root directory. `SIMVLA_TRAIN_METAS` (optional) points to the generated JSON metadata file — these are two different paths.

## Data Preparation

```bash
# 1. Generate training metadata from LIBERO HDF5 files
python create_libero_meta.py \
    --data_dir "$LIBERO_DATASETS" \
    --subsets libero_goal \
    --output ./datasets/metas/libero_goal_train.json

# 2. Compute action/state normalization statistics
python compute_libero_norm_stats.py \
    --data_dir "$LIBERO_DATASETS" \
    --subsets libero_goal \
    --output ./datasets/metas/libero_goal_norm.json
```

Both scripts are called automatically by the training shell scripts if their outputs don't exist.

## Training

```bash
source paths.env

# Small model (hidden=768, depth=12, heads=12) — single GPU
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Large model (hidden=1024, depth=24, heads=16) — multi GPU
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Direct invocation
accelerate launch --mixed_precision bf16 train_smolvlm.py \
    --output_dir ./runs/my_run \
    --train_metas_path ./datasets/metas/libero_goal_train.json \
    --norm_stats_path ./datasets/metas/libero_goal_norm.json \
    --batch_size 8 --hidden_size 768 --depth 12 --num_heads 12
```

Key training flags: `--freeze_steps` (steps before unfreezing VLM backbone, default 1000), `--learning_coef` (LR multiplier for VLM backbone, default 0.1), `--num_actions` (action horizon, default 10), `--image_size` (384 or 512), `--use_adaln` / `--use_ctaf` / `--use_psca` (optional architecture extensions).

WandB is automatically enabled when `WANDB_API_KEY` is set in the environment; disabled otherwise.

## Evaluation (LIBERO)

```bash
source paths.env

# Step 1: Start the FastAPI model server
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/my_run/ckpt-150000 \
    --norm_stats ./datasets/metas/libero_goal_norm.json \
    --port 8102

# Step 2: Run parallel evaluation across all 4 LIBERO task suites
bash evaluation/libero/run_eval_all.sh 8102 50 "eval_run_name" "0 1 2 3"
# Args: <port> <num_trials> <eval_name> <gpu_ids>
```

## Architecture

### Model (`models/`)

**`modeling_smolvlm_vla.py`** — `SmolVLMVLA` is the top-level `PreTrainedModel`. It composes:
1. SmolVLM-500M backbone (SigLIP vision encoder + language transformer)
2. `SmolVLMActionTransformer` — temporal/action decoder head
3. Action space (normalization + loss)

Training uses **conditional flow matching**: interpolates between noise and target action, trains the model to predict the velocity field, then runs Euler integration at inference.

**`transformer_smolvlm.py`** — `SmolVLMActionTransformer`. Supports four modes (controlled by config flags):
- Baseline (`use_adaln=False`, `use_ctaf=False`, `use_psca=False`): standard concat-mode transformer
- `use_adaln=True`: DiT-style Adaptive LayerNorm conditioning
- `use_ctaf=True`: Fourier coefficient decoder — pools action features → predicts Fourier coefficients → reconstructs smooth trajectory via `query_fourier()`
- `use_psca=True`: LoRA adapters (rank=8, B-init=0) on every MLP block for parameter-efficient adaptation

**`configuration_smolvlm_vla.py`** — HuggingFace `PretrainedConfig` subclass. Key fields: `hidden_size`, `depth`, `num_heads`, `action_mode`, `image_size`, `use_adaln`, `use_ctaf`, `use_psca`, `num_fourier_freqs`, `psca_rank`.

**`processing_smolvlm_vla.py`** — `SmolVLMVLAProcessor` prepares multimodal inputs: 3 camera views (agentview, eye_in_hand, third_person) at 384×384 or 512×512, plus language tokenization.

**`action_hub.py`** — Action space registry. `libero_joint` mode = 7-dim actions (Δxyz, Δeuler, gripper), 8-dim proprio. Handles z-score normalization/denormalization.

### Datasets (`datasets/`)

**`dataset_smolvlm.py`** — `SmolVLMDataReader` (`IterableDataset`): reads episode windows from HDF5 files using metadata JSONs, applies normalization, returns batches with keys `image_input [B,V,C,H,W]`, `proprio [B,8]`, `action [B,T,7]`, `language_instruction`.

**`domain_handler/`** — Pluggable data handler pattern. `libero_hdf5.py` implements LIBERO-specific HDF5 loading. New datasets register via `registry.py`.

### Data Flow

```
LIBERO HDF5 → create_libero_meta.py → JSON metadata
                                           ↓
compute_libero_norm_stats.py → norm_stats JSON
                                           ↓
SmolVLMDataReader → SmolVLMVLAProcessor → SmolVLMVLA → flow matching loss
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
| `paths.env` | Local machine paths & credentials (git-ignored) |
| `train_smolvlm.py` | Main training entry point |
| `train_smolvlm_small.sh` | Single-GPU training (Small model) |
| `train_smolvlm_large.sh` | Multi-GPU training (Large model) |
| `models/modeling_smolvlm_vla.py` | Top-level model class |
| `models/transformer_smolvlm.py` | Action transformer head (CTAF + PSCA) |
| `datasets/dataset_smolvlm.py` | Data loading pipeline |
| `evaluation/libero/serve_smolvlm_libero.py` | Inference server |
