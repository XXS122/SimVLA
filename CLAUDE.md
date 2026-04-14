# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
conda create -n simvla python=3.10 -y && conda activate simvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

### Data Preparation (LIBERO)
```bash
# 1. Generate dataset metadata JSON
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json

# 2. Compute action/state normalization statistics
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

### Training
```bash
# Small model (768 hidden, 12 layers, 12 heads) - single GPU
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Large model (1024 hidden, 24 layers, 16 heads) - 4 GPUs
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Direct invocation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes=1 --mixed_precision bf16 \
    train_smolvlm.py \
    --train_metas_path ./datasets/metas/libero_train.json \
    --smolvlm_model_path /path/to/SmolVLM-500M-Instruct \
    --action_mode libero_joint \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --output_dir ./runs/my_run
```

### Evaluation (LIBERO - WebSocket server)
```bash
cd evaluation/libero
# Start inference server
python serve_smolvlm_libero.py \
    --checkpoint /path/to/checkpoint \
    --norm_stats /path/to/libero_norm.json \
    --smolvlm_model /path/to/SmolVLM-500M-Instruct \
    --port 8000
```

Key training args: `--use_adaln` (DiT-style conditioning), `--image_size 384|512`, `--num_actions 10`, `--freeze_steps 1000` (VLM frozen for first N steps), `--learning_coef 0.1` (VLM LR multiplier).

## Architecture

SimVLA is a Vision-Language-Action (VLA) model for robot manipulation. It has two main components: a frozen/fine-tuned VLM backbone and a Flow Matching action head.

### Data Flow
```
HDF5 file → LiberoHDF5Handler.iter_episode()
  ├─ obs: agentview_rgb[T,128,128,3], eye_in_hand_rgb[T,128,128,3]
  ├─ proprio: ee_pos(3) + euler→axis_angle(3) + gripper(2) = 8-dim
  └─ actions: delta_xyz(3) + delta_euler(3) + gripper(1) = 7-dim
       ↓ image_aug (resize→384, ColorJitter, ImageNet normalize)
SmolVLMDataReader → DataLoader → batch
       ↓ processor.encode_language() → input_ids
SmolVLMVLA.forward()
  ├─ forward_vlm_efficient(): SigLIP → connector → concat text_embeds → LM → vlm_features[B, seq, 576]
  ├─ Flow Matching: t~Beta(1.5,1), x_t = t*noise + (1-t)*action_norm, target u_t = noise - action
  └─ SmolVLMActionTransformer → MSE(v_t, u_t)
```

### Module Map

| File | Role |
|---|---|
| `models/modeling_smolvlm_vla.py` | Top-level `SmolVLMVLA(PreTrainedModel)` — VLM forward, Flow Matching training loop, Euler inference, FastAPI service |
| `models/transformer_smolvlm.py` | `SmolVLMActionTransformer` — two modes: Concat (`TransformerBlock`) or AdaLN/DiT (`DiTBlock`). Also `timestep_embedding`, `FinalLayer` |
| `models/action_hub.py` | `LiberoJointActionSpace` — action/state normalization (z-score or quantile). Registry pattern via `@register_action`. |
| `models/configuration_smolvlm_vla.py` | HuggingFace `PretrainedConfig` subclass, serialized with `save_pretrained` |
| `models/processing_smolvlm_vla.py` | `SmolVLMVLAProcessor` — wraps SmolVLM processor for `encode_language()` at training time |
| `datasets/dataset_smolvlm.py` | `SmolVLMDataReader(IterableDataset)` — infinite weighted multi-dataset sampler |
| `datasets/domain_handler/registry.py` | Dict mapping `dataset_name → HandlerClass`. **Edit here to add new datasets.** |
| `datasets/domain_handler/libero_hdf5.py` | `LiberoHDF5Handler(DomainHandler)` — reads LIBERO HDF5 format |
| `datasets/domain_handler/base.py` | Abstract `DomainHandler` + `BaseHDF5Handler` with interpolation-based trajectory sampling |
| `datasets/domain_config.py` | `DATA_WEIGHTS` dict for multi-dataset sampling ratios |
| `datasets/utils.py` | `action_slice()`, `read_parquet()`, `decode_image_from_bytes()`, rotation converters |
| `train_smolvlm.py` | Training loop using `accelerate`. Three optimizer param groups: `vlm` (frozen first N steps), `transformer_core`, `action_heads` |
| `evaluation/libero/serve_smolvlm_libero.py` | WebSocket inference server (msgpack_numpy serialization) |

### Action Transformer Modes

**Concat mode** (default, `use_adaln=False`): action tokens + time + proprio are concatenated, then VLM features are appended to sequence: `x = cat([action_tokens, vlm_proj(vlm_features)], dim=1)`. Only action positions are decoded.

**AdaLN/DiT mode** (`--use_adaln`): condition `c = time_emb + vlm_pool + proprio_emb` injected into each `DiTBlock` via adaptive layer norm. Cleaner separation of conditioning signal.

### Adding a New Dataset

1. Create `datasets/domain_handler/mydata.py` implementing `DomainHandler.iter_episode()` — must yield dicts with keys: `language_instruction`, `image_input[V,C,H,W]`, `image_mask[V]`, `abs_trajectory[T+1,D]` (state at [0], actions at [1:])
2. Register in `datasets/domain_handler/registry.py`: add `"mydata_name": MyHandler` to `_REGISTRY`
3. Add sampling weight in `datasets/domain_config.py`
4. If action space differs, add a new `@register_action("mydata_joint")` class in `models/action_hub.py`
5. Create meta JSON with fields: `dataset_name`, `datalist` (list of `{path, task}`), `data_dir`

### Key Design Decisions

- **No aux_visual_inputs**: Unlike FlorenceVLA, all camera views go through SmolVLM directly. The VLM processes `[image1_patches, image2_patches, ..., text_tokens]` as a single sequence.
- **Flow Matching, not DDPM**: Uses linear interpolation ODE with Beta(1.5,1) time distribution and Euler integration at inference (10 steps by default).
- **Lazy VLM unfreeze**: VLM parameters have `lr=0` for the first `freeze_steps` iterations, then are unfrozen at `learning_rate * learning_coef`.
- **`abs_trajectory` convention**: Handlers yield `[T+1, D]` where index 0 is the current state (proprio) and indices 1..T are future actions. `action_slice()` separates these and optionally computes deltas.
- `datasets/utils.py` already contains `read_parquet()` for Parquet support — the infrastructure for non-HDF5 formats is partially in place.
