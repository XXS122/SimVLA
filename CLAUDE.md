# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供该仓库的开发指引。

## 项目概述

SimVLA 是一个面向机器人操作任务的视觉-语言-动作（VLA）模型。以 SmolVLM-500M-Instruct 作为视觉语言主干，配合可配置的动作 Transformer 头，从相机观测和语言指令中预测机器人动作。主要评测基准为 [LIBERO](https://libero-project.github.io/)（基于 HDF5 的机器人示例数据）。

## 分支

所有开发工作在 **`feature/ctaf-psca`** 分支上进行，不要创建或推送到其他分支。

## 环境配置

```bash
pip install -r requirements.txt
pip install flash-attn>=2.5.0   # 需要 CUDA
```

所有路径和凭证配置存放在 `paths.env`（已加入 .gitignore），运行任何命令前先加载：

```bash
source paths.env
```

`paths.env` 中的关键变量：

| 变量 | 用途 |
|------|------|
| `SIMVLA_SMOLVLM_MODEL` | SmolVLM 模型的 HF 仓库名或本地路径 |
| `LIBERO_DATASETS` | LIBERO HDF5 文件的根目录 |
| `SIMVLA_CHECKPOINTS` | 预训练 SimVLA checkpoint（可选） |
| `SIMVLA_RESUME_CKPT` | 恢复训练的 checkpoint 路径（可选） |
| `WANDB_API_KEY` | WandB API key，设置后训练自动启用 WandB |
| `WANDB_PROJECT` | WandB 项目名 |
| `CUDA_DEVICES` | GPU 编号（如 `"0"` 或 `"0,1,2,3"`） |
| `NUM_GPUS` | 使用的 GPU 数量 |

注意：`LIBERO_DATASETS` 指向原始 HDF5 根目录，`SIMVLA_TRAIN_METAS`（可选）指向生成的 JSON 元数据文件，两者是不同的路径。

## 数据准备

```bash
# 1. 从 LIBERO HDF5 文件生成训练元数据
python create_libero_meta.py \
    --data_dir "$LIBERO_DATASETS" \
    --subsets libero_goal \
    --output ./datasets/metas/libero_goal_train.json

# 2. 计算动作/状态的归一化统计量
python compute_libero_norm_stats.py \
    --data_dir "$LIBERO_DATASETS" \
    --subsets libero_goal \
    --output ./datasets/metas/libero_goal_norm.json
```

如果输出文件不存在，训练脚本会自动调用以上两个脚本。

## 训练

```bash
source paths.env

# Small 模型（hidden=768, depth=12, heads=12）— 单卡
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# Large 模型（hidden=1024, depth=24, heads=16）— 多卡
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# 直接调用
accelerate launch --mixed_precision bf16 train_smolvlm.py \
    --output_dir ./runs/my_run \
    --train_metas_path ./datasets/metas/libero_goal_train.json \
    --norm_stats_path ./datasets/metas/libero_goal_norm.json \
    --batch_size 8 --hidden_size 768 --depth 12 --num_heads 12
```

关键训练参数：`--freeze_steps`（解冻 VLM 主干前的步数，默认 1000）、`--learning_coef`（VLM 主干的学习率倍率，默认 0.1）、`--num_actions`（动作预测时域，默认 10）、`--image_size`（384 或 512）、`--use_adaln` / `--use_ctaf` / `--use_psca`（可选架构扩展，脚本默认全开）。

设置 `WANDB_API_KEY` 后 WandB 自动启用，否则关闭。

## 评估（LIBERO）

```bash
source paths.env

# Step 1：启动 FastAPI 模型推理服务
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/my_run/ckpt-150000 \
    --norm_stats ./datasets/metas/libero_goal_norm.json \
    --port 8102

# Step 2：并行评估全部 4 个 LIBERO 任务套件
bash evaluation/libero/run_eval_all.sh 8102 50 "eval_run_name" "0 1 2 3"
# 参数：<port> <num_trials> <eval_name> <gpu_ids>
```

评估结果保存在带时间戳的子目录中，包含 `results.json`（主结果）、`episodes_detail.json`（每 episode 中间变量）、`action_stats.npz`（动作分布统计）。

## 架构

### 模型（`models/`）

**`modeling_smolvlm_vla.py`** — `SmolVLMVLA` 是顶层 `PreTrainedModel`，由以下模块组成：
1. SmolVLM-500M 主干（SigLIP 视觉编码器 + 语言 Transformer）
2. `SmolVLMActionTransformer` — 时序/动作解码头
3. 动作空间（归一化 + 损失）

训练采用**条件流匹配**：在噪声与目标动作之间插值，训练模型预测速度场，推理时通过 Euler 积分生成动作。

**`transformer_smolvlm.py`** — `SmolVLMActionTransformer`，支持四种模式（由配置参数控制）：
- 基线（`use_adaln=False`, `use_ctaf=False`, `use_psca=False`）：标准 concat 模式 Transformer
- `use_adaln=True`：DiT 风格的自适应 LayerNorm 条件
- `use_ctaf=True`：傅里叶系数解码器 — 池化动作特征 → 预测傅里叶系数 → 通过 `query_fourier()` 重建平滑轨迹
- `use_psca=True`：每个 MLP 块加 LoRA 适配器（rank=8，B 初始化为 0），参数高效适配

**`configuration_smolvlm_vla.py`** — HuggingFace `PretrainedConfig` 子类。关键字段：`hidden_size`、`depth`、`num_heads`、`action_mode`、`image_size`、`use_adaln`、`use_ctaf`、`use_psca`、`num_fourier_freqs`、`psca_rank`。

**`processing_smolvlm_vla.py`** — `SmolVLMVLAProcessor`，准备多模态输入：3路相机视角（agentview、eye_in_hand、third_person），分辨率 384×384 或 512×512，以及语言指令的 tokenization。

**`action_hub.py`** — 动作空间注册表。`libero_joint` 模式 = 7 维动作（Δxyz、Δeuler、夹爪），8 维本体感知状态，支持 z-score 归一化/反归一化。

### 数据集（`datasets/`）

**`dataset_smolvlm.py`** — `SmolVLMDataReader`（`IterableDataset`）：从 HDF5 文件读取 episode 窗口，应用归一化，返回 batch，键为 `image_input [B,V,C,H,W]`、`proprio [B,8]`、`action [B,T,7]`、`language_instruction`。

**`domain_handler/`** — 可插拔数据处理器模式。`libero_hdf5.py` 实现 LIBERO HDF5 加载，新数据集通过 `registry.py` 注册。

### 数据流

```
LIBERO HDF5 → create_libero_meta.py → JSON 元数据
                                           ↓
compute_libero_norm_stats.py → 归一化统计 JSON
                                           ↓
SmolVLMDataReader → SmolVLMVLAProcessor → SmolVLMVLA → 流匹配损失
```

### 评估流程

```
serve_smolvlm_libero.py（FastAPI + WebSocket 推理服务）
         ↑ WebSocket
libero_client.py × 4 并行（每个 LIBERO 套件一个）
```

## 关键文件速查

| 文件 | 用途 |
|------|------|
| `paths.env` | 本机路径与凭证（已 git-ignore） |
| `train_smolvlm.py` | 训练主入口 |
| `train_smolvlm_small.sh` | 单卡训练（Small 模型） |
| `train_smolvlm_large.sh` | 多卡训练（Large 模型） |
| `models/modeling_smolvlm_vla.py` | 顶层模型类 |
| `models/transformer_smolvlm.py` | 动作 Transformer 头（CTAF + PSCA） |
| `datasets/dataset_smolvlm.py` | 数据加载流水线 |
| `evaluation/libero/serve_smolvlm_libero.py` | 推理服务 |
