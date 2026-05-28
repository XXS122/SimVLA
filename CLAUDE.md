# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本文件为 Claude Code 在此仓库中工作提供指导。

**重要：所有回答请使用中文。**

## 环境安装

### simvla 训练环境（Python 3.10）

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets -i https://mirrors.aliyun.com/pypi/simple/
```

### libero 评估环境（Python 3.8，仅评估时需要）

```bash
conda create -n libero python=3.8.13 -y
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

### 路径配置

训练和评估脚本通过 `paths.env` 读取路径，首次使用需配置（**不提交 git**）：

```bash
# paths.env
export SIMVLA_SMOLVLM_MODEL='/datasets/models/smolvlm/SmolVLM-500M-Instruct'
export LIBERO_DATASETS='/datasets/liber-datasets'
export SIMVLA_CHECKPOINTS='/datasets/models/simvla-model/model'
export SIMVLA_RESUME_CKPT=''          # 续训 checkpoint 路径，留空则从头训练
export CUDA_DEVICES="0,1"             # 训练脚本使用的 GPU 列表
export NUM_GPUS=2                     # 对应 GPU 数量
export WANDB_API_KEY="<your_wandb_api_key>"
export WANDB_PROJECT="simvla"
```

---

## 常用命令

**准备数据集元数据**（一次性）：
```bash
python create_libero_meta.py \
    --data_dir /datasets/liber-datasets \
    --subsets libero_goal  \
    --output ./datasets/metas/libero_train.json
```

**计算归一化统计量**（一次性）：
```bash
python compute_libero_norm_stats.py \
    --data_dir /datasets/liber-datasets \
    --subsets libero_goal \
    --output ./norm_stats/libero_norm.json
```

**训练：**
```bash
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
```

**AWR 离线强化学习微调：**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes=2 --mixed_precision bf16 \
    finetune_offline_rl.py \
    --checkpoint ./runs/simvla_hypernet/ckpt-50000 \
    --train_metas_path ./datasets/metas/libero_train.json \
    --norm_stats_path ./norm_stats/libero_norm.json \
    --temperature 0.5 --iters 20000 --learning_rate 5e-5 \
    --output_dir ./runs/awr_finetune
# 奖励 = max_traj_len / traj_len（轨迹越短权重越高），用加权 BC 损失微调
```

**评估（串行，2 张 GPU）：**

路径从 `paths.env` 自动读取。在 `simvla` 环境下启动服务器，在 `libero` 环境下跑客户端。

```bash
# 加载路径配置
source paths.env
```

```bash
# 方式一：一键串行跑全部 4 个套件（脚本自动启动服务器）
# 注意：libero_client.py 需要 libero conda 环境
cd evaluation/libero
bash run_eval_serial.sh [port] [num_trials] [output_prefix] [gpu_server] [gpu_client]
# 默认：port=8102, trials=50, gpu_server=0, gpu_client=1
bash run_eval_serial.sh 8102 50 eval_simvla 0 1

# 方式二：手动分两个终端（推荐，便于调试）
# 终端 1（simvla 环境）— 启动推理服务器
conda activate simvla
python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/simvla_hypernet/ckpt-80000 \
    --norm_stats ./norm_stats/libero_norm.json \
    --smolvlm_model /datasets/models/smolvlm/SmolVLM-500M-Instruct \
    --port 9999

# 终端 2（libero 环境）— 跑单个套件
conda activate libero
cd evaluation/libero
python libero_client.py \
    --host 127.0.0.1 --port 9999 \
    --client_type websocket \
    --task_suite libero_goal \
    --num_trials 100
```

可选 `--task_suite`：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`

评估结果保存在 `evaluation/libero/eval_results/` 目录下。

`run_eval_all.sh` 需要 4 张卡并行，本机双卡不适用。

```bash
# 恢复备份文件（models/ 目录被修改后可用此命令回退）
cp .backup/* models/ && cp .backup/train_smolvlm.py . && cp .backup/serve_smolvlm_libero.py evaluation/libero/ && cp .backup/libero_hdf5.py datasets/domain_handler/
```

## 架构

### 模型（`models/`）

**`SmolVLMVLA`**（`modeling_smolvlm_vla.py`）— 顶层 `PreTrainedModel`，包含两个子模块：

1. **`self.vlm`** — SmolVLM-500M-Instruct（Idefics3 架构，SigLIP 视觉编码器 + Mistral 文本骨干，VLM 隐层 576 维）。通过 `forward_vlm_efficient()` 将多视角图像 + 语言编码为统一特征序列，手动执行 SigLIP → connector → text_model，避免 HuggingFace `generate()` 的额外开销。

2. **`self.transformer`** — `SmolVLMActionTransformer`（`transformer_smolvlm.py`）。ViT 风格的 Transformer，接收 VLM 特征、带噪动作、时间步和本体感知，预测 flow matching 速度场。通过 `use_adaln` 控制两种条件注入模式：
   - `False`（默认）：concat 条件——动作 token 与 VLM 特征拼接
   - `True`：DiT 风格 AdaLN——时间步/本体感知通过自适应层归一化注入

**模型规格：**

| 变体 | hidden_size | depth | num_heads | 训练 GPU |
|------|-------------|-------|-----------|---------|
| Small | 768 | 12 | 12 | GPU 0–3 |
| Large | 1024 | 24 | 16 | GPU 4–7 |

**`SmolVLMVLAConfig`**（`configuration_smolvlm_vla.py`）— 随每个 checkpoint 序列化保存。关键字段：`hidden_size`、`depth`、`num_heads`、`action_mode`、`num_actions`、`use_adaln`、`use_hypernet`、`hypernet_rank`、`image_size`。**`use_adaln` 和 `use_hypernet` 在 checkpoint 创建后不可更改，且二者互斥。**

**`SmolVLMVLAProcessor`**（`processing_smolvlm_vla.py`）— 处理图像预处理（GPU 端 resize 到 384×384 或 512×512，ImageNet 归一化）和文本 tokenization（max_length=50）。`encode_image()` 是快速路径；`encode_image_legacy()` 使用 HuggingFace processor 以保持兼容性。

### 动作空间（`models/action_hub.py`）

动作空间通过 `@register_action("name")` 注册，用 `build_action_space(name)` 实例化。当前已注册：`libero_joint` — 7 维 delta 动作（xyz_delta + euler_delta + gripper），8 维本体感知（ee_pos + axis_angle + gripper_states）。归一化统计量从 `norm_stats/libero_norm.json` 加载，格式为 `{"norm_stats": {"state": {...}, "actions": {...}}}`，包含 mean/std/q01/q99。

### HyperNet 变体（`transformer_smolvlm.py`）

`use_hypernet=True` 时启用 `SmolVLMActionTransformerV2`：VLM 特征经 HyperNet 生成低秩权重增量（rank 由 `hypernet_rank` 控制，默认 4），注入动作 Transformer 的 MLP 层，实现任务条件化权重调制。与 `use_adaln` 互斥，不可同时启用。

### 训练（`train_smolvlm.py`）

Flow matching：采样 `t ~ Beta(1.5, 1)`，插值 `x_t = t*noise + (1-t)*action`，用 MSE 回归速度场 `u_t = noise - action`。三个优化器参数组，各自独立学习率：

| 参数组 | 冻结期行为 | 解冻后 |
|---|---|---|
| `vlm` | lr=0（冻结） | `lr * learning_coef` |
| `transformer_core` | lr=0（冻结） | `lr` |
| `action_heads` | `lr`（始终训练） | `lr` |

`freeze_steps`（默认 1000）先只热身动作头，再解冻整个模型。支持 WandB 和 TensorBoard 日志，checkpoint 保存为 `./runs/{run_name}/ckpt-{step}/model.safetensors` + `state.json`。

### 数据集（`datasets/`）

数据管道流程：`LiberoHDF5 文件` → `LiberoHDF5Handler`（解码、euler→axis-angle 转换）→ `SmolVLMDataReader`（采样、拼接多视角）→ `SmolVLMVLAProcessor`（resize、归一化）→ 训练 batch。

- **`SmolVLMDataReader`**（`dataset_smolvlm.py`）— 无限 `IterableDataset`，输出 `{language_instruction, image_input [V,C,384,384], image_mask, proprio [8], action [T,7]}`
- **`LiberoHDF5Handler`**（`datasets/domain_handler/libero_hdf5.py`）— 读取 agentview_rgb + eye_in_hand_rgb（原始 128×128 → resize 到 384×384），提取 7 维 delta 动作，ee_ori（euler）→ axis-angle 转换
- **领域处理器**通过 `registry.py` 注册/查找；数据集权重在 `domain_config.py` 中配置（各 LIBERO 子集默认权重均为 1.0）

### 推理与部署

`generate_actions()` — 对 flow matching ODE 做 Euler 积分，默认 10 步。评估服务器（`evaluation/libero/serve_smolvlm_libero.py`）通过 WebSocket 暴露推理接口，使用 `msgpack_numpy` 序列化，支持每个连接的静态 VLM 特征缓存以提升效率。

客户端输入：8 维状态 [ee_pos(3), axis_angle(3), gripper_qpos(2)]；服务器输出：7 维动作 [delta_xyz(3), delta_axisangle(3), gripper_cmd(1)]。图像在客户端旋转 180°（LIBERO 相机朝向约定）。

各套件最大步数：libero_spatial/object/goal=800，libero_10/90=900。

## 关键路径（见 `paths.env`）

- 基础模型：`/datasets/models/smolvlm/SmolVLM-500M-Instruct`
- 评估 checkpoint：`/datasets/models/simvla-model/model`
- 训练 checkpoint：`./runs/{run_name}/ckpt-{step}/`，保存为 `model.safetensors` + `state.json`
- LIBERO HDF5 数据：`/datasets/liber-datasets`
- 归一化统计量：`./norm_stats/libero_norm.json`
- GPU：0 和 1（双卡机器）

## 约束

- 需要 `transformers>=4.57.0`（该版本起支持 SmolVLM）
- `flash-attn==2.5.6` 版本固定，安装时加 `--no-build-isolation`
- 评估需要独立的 `libero` conda 环境（Python 3.8）运行 LIBERO 模拟器；推理服务器在 `simvla` 环境（Python 3.10）中运行
- 本机双卡（GPU 0/1），串行跑 4 个评估套件；`run_eval_all.sh` 需要 4 张卡并行，本机不适用
- 小模型在 GPU 0–3 训练，大模型在 GPU 4–7 训练（在 shell 脚本中配置）
- 无单元测试或 lint 配置文件

## 单卡（4090）评估示例

```bash
# 终端 1（simvla 环境）— 启动推理服务器
conda activate simvla
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint ./runs/simvla_libero_small/ckpt-100000 \
    --norm_stats ./norm_stats/libero_norm.json \
    --smolvlm_model /datasets/models/smolvlm/SmolVLM-500M-Instruct \
    --port 8102

# 终端 2（libero 环境）— 跑单个套件
conda activate libero
cd evaluation/libero
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
    --host 127.0.0.1 --port 8102 \
    --client_type websocket \
    --task_suite libero_goal \
    --num_trials 50
```