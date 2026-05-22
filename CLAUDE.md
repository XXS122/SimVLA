# CLAUDE.md

本文件为 Claude Code 在此仓库中工作提供指导。

**重要：所有回答请使用中文。**

## 常用命令

**准备数据集元数据**（一次性）：
```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

**计算归一化统计量**（一次性）：
```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

**训练：**
```bash
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
```

**评估（串行，2 张 GPU）：**

路径从 `paths.env` 自动读取。在 `simvla` 环境下启动服务器，在 `libero` 环境下跑客户端。

```bash
# 方式一：一键串行跑全部 4 个套件（脚本自动启动服务器）
# 注意：libero_client.py 需要 libero conda 环境
cd evaluation/libero
bash run_eval_serial.sh [port] [num_trials] [output_prefix] [gpu_server] [gpu_client]
# 默认：port=8102, trials=50, gpu_server=0, gpu_client=1
bash run_eval_serial.sh 8102 50 eval_simvla 0 1

# 方式二：手动分两个终端
# 终端 1（simvla 环境）— 启动推理服务器
cd /datasets/code/SimVLA
CUDA_VISIBLE_DEVICES=0 python evaluation/libero/serve_smolvlm_libero.py \
    --checkpoint /datasets/models/simvla-model/model \
    --norm_stats ./norm_stats/libero_norm.json \
    --smolvlm_model /datasets/models/smolvlm/SmolVLM-500M-Instruct \
    --port 8102

# 终端 2（libero 环境）— 跑单个套件
cd /datasets/code/SimVLA/evaluation/libero
CUDA_VISIBLE_DEVICES=1 python libero_client.py \
    --host 127.0.0.1 --port 8102 \
    --client_type websocket \
    --task_suite libero_spatial \
    --num_trials 20
```


```bash
# 恢复改动
cp .backup/* models/ && cp .backup/train_smolvlm.py . && cp .backup/serve_smolvlm_libero.py evaluation/libero/ && cp .backup/libero_hdf5.py datasets/domain_handler/
```

## 架构

### 模型（`models/`）

**`SmolVLMVLA`**（`modeling_smolvlm_vla.py`）— 顶层 `PreTrainedModel`，包含两个子模块：

1. **`self.vlm`** — SmolVLM-500M-Instruct（Idefics3/Mistral 骨干）。通过 `forward_vlm_efficient()` 将多视角图像 + 语言编码为统一特征序列。该方法手动执行 SigLIP → connector → text_model，避免 HuggingFace `generate()` 的额外开销。

2. **`self.transformer`** — `SmolVLMActionTransformer`（`transformer_smolvlm.py`）。ViT 风格的 Transformer，接收 VLM 特征、带噪动作、时间步和本体感知，预测 flow matching 速度场。通过 `use_adaln` 控制两种条件注入模式：
   - `False`（默认）：concat 条件——动作 token 与 VLM 特征拼接
   - `True`：DiT 风格 AdaLN——时间步/本体感知通过自适应层归一化注入

**`SmolVLMVLAConfig`**（`configuration_smolvlm_vla.py`）— 随每个 checkpoint 序列化保存。关键字段：`hidden_size`、`depth`、`num_heads`、`action_mode`、`num_actions`、`use_adaln`、`use_hypernet`、`hypernet_rank`、`image_size`。`use_adaln` 和 `use_hypernet` 在 checkpoint 创建后不可更改。

**`SmolVLMVLAProcessor`**（`processing_smolvlm_vla.py`）— 处理图像预处理（GPU 端 resize 到 384×384 或 512×512，ImageNet 归一化）和文本 tokenization。`encode_image()` 是快速路径；`encode_image_legacy()` 使用 HuggingFace processor 以保持兼容性。

### 动作空间（`models/action_hub.py`）

动作空间通过 `@register_action("name")` 注册，用 `build_action_space(name)` 实例化。当前已注册：`libero_joint` — 7 维 delta 动作（xyz_delta + euler_delta + gripper），8 维本体感知（ee_pos + axis_angle + gripper_states）。归一化统计量从 `norm_stats/libero_norm.json` 加载。

### HyperNet 变体（`transformer_smolvlm.py`）

`use_hypernet=True` 时启用 `SmolVLMActionTransformerV2`：VLM 特征经 HyperNet 生成低秩权重增量（rank 由 `hypernet_rank` 控制，默认 4），注入动作 Transformer 的 MLP 层，实现任务条件化权重调制。与 `use_adaln` 互斥，不可同时启用。

### 训练（`train_smolvlm.py`）

Flow matching：采样 `t ~ Beta(1.5, 1)`，插值 `x_t = t*noise + (1-t)*action`，用 MSE 回归速度场 `u_t = noise - action`。三个优化器参数组，各自独立学习率：

| 参数组 | 冻结期 | 解冻后 |
|---|---|---|
| `vlm` | 0 | `lr * learning_coef` |
| `transformer_core` | 0 | `lr` |
| `action_heads` | `lr` | `lr` |

`freeze_steps`（默认 1000）先只热身动作头，再解冻整个模型。

### 数据集（`datasets/`）

`SmolVLMDataReader`（`dataset_smolvlm.py`）— 无限 `IterableDataset`，读取 LIBERO HDF5 文件。`datasets/domain_handler/` 中的领域处理器通过 `registry.py` 查找。`LiberoHDF5Handler` 读取 agentview_rgb + eye_in_hand_rgb（原始 128×128，resize 到 384×384），提取 7 维 delta 动作并将 euler 角转换为 axis-angle。数据集权重在 `domain_config.py` 中配置。

### 推理与部署

`generate_actions()` — 对 flow matching ODE 做 Euler 积分，默认 10 步。FastAPI 服务通过 `model.run(processor, host, port)` 暴露 `/act` 端点。评估服务器（`evaluation/libero/serve_smolvlm_libero.py`）封装此接口用于 LIBERO rollout。

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
- 评估需要独立的 `libero` conda 环境（Python 3.8）运行 LIBERO 模拟器；推理服务器在 `simvla` 环境中运行
- 本机双卡（GPU 0/1），串行跑 4 个评估套件；`run_eval_all.sh` 需要 4 张卡并行，本机不适用
- 小模型在 GPU 0–3 训练，大模型在 GPU 4–7 训练（在 shell 脚本中配置）
