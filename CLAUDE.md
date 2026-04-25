# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本文件为 Claude Code 在此仓库中工作时提供指导。

## 语言要求

**重要：所有回答、解释、代码注释均必须使用中文。**

## 文档维护要求

**重要：每次对话结束后，若出现了新的命令、路径、参数或操作流程，必须立即更新到本 CLAUDE.md 文档对应章节中，保持文档与实际使用一致。**

## 多机器路径配置

代码库通过 `paths.env` 文件管理机器特定路径，该文件已加入 `.gitignore`，不会提交到 git。

### 配置方式

复制模板并填入本机路径：
```bash
cp paths.env.example paths.env
# 编辑 paths.env，填入本机实际路径
```

`paths.env` 中定义的环境变量：

| 变量 | 用途 |
|---|---|
| `SIMVLA_SMOLVLM_MODEL` | SmolVLM 预训练模型路径 |
| `SIMVLA_VLABENCH_DATA` | VLABench 数据集路径（含 .tfrecord 文件的目录） |
| `SIMVLA_LIBERO_DATA` | LIBERO 数据集根目录（含 libero_10/goal/object/spatial 子目录） |
| `SIMVLA_VLABENCH_CODE` | VLABench 代码库路径（评估时需要） |
| `SIMVLA_EVAL_RESULTS` | 评估结果保存目录 |
| `SIMVLA_CUDA_DEVICES` | 训练使用的 GPU 编号（如 `"0"` / `"0,1"` / `"0,1,2,3,4,5,6,7"`） |
| `SIMVLA_NUM_GPUS` | 训练进程数，须与 `SIMVLA_CUDA_DEVICES` 中 GPU 数量一致 |

### 当前机器路径（/datasets/...，单卡）
```
SIMVLA_SMOLVLM_MODEL=/datasets/models/smolvlm/SmolVLM-500M-Instruct
SIMVLA_VLABENCH_DATA=/datasets/vlabench/data/1.0.0
SIMVLA_VLABENCH_CODE=/datasets/code/VLABench
SIMVLA_EVAL_RESULTS=/datasets/simvla_output/eval_results
SIMVLA_CUDA_DEVICES=0
SIMVLA_NUM_GPUS=1
```

### 工作原理
- **Shell 脚本**：启动时自动 `source paths.env`，用 `${SIMVLA_XXX:-/root/fallback}` 语法，未设置时回退到原始路径
- **Python 脚本**：argparse `default=os.environ.get("SIMVLA_XXX", "/root/fallback")`，也可手动 `source paths.env && python ...`

### VLABench 项目路径配置（/datasets/code/VLABench）

VLABench 项目采用相同的 `paths.env` 机制，变量前缀为 `VLABENCH_`：

| 变量 | 用途 | 默认回退值 |
|---|---|---|
| `VLABENCH_ROOT` | VLABench 包根目录（含 configs/ tasks/ 等） | `/datasets/code/VLABench/VLABench` |
| `VLABENCH_DATA` | 轨迹数据集保存/读取目录 | `./datasets` |
| `VLABENCH_CUDA_DEVICES` | 使用的 GPU 编号 | `0` |
| `VLABENCH_NUM_GPUS` | GPU 数量（用于计算进程数） | `1`（或 nvidia-smi 自动检测） |
| `VLABENCH_PROCS_PER_GPU` | 数据生成每卡并行进程数 | `8` |
| `VLABENCH_EVAL_PROCS_PER_GPU` | 评估每卡并行进程数 | `2` |
| `VLABENCH_MODEL_CKPT` | 基础模型路径（evaluate_policy.py） | 空 |
| `VLABENCH_LORA_CKPT` | LoRA checkpoint 路径 | 空 |
| `VLABENCH_OPENVLA_LORA_CKPT` | OpenVLA LoRA checkpoint 路径 | 空 |

当前机器配置（写入 `/datasets/code/VLABench/paths.env`）：
```
VLABENCH_ROOT=/datasets/code/VLABench/VLABench
VLABENCH_DATA=/datasets/vlabench/datasets
VLABENCH_CUDA_DEVICES=0
VLABENCH_NUM_GPUS=1
VLABENCH_PROCS_PER_GPU=8
VLABENCH_EVAL_PROCS_PER_GPU=2
```

受影响脚本：`sh/data_generation/multi_gpu_data_generation.sh`、`sh/evaluation/example_multi_gpu_eval.sh`、`evaluate_openvla.sh`、`dataset_generation.sh`、`scripts/evaluate_policy.py`。

## 命令

### 环境配置
```bash
conda create -n simvla python=3.10 -y && conda activate simvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

### 数据准备（LIBERO）
```bash
# 1. 生成数据集元数据 JSON
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json

# 2. 计算动作/状态归一化统计量
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

### 数据准备（VLABench）
```bash
# 数据位置：/root/dataset/vlabench-data/1.0.0（RLDS TFRecord 格式，512个shard）

# 1. 生成数据集元数据 JSON
python create_vlabench_meta.py \
    --data_dir /root/dataset/vlabench-data/1.0.0 \
    --output ./datasets/metas/vlabench_train.json

# 2. 计算动作/状态归一化统计量
python compute_vlabench_norm_stats.py \
    --data_dir /root/dataset/vlabench-data/1.0.0 \
    --output ./norm_stats/vlabench_norm.json
# 可加 --max_shards 50 用部分数据加速估算
```

### 训练
```bash
# 小模型（隐藏层768，12层，12头）- LIBERO，双 GPU
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# 大模型（隐藏层1024，24层，16头）- 4 GPU
bash train_smolvlm_large.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# VLABench 基础训练（AdaLN 关闭，自动完成 meta 生成和 norm stats 计算）
bash train_smolvlm_vlabench.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# VLABench + CVAE 子目标潜变量训练（AdaLN + SubgoalVAE，双 GPU A800）
bash train_smolvlm_subgoal.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# 示例：
bash train_smolvlm_subgoal.sh 32 0.1 ./simvla_output/simvla_subgoal
# 从断点续训：
bash train_smolvlm_subgoal.sh 32 0.1 ./simvla_output/simvla_subgoal ./simvla_output/simvla_subgoal/ckpt-10000

# HiPhys-VLA 训练（HistoryEncoder + PhysicsPredicateDecoder，均需配合 --use_adaln）
# 仅 PhysCoT（物理谓词嵌入，track_3）：
python train_smolvlm.py \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --use_adaln --use_physics_cot --physics_weight 0.01 \
    --batch_size 16 --output_dir ./simvla_output/simvla_physics

# 仅 HistoryEncoder（GRU 历史感知，track_5）：
python train_smolvlm.py \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --use_adaln --use_history_encoder --history_seq_len 4 --switch_loss_weight 0.05 \
    --batch_size 16 --output_dir ./simvla_output/simvla_history

# HiPhys-VLA 完整（两者合并，track_3 + track_5）：
python train_smolvlm.py \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --use_adaln --use_physics_cot --use_history_encoder --history_seq_len 4 \
    --physics_weight 0.01 --switch_loss_weight 0.05 \
    --batch_size 16 --output_dir ./simvla_output/simvla_hiphys
```

所有训练脚本均从 `paths.env` 读取 `SIMVLA_CUDA_DEVICES` 和 `SIMVLA_NUM_GPUS`，使用 `accelerate` DDP，`bf16` 混合精度。各脚本的默认回退值：`small/vlabench` 双卡（6,7）、`large` 四卡（4,5,6,7）、`subgoal` 单卡（0）。

### 评估（LIBERO - WebSocket 服务器）
```bash
cd evaluation/libero
# 终端1：启动推理服务器（simvla 环境）
python serve_smolvlm_libero.py \
    --checkpoint /path/to/checkpoint \
    --norm_stats /path/to/libero_norm.json \
    --smolvlm_model /path/to/SmolVLM-500M-Instruct \
    --port 8000

# 终端2：运行评估客户端（需先安装 LIBERO，见 evaluation/libero/README.md）
# task_suite 可选：libero_spatial, libero_object, libero_goal, libero_10
CUDA_VISIBLE_DEVICES=0 python libero_client.py \
    --host 127.0.0.1 \
    --port 8000 \
    --client_type websocket \
    --task_suite libero_object \
    --num_trials 50 \
    --video_out ./eval_output

# 或使用 run_eval_all.sh 自动化（参数：port num_trials output_prefix "gpu_ids"）
bash run_eval_all.sh 8000 50 eval_simvla "0"
```

### 评估（VLABench）

VLABench 评估需要两个环境：
- **simvla 环境**：运行推理服务器
- **vlabench 环境**：运行评估客户端

#### 1. 准备 VLABench 环境
```bash
# 创建 vlabench 环境
conda create -n vlabench python=3.10 -y && conda activate vlabench
pip install dm_control==1.0.22 mujoco==3.2.6
pip install numpy scipy pillow matplotlib
pip install tensorflow tensorflow-datasets
pip install websockets msgpack-numpy

# 安装 EGL 渲染支持（headless 服务器必需）
apt-get update && apt-get install -y libegl1 libgl1-mesa-glx libglib2.0-0

# 下载 VLABench 资源文件（需手动从 Google Drive 下载并解压）
# obj.zip: https://drive.google.com/file/d/1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-/view
# scene.zip: https://drive.google.com/file/d/1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw/view
# 解压到 /root/code/VLABench/VLABench/assets/
```

#### 2. 生成 track_5_long_horizon 测试集
```bash
conda activate vlabench
cd /root/code/VLABench
VLABENCH_ROOT=/root/code/VLABench/VLABench MUJOCO_GL=egl \
python generate_track5_long_horizon.py --n-episodes 50

# 输出：VLABench/configs/evaluation/tracks/track_5_long_horizon.json
# 包含 7 个长期任务，每任务 50 个 episode，共 350 个测试样本
```

**VLABench 测试集统计**：
| Track | 任务数 | 每任务 episode | 总 episode |
|---|---|---|---|
| track_1_in_distribution | 10 | 50 | 500 |
| track_2_cross_category | 10 | 50（部分10） | 460 |
| track_3_common_sense | 10 | 50 | 500 |
| track_4_semantic_instruction | 10 | 50 | 500 |
| track_5_long_horizon | 7 | 50 | 350 |
| track_6_unseen_texture | 10 | 50 | 500 |

#### 3. 启动推理服务器（simvla 环境）
```bash
conda activate simvla
cd /root/code/SimVLA/evaluation/vlabench

CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_vlabench.py \
    --checkpoint /root/checkpoint/simvla_output/ckpt-100000 \
    --norm_stats ../../norm_stats/vlabench_norm.json \
    --smolvlm_model /root/model/smolvlm-500M \
    --port 8001

# 服务器监听 0.0.0.0:8001，使用 WebSocket + msgpack 协议
# 与 VLABench 的 OpenPiPolicy 客户端完全兼容
```

#### 4. 运行评估（vlabench 环境，另开终端）
```bash
conda activate vlabench
cd /root/code/VLABench

python /root/code/SimVLA/evaluation/vlabench/evaluate_simvla.py \
    --eval-track track_1_in_distribution \
    --n-episode 10 \
    --port 8001 \
    --save-dir /root/eval_results

# 支持的 track：
#   track_1_in_distribution, track_2_cross_category, track_3_common_sense,
#   track_4_semantic_instruction, track_5_long_horizon, track_6_unseen_texture

# 可选参数：
#   --replan-steps 4        # 每隔多少步重新推理动作 chunk
#   --metrics success_rate  # 评估指标（success_rate, intention_score, progress_score）
#   --visualization         # 保存可视化视频
```

#### 5. 查看结果
```bash
# 结果保存在：
# /root/eval_results/{track}_{timestamp}/simvla/evaluation_result.json

cat /root/eval_results/track_5_long_horizon_20260415_1100/simvla/evaluation_result.json

# 每个 task 完成后也会增量保存到：
# /root/eval_results/{track}_{timestamp}/metrics.json
# /root/eval_results/{track}_{timestamp}/{task}/detail_info.json
```

#### 6. 并行评估（同时跑多个 track，需多张 GPU）
```bash
# 终端1：推理服务器1（GPU 0，port 8001）
conda activate simvla
cd /root/code/SimVLA/evaluation/vlabench
CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_vlabench.py \
    --checkpoint /root/checkpoint/simvla_output/ckpt-100000 \
    --norm_stats ../../norm_stats/vlabench_norm.json \
    --smolvlm_model /root/model/smolvlm-500M \
    --port 8001

# 终端2：推理服务器2（GPU 1，port 8002）
conda activate simvla
cd /root/code/SimVLA/evaluation/vlabench
CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_vlabench.py \
    --checkpoint /root/checkpoint/simvla_output/ckpt-100000 \
    --norm_stats ../../norm_stats/vlabench_norm.json \
    --smolvlm_model /root/model/smolvlm-500M \
    --port 8002

# 终端3：评估客户端1（连 port 8001，跑 track_1）
(choose from 'track_1_in_distribution', 'track_2_cross_category', 'track_3_common_sense', 'track_4_semantic_instruction', 'track_5_long_horizon', 'track_6_unseen_texture')
conda activate vlabench
cd /root/code/VLABench
python /root/code/SimVLA/evaluation/vlabench/evaluate_simvla.py \
    --eval-track track_2_cross_category \
    --n-episode 10 --port 8001 --save-dir /root/eval_results

# 终端4：评估客户端2（连 port 8002，跑 track_2）
conda activate vlabench
cd /root/code/VLABench
python /root/code/SimVLA/evaluation/vlabench/evaluate_simvla.py \
    --eval-track track_5_long_horizon \
    --n-episode 10 --port 8002 --save-dir /root/eval_results
# 注意：每个客户端用不同 port，save-dir 相同但 track 名不同，结果目录自动区分
```

关键训练参数：`--use_adaln`（DiT 风格条件注入）、`--image_size 384|512`、`--num_actions 10`、`--freeze_steps 1000`（前 N 步冻结 VLM）、`--learning_coef 0.1`（VLM 学习率倍率）。

CVAE 子目标参数：`--use_subgoal_vae`（启用，需配合 `--use_adaln`）、`--subgoal_latent_dim 64`（潜变量维度）、`--kl_weight 0.001`（KL 权重上限）、`--kl_warmup_steps 10000`（KL annealing 步数）。

损失函数与采样参数（新增）：`--use_huber_loss`（Huber loss 替代 MSE，对噪声演示更鲁棒）、`--huber_delta 1.0`（Huber delta）、`--gripper_weight 5.0`（gripper 维度损失权重倍率，建议 3~10）、`--time_sampling logit_normal|cosine|beta`（Flow Matching 时间步采样策略，默认 beta）。

Cross-Attention 参数（新增）：`--use_adaln` 模式下默认启用 cross-attention（动作 token → VLM 全序列），可用 `--no_cross_attn` 禁用（消融实验用）。

LDM 参数：`--use_latent_flow`（启用 z 空间 Flow Matching，需配合 `--use_subgoal_vae`）、`--latent_flow_steps 5`（推理时 z 空间 Euler 积分步数）、`--latent_fm_weight 1.0`（z 空间 FM 损失权重）。

## 架构

SimVLA 是用于机器人操作的视觉-语言-动作（VLA）模型，由两个主要组件构成：冻结/微调的 VLM 骨干网络和 Flow Matching 动作头。

### 数据流
```
HDF5 文件 → LiberoHDF5Handler.iter_episode()
  ├─ 观测: agentview_rgb[T,128,128,3], eye_in_hand_rgb[T,128,128,3]
  ├─ 本体感知: ee_pos(3) + euler→axis_angle(3) + gripper(2) = 8维
  └─ 动作: delta_xyz(3) + delta_euler(3) + gripper(1) = 7维
       ↓ 图像增强（resize→384, ColorJitter, ImageNet 归一化）
SmolVLMDataReader → DataLoader → batch
       ↓ processor.encode_language() → input_ids
SmolVLMVLA.forward()
  ├─ forward_vlm_efficient(): SigLIP → connector → 拼接 text_embeds → LM → vlm_features[B, seq, 576]
  ├─ Flow Matching: t~Beta(1.5,1), x_t = t*noise + (1-t)*action_norm, 目标 u_t = noise - action
  └─ SmolVLMActionTransformer → MSE(v_t, u_t)
```

### 模块说明

| 文件 | 作用 |
|---|---|
| `models/modeling_smolvlm_vla.py` | 顶层 `SmolVLMVLA(PreTrainedModel)` — VLM 前向传播、Flow Matching 训练循环、Euler 推理、FastAPI 服务 |
| `models/transformer_smolvlm.py` | `SmolVLMActionTransformer` — 两种模式：Concat（`TransformerBlock`）或 AdaLN/DiT（`DiTBlock` / `DiTBlockWithCrossAttn`）。包含 `timestep_embedding`、`FinalLayer`、`SubgoalVAE`、`LatentFlowNet`、`HistoryEncoder`（GRU 历史感知）、`PhysicsPredicateDecoder`（物理谓词嵌入） |
| `models/action_hub.py` | `LiberoJointActionSpace`、`VLABenchJointActionSpace` — 动作/状态归一化（z-score 或分位数）。通过 `@register_action` 注册 |
| `models/configuration_smolvlm_vla.py` | HuggingFace `PretrainedConfig` 子类，通过 `save_pretrained` 序列化。含 CVAE 和 LDM 配置字段 |
| `models/processing_smolvlm_vla.py` | `SmolVLMVLAProcessor` — 封装 SmolVLM processor，训练时调用 `encode_language()` |
| `datasets/dataset_smolvlm.py` | `SmolVLMDataReader(IterableDataset)` — 无限加权多数据集采样器 |
| `datasets/domain_handler/registry.py` | `dataset_name → HandlerClass` 映射字典。**添加新数据集时在此修改。** |
| `datasets/domain_handler/libero_hdf5.py` | `LiberoHDF5Handler(DomainHandler)` — 读取 LIBERO HDF5 格式 |
| `datasets/domain_handler/vlabench_rlds.py` | `VLABenchRLDSHandler(DomainHandler)` — 读取 VLABench RLDS TFRecord 格式 |
| `datasets/domain_handler/base.py` | 抽象类 `DomainHandler` + `BaseHDF5Handler`，支持基于插值的轨迹采样 |
| `datasets/domain_config.py` | 多数据集采样比例 `DATA_WEIGHTS` 字典 |
| `datasets/utils.py` | `action_slice()`、`read_parquet()`、`decode_image_from_bytes()`、旋转转换工具 |
| `train_smolvlm.py` | 使用 `accelerate` 的训练循环。三个优化器参数组：`vlm`（前 N 步冻结）、`transformer_core`、`action_heads` |
| `train_smolvlm_vlabench.sh` | VLABench 基础训练入口脚本，自动完成 meta 生成和 norm stats 计算 |
| `train_smolvlm_subgoal.sh` | VLABench + CVAE 子目标潜变量训练脚本（AdaLN + SubgoalVAE，双卡 A800） |
| `create_vlabench_meta.py` | 生成 VLABench 训练元数据 JSON（扫描 TFRecord shard 列表） |
| `compute_vlabench_norm_stats.py` | 计算 VLABench 动作/状态归一化统计量，复用 `RunningStats`，输出格式与 LIBERO 一致 |
| `/root/code/VLABench/generate_track5_long_horizon.py` | 生成 track_5_long_horizon.json（7个长期任务，每任务50 episode），在 vlabench 环境下运行 |
| `evaluation/libero/serve_smolvlm_libero.py` | LIBERO WebSocket 推理服务器（msgpack_numpy 序列化） |
| `evaluation/vlabench/serve_smolvlm_vlabench.py` | VLABench WebSocket 推理服务器，含 quat→axis_angle 状态转换 |
| `evaluation/libero/libero_client.py` | LIBERO 评估客户端（WebSocket），搭配 `serve_smolvlm_libero.py` 使用 |
| `evaluation/vlabench/evaluate_simvla.py` | VLABench 评估客户端，复用 OpenPiPolicy 协议，支持全部6个 track |
| `read_rlds.py` | RLDS TFRecord 数据格式调试工具，可用于验证数据读取 |
| `data_process/view_data.py` | 数据可视化工具，用于检查 HDF5 轨迹图像和动作 |

### 设计文档

| 文件 | 内容 |
|---|---|
| `docs/simvla_architecture.md` | 完整架构图（Mermaid）和各模块说明 |
| `docs/subgoal_vae_design.md` | SubgoalVAE + LatentFlow 设计细节，含 KL annealing 策略 |
| `docs/pi0_comparison_experiment.md` | 与 Pi0 基线的对比实验设计 |

### 动作 Transformer 模式

**Concat 模式**（默认，`use_adaln=False`）：动作 token + 时间步 + 本体感知拼接后，将 VLM 特征追加到序列末尾：`x = cat([action_tokens, vlm_proj(vlm_features)], dim=1)`，仅解码动作位置。

**AdaLN/DiT 模式**（`--use_adaln`）：条件 `c = time_emb + vlm_pool + proprio_emb` 通过自适应层归一化注入每个 `DiTBlock`，条件信号与主序列分离更清晰。默认同时启用 `DiTBlockWithCrossAttn`，让动作 token 通过 cross-attention 直接 attend 到 VLM 全序列（而非仅 pooled 向量），显著提升视觉特征利用率。用 `--no_cross_attn` 可退回纯 AdaLN 模式（消融实验）。

**CVAE 子目标模式**（`--use_adaln --use_subgoal_vae`）：在 AdaLN 条件中额外注入子目标潜变量 `z_goal ∈ R^64`。训练时从后验 `q(z | vlm, action)` 采样，推理时从先验 `p(z | vlm)` 采样。条件变为 `c = time_emb + vlm_pool + proprio_emb + subgoal_proj(z_goal)`，使模型能表示子任务的多模态分布，对长程任务阶段切换更鲁棒。

**层次化双扩散模式**（`--use_adaln --use_subgoal_vae --use_latent_flow`）：在 CVAE 基础上，用 `LatentFlowNet` 在 z 空间（64维）做 Flow Matching，替换简单高斯先验采样。推理时先用 5 步 Euler 积分从噪声生成 `z_goal`，再用 10 步 Euler 积分生成动作序列，形成两层嵌套扩散结构（z 空间子目标层 + 动作空间执行层）。详见 `docs/subgoal_vae_design.md`。

### 添加新数据集

1. 创建 `datasets/domain_handler/mydata.py`，实现 `DomainHandler.iter_episode()` — 必须 yield 包含以下键的字典：`language_instruction`、`image_input[V,C,H,W]`、`image_mask[V]`、`abs_trajectory[T+1,D]`（索引0为当前状态，1:为未来动作）
2. 在 `datasets/domain_handler/registry.py` 注册：将 `"mydata_name": MyHandler` 添加到 `_REGISTRY`
3. 在 `datasets/domain_config.py` 添加采样权重
4. 若动作空间不同，在 `models/action_hub.py` 添加新的 `@register_action("mydata_joint")` 类
5. 创建元数据 JSON，包含字段：`dataset_name`、`datalist`（`{path, task}` 列表）、`data_dir`

### 关键设计决策

- **无 aux_visual_inputs**：与 FlorenceVLA 不同，所有摄像头视角直接通过 SmolVLM 处理。VLM 将 `[image1_patches, image2_patches, ..., text_tokens]` 作为单一序列处理。
- **Flow Matching 而非 DDPM**：使用线性插值 ODE，时间分布为 Beta(1.5,1)，推理时采用 Euler 积分（默认10步）。
- **延迟 VLM 解冻**：VLM 参数在前 `freeze_steps` 步 `lr=0`，之后以 `learning_rate * learning_coef` 解冻。
- **`abs_trajectory` 约定**：Handler yield `[T+1, D]`，索引0为当前状态（本体感知），索引1..T为未来动作。`action_slice()` 负责分离并可选计算增量。
- `datasets/utils.py` 已包含 `read_parquet()` 支持 Parquet 格式 — 非 HDF5 格式的基础设施已部分就绪。
- **CVAE 子目标潜变量**：`SubgoalVAE` 位于 `models/transformer_smolvlm.py`。训练时后验 encoder 接收 `vlm_pooled + action_chunk`，推理时先验 encoder 仅接收 `vlm_pooled`。使用 Free Bits（每维 KL ≥ 0.5）防止 KL 崩溃，KL annealing 前 `kl_warmup_steps` 步线性增加权重。`vlm_pooled` 在 `freeze_steps` 内 `.detach()` 防止梯度干扰 VLM。
- **Latent Diffusion Model（LDM）**：`LatentFlowNet` 位于 `models/transformer_smolvlm.py`，在 64 维 z 空间做 Flow Matching，以 `vlm_pooled` 为条件。训练时计算 z 空间速度场损失 `L_latent_fm = MSE(v_z, u_z)`；推理时用 5 步 Euler 积分从噪声生成 `z_goal`，替换简单高斯先验采样。输出层初始化为零，训练初期不干扰 CVAE 收敛。总损失：`L = L_velocity + λ_kl · L_KL + λ_fm · L_latent_fm`。

### VLABench 数据集说明

数据位置：`/root/dataset/vlabench-data/1.0.0`，RLDS TFRecord 格式，512个 shard。

| 字段 | 说明 |
|---|---|
| `steps/action` | 7维动作：xyz(3) + axis_angle(3) + gripper(1) |
| `steps/observation/ee_state` | 7维本体感知（与 action 同维） |
| `steps/observation/front` | 前置摄像头 JPEG |
| `steps/observation/wrist` | 腕部摄像头 JPEG |
| `steps/observation/image_0/1` | 额外摄像头 JPEG |
| `steps/language_instruction` | 语言指令（每步重复，取第0步） |

动作空间 `vlabench_joint`：proprio 7维（LIBERO 是8维，少一个 gripper 状态），action 7维相同。

### 训练监控指标

TensorBoard 日志写入 `runs/` 目录，查看命令：`tensorboard --logdir runs/`

关键指标：

| 指标 | 正常范围 | 说明 |
|---|---|---|
| `v_loss` | 与 baseline 相近 | 动作空间 velocity loss（Flow Matching 主损失） |
| `kl_loss` | 0.5 ~ 2.0 | 原始 KL 散度（CVAE 模式，Free Bits 每维 ≥ 0.5） |
| `z_fm_loss` | 0.1 ~ 1.0 | z 空间 FM loss（LDM 模式） |
| `kl_weight` | 0 → 0.001 | KL annealing 权重，前 `kl_warmup_steps` 步线性增加 |

消融实验对比（三种配置）：
- **Baseline**：`bash train_smolvlm_vlabench.sh`（无 CVAE，无 LDM）
- **Ablation-CVAE**：加 `--use_adaln --use_subgoal_vae`（仅 CVAE）
- **完整模型**：`bash train_smolvlm_subgoal.sh`（CVAE + LDM）
