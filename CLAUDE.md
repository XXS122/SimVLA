# CLAUDE.md

Claude Code 在此仓库工作时的快速参考指南。详细设计文档见 `docs/` 目录。

**语言要求：所有回答、代码注释均必须使用中文。**

**文档维护：每次对话结束后，若出现新命令/路径/参数，立即更新本文档。**

---

## 路径配置（paths.env）

复制模板并填入本机路径（已加入 `.gitignore`）：
```bash
cp paths.env.example paths.env
```

| 变量 | 用途 |
|---|---|
| `SIMVLA_SMOLVLM_MODEL` | SmolVLM 预训练模型路径 |
| `SIMVLA_VLABENCH_DATA` | VLABench 数据集路径 |
| `SIMVLA_LIBERO_DATA` | LIBERO 数据集根目录 |
| `SIMVLA_VLABENCH_CODE` | VLABench 代码库路径 |
| `SIMVLA_EVAL_RESULTS` | 评估结果保存目录 |
| `SIMVLA_CUDA_DEVICES` | GPU 编号（如 `"0"` / `"0,1"`） |
| `SIMVLA_NUM_GPUS` | 训练进程数 |

当前机器配置示例：
```
SIMVLA_SMOLVLM_MODEL=/datasets/models/smolvlm/SmolVLM-500M-Instruct
SIMVLA_VLABENCH_DATA=/datasets/vlabench/data/1.0.0
SIMVLA_VLABENCH_CODE=/datasets/code/VLABench
SIMVLA_EVAL_RESULTS=/datasets/simvla_output/eval_results
SIMVLA_CUDA_DEVICES=0
SIMVLA_NUM_GPUS=1
```

---

## 环境安装

```bash
conda create -n simvla python=3.10 -y && conda activate simvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

---

## 数据准备

### VLABench
```bash
python create_vlabench_meta.py \
    --data_dir /root/dataset/vlabench-data/1.0.0 \
    --output ./datasets/metas/vlabench_train.json

python compute_vlabench_norm_stats.py \
    --data_dir /root/dataset/vlabench-data/1.0.0 \
    --output ./norm_stats/vlabench_norm.json
```

### LIBERO
```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json

python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

---

## 训练命令

### 基础训练
```bash
# VLABench 基础（无 AdaLN）
bash train_smolvlm_vlabench.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# VLABench + CVAE 子目标
bash train_smolvlm_subgoal.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]

# LIBERO 小模型
bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
```

### HiPhys-VLA 训练（新）
```bash
# 完整 HiPhys-VLA（HistoryEncoder + PhysicsDecoder）
bash train_smolvlm_hiphys.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# 示例：
bash train_smolvlm_hiphys.sh 16 0.1 ./simvla_output/simvla_hiphys
# 从断点续训：
bash train_smolvlm_hiphys.sh 16 0.1 ./simvla_output/simvla_hiphys ./simvla_output/simvla_hiphys/ckpt-50000

# 仅 HistoryEncoder（消融，track_5）
python train_smolvlm.py \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --action_mode vlabench_joint \
    --use_adaln --use_history_encoder --history_seq_len 4 --switch_loss_weight 0.05 \
    --batch_size 16 --output_dir ./simvla_output/simvla_history

# 仅 PhysicsDecoder（消融，track_3）
python train_smolvlm.py \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --action_mode vlabench_joint \
    --use_adaln --use_physics_cot --physics_weight 0.01 \
    --batch_size 16 --output_dir ./simvla_output/simvla_physics
```

关键训练参数：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--use_adaln` | DiT 风格条件注入 | False |
| `--use_history_encoder` | GRU 历史感知（需 adaln） | False |
| `--history_seq_len` | 历史帧数 K | 4 |
| `--switch_loss_weight` | gripper 切换辅助损失权重 | 0.05 |
| `--use_physics_cot` | 物理谓词嵌入（需 adaln） | False |
| `--physics_weight` | 物理谓词辅助损失权重 | 0.01 |
| `--use_subgoal_vae` | CVAE 子目标潜变量（需 adaln） | False |
| `--no_cross_attn` | 禁用 cross-attention（消融用） | False |
| `--use_huber_loss` | Huber loss 替代 MSE | False |
| `--gripper_weight` | gripper 维度损失倍率 | 1.0 |
| `--time_sampling` | FM 时间步采样：beta/logit_normal/cosine | beta |

---

## 评估

### VLABench（两个环境）
```bash
# 终端1（simvla 环境）：启动推理服务器
conda activate simvla
CUDA_VISIBLE_DEVICES=0 python evaluation/vlabench/serve_smolvlm_vlabench.py \
    --checkpoint /path/to/ckpt \
    --norm_stats ./norm_stats/vlabench_norm.json \
    --smolvlm_model /path/to/SmolVLM-500M-Instruct \
    --port 8001

# 终端2（vlabench 环境）：运行评估
conda activate vlabench
cd /path/to/VLABench
python /path/to/SimVLA/evaluation/vlabench/evaluate_simvla.py \
    --eval-track track_1_in_distribution \
    --n-episode 10 --port 8001 --save-dir /path/to/results

# 支持的 track：
# track_1_in_distribution, track_2_cross_category, track_3_common_sense,
# track_4_semantic_instruction, track_5_long_horizon, track_6_unseen_texture
```

### LIBERO
```bash
cd evaluation/libero
python serve_smolvlm_libero.py --checkpoint /path/to/ckpt --port 8000
# 另一终端：
python libero_client.py --port 8000 --task_suite libero_object --num_trials 50
```

VLABench 测试集统计：

| Track | 任务数 | 总 episode |
|---|---|---|
| track_1_in_distribution | 10 | 500 |
| track_2_cross_category | 10 | 460 |
| track_3_common_sense | 10 | 500 |
| track_4_semantic_instruction | 10 | 500 |
| track_5_long_horizon | 7 | 350 |
| track_6_unseen_texture | 10 | 500 |

---

## Git 推送（本环境代理问题）

本环境 HTTP 代理无推送权限，需用 GitHub token 临时推送：

```bash
# 1. 设置 token（token 在 https://github.com/settings/tokens 生成，勾选 repo 权限）
git remote set-url origin https://YOUR_TOKEN@github.com/XXS122/SimVLA.git
git push -u origin claude/init-project-kZV1L

# 2. 推送后立即清除 token
git remote set-url origin https://github.com/XXS122/SimVLA.git
```

---

## 架构概览

详见 `docs/` 目录：

| 文档 | 内容 |
|---|---|
| `docs/hiphys_vla_design.md` | HiPhys-VLA 完整设计（HistoryEncoder + PhysicsDecoder） |
| `docs/related_papers.md` | 相关论文综述（35篇，覆盖 track_3/5/2/6） |
| `docs/simvla_architecture.md` | SimVLA 整体架构图 |
| `docs/subgoal_vae_design.md` | SubgoalVAE + LatentFlow 设计细节 |

### 核心模块

| 文件 | 作用 |
|---|---|
| `models/modeling_smolvlm_vla.py` | 顶层 `SmolVLMVLA` — 训练/推理主循环 |
| `models/transformer_smolvlm.py` | 动作 Transformer — DiTBlock/HistoryEncoder/PhysicsDecoder |
| `models/configuration_smolvlm_vla.py` | HuggingFace PretrainedConfig 子类 |
| `models/action_hub.py` | 动作空间归一化（libero_joint / vlabench_joint） |
| `models/processing_smolvlm_vla.py` | SmolVLMVLAProcessor — 语言编码 |
| `datasets/dataset_smolvlm.py` | SmolVLMDataReader — 无限加权多数据集采样 |
| `datasets/domain_handler/vlabench_rlds.py` | VLABench TFRecord 读取 + 弱监督标签计算 |
| `datasets/domain_handler/libero_hdf5.py` | LIBERO HDF5 读取 |
| `train_smolvlm.py` | accelerate 训练循环 |
| `evaluation/vlabench/serve_smolvlm_vlabench.py` | VLABench 推理服务器（含 GRU 状态持久化） |

### 动作 Transformer 模式

- **Concat 模式**（`use_adaln=False`）：默认，动作 token 与 VLM 特征拼接
- **AdaLN/DiT 模式**（`--use_adaln`）：条件 `c = t_emb + vlm_pool + proprio_emb` 通过自适应层归一化注入
- **HiPhys-VLA 模式**（`--use_adaln --use_history_encoder --use_physics_cot`）：`c` 额外加入 `h_cond`（GRU历史）和 `physics_cond`（物理谓词）
- **CVAE 子目标模式**（`--use_adaln --use_subgoal_vae`）：`c` 额外加入 `subgoal_proj(z_goal)`

### 训练监控

```bash
tensorboard --logdir runs/
```

| 指标 | 说明 |
|---|---|
| `velocity_loss` | Flow Matching 主损失 |
| `switch_loss` | GRU gripper 切换预测损失 |
| `physics_loss` | 物理谓词弱监督损失 |
| `kl_loss` | CVAE KL 散度（仅 subgoal 模式） |
