# SimVLA-HyperNet 算法创新文档

## 背景

本工作基于 SimVLA 框架（main 分支）进行改进。SimVLA 原始方法已包含：
- SmolVLM-500M 作为视觉语言骨干（SigLIP + Mistral）
- Flow Matching 动作生成框架
- 两种条件化方式：Concat（默认）和 AdaLN

本文档描述在此基础上的全部创新点。

---

## 原始 SimVLA 的局限

| 问题 | 具体表现 |
|------|---------|
| 条件化能力弱 | Concat 只在注意力层融合，无法直接调制 MLP 权重 |
| 时间采样固定 | 全程 Beta(1.5,1)，不随训练进度调整 |
| 无辅助监督 | 只有 flow matching 损失，缺乏对环境动态的建模 |
| 推理效率低 | 每步重复跑完整 VLM，静态特征重复计算 |
| 无后训练机制 | 缺乏利用轨迹质量差异的微调方法 |

---

## 创新点一：HyperNet 参数级任务条件化

### 思路

原始 Concat 方式只在注意力层融合 VLM 特征，MLP 权重对所有任务完全相同。HyperNet 让 VLM 特征直接生成每一层 MLP 的权重增量，实现**参数级的任务条件化**。

### 实现

```
VLM features [B, T_enc, 576]
      ↓ mean pool → task_vec [B, 576]
      ↓ shared trunk: Linear(576→768)→SiLU→Linear(768→32)
task_emb [B, 32]
      ↓ 12 个独立 per-layer head
(A_fc1, B_fc1, A_fc2, B_fc2) × 12层
```

每层 MLP 的前向计算变为：

```
x @ (W + B@A)^T  =  x @ W^T  +  (x @ A^T) @ B^T
```

LoRA 风格分解，rank=4 时显存节省约 1000x（相比展开完整 delta 矩阵）。

**双路径并行**：保留原始 Concat token-level 路径，HyperNet 作为额外的参数级路径叠加，两者互补。

**零初始化**：所有 head 权重初始化为 0，训练初期 delta=0，不破坏预训练权重。

新增参数量：~12.3M（rank=4，hidden=768）。

---

## 创新点二：课程学习时间采样

### 思路

Flow Matching 的时间步 t 决定了训练时关注哪个阶段的去噪过程。固定的 Beta(1.5,1) 无法适应训练不同阶段的需求。

### 实现

```python
def sample_time(B, global_step, total_steps, device):
    progress = global_step / total_steps
    if progress < 0.3:
        alpha = 1.0   # 均匀采样，探索全局动作结构
    elif progress < 0.7:
        alpha = 1.5   # 标准设置，平衡探索
    else:
        alpha = 2.5   # 偏向小 t，强化精细动作控制
```

| 训练阶段 | alpha | 效果 |
|---------|-------|------|
| 0~30% | 1.0 | 均匀覆盖全时间范围，学习粗粒度结构 |
| 30~70% | 1.5 | 原始设置，平衡 |
| 70~100% | 2.5 | 集中在小 t（精细阶段），提升动作精度 |

---

## 创新点三：辅助世界模型损失

### 思路

仅用 flow matching 损失训练，模型对环境动态没有显式建模。增加对下一时刻本体感知的预测，作为辅助监督信号，促使模型学习环境动态。

### 实现

在 Action Transformer 的第一个 token 上接一个状态预测头：

```python
self.state_pred_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.SiLU(),
    nn.Linear(hidden_size // 2, dim_proprio),
)
```

训练时同时优化：

```
loss = velocity_loss + state_loss_weight * state_loss
```

其中 `state_loss = MSE(predicted_next_proprio, actual_next_proprio)`。

数据集侧同步提供 `next_proprio` 标签（`libero_hdf5.py`）。

---

## 创新点四：推理加速——静态-动态特征解耦

### 思路

LIBERO 评估时，同一 episode 内语言指令和 agentview 图像基本不变，但每步都重新跑完整 VLM 造成大量冗余计算。

### 实现

将 VLM 特征拆分为两部分：

| 特征类型 | 内容 | 更新频率 | 计算量 |
|---------|------|---------|-------|
| 静态特征 | agentview + 语言 | episode 开始时一次 | 重（text_model） |
| 动态特征 | eye_in_hand | 每步更新 | 轻（vision_encoder only） |

```python
# episode 开始时调用一次
static_context = model.encode_static_context(input_ids, image_input, image_mask)

# 每步只跑动态视角
dynamic_feats = model.encode_dynamic_view(wrist_image)

# 拼接后推理
actions = model.generate_actions_with_cache(static_context, dynamic_feats, proprio)
```

**自适应推理停止**：当相邻两步速度向量的余弦相似度超过阈值（默认 0.97）时提前停止 Euler 积分，减少推理步数。

---

## 创新点五：AWR 离线强化学习微调

### 思路

行为克隆（BC）对所有轨迹一视同仁，但数据集中轨迹质量差异显著（有的高效、有的冗余）。Advantage-Weighted Regression（AWR）利用轨迹质量差异，对高质量轨迹赋予更高权重。

### 实现

以轨迹长度作为质量代理（更短 = 更高效）：

```python
reward_i = max_traj_len / traj_len_i
advantage_i = reward_i - mean(rewards)
weight_i = clip(exp(advantage_i / temperature), 0.1, 10.0)

loss = weight_i * MSE(v_t, u_t)  # 加权 flow matching 损失
```

在已有 checkpoint 上微调，无需环境交互：

```bash
python finetune_offline_rl.py \
    --checkpoint ./runs/simvla_hypernet/ckpt-50000 \
    --temperature 0.5 --iters 20000 --learning_rate 5e-5
```

---

## 创新点汇总

| 创新点 | 解决的问题 | 关键文件 |
|--------|-----------|---------|
| HyperNet 参数级条件化 | 条件化能力弱，MLP 权重任务无关 | `models/transformer_smolvlm.py` |
| 课程学习时间采样 | 固定采样分布无法适应训练进度 | `train_smolvlm.py` |
| 辅助世界模型损失 | 缺乏对环境动态的显式建模 | `models/modeling_smolvlm_vla.py` |
| 静态-动态特征解耦 | 推理时重复计算静态 VLM 特征 | `models/modeling_smolvlm_vla.py` |
| AWR 离线 RL 微调 | BC 忽略轨迹质量差异 | `finetune_offline_rl.py` |
