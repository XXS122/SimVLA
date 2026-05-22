# HyperNet Policy 设计文档

**目标：** 用 Hypernetwork 替代当前 SimVLA 的 VLM-action 融合方式，让 VLM 的语义理解直接编码进 action transformer 的 MLP 权重，而不是作为输入特征。

**核心创新：** VLM 不再输出"特征"，而是输出"权重增量"——为每个任务动态生成专属的 action transformer MLP 参数。

---

## 1. 当前架构的问题

当前 Concat 模式：action tokens（10个）和 VLM features（500+个）拼成长序列做自注意力。所有任务共享同一套固定的 transformer 权重，VLM 的语义理解只能通过特征输入影响输出，无法改变网络的计算方式本身。

---

## 2. 新架构：HyperNet Policy

### 信息流

```
VLM features [B, T, 576]
      ↓ mean pool
task_vec [B, 576]
      ↓ HyperNet（3层 MLP）
      ↓
per_layer_deltas: 12 组低秩增量
  每组：A_fc1[r,768], B_fc1[3072,r], A_fc2[r,3072], B_fc2[768,r]
  ΔW_fc1 = B_fc1 @ A_fc1  →  [3072, 768]
  ΔW_fc2 = B_fc2 @ A_fc2  →  [768, 3072]

action tokens [B, 10, 7]
      ↓ action_encoder
      ↓ pos_emb
      ↓ 12 × TransformerBlock（MLP 权重 = 基础权重 + ΔW）
      ↓ action_decoder
velocity [B, 10, 7]
```

### HyperNet 结构

```python
HyperNet(
  Linear(576, 768), SiLU,
  Linear(768, 768), SiLU,
  Linear(768, num_layers * 4 * rank * (768 + 3072))
  # 12层 × 4矩阵 × rank=16 × (768+3072) = 12×4×16×3840 = 2,949,120 输出
)
```

rank=16，每层生成 4 个低秩矩阵（fc1 的 A/B，fc2 的 A/B）。

### 修改后的 MLP Forward

```python
# 原来：
x = fc2(fc1(x))

# 改后：
# delta_fc1 = B_fc1 @ A_fc1，shape [3072, 768]
# delta_fc2 = B_fc2 @ A_fc2，shape [768, 3072]
h = F.linear(x, fc1.weight + delta_fc1, fc1.bias)
h = act(h)
out = F.linear(h, fc2.weight + delta_fc2, fc2.bias)
```

### VLM features 的处理

VLM features 仍然通过 `vlm_proj` 投影后拼接到 action 序列（保留原有 concat 路径），HyperNet 是**额外的**权重调制路径，两条路径并行：

- Concat 路径：提供细粒度的视觉-语言上下文（token 级别）
- HyperNet 路径：提供任务级别的网络结构调制（参数级别）

这样不需要删除原有 concat 逻辑，风险更低。

---

## 3. 改动文件

### `models/transformer_smolvlm.py`

新增：
- `HyperNet` 类：`task_vec [B, 576] → per_layer_deltas`
- `HyperNetMlp` 类：接收 `delta_fc1, delta_fc2`，在 forward 里用 `F.linear` 加增量
- `HyperNetTransformerBlock` 类：用 `HyperNetMlp` 替换 `Mlp`，forward 接收 `deltas` 参数
- `SmolVLMActionTransformerV2` 类：包含 `HyperNet` + 12层 `HyperNetTransformerBlock`

原有类全部保留，不删除。

### `models/modeling_smolvlm_vla.py`

- `__init__`：根据 `config.use_hypernet` 选择实例化 V2 或原版 transformer
- `forward` / `generate_actions`：接口不变，内部多一步 `hyper_deltas = transformer.hypernet(task_vec)`

### `models/configuration_smolvlm_vla.py`

新增字段：
- `use_hypernet: bool = False`（默认关闭，不破坏旧 checkpoint）
- `hypernet_rank: int = 16`（低秩矩阵的 rank）

### `train_smolvlm.py`

新增完整的训练诊断日志（见第 4 节）。

---

## 4. 训练诊断日志（核心需求）

训练两三天，结果不好必须能定位原因。以下日志覆盖所有常见失败模式。

### 4.1 HyperNet 输出监控

每 `log_interval` 步记录：

```python
# HyperNet 生成的增量范数——判断 HyperNet 是否在学习
for layer_idx in range(num_layers):
    delta_fc1_norm = delta_fc1[layer_idx].norm().item()
    delta_fc2_norm = delta_fc2[layer_idx].norm().item()
    logs[f"hypernet/layer{layer_idx}_delta_fc1_norm"] = delta_fc1_norm
    logs[f"hypernet/layer{layer_idx}_delta_fc2_norm"] = delta_fc2_norm

# 所有层的平均增量范数
logs["hypernet/mean_delta_norm"] = mean of all delta norms

# 增量与基础权重的比值——判断增量是否过大（>1.0 说明 HyperNet 主导了权重）
logs["hypernet/delta_to_base_ratio"] = delta_norm / base_weight_norm
```

**诊断意义：**
- `mean_delta_norm` 接近 0 → HyperNet 没有学到任何东西，梯度消失
- `delta_to_base_ratio` > 1.0 → 增量过大，基础权重被覆盖，训练不稳定
- 各层 delta_norm 差异极大 → 某些层过载，需要调整 rank 或学习率

### 4.2 损失分解监控

```python
logs["loss/velocity"] = velocity_loss.item()
logs["loss/total"] = total_loss.item()

# 损失的滑动标准差（每100步）——判断训练是否稳定
# 如果 std/mean > 0.5，说明训练震荡
logs["loss/velocity_std100"] = std of last 100 velocity_loss values
```

### 4.3 梯度监控

```python
# HyperNet 参数的梯度范数——判断梯度是否正常流过 HyperNet
hypernet_grad_norm = sum(p.grad.norm()**2 for p in hypernet.parameters())**0.5
logs["grad/hypernet_norm"] = hypernet_grad_norm.item()

# action transformer 基础权重的梯度范数
transformer_grad_norm = sum(p.grad.norm()**2 for p in transformer_core.parameters())**0.5
logs["grad/transformer_core_norm"] = transformer_grad_norm.item()

# VLM 的梯度范数（解冻后）
vlm_grad_norm = sum(p.grad.norm()**2 for p in vlm.parameters() if p.grad is not None)**0.5
logs["grad/vlm_norm"] = vlm_grad_norm.item()
```

**诊断意义：**
- `hypernet_grad_norm` 接近 0 → HyperNet 梯度消失，增大学习率或检查初始化
- `hypernet_grad_norm` >> `transformer_core_norm` → HyperNet 更新过快，降低其学习率

### 4.4 Task Vector 多样性监控

每 1000 步记录一次（计算量稍大）：

```python
# 收集一个 batch 内所有样本的 task_vec，计算两两余弦相似度的均值
# 如果均值接近 1.0，说明 VLM 对不同任务输出了几乎相同的 task_vec
# HyperNet 就无法区分任务，等于没有作用
task_vecs = []  # 收集 batch 内的 task_vec
cos_sim_matrix = pairwise_cosine_similarity(task_vecs)
logs["hypernet/task_vec_diversity"] = 1.0 - cos_sim_matrix.mean().item()
# 接近 1.0 = 多样性高（好），接近 0.0 = 多样性低（坏）
```

### 4.5 权重直方图（TensorBoard 专用）

每 5000 步记录一次：

```python
# HyperNet 各层权重分布
for name, param in hypernet.named_parameters():
    writer.add_histogram(f"weights/hypernet/{name}", param.data, global_step)

# 第 0 层和最后一层的 delta 分布（代表性采样）
writer.add_histogram("delta/layer0_fc1", delta_fc1[0], global_step)
writer.add_histogram("delta/last_fc2", delta_fc2[-1], global_step)
```

### 4.6 学习率分组

HyperNet 需要独立的学习率组（通常比 transformer_core 高 2-5 倍，因为它从随机初始化开始）：

```python
param_groups = [
    {"name": "vlm",              "params": vlm_params,              "lr": 0.0},
    {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0},
    {"name": "action_heads",     "params": action_head_params,      "lr": lr},
    {"name": "hypernet",         "params": hypernet_params,         "lr": lr * 3.0},  # 新增
]
```

### 4.7 检查点保存策略

除了每 `save_interval` 步保存，额外在以下情况保存：
- `velocity_loss` 达到历史最低时（best checkpoint）
- 训练前 5000 步每 1000 步保存一次（早期诊断用）

```python
# 在 state.json 里额外记录
{
    "global_step": global_step,
    "best_velocity_loss": best_loss,
    "hypernet_mean_delta_norm": last_delta_norm,  # 方便事后分析
}
```

---

## 5. 参数量估算

| 组件 | 参数量 |
|------|--------|
| HyperNet（3层 MLP） | ~2.3M |
| 低秩矩阵 A/B（12层×4矩阵×rank=16） | ~2.9M 输出维度，但存在 HyperNet 里 |
| 基础 transformer MLP 权重（不变） | 56.6M |
| 新增总参数 | ~2.3M（仅 HyperNet） |

HyperNet 只增加约 2.3M 参数，占原模型比例极小。

---

## 6. 验证方案

1. **单步前向**：构造两个不同任务的 batch，确认生成的 `delta_fc1/fc2` 不同（task_vec 多样性 > 0）
2. **梯度流验证**：确认 `hypernet_grad_norm > 0`，梯度能从 velocity_loss 流回 HyperNet
3. **训练 1000 步**：观察 `mean_delta_norm` 是否从 0 开始增长，`delta_to_base_ratio` 是否稳定在 0.01~0.1 之间
4. **LIBERO 评估**：与原版 checkpoint 对比成功率

---

## 7. 失败模式速查表

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| `mean_delta_norm` 始终接近 0 | HyperNet 梯度消失 | 检查 `hypernet_grad_norm`，增大 HyperNet 学习率 |
| `delta_to_base_ratio` > 1.0 | 增量过大，训练不稳定 | 降低 HyperNet 学习率，或加 L2 正则 |
| `task_vec_diversity` < 0.1 | VLM 输出无差异 | 检查 VLM 是否被冻结，或 mean pool 是否正确 |
| `velocity_loss` 震荡（std/mean > 0.5） | 学习率过高 | 降低 HyperNet 或 transformer_core 学习率 |
| 各层 `delta_norm` 差异 > 10× | 某些层过载 | 对 HyperNet 输出加 per-layer LayerNorm |
