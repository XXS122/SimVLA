# WandB 指标说明与曲线解读指南

所有指标每 `log_interval`（默认 20 步）上传一次，除非特别注明。

---

## 一、Loss 类

### `velocity_loss`
Flow matching 的核心损失，MSE(预测速度场, 真实速度场)。

- **正常趋势**：持续下降，前期下降快，后期趋于平稳
- **异常信号**：
  - 长时间不下降 → 学习率过低，或 VLM 冻结期太长
  - 剧烈震荡 → 学习率过高，调低 `learning_rate` 或 `learning_coef`
  - 突然跳升后不恢复 → 梯度爆炸，检查 `grad/*_norm`

### `state_loss`（仅 `--use_state_loss` 时存在）
辅助世界模型损失，预测下一时刻本体感知。权重由 `--state_loss_weight` 控制（默认 0.1）。

- 该值应远小于 `velocity_loss`（因为乘了权重系数）
- 如果 `state_loss` 异常大，说明本体感知归一化有问题

### `loss_total`
`velocity_loss + state_loss`（如有）的总和，是优化器实际优化的目标。

### `loss/velocity_std100`
最近 100 步 velocity_loss 的**变异系数**（标准差 / 均值）。

- **正常范围**：< 0.3
- **异常信号**：持续 > 0.5 → 训练不稳定，考虑降低学习率或增大 batch size

---

## 二、学习率类

### `lr_action_heads`
动作头的学习率。冻结期（前 `freeze_steps` 步，默认 1000 步）内为设定值，之后保持不变。

### `lr_transformer_core`
Action Transformer 主体的学习率。冻结期内为 0，解冻后按 warmup-cosine 调度上升。

### `lr_vlm`
SmolVLM 骨干的学习率。冻结期内为 0，解冻后为 `lr_transformer_core * learning_coef`。

**看法**：三条曲线应在 step=1000 附近同时从 0 开始上升（除 `lr_action_heads`）。如果某条一直是 0，说明解冻逻辑有问题。

---

## 三、梯度范数类（每 20 步）

### `grad/vlm_norm`
SmolVLM 骨干参数的梯度 L2 范数（clip 后）。

### `grad/transformer_core_norm`
Action Transformer 主体的梯度范数。

### `grad/action_heads_norm`
动作头的梯度范数。

**正常范围**：0.1 ~ 10。具体数值因模型而异，重要的是**趋势稳定**。

**异常信号**：
- 某组突然跳到 100+ → 该模块梯度爆炸，检查对应模块的输入是否有异常值
- 某组长期为 0 → 该模块没有参与训练（可能冻结未解除，或梯度没有流过去）
- `grad/vlm_norm` 在解冻后仍为 0 → VLM 梯度没有回传，检查 `learning_coef` 是否为 0

---

## 四、动作维度类（每 20 步）

libero_joint 动作空间：7 维 delta 动作，含义如下：

| 维度 | 含义 |
|------|------|
| dim_0 | Δx（末端执行器 x 方向位移） |
| dim_1 | Δy |
| dim_2 | Δz |
| dim_3 | Δroll（axis-angle x） |
| dim_4 | Δpitch（axis-angle y） |
| dim_5 | Δyaw（axis-angle z） |
| dim_6 | 夹爪开合（0=关, 1=开） |

### `action_dim/mean_N`
第 N 维动作在当前 batch 中的均值（归一化前原始值）。

- 应在 0 附近波动（delta 动作大多数时候接近 0）
- dim_6（夹爪）均值应在 0~1 之间

### `action_dim/std_N`
第 N 维动作在当前 batch 中的标准差。

**异常信号**：
- 某维 std 长期接近 0 → 该维度动作几乎没有变化，数据集中该维度缺乏多样性，或归一化出错
- dim_6 std 接近 0 → 夹爪动作单一，模型可能学不到开关夹爪

---

## 五、本体感知类（每 20 步）

### `proprio/mean`
当前 batch 本体感知输入（8 维）的整体均值。

### `proprio/std`
当前 batch 本体感知输入的整体标准差。

**正常范围**：归一化后应接近 0 均值、1 标准差。

**异常信号**：
- `proprio/mean` 持续偏离 0（如 > 2）→ 归一化统计量有误，重新运行 `compute_libero_norm_stats.py`
- `proprio/std` 接近 0 → 本体感知输入几乎没有变化，数据读取可能有问题

---

## 六、VLM 特征类（每 500 步）

### `vlm/feature_norm_mean`
VLM 输出特征序列中，每个 token 的 L2 范数的均值。

### `vlm/feature_norm_std`
VLM 输出特征 L2 范数的标准差（衡量不同 token 之间的差异程度）。

**正常范围**：`feature_norm_mean` 通常在 5~50 之间（取决于模型规模）。

**异常信号**：
- `feature_norm_mean` 持续单调下降趋近 0 → VLM 特征退化（representation collapse），是 VLA 训练中最严重的失败模式，需要降低 VLM 学习率（减小 `learning_coef`）
- `feature_norm_std` 接近 0 → 所有 token 特征趋同，同样是退化信号

---

## 七、时间采样类

### `time_sampling/alpha`
Flow matching 时间采样 Beta 分布的 alpha 参数，由课程学习控制：

| 训练进度 | alpha | 含义 |
|----------|-------|------|
| 0~30% | 1.0 | 均匀采样，覆盖全时间范围 |
| 30~70% | 1.5 | 偏向中间时间步 |
| 70~100% | 2.5 | 更集中在中间时间步 |

这条曲线是阶梯形，属于正常现象。

---

## 八、HyperNet 类（仅 `--use_hypernet` 时存在）

### `hypernet/mean_delta_norm`
HyperNet 生成的权重增量的平均 L2 范数。

### `hypernet/delta_to_base_ratio`
权重增量范数 / 基础权重范数。衡量 HyperNet 对模型的调制强度。

- **正常范围**：0.01 ~ 0.3
- 过大（> 1）→ HyperNet 主导了权重，基础模型失效
- 过小（< 0.001）→ HyperNet 几乎没有效果

### `hypernet/task_vec_diversity`（每 1000 步）
batch 内不同样本 task vector 的多样性（1 - 平均余弦相似度）。

- 接近 1 → 不同任务的表示差异大，HyperNet 能区分任务
- 接近 0 → 所有任务的 task vector 趋同，HyperNet 退化为固定偏置

### `hypernet/layerN_fc1_norm` / `hypernet/layerN_fc2_norm`
各层 fc1/fc2 权重增量的范数，用于定位哪一层的调制最强。

---

## 快速复盘流程

训练结果不好时，按以下顺序排查：

1. **看 `velocity_loss`**：是否持续下降？有没有异常跳升？
2. **看 `loss/velocity_std100`**：是否稳定？震荡大说明训练不稳定
3. **看 `grad/*_norm`**：有没有某组梯度爆炸或长期为 0？
4. **看 `vlm/feature_norm_mean`**：是否在下降？下降说明 VLM 退化
5. **看 `action_dim/std_N`**：有没有某维接近 0？说明该维度动作没有学到
6. **看 `proprio/mean` 和 `proprio/std`**：归一化是否正常？
7. **（HyperNet）看 `hypernet/delta_to_base_ratio`**：调制强度是否合理？
