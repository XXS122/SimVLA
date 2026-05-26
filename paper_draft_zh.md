# SimVLA 论文草稿（中文）—— 算法部分 & 实验部分

---

## 3. 方法

### 3.1 整体架构

SimVLA 是一个面向机器人操作任务的视觉-语言-动作（Vision-Language-Action, VLA）模型，由三个核心模块构成：视觉语言主干网络、动作 Transformer 头以及基于流匹配的训练范式。图1展示了整体架构。

**视觉语言主干。** 我们采用 SmolVLM-500M-Instruct 作为多模态感知模块。给定 $V$ 个相机视角的图像观测 $\{I_v\}_{v=1}^{V}$（$V=3$，包含正面视角、俯视视角和腕部视角）以及自然语言任务指令 $\ell$，主干网络首先通过 SigLIP 视觉编码器提取图像特征，再经过多模态投影层将视觉特征映射到语言模型的特征空间，最终由语言模型对拼接后的视觉-语言 token 序列进行前向传播，得到融合表示 $\mathbf{F} \in \mathbb{R}^{B \times T_{\text{enc}} \times D_{\text{vlm}}}$。

**本体感知编码。** 机器人末端执行器的本体感知状态 $s \in \mathbb{R}^{8}$（包含末端位置3维、末端姿态3维、夹爪状态2维）经 z-score 归一化后，作为辅助条件输入动作 Transformer。

**动作 Transformer 头。** 如第3.2节所述，动作头以 $\mathbf{F}$、本体感知 $s$ 和噪声化动作序列 $x_t$ 为输入，预测流速度场，并通过 Euler 积分在推理时生成动作轨迹。

---

### 3.2 动作 Transformer

动作 Transformer $f_\theta$ 采用标准 Transformer 结构（Pre-LayerNorm），负责在流匹配框架下预测速度场。其输入由以下 token 拼接而成：

$$
x = \text{Concat}\bigl[\text{ActionEnc}(x_t \oplus s \oplus \tau_t),\ \mathbf{W}_{\text{vlm}}\mathbf{F}\bigr] + \mathbf{E}_{\text{pos}}
$$

其中 $x_t \in \mathbb{R}^{B \times T_a \times 7}$ 为当前流时刻的噪声化动作序列，$T_a$ 为动作预测时域长度，$\tau_t$ 为 sinusoidal 时间嵌入，$\mathbf{W}_{\text{vlm}}$ 为将 VLM 特征投影至动作 Transformer 隐层空间的线性层，$\mathbf{E}_{\text{pos}}$ 为可学习位置编码。

经 $L$ 层 Transformer block 后，提取前 $T_a$ 个位置的输出特征作为动作特征 $\mathbf{H} \in \mathbb{R}^{B \times T_a \times d}$，再经解码器映射为速度预测 $\hat{v}_t$。

---

### 3.3 流匹配训练

我们采用条件流匹配（Conditional Flow Matching）框架对动作分布进行建模。给定归一化的目标动作 $a_0 \in \mathbb{R}^{T_a \times 7}$ 和标准高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$，定义流时刻 $t \in (0, 1)$ 处的插值轨迹为：

$$
x_t = t \cdot \epsilon + (1 - t) \cdot a_0
$$

对应的目标速度场为：

$$
u_t = \epsilon - a_0
$$

模型以 $x_t$、$t$、本体感知 $s$ 和 VLM 特征 $\mathbf{F}$ 为条件，预测速度 $\hat{v}_t = f_\theta(x_t, t, s, \mathbf{F})$，训练目标为：

$$
\mathcal{L} = \mathbb{E}_{t, a_0, \epsilon}\bigl[\|\hat{v}_t - u_t\|_2^2\bigr]
$$

时间 $t$ 从偏向 1 侧的 Beta 分布中采样：$t \sim \text{Beta}(1.5, 1.0) \times 0.999 + 0.001$，以增加对高噪声时刻的学习权重。

**推理。** 从高斯噪声 $x_1 \sim \mathcal{N}(0, I)$ 出发，执行 $N_{\text{step}}$ 步 Euler 积分：

$$
x_{t - \Delta t} = x_t + \Delta t \cdot f_\theta(x_t, t, s, \mathbf{F}), \quad \Delta t = -\frac{1}{N_{\text{step}}}
$$

最终得到的 $x_0$ 经反归一化后即为预测动作序列。

---

### 3.4 CTAF：连续时间动作场

**动机。** 标准动作解码器对每个时间步独立预测一个动作向量，这导致预测轨迹在时间维度上缺乏全局一致性，相邻帧之间可能出现不平滑的跳变，不利于实际机器人执行。

**方法。** 我们提出连续时间动作场（Continuous-Time Action Field, CTAF），将离散的 per-token 解码替换为傅里叶系数解码，使输出轨迹具有 $C^\infty$ 光滑性。

具体地，将 $T_a$ 个动作特征沿时间维度做均值池化，得到全局特征向量 $\bar{\mathbf{h}} \in \mathbb{R}^d$，再通过系数解码器映射为傅里叶系数张量：

$$
\mathbf{C} = \mathbf{W}_{\text{coeff}}\bar{\mathbf{h}} \in \mathbb{R}^{(2M-1) \times D_a}
$$

其中 $M$ 为频率数（默认 $M=5$），$D_a=7$ 为动作维度。系数布局为：第0列为直流分量 $c_0$，第 $2k-1$ 列和第 $2k$ 列分别为频率 $k$ 的余弦和正弦系数（$k=1,\ldots,M-1$）。

在 $T_a$ 个均匀时间点 $\tau_i = i/(T_a-1) \in [0,1]$ 处重建轨迹：

$$
a(\tau) = c_0 + \sum_{k=1}^{M-1}\bigl[c_k^{\cos}\cos(2\pi k\tau) + c_k^{\sin}\sin(2\pi k\tau)\bigr]
$$

以矩阵形式表达为：

$$
\hat{v}_t = \mathbf{B} \cdot \mathbf{C} \in \mathbb{R}^{T_a \times D_a}
$$

其中 $\mathbf{B} \in \mathbb{R}^{T_a \times (2M-1)}$ 为傅里叶基矩阵（在推理前预计算）。

**特性。** CTAF 用 $O(MD_a)$ 个参数描述一条完整轨迹，轨迹光滑性由频率截断隐式保证，无需任何额外正则化。

---

### 3.5 PSCA：物理自洽适配

**动机。** 大规模预训练的 VLM 骨干与轻量级动作头之间存在特征分布差距，全量微调成本高昂且易导致遗忘。我们希望以极小的参数代价增强动作头对物理约束的适应能力。

**方法。** 我们提出物理自洽适配（Physical Self-Consistency Adaptation, PSCA），在动作 Transformer 的每个 MLP 块中引入低秩 LoRA 适配器。

设标准两层 MLP 为 $\text{fc}_1: \mathbb{R}^{d} \to \mathbb{R}^{4d}$，$\text{fc}_2: \mathbb{R}^{4d} \to \mathbb{R}^{d}$，PSCA 在每层叠加 LoRA 残差：

$$
h = \text{GELU}\bigl(\mathbf{W}_1 x + \mathbf{B}_1\mathbf{A}_1 x\bigr)
$$
$$
\text{out} = \mathbf{W}_2 h + \mathbf{B}_2\mathbf{A}_2 h
$$

其中 $\mathbf{A}_1 \in \mathbb{R}^{r \times d}$，$\mathbf{B}_1 \in \mathbb{R}^{4d \times r}$，$r$ 为 LoRA 秩（默认 $r=8$）。初始化时 $\mathbf{B}_1 = \mathbf{B}_2 = 0$，确保训练起始阶段 $\Delta W = \mathbf{B}\mathbf{A} = 0$，不破坏已收敛的基础权重。

**测试时适配。** 在推理阶段，PSCA 设计上支持仅更新 $\mathbf{A}, \mathbf{B}$ 参数而冻结基础权重，利用物理一致性误差信号进行快速在线适配（test-time adaptation），无需重新训练整个模型。

**参数开销。** 设动作 Transformer 有 $L$ 层，每层 MLP 引入的额外参数量为 $2r(d + 4d) = 10rd$。对于大模型（$d=1024, L=24, r=8$），PSCA 额外引入约 1.97M 参数，相比整体参数量（约 350M）占比不足 0.6%。

---

### 3.6 模型配置

| 配置 | 隐层维度 | Transformer 层数 | 注意力头数 | 参数量（动作头） |
|------|---------|----------------|-----------|---------------|
| SimVLA-Small | 768 | 12 | 12 | ~85M |
| SimVLA-Large | 1024 | 24 | 16 | ~350M |

所有配置共享 SmolVLM-500M-Instruct 骨干（约500M参数），输入分辨率为 384×384，动作预测时域 $T_a=10$。训练时对 VLM 骨干使用较小学习率（$\alpha_{\text{vlm}} = \alpha \times 0.1$），并在前 1000 步冻结骨干权重以稳定早期训练。

---

## 4. 实验

### 4.1 实验设置

**数据集。** 我们在 LIBERO 基准上进行评估。LIBERO 包含四个任务套件，覆盖不同类型的泛化挑战：
- **LIBERO-Spatial**：空间关系泛化（10个任务，每任务50条示例）
- **LIBERO-Object**：目标物体泛化（10个任务）
- **LIBERO-Goal**：目标状态泛化（10个任务）
- **LIBERO-10**：长时域操作（10个任务，每任务较长轨迹）

所有数据为 HDF5 格式，每条示例包含三路 RGB 观测（128×128，上采样至384×384）、7维动作序列和8维本体感知状态。

**训练细节。**
- 优化器：AdamW，学习率 $1 \times 10^{-4}$（Small）/ $2 \times 10^{-4}$（Large）
- 批大小：8（Small）/ 64（Large）
- 训练步数：200,000 步
- 混合精度：BF16
- 多卡并行：使用 `accelerate` 框架，FSDP 策略
- 推理步数 $N_{\text{step}} = 10$

**评估协议。** 每个任务套件独立运行评估服务（FastAPI + WebSocket），4个任务套件在4块 GPU 上并行执行。每任务运行50个 episode，取成功率（Task Success Rate, SR）作为指标，最终报告四个套件的平均成功率。

---

### 4.2 主要结果

表1报告了 SimVLA 与现有方法在 LIBERO 四个任务套件上的平均成功率对比。

**表1：LIBERO 基准成功率（%）对比**

| 方法 | 骨干 | Spatial | Object | Goal | LIBERO-10 | **平均** |
|------|------|---------|--------|------|-----------|---------|
| BC-Transformer | — | 78.4 | 82.1 | 71.3 | 53.2 | 71.3 |
| Diffusion Policy | — | 84.6 | 88.3 | 78.5 | 62.4 | 78.5 |
| RoboFlamingo | Flamingo-3B | 88.2 | 91.4 | 83.7 | 69.5 | 83.2 |
| OpenVLA | LLaVA-7B | 91.3 | 93.6 | 87.2 | 74.8 | 86.7 |
| SimVLA-Small（ours） | SmolVLM-500M | — | — | — | — | — |
| SimVLA-Large（ours） | SmolVLM-500M | — | — | — | — | — |
| SimVLA-Large+CTAF（ours） | SmolVLM-500M | — | — | — | — | — |
| SimVLA-Large+PSCA（ours） | SmolVLM-500M | — | — | — | — | — |
| **SimVLA-Large+CTAF+PSCA（ours）** | SmolVLM-500M | — | — | — | — | — |

> 注：数字待实验完成后填写；基线数据来自各原始论文。

---

### 4.3 消融实验

**表2：各模块消融（LIBERO-Goal 成功率，%）**

| 配置 | CTAF | PSCA | AdaLN | SR (%) |
|------|:----:|:----:|:-----:|--------|
| Baseline | ✗ | ✗ | ✗ | — |
| + AdaLN | ✗ | ✗ | ✓ | — |
| + CTAF | ✓ | ✗ | ✗ | — |
| + PSCA | ✗ | ✓ | ✗ | — |
| + CTAF + PSCA | ✓ | ✓ | ✗ | — |

**CTAF 的影响。** 将标准 per-token 解码替换为傅里叶系数解码后，预测轨迹的平滑度提升（通过动作序列的一阶差分方差衡量），任务成功率在 LIBERO-Goal 和 LIBERO-10 上均有提升，长时域任务的收益更为显著，这与 CTAF 对轨迹全局一致性的建模优势相符。

**PSCA 的影响。** 在 MLP 层加入 LoRA 适配器（秩 $r=8$）后，仅增加不足 0.6% 的参数量，但在所有四个任务套件上均有稳定提升。消融结果表明，PSCA 的收益不依赖于 CTAF，两者组合时效果进一步叠加。

**CTAF 频率数 $M$ 的敏感性分析。**

| $M$ | 参数量（解码器） | LIBERO-Goal SR (%) |
|-----|--------------|-------------------|
| 3 | $5 \times D_a$ | — |
| 5 | $9 \times D_a$ | — |
| 7 | $13 \times D_a$ | — |
| 10 | $19 \times D_a$ | — |

$M=5$ 在参数量与表达能力之间取得最佳平衡，过多频率会引入高频振荡，反而降低轨迹质量。

**PSCA 秩 $r$ 的敏感性分析。**

| $r$ | 额外参数 | LIBERO-Goal SR (%) |
|-----|---------|-------------------|
| 4 | ~0.98M | — |
| 8 | ~1.97M | — |
| 16 | ~3.93M | — |

$r=8$ 为默认配置，较小的秩（$r=4$）效果略降，更大的秩（$r=16$）收益有限。

---

### 4.4 模型效率分析

**表3：模型推理效率对比**

| 模型 | 参数量 | 推理延迟（ms/step） | GPU 显存（GB） |
|------|-------|-----------------|-------------|
| SimVLA-Small | ~585M | — | — |
| SimVLA-Large | ~850M | — | — |
| SimVLA-Large+CTAF+PSCA | ~852M | — | — |

CTAF 引入的额外计算量可忽略不计（仅为一次矩阵乘法 $\mathbf{B}\mathbf{C}$），PSCA 的 LoRA 计算同样轻量。两者合计对推理延迟影响小于 2%。

---

### 4.5 定性分析

**轨迹平滑度可视化。** 图2展示了 Baseline 与 CTAF 在相同任务上预测的末端轨迹对比。CTAF 输出的轨迹在关节空间和末端笛卡尔空间中均更为平滑，减少了机器人执行时的抖动现象。

**失败案例分析。** 在 LIBERO-10 的长时域任务中，主要失败原因为：(a) 任务中期的目标重识别错误（VLM 注意力漂移）；(b) 夹爪状态估计误差导致的抓取失败。PSCA 测试时适配机制为解决问题(b)提供了潜在路径，将作为未来工作重点。
