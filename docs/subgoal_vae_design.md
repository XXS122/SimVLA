# SimVLA 创新点：层次化双扩散架构（CVAE SubgoalVAE + Latent Diffusion Model）

## 1. 动机与问题

### 1.1 原始 SimVLA 的局限

原始 SimVLA 采用单帧无状态推理：每次调用 `generate_actions()` 时，模型只看当前帧的图像和语言指令，完全不感知历史。这在短程任务（单一动作目标）上表现良好，但在长程任务上存在根本性缺陷：

- **无子任务感知**：VLABench `track_5_long_horizon` 包含 7 个顺序子任务，模型无法判断当前处于哪个阶段
- **条件信息单一**：AdaLN 条件 `c = time_emb + vlm_pool + proprio_emb` 中，`vlm_pool` 是 VLM 特征的简单平均池化，压缩了大量语义信息
- **动作分布单峰假设**：Flow Matching 隐式假设给定条件下动作分布是单峰的，但长程任务中同一视觉状态可能对应多个合理的子任务行为
- **先验分布表达能力弱**：即使引入 CVAE，若推理时先验 `p(z|vlm)` 只是一个简单高斯（两层 MLP 输出），其表达能力仍然有限，无法建模 z 空间的复杂多模态分布

### 1.2 核心洞察

长程任务中，机器人在不同子任务阶段面对相似的视觉输入时，应该执行截然不同的动作。例如"先拿杯子，再放到托盘上"——拿起阶段和放置阶段的视觉输入可能相似，但动作完全不同。

**子目标潜变量** `z_goal ∈ R^64` 的作用是编码"当前处于哪个子任务阶段"这一隐含信息，使模型能够在潜在空间中区分这些多模态情况。

进一步地，**Latent Diffusion Model（LDM）** 在 z 空间上做 Flow Matching，让 z_goal 的生成过程本身也是一个扩散过程，以 vlm_pooled 为条件从噪声逐步去噪到有意义的子目标潜变量，使先验分布更灵活、更准确。

---

## 2. 整体架构：层次化双扩散

本方案形成**两层嵌套的扩散结构**：

```
第一层（z 空间，64维，5步 Euler）：
  噪声 z_T ~ N(0,I)
       ↓  LatentFlowNet（以 vlm_pooled 为条件）
  子目标潜变量 z_goal ∈ R^64

第二层（动作空间，7维×10步，10步 Euler）：
  噪声 x_T ~ N(0,I)
       ↓  SmolVLMActionTransformer（以 c = t_emb + vlm_pool + proprio + subgoal_proj(z_goal) 为条件）
  动作序列 a_{1:10} ∈ R^{10×7}
```

### 2.1 训练时数据流

```
输入：vlm_features[B, T_seq, 576]  +  action_chunk[B, 10, 7]  +  proprio[B, 7]

Step 1 — CVAE 后验采样：
  vlm_pooled = vlm_features.mean(dim=1).detach()          [B, 576]
  post_mu, post_log_var = SubgoalVAE.posterior(vlm_pooled, action_chunk)
  prior_mu, prior_log_var = SubgoalVAE.prior(vlm_pooled)
  z_0 = reparameterize(post_mu, post_log_var)              [B, 64]  ← 后验采样，作为 FM 目标
  L_KL = KL(q(z|vlm,action) || p(z|vlm))                  标量

Step 2 — z 空间 Flow Matching（LatentFlowNet）：
  z_noise ~ N(0, I)                                        [B, 64]
  t_z ~ Beta(1.5, 1) * 0.999 + 0.001                      [B]
  z_t = t_z * z_noise + (1 - t_z) * z_0                   线性插值
  u_z = z_noise - z_0                                      目标速度场
  v_z = LatentFlowNet(z_t, t_z, vlm_pooled)               预测速度场
  L_latent_fm = MSE(v_z, u_z)                              标量

Step 3 — 动作空间 Flow Matching（SmolVLMActionTransformer）：
  z_goal = z_0                                             训练时直接用后验 z_0
  c = t_emb + vlm_cond + proprio_cond + subgoal_proj(z_goal)
  x_t = t * noise + (1-t) * action_norm
  v_t = SmolVLMActionTransformer(x_t, t, vlm_features, proprio, z_goal)
  L_velocity = MSE(v_t, u_t)                               标量

总损失：
  L = L_velocity + λ_kl(step) × L_KL + λ_fm × L_latent_fm
```

### 2.2 推理时数据流

```
输入：vlm_features[B, T_seq, 576]  +  proprio[B, 7]

Step 1 — z 空间 LDM 积分（5步 Euler）：
  z_t = randn(B, 64)                                       从纯噪声出发
  dt = -1/5 = -0.2
  for t in [1.0, 0.8, 0.6, 0.4, 0.2]:
      v_z = LatentFlowNet(z_t, t, vlm_pooled)
      z_t = z_t + dt * v_z
  z_goal = z_t                                             最终 z_0

Step 2 — 动作空间 Euler 积分（10步）：
  x_t = randn(B, 10, 7)
  dt = -1/10 = -0.1
  for t in [1.0, 0.9, ..., 0.1]:
      c = t_emb + vlm_cond + proprio_cond + subgoal_proj(z_goal)
      v_t = SmolVLMActionTransformer(x_t, t, vlm_features, proprio, z_goal)
      x_t = x_t + dt * v_t
  actions = x_t                                            最终动作序列
```

---

## 3. 模块详解

### 3.1 SubgoalVAE

```python
class SubgoalVAE(nn.Module):
    # 先验网络：p(z | vlm_pooled)
    prior_net:     Linear(576, 256) → SiLU → Linear(256, 128)   # 输出 [mu, log_var]，各 64 维

    # 后验网络：q(z | vlm_pooled, action_chunk)
    posterior_net: Linear(576 + 70, 256) → SiLU → Linear(256, 128)
    #                      ↑ 576=vlm_hidden  ↑ 70=10步×7维动作（展平）

    latent_dim = 64   # z_goal 维度
```

**重参数化采样**：

```
z = μ + ε · σ,   ε ~ N(0, I)
σ = exp(0.5 · log_var)
```

**KL 散度（Free Bits）**：

```
KL(q || p) = 0.5 · [log(σ_p²/σ_q²) + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1]

每维 Free Bits：KL_per_dim = max(KL_per_dim, 0.5)
L_KL = mean(KL_per_dim)   ← 对 batch 和 latent_dim 取均值
```

Free Bits 保证每个潜变量维度至少携带 0.5 nats 的信息，防止后验退化为先验（KL 崩溃）。

### 3.2 LatentFlowNet

轻量 MLP，在 64 维 z 空间做 Flow Matching，以 vlm_pooled 为条件：

```python
class LatentFlowNet(nn.Module):
    vlm_proj: Linear(576, 64)          # 压缩 VLM 条件到 latent_dim
    net:
        Linear(64 × 3, 256) → SiLU    # 输入：[z_t, t_emb, vlm_cond] 拼接
        Linear(256, 256) → SiLU
        Linear(256, 64)                # 输出：速度场 v_t，初始化为零
```

时间嵌入复用 `timestep_embedding(t, latent_dim=64)`（正弦位置编码）。

**输出层初始化为零**：训练初期 `v_z ≈ 0`，不干扰 CVAE 的收敛。

**Flow Matching 目标**（Rectified Flow / OT-CFM）：

```
z_t = t · z_noise + (1 - t) · z_0,   t ~ Beta(1.5, 1) · 0.999 + 0.001
u_z = z_noise - z_0                   目标速度场（从 z_0 到 z_noise 的方向）
v_z = LatentFlowNet(z_t, t, vlm_pooled)
L_latent_fm = E[||v_z - u_z||²]
```

### 3.3 AdaLN 条件注入

```
c = t_emb + vlm_cond + proprio_cond + subgoal_proj(z_goal)
  ↓
DiTBlock: LayerNorm → scale/shift by c → Attention/FFN
```

`subgoal_proj` 是 `Linear(64, hidden_size)`，将 z_goal 投影到与其他条件相同的维度后相加。

---

## 4. 训练损失与调度

### 4.1 总损失

```
L = L_velocity + λ_kl(step) · L_KL + λ_fm · L_latent_fm

L_velocity   = MSE(v_t, u_t)                    动作空间 Flow Matching
L_KL         = KL(q(z|vlm,action) || p(z|vlm))  CVAE 正则化（Free Bits）
L_latent_fm  = MSE(v_z, u_z)                    z 空间 Flow Matching

λ_kl(step)   = kl_weight × min(1.0, step / kl_warmup_steps)   KL annealing
λ_fm         = latent_fm_weight（默认 1.0）
```

### 4.2 KL Annealing

```
λ_kl(step) = 0.001 × min(1.0, step / 10000)

step=0:      λ_kl = 0.0      （纯 Flow Matching，让动作头先收敛）
step=5000:   λ_kl = 0.0005   （KL 约束逐渐引入）
step=10000+: λ_kl = 0.001    （达到最终权重，稳定训练）
```

### 4.3 关键工程细节

| 细节 | 原因 |
|------|------|
| `vlm_pooled.detach()` | 前 `freeze_steps` 步 VLM 被冻结，detach 防止 VAE/LDM 梯度反传到 VLM |
| Free Bits（每维 KL ≥ 0.5） | 防止后验退化为先验（KL 崩溃），保证 z_goal 携带有效信息 |
| KL annealing（前 10k 步线性增加） | 训练初期 KL 权重为 0，让 Flow Matching 先收敛，再逐步引入 KL 约束 |
| action_chunk 输入加 Dropout(0.1) | 防止后验 encoder 过拟合到训练集的动作模式 |
| LatentFlowNet 输出层初始化为零 | 训练初期 z 空间速度场为零，不干扰 CVAE 收敛 |
| 训练时 z_goal = z_0（后验直接用） | 避免训练时走 LDM 推理路径，节省计算；LDM 只学速度场 |
| Beta(1.5, 1) 时间分布 | 与动作空间 FM 保持一致，偏向 t→1（噪声端），提升训练稳定性 |

---

## 5. 代码实现

### 5.1 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `models/transformer_smolvlm.py` | 新增 `SubgoalVAE` 类、`LatentFlowNet` 类；`SmolVLMActionTransformer.__init__` 新增 `use_subgoal_vae`、`subgoal_latent_dim`、`num_actions` 参数，AdaLN 分支新增 `subgoal_vae`、`subgoal_proj`、`latent_flow_net`；`_forward_adaln` 接受可选 `z_goal` |
| `models/modeling_smolvlm_vla.py` | `forward()` 加入后验采样 + KL loss + z 空间 FM loss；`generate_actions()` 替换先验采样为 LDM Euler 积分 |
| `models/configuration_smolvlm_vla.py` | 新增 `use_subgoal_vae`、`subgoal_latent_dim`、`kl_weight`、`use_latent_flow`、`latent_flow_steps`、`latent_fm_weight` 配置字段 |
| `train_smolvlm.py` | 新增 7 个命令行参数；加入 KL annealing 调度；日志打印 `v_loss`、`kl_loss`、`z_fm_loss` |
| `train_smolvlm_subgoal.sh` | 新训练脚本，包含所有 CVAE + LDM 参数，自动生成 meta 和 norm stats |

### 5.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_adaln` | False | 启用 DiT 风格 AdaLN 条件注入（SubgoalVAE 依赖此项） |
| `--use_subgoal_vae` | False | 启用 CVAE 子目标潜变量（需配合 `--use_adaln`） |
| `--subgoal_latent_dim` | 64 | 潜变量 z_goal 的维度 |
| `--kl_weight` | 0.001 | KL 散度权重上限 |
| `--kl_warmup_steps` | 10000 | KL annealing 步数 |
| `--use_latent_flow` | False | 启用 z 空间 LDM（需配合 `--use_subgoal_vae`） |
| `--latent_flow_steps` | 5 | 推理时 z 空间 Euler 积分步数 |
| `--latent_fm_weight` | 1.0 | z 空间 FM 损失权重 |

---

## 6. 训练命令

### 完整训练（推荐，双卡 A800）

```bash
cd /root/SimVLA
bash train_smolvlm_subgoal.sh 32 0.1 ./simvla_output/simvla_subgoal
```

### 从断点续训

```bash
bash train_smolvlm_subgoal.sh 32 0.1 ./simvla_output/simvla_subgoal ./simvla_output/simvla_subgoal/ckpt-10000
```

### 消融实验对比

```bash
# Baseline：原始 AdaLN，无 CVAE，无 LDM
bash train_smolvlm_vlabench.sh 32 0.1 ./simvla_output/baseline_adaln

# Ablation-CVAE：仅 CVAE，无 LDM（关闭 use_latent_flow）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes=2 --main_process_port 29507 --mixed_precision bf16 \
    train_smolvlm.py \
    --output_dir ./simvla_output/ablation_cvae \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --smolvlm_model_path /root/model/smolvlm-500M \
    --action_mode vlabench_joint \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --use_adaln --use_subgoal_vae \
    --kl_weight 0.001 --kl_warmup_steps 10000 \
    --batch_size 32 --iters 100000

# Ours（完整）：CVAE + LDM
bash train_smolvlm_subgoal.sh 32 0.1 ./simvla_output/ours_full
```

---

## 7. 训练监控

训练日志中会同时记录三个损失：

```
[20/100000] loss=2.08 v_loss=2.08 kl_loss=0.00 z_fm_loss=0.00 lr_core=0.00e+00 lr_action=1.00e-04 lr_vlm=0.00e+00 (0.42s/it)
[1000/100000] loss=0.42 v_loss=0.40 kl_loss=0.02 z_fm_loss=0.01 lr_core=1.00e-04 lr_action=1.00e-04 lr_vlm=1.00e-05 (0.38s/it)
```

**正常收敛的判断标准：**

| 指标 | 正常范围 | 异常情况 |
|------|---------|---------|
| `v_loss`（velocity_loss） | 与 baseline 相近，逐步下降 | 显著高于 baseline → CVAE/LDM 干扰了 FM |
| `kl_loss`（原始，未乘权重） | 0.5 ~ 2.0（稳定后） | < 0.1 → KL 崩溃；> 10 → 后验过于分散 |
| `kl_loss` 趋势 | 前 10k 步从 0 线性增加 | 一直为 0 → `use_subgoal_vae` 未生效 |
| `z_fm_loss`（latent_fm_loss） | 0.1 ~ 1.0（稳定后） | 不下降 → LatentFlowNet 未正常训练 |

---

## 8. 评估

推理服务器接口**无需修改**，z_goal 在 `generate_actions()` 内部自动通过 LDM 积分生成，对外完全透明。

```bash
# 启动推理服务器（与原来完全相同的命令）
conda activate simvla
CUDA_VISIBLE_DEVICES=0 python evaluation/vlabench/serve_smolvlm_vlabench.py \
    --checkpoint ./simvla_output/simvla_subgoal/ckpt-100000 \
    --norm_stats ./norm_stats/vlabench_norm.json \
    --smolvlm_model /root/model/smolvlm-500M \
    --port 8001

# 评估（重点关注 track_5_long_horizon）
conda activate vlabench
cd /root/VLABench
python /root/SimVLA/evaluation/vlabench/evaluate_simvla.py \
    --eval-track track_5_long_horizon \
    --n-episode 50 \
    --port 8001 \
    --save-dir /root/SimVLA/simvla_output/eval_results
```

---

## 9. 预期效果与局限

### 预期改善

- **长程任务（track_5）**：子任务阶段切换更准确，success_rate 预期提升
- **多模态动作分布**：同一视觉状态下不同子任务阶段的动作不再混淆
- **先验质量提升**：LDM 先验比简单高斯先验更灵活，能建模 z 空间的复杂多模态分布
- **层次化扩散**：z 空间（子目标层）+ 动作空间（执行层）形成两级扩散，与人类"先规划子目标，再执行动作"的认知结构对齐
- **可解释性**：可通过 t-SNE 可视化 z_goal 的分布，验证不同子任务阶段是否形成聚类

### 局限与后续工作

- **训练数据无历史帧**：当前每个 sample 是单帧，z_goal 只能从当前帧推断子任务阶段，无法利用历史轨迹
- **先验质量依赖 VLM**：推理时 LDM 的条件 `vlm_pooled` 的质量取决于 VLM 特征能否区分不同子任务阶段
- **后续方向**：结合历史帧缓冲区（History-Conditioned Generation），让 LatentFlowNet 接收历史 VLM 特征序列作为条件，进一步提升长程任务性能
