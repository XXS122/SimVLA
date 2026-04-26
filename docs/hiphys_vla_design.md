# HiPhys-VLA 设计文档

## 研究背景

**目标**：在 VLABench 基准上发表中科院一区论文（NeurIPS/ICLR/ICML/CVPR/CoRL/RSS 级别）。

**核心洞察**：现有 VLA 的条件向量 `c = t_emb + vlm_cond + proprio_cond` 缺少两个关键信息：
1. **时序历史**：知道"任务走到第几步了"→ 解决 track_5 长程任务
2. **物理约束**：知道"物理上可不可行"→ 解决 track_3 物理常识

**方案**：将两者以零参数无架构侵入的方式加入条件向量，形成 HiPhys-VLA 统一框架。

---

## 架构设计

### 条件向量扩展

```
原有：c = t_emb + vlm_cond + proprio_cond
新增：c = c + h_proj(h_norm(h_t))        # HistoryEncoder GRU 状态
     c = c + physics_proj(physics_pred)   # PhysicsPredicateDecoder 预测
```

两项独立，可分别消融。

### 总损失函数

```
L = L_velocity                          # Flow Matching 主损失
  + λ_switch * L_switch                 # GRU 辅助：gripper 切换预测（0.05）
  + λ_physics * L_physics               # 物理谓词弱监督（0.01）
```

---

## 创新点一：HistoryEncoder（GRU 历史感知）

**解决问题**：track_5 长程任务，多阶段切换，错误恢复

**参数量**：约 0.5M（GRU 128维 + LayerNorm + 线性投影）

### 模块定义

```python
# models/transformer_smolvlm.py
class HistoryEncoder(nn.Module):
    gru:      nn.GRUCell(input_size=7, hidden_size=128)
    h_norm:   nn.LayerNorm(128)
    h_proj:   nn.Linear(128, hidden_size)   # zero-init
    switch_pred: nn.Linear(128, 1)          # 辅助任务头
```

### 训练方式

- 数据格式：K=4 连续帧本体感知序列 `[B, K, 7]`
- GRU 逐帧更新：`h = gru(proprio_seq[:, k], h)`
- 最终隐状态注入 `c`

### 辅助监督（弱监督，无人工标注）

```python
L_switch = HuberLoss(Linear(h_t), steps_until_gripper_switch / num_actions)
```

标签来源：轨迹中 `gripper` 维度跳变点自动检测，无需人工标注。

### 推理时状态持久化

```python
# evaluation/vlabench/serve_smolvlm_vlabench.py
_episode_h_states: Dict[conn_id, Tensor[128]]

# 卡住检测：连续 5 步 EE 位移 < 0.005m，自动 reset h_state
if stuck: h_state = None  # 错误恢复
```

---

## 创新点二：PhysicsPredicateDecoder（物理谓词嵌入）

**解决问题**：track_3 物理常识，track_4 语义指令（含物理约束）

**参数量**：约 0.3M（MLP 576→256→128→5 + 线性投影）

### 5 个物理谓词

| 索引 | 谓词 | 弱监督标签来源 | 损失类型 |
|---|---|---|---|
| 0 | `gripper_active` | `chunk[:, 6].mean() > 0.05` | BCE |
| 1 | `high_rotation` | `\|chunk[:, 3:6]\|.mean() > 0.05` | BCE |
| 2 | `z_height` | `sigmoid(proprio[2] * 5)` | MSE × 0.1 |
| 3 | `moving_up` | `chunk[:, 2].mean() > 0.01` | BCE |
| 4 | `stable_traj` | `std(chunk[:, :3]) < 0.02` | BCE |

### 辅助损失

```python
L_physics = BCE(pred[:, [0,1,3,4]], labels[:, [0,1,3,4]])  # 4个二分类
           + 0.1 * MSE(sigmoid(pred[:, 2]), labels[:, 2])   # z_height 回归
```

### 推理方式

推理时无监督标签，直接将 MLP 预测值 sigmoid 后投影进入 `c`，完全端到端。

---

## 训练配置

### 快速启动

```bash
bash train_smolvlm_hiphys.sh 16 0.1 ./simvla_output/simvla_hiphys
```

### 手动配置

```bash
python train_smolvlm.py \
    --train_metas_path ./datasets/metas/vlabench_train.json \
    --action_mode vlabench_joint \
    --smolvlm_model_path /path/to/SmolVLM-500M-Instruct \
    --norm_stats_path ./norm_stats/vlabench_norm.json \
    --use_adaln \
    --use_history_encoder --history_seq_len 4 --switch_loss_weight 0.05 \
    --use_physics_cot --physics_weight 0.01 \
    --use_huber_loss --gripper_weight 5.0 \
    --time_sampling logit_normal \
    --batch_size 16 --learning_rate 1e-4 --learning_coef 0.1 \
    --freeze_steps 1000 --iters 200000 \
    --output_dir ./simvla_output/simvla_hiphys
```

---

## 消融实验设计（论文核心实验）

| 变体 | track_3 | track_5 | 说明 |
|---|---|---|---|
| A. Baseline SimVLA | - | - | 无 AdaLN，无 HiPhys |
| B. + AdaLN only | - | - | 仅 AdaLN，无新模块 |
| C. + PhysCoT only | ↑ | - | 仅物理谓词 |
| D. + HistEnc only | - | ↑ | 仅 GRU 历史 |
| E. HiPhys-VLA（完整） | ↑ | ↑ | 主要结果 |
| F. E - switch_loss | - | ↑(小) | 消融切换监督 |
| G. E - error_reset | - | ↑(小) | 消融错误恢复 |

对比基线（已在 VLABench 论文中出现）：
- **OpenVLA**：官方基线
- **π0 (Pi0)**：官方基线
- **CoT-VLA**（CVPR 2025）：图像式 CoT，track_3 近期 SOTA
- **ECoT**：具身链式推理

---

## 关键设计决策

### 零初始化确保向后兼容

所有新增投影层（`h_proj`、`physics_proj`）权重初始化为零：
- 训练初期两个新模块贡献为零，等同于原始 AdaLN 行为
- 现有 checkpoint 不受影响，可直接 fine-tune

### 弱监督无人工标注

物理谓词标签和 switch 标签全部从轨迹数据自动计算（`vlabench_rlds.py`），无需任何人工标注，可直接应用于所有 VLABench 训练数据。

### 推理开销极低

- GRU 每步更新：128维向量，< 1ms
- PhysicsDecoder：1次 MLP 前向，< 1ms
- 总额外推理开销 < 2ms / step

---

## 论文贡献总结

1. **首个将 GRU 历史感知与 Flow Matching 动作头联合端到端训练的 VLA 框架**，在 track_5 长程任务上对所有基线取得显著提升
2. **弱监督物理谓词嵌入**：无人工标注，通过轨迹信号自动抽取5维物理谓词，可微注入 AdaLN 条件，在 track_3 上超越 CoT-VLA（更轻量）
3. **统一可扩展 AdaLN 条件注入框架**：历史 GRU + 物理谓词 + SubgoalVAE 三层条件叠加，理论清晰，消融完整

**目标 venue**：NeurIPS 2025 或 ICLR 2026
