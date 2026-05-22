# SimVLA 创新点实现计划

**目标：** 按顺序实现 5 个创新点，每个独立可测试，不破坏现有训练/评估流程。

**实现顺序：** ③ 自适应推理步数 → ① 静态动态解耦 → ⑤ 课程学习时间采样 → ⑥ 辅助世界模型损失 → ⑩ 离线 RL 微调

---

## Task 1：③ 自适应推理步数（ProbeFlow）

**改动文件：** `models/modeling_smolvlm_vla.py`（仅改 `generate_actions`）

**原理：** 每步积分后计算速度向量余弦相似度，相似度 > 阈值说明轨迹已收敛，提前停止。无需重新训练。

### Step 1：在 `generate_actions` 加入自适应逻辑

将 `generate_actions` 方法替换为：

```python
@torch.no_grad()
def generate_actions(
    self,
    input_ids: torch.LongTensor,
    image_input: torch.FloatTensor,
    image_mask: torch.Tensor,
    proprio: torch.Tensor,
    steps: int = 10,
    adaptive: bool = False,
    cos_threshold: float = 0.97,
    min_steps: int = 2,
) -> torch.Tensor:
    self.eval()
    enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

    B = input_ids.shape[0]
    D = self.action_space.dim_action
    device = proprio.device
    dtype = proprio.dtype

    if hasattr(self.action_space, 'normalize_state'):
        proprio_norm = self.action_space.normalize_state(proprio)
    else:
        proprio_norm = proprio

    steps = max(1, int(steps))
    dt = -1.0 / steps

    x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
    t = 1.0
    v_prev = None
    actual_steps = 0

    while t > -dt / 2:
        t_tensor = torch.full((B,), t, device=device, dtype=dtype)
        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            proprio=proprio_norm,
            t=t_tensor,
        )
        x_t = x_t + dt * v_t
        t = t + dt
        actual_steps += 1

        # 自适应早停：速度方向收敛则停止
        if adaptive and v_prev is not None and actual_steps >= min_steps:
            # v_t: [B, T_action, dim_action] → 展平后计算余弦相似度
            v_flat = v_t.reshape(B, -1)
            v_prev_flat = v_prev.reshape(B, -1)
            cos_sim = torch.nn.functional.cosine_similarity(v_flat, v_prev_flat, dim=-1).mean()
            if cos_sim.item() > cos_threshold:
                break

        v_prev = v_t

    return self.action_space.postprocess(x_t)
```

### Step 2：在 `serve_smolvlm_libero.py` 的 `infer()` 启用自适应

找到 `generate_actions` 调用处，加 `adaptive=True`：

```python
actions = model.generate_actions(
    input_ids=lang['input_ids'],
    image_input=images,
    image_mask=image_mask,
    proprio=proprio_tensor,
    steps=CONFIG["action_horizon"],
    adaptive=True,
    cos_threshold=0.97,
    min_steps=2,
)
```

### Step 3：验证

用现有 checkpoint 跑推理，打印实际步数：

```python
# 临时在 generate_actions 末尾加一行调试
print(f"[adaptive] actual_steps={actual_steps}/{steps}")
```

预期：简单直线运动 2-4 步收敛，复杂抓取 8-10 步。

### Step 4：调参建议

| cos_threshold | 效果 |
|---|---|
| 0.99 | 保守，接近原始 10 步 |
| 0.97 | 推荐，平均约 3-5 步 |
| 0.95 | 激进，可能影响精度 |

---

## Task 2：① 静态-动态特征解耦

**改动文件：**
- `models/modeling_smolvlm_vla.py`：新增 `encode_static_context()` 方法，修改 `generate_actions()`
- `evaluation/libero/serve_smolvlm_libero.py`：episode 开始时缓存静态特征

**原理：** 语言指令 + agentview（第三视角，index=0）只在 episode 开始时算一次并缓存；每步只跑 eye_in_hand（index=1）的动态路，再与静态特征拼接。

### Step 1：在 `SmolVLMVLA` 新增 `encode_static_context()`

在 `generate_actions` 之前插入：

```python
@torch.no_grad()
def encode_static_context(
    self,
    input_ids: torch.LongTensor,       # [B, L]
    image_input: torch.FloatTensor,    # [B, V, C, H, W]
    image_mask: torch.Tensor,          # [B, V]
    static_view_idx: int = 0,          # agentview 的视角索引
) -> torch.Tensor:
    """
    只编码静态视角（agentview）+ 语言，返回缓存特征。
    episode 开始时调用一次，后续每步复用。
    """
    # 只保留 static_view_idx 对应的视角
    static_images = image_input[:, static_view_idx:static_view_idx+1, ...]  # [B, 1, C, H, W]
    static_mask = image_mask[:, static_view_idx:static_view_idx+1]          # [B, 1]
    enc = self.forward_vlm_efficient(static_images, static_mask, input_ids)
    return enc["vlm_features"]  # [B, T_static, D]
```

### Step 2：新增 `encode_dynamic_view()`

```python
@torch.no_grad()
def encode_dynamic_view(
    self,
    image_input: torch.FloatTensor,    # [B, V, C, H, W]
    image_mask: torch.Tensor,          # [B, V]
    dynamic_view_idx: int = 1,         # eye_in_hand 的视角索引
) -> torch.Tensor:
    """
    只编码动态视角（eye_in_hand），每步调用。
    不经过 text_model，只跑 vision_encoder + connector。
    """
    B = image_input.shape[0]
    dyn_img = image_input[:, dynamic_view_idx, ...]  # [B, C, H, W]

    vision_outputs = self.vlm.model.vision_model(
        pixel_values=dyn_img,
        output_hidden_states=True,
        return_dict=True,
    )
    feats = vision_outputs.last_hidden_state  # [B, num_patches, vision_hidden]

    if hasattr(self.vlm.model, 'connector'):
        feats = self.vlm.model.connector(feats)
    elif hasattr(self.vlm.model, 'multi_modal_projector'):
        feats = self.vlm.model.multi_modal_projector(feats)

    return feats  # [B, num_patches, D]
```

### Step 3：新增 `generate_actions_with_cache()`

```python
@torch.no_grad()
def generate_actions_with_cache(
    self,
    static_context: torch.Tensor,      # [B, T_static, D]，来自 encode_static_context()
    dynamic_feats: torch.Tensor,       # [B, num_patches, D]，来自 encode_dynamic_view()
    proprio: torch.Tensor,             # [B, dim_proprio]
    steps: int = 10,
    adaptive: bool = True,
    cos_threshold: float = 0.97,
    min_steps: int = 2,
) -> torch.Tensor:
    """
    使用缓存的静态特征 + 当前动态特征推理，避免重复跑 text_model。
    """
    # 拼接静态 + 动态特征
    vlm_features = torch.cat([static_context, dynamic_feats], dim=1)  # [B, T_static+num_patches, D]

    B = static_context.shape[0]
    D = self.action_space.dim_action
    device = proprio.device
    dtype = proprio.dtype

    if hasattr(self.action_space, 'normalize_state'):
        proprio_norm = self.action_space.normalize_state(proprio)
    else:
        proprio_norm = proprio

    steps = max(1, int(steps))
    dt = -1.0 / steps
    x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
    t = 1.0
    v_prev = None
    actual_steps = 0

    while t > -dt / 2:
        t_tensor = torch.full((B,), t, device=device, dtype=dtype)
        v_t = self.transformer(
            vlm_features=vlm_features,
            action_with_noise=x_t,
            proprio=proprio_norm,
            t=t_tensor,
        )
        x_t = x_t + dt * v_t
        t = t + dt
        actual_steps += 1

        if adaptive and v_prev is not None and actual_steps >= min_steps:
            v_flat = v_t.reshape(B, -1)
            v_prev_flat = v_prev.reshape(B, -1)
            cos_sim = torch.nn.functional.cosine_similarity(v_flat, v_prev_flat, dim=-1).mean()
            if cos_sim.item() > cos_threshold:
                break
        v_prev = v_t

    return self.action_space.postprocess(x_t)
```

### Step 4：修改 `serve_smolvlm_libero.py` 的 `infer()`

在 `infer()` 函数外部维护 episode 级缓存：

```python
# 全局缓存（每个 WebSocket 连接对应一个 episode）
_static_cache: dict = {}   # key: connection_id, value: static_context tensor

async def handle_connection(websocket, path=None):
    conn_id = id(websocket)
    _static_cache[conn_id] = None   # 新连接，清空缓存
    try:
        ...
    finally:
        _static_cache.pop(conn_id, None)
```

在 `infer()` 里：

```python
def infer(observation, conn_id=None):
    ...
    # 第一步：计算并缓存静态特征
    if conn_id is not None and _static_cache.get(conn_id) is None:
        static_context = model.encode_static_context(
            input_ids=lang['input_ids'],
            image_input=images,
            image_mask=image_mask,
            static_view_idx=0,
        )
        _static_cache[conn_id] = static_context
    else:
        static_context = _static_cache.get(conn_id)

    # 每步：只编码动态视角
    dynamic_feats = model.encode_dynamic_view(
        image_input=images,
        image_mask=image_mask,
        dynamic_view_idx=1,
    )

    # 推理
    with torch.no_grad():
        actions = model.generate_actions_with_cache(
            static_context=static_context,
            dynamic_feats=dynamic_feats,
            proprio=proprio_tensor,
            steps=CONFIG["action_horizon"],
            adaptive=True,
        )
```

### Step 5：验证

对比两种推理方式的输出差异（应该接近但不完全相同，因为静态特征不再包含动态视角信息）：

```python
# 用原始 generate_actions 和新 generate_actions_with_cache 各跑一次
# 打印动作的 L2 距离，确认在合理范围内（< 0.5）
```

---

## Task 3：⑤ 非均匀时间采样 + 课程学习

**改动文件：** `train_smolvlm.py`（仅改时间采样约 10 行）

**原理：** 根据训练进度动态调整 Beta 分布参数，早期均匀探索，后期偏向精细动作端（小 t）。

### Step 1：在 `train_smolvlm.py` 的 `forward()` 替换时间采样

找到：
```python
beta_dist = torch.distributions.Beta(
    torch.tensor(1.5, device=device),
    torch.tensor(1.0, device=device)
)
t = beta_dist.sample((B,)) * 0.999 + 0.001
```

替换为（需要把 `global_step` 和 `total_steps` 传入 `forward`，或在训练循环里采样 t 后传入）：

**推荐做法：在训练循环里采样 t，作为参数传入 `forward`。**

在 `train_smolvlm.py` 的训练循环里，`model(**inputs)` 之前加：

```python
def sample_time(B: int, global_step: int, total_steps: int, device) -> torch.Tensor:
    """根据训练进度动态调整 Beta 分布参数。"""
    progress = global_step / max(total_steps, 1)
    if progress < 0.3:
        alpha = 1.0   # 均匀采样，探索全局结构
    elif progress < 0.7:
        alpha = 1.5   # 原始设置
    else:
        alpha = 2.5   # 偏向小 t，强化精细控制
    beta_dist = torch.distributions.Beta(
        torch.tensor(alpha, device=device),
        torch.tensor(1.0, device=device),
    )
    return beta_dist.sample((B,)) * 0.999 + 0.001
```

然后在训练循环里：

```python
B = inputs["action"].shape[0]
t_sample = sample_time(B, global_step, args.iters, device)
inputs["t_override"] = t_sample   # 传入 forward
```

### Step 2：修改 `SmolVLMVLA.forward()` 接受外部 t

在 `forward` 签名加 `t_override=None`：

```python
def forward(
    self,
    input_ids, image_input, image_mask, proprio, action,
    t_override: torch.Tensor | None = None,   # 新增
) -> Dict[str, torch.Tensor]:
    ...
    # 原来的 Beta 采样改为：
    if t_override is not None:
        t = t_override.to(device)
    else:
        beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, device=device),
            torch.tensor(1.0, device=device),
        )
        t = beta_dist.sample((B,)) * 0.999 + 0.001
```

### Step 3：验证

在训练日志里打印当前 alpha 值，确认随 step 变化：

```python
if global_step % 1000 == 0:
    progress = global_step / args.iters
    alpha = 1.0 if progress < 0.3 else (1.5 if progress < 0.7 else 2.5)
    accelerator.log({"time_sampling/alpha": alpha}, step=global_step)
```

---

## Task 4：⑥ 辅助世界模型预测损失

**改动文件：**
- `models/transformer_smolvlm.py`：在 `SmolVLMActionTransformer` 加 `state_pred_head`
- `models/modeling_smolvlm_vla.py`：`forward()` 加 state_loss
- `datasets/dataset_smolvlm.py`：`__iter__` 返回 `next_proprio`
- `datasets/domain_handler/libero_hdf5.py`：提取 next_proprio

### Step 1：在 `SmolVLMActionTransformer.__init__` 加预测头

在 `__init__` 末尾加：

```python
# 辅助世界模型头：预测执行动作后的下一个 proprio 状态
self.state_pred_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 2),
    nn.SiLU(),
    nn.Linear(hidden_size // 2, dim_propio),
)
```

### Step 2：在 `SmolVLMActionTransformer.forward()` 返回中间特征

修改 `forward` 返回值，同时返回动作 token 的特征（用于 state_pred_head）：

```python
def forward(self, vlm_features, action_with_noise, proprio, t, return_features=False):
    if self.use_adaln:
        result = self._forward_adaln(vlm_features, action_with_noise, proprio, t, return_features)
    else:
        result = self._forward_concat(vlm_features, action_with_noise, proprio, t, return_features)
    return result
```

在 `_forward_concat` 末尾：

```python
action_feats = self.norm(x[:, :num_actions])   # [B, T_action, H]
velocity = self.action_decoder(action_feats)    # [B, T_action, dim_action]

if return_features:
    return velocity, action_feats
return velocity
```

`_forward_adaln` 同理修改 `final_layer` 前返回特征。

### Step 3：修改 `SmolVLMVLA.forward()` 加 state_loss

```python
def forward(
    self,
    input_ids, image_input, image_mask, proprio, action,
    next_proprio: torch.Tensor | None = None,   # 新增
    t_override: torch.Tensor | None = None,
    state_loss_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    ...
    # 原来的 transformer 调用改为：
    use_state_loss = next_proprio is not None
    if use_state_loss:
        v_t, action_feats = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
            return_features=True,
        )
    else:
        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
        )

    velocity_loss = torch.mean(torch.square(v_t - u_t))
    loss_dict = {"velocity_loss": velocity_loss}

    # 辅助 state 预测损失
    if use_state_loss:
        # 用第一个动作 token 的特征预测下一个 proprio
        state_pred = self.transformer.state_pred_head(action_feats[:, 0, :])  # [B, dim_proprio]
        if hasattr(self.action_space, 'normalize_state'):
            next_proprio_norm = self.action_space.normalize_state(next_proprio)
        else:
            next_proprio_norm = next_proprio
        state_loss = torch.mean(torch.square(state_pred - next_proprio_norm))
        loss_dict["state_loss"] = state_loss_weight * state_loss

    return loss_dict
```

### Step 4：在 `LiberoHDF5Handler` 提取 next_proprio

在 `datasets/domain_handler/libero_hdf5.py` 的数据提取逻辑里，在返回 `proprio` 的同时返回 `next_proprio`（即下一时刻的 proprio）：

```python
# 假设当前 step 索引为 idx，proprio 维度为 [T, dim_proprio]
proprio = states[idx]           # 当前状态
next_proprio = states[min(idx + args.num_actions, len(states) - 1)]  # num_actions 步后的状态
```

### Step 5：在 `dataset_smolvlm.py` 的 `__iter__` 返回 `next_proprio`

在 yield 的 dict 里加：

```python
yield {
    'language_instruction': ...,
    'image_input': ...,
    'image_mask': ...,
    'proprio': ...,
    'action': ...,
    'next_proprio': next_proprio,   # 新增
}
```

### Step 6：验证

训练时观察 `state_loss` 是否在合理范围（应该比 `velocity_loss` 小一个量级），并确认总 loss 下降正常：

```
velocity_loss: ~0.5 (初始)
state_loss (weighted): ~0.05 (初始)
```

---

## Task 5：⑩ 离线 RL 微调（IQL）

**改动文件：**
- 新增 `train_rl_finetune.py`：IQL 训练脚本
- 新增 `collect_rollouts.py`：用 BC 模型收集 rollout 数据
- `models/modeling_smolvlm_vla.py`：加 Q 网络头

**注意：** 这是最复杂的一步，需要先完成前 4 个创新点并训练出一个较好的 BC checkpoint，再做 RL 微调。

### Step 1：收集 rollout 数据

新建 `collect_rollouts.py`，复用 `evaluation/libero/libero_client.py` 的 rollout 逻辑，将每条轨迹的 `(obs, action, reward, next_obs)` 保存为 HDF5：

```python
# 成功 episode：所有步 reward = 1.0
# 失败 episode：所有步 reward = 0.0
# 混合原始演示数据（reward = 1.0）
```

### Step 2：在 `SmolVLMVLA` 加 Q 网络头

```python
# 在 __init__ 末尾加
self.q_head = nn.Sequential(
    nn.Linear(config.hidden_size, config.hidden_size // 2),
    nn.SiLU(),
    nn.Linear(config.hidden_size // 2, 1),
)
```

新增 `forward_q()` 方法：

```python
def forward_q(self, input_ids, image_input, image_mask, proprio, action) -> torch.Tensor:
    """计算 Q(s, a)，用于 IQL 训练。"""
    enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)
    _, action_feats = self.transformer(
        vlm_features=enc["vlm_features"],
        action_with_noise=action,
        t=torch.zeros(action.shape[0], device=action.device),
        proprio=proprio,
        return_features=True,
    )
    q_value = self.q_head(action_feats[:, 0, :])  # [B, 1]
    return q_value.squeeze(-1)
```

### Step 3：IQL 训练循环（`train_rl_finetune.py`）

IQL 核心：不需要与环境交互，只用离线数据。

```python
# IQL 损失（简化版）
# 1. Value 网络：V(s) = E_a[Q(s,a)]，用 expectile regression
# 2. Q 网络：Q(s,a) = r + γ * V(s')
# 3. Actor：最大化 exp(β*(Q(s,a)-V(s))) * log π(a|s)

# 只微调 transformer_core 和 action_heads，VLM 冻结
# 学习率 = BC 阶段的 1/10
```

### Step 4：验证

在 LIBERO 上对比 BC checkpoint 和 RL 微调后的成功率，预期提升 5-15%（尤其是 libero_10 长视野任务）。

---

## 实施注意事项

1. **Task 1（自适应步数）** 无需重训，直接在现有 checkpoint 上验证，风险最低
2. **Task 2（静态动态解耦）** 改变了推理时的特征分布，需要重新训练才能发挥最大效果；但可以先在现有 checkpoint 上测试接口是否正确
3. **Task 3、4** 需要从头训练（或从现有 checkpoint 继续训练），建议在 Task 1/2 验证后再启动
4. **Task 5** 依赖 Task 3/4 训练出的更好 checkpoint，放最后
5. 每个 Task 完成后建议单独评估一次 LIBERO 成功率，记录基线对比
