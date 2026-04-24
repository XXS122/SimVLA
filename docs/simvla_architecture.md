# SimVLA 架构图

## 图1：训练数据流

```mermaid
flowchart TD
    subgraph DATA["数据层"]
        TF["VLABench RLDS TFRecord\n/datasets/vlabench/data/1.0.0\n512 shards"]
        HANDLER["VLABenchRLDSHandler\ndatasets/domain_handler/vlabench_rlds.py"]
        READER["SmolVLMDataReader\ndatasets/dataset_smolvlm.py\nIterableDataset · 无限采样"]
    end

    subgraph PREPROC["预处理"]
        IMG["图像处理\nresize → 384×384\nColorJitter 增强\nImageNet 归一化"]
        PROC["SmolVLM Processor\nprocessing_smolvlm_vla.py\nencode_language() → input_ids"]
        NORM["动作归一化\naction_hub.py · VLABenchJointActionSpace\nz-score 归一化"]
    end

    subgraph BATCH["Batch 结构"]
        B1["input_ids [B, L]"]
        B2["pixel_values [B, V, C, H, W]\nV=2 (front + wrist)"]
        B3["proprio [B, 7]"]
        B4["action_norm [B, 10, 7]"]
    end

    subgraph LOSS["损失计算"]
        FM["Flow Matching\nx_t = t·noise + (1-t)·action_norm\nu_t = noise - action_norm"]
        TS["时间步采样\nlogit_normal: t=sigmoid(N(0,1))"]
        HLOSS["Huber Loss + Gripper 加权\nL = huber(v_t, u_t)\nweight[-1] = 5.0 (gripper)"]
        KL["KL Loss (CVAE)\nFree Bits: max(KL, 0.5/dim)\nKL annealing: 0→0.001"]
        ZFM["z空间 FM Loss (LDM)\nL_z = MSE(v_z, u_z)"]
        TOTAL["总损失\nL = L_velocity + λ_kl·L_KL + λ_fm·L_z"]
    end

    TF --> HANDLER --> READER
    READER --> IMG & PROC & NORM
    IMG --> B2
    PROC --> B1
    NORM --> B3 & B4
    B1 & B2 & B3 & B4 --> FM
    FM --> TS --> HLOSS
    HLOSS & KL & ZFM --> TOTAL
```

---

## 图2：模型架构（完整模型）

```mermaid
flowchart TD
    subgraph INPUT["输入"]
        I1["front 图像\n[B, 3, 384, 384]"]
        I2["wrist 图像\n[B, 3, 384, 384]"]
        I3["语言指令\ninput_ids [B, L]"]
        I4["本体感知 proprio\n[B, 7]"]
        I5["噪声 ε ~ N(0,I)\n[B, 10, 7]"]
        I6["时间步 t ∈ (0,1)"]
    end

    subgraph VLM["SmolVLM 骨干 (冻结→微调)"]
        SIGLIP["SigLIP 视觉编码器\n图像 → patch embeddings"]
        CONN["Connector (MLP)\npatch embeddings → VLM 空间"]
        LM["SmolLM2 语言模型\n[image_patches, text_tokens] → hidden states"]
        VF["vlm_features\n[B, seq_len, 576]"]
        VP["vlm_pooled\n[B, 576] (mean pool)"]
    end

    subgraph CVAE["SubgoalVAE (训练/推理分支)"]
        POST["后验 Encoder q(z|vlm,action)\n训练时: vlm_pooled + action_chunk → μ,σ"]
        PRIOR["先验 Encoder p(z|vlm)\n推理时: vlm_pooled → μ,σ"]
        SAMPLE["重参数化采样\nz_goal = μ + σ·ε, z∈R^64"]
        ZPROJ["SubgoalProj\nLinear(64→hidden_size)"]
    end

    subgraph LDM["LatentFlowNet (推理时替换先验采样)"]
        ZNOISE["z噪声 ~ N(0,I)\n[B, 64]"]
        ZEULER["z空间 Euler 积分\n5步 · 条件: vlm_pooled"]
        ZOUT["z_goal [B, 64]"]
    end

    subgraph COND["条件向量 c"]
        TE["时间步嵌入\ntimestep_embedding(t, hidden_size)"]
        PE["本体感知嵌入\nLinear(7→hidden_size)"]
        CCAT["c = time_emb + vlm_pool_proj + proprio_emb + subgoal_proj(z)"]
    end

    subgraph TRANSFORMER["SmolVLMActionTransformer (AdaLN + CrossAttn)"]
        AT["动作 Token 嵌入\n[B, 10, hidden_size=768]"]
        D1["DiTBlockWithCrossAttn × 12\n① AdaLN Self-Attention\n② AdaLN MLP\n③ Cross-Attention → vlm_features"]
        FL["FinalLayer\nAdaLN → Linear(hidden_size→7)"]
        VT["预测速度场 v_t\n[B, 10, 7]"]
    end

    subgraph INFER["推理：Euler 积分 (10步)"]
        X0["x_T = ε ~ N(0,I)"]
        EULER["x_{t-Δt} = x_t - Δt · v_θ(x_t, t, c)"]
        XOUT["动作序列 [B, 10, 7]"]
        DENORM["反归一化\nVLABenchJointActionSpace.denormalize()"]
        ACT["机器人动作\nxyz(3) + axis_angle(3) + gripper(1)"]
    end

    I1 & I2 --> SIGLIP --> CONN --> LM
    I3 --> LM
    LM --> VF & VP

    VP --> POST & PRIOR
    I4 --> POST
    POST -->|训练| SAMPLE
    PRIOR -->|推理(无LDM)| SAMPLE
    ZNOISE --> ZEULER
    VP --> ZEULER
    ZEULER --> ZOUT -->|推理(有LDM)| SAMPLE
    SAMPLE --> ZPROJ

    I6 --> TE
    I4 --> PE
    VP --> CCAT
    TE & PE & ZPROJ --> CCAT

    I5 --> AT
    CCAT --> D1
    VF --> D1
    AT --> D1 --> FL --> VT

    VT -->|训练| HLOSS2["Huber Loss\n对比目标 u_t = ε - action"]
    X0 --> EULER
    CCAT --> EULER
    VF --> EULER
    EULER --> XOUT --> DENORM --> ACT
```

---

## 图3：推理服务流程

```mermaid
sequenceDiagram
    participant ENV as VLABench 环境
    participant CLI as 评估客户端<br/>evaluate_simvla.py
    participant SRV as 推理服务器<br/>serve_smolvlm_vlabench.py
    participant MDL as SimVLA 模型

    ENV->>CLI: 观测 {front_img, wrist_img, proprio, instruction}
    CLI->>SRV: WebSocket + msgpack<br/>{images[2,H,W,3], state[7], instruction}
    SRV->>SRV: 状态转换: quat→axis_angle
    SRV->>MDL: forward_inference()
    MDL->>MDL: VLM 编码 → vlm_features
    MDL->>MDL: LatentFlowNet: 5步 z空间 Euler
    MDL->>MDL: 动作 Euler 积分: 10步
    MDL-->>SRV: actions [10, 7]
    SRV-->>CLI: msgpack {actions[10,7]}
    loop replan_steps=4 次
        CLI->>ENV: 执行 action[i]
        ENV-->>CLI: 下一帧观测
    end
    Note over CLI,ENV: 每4步重新推理一次动作 chunk
```

---

## 图4：消融实验对比

```mermaid
graph LR
    subgraph B["Baseline\ntrain_smolvlm_vlabench.sh"]
        B1["Concat 模式\n动作token + VLM特征拼接"]
        B2["MSE Loss"]
        B3["Beta(1.5,1) 时间步"]
    end

    subgraph C["Ablation-CVAE\n--use_adaln --use_subgoal_vae"]
        C1["AdaLN/DiT 模式\n+ CrossAttn"]
        C2["Huber Loss\n+ Gripper×5"]
        C3["Logit-Normal 时间步"]
        C4["SubgoalVAE\nz∈R^64"]
    end

    subgraph F["完整模型\ntrain_smolvlm_subgoal.sh"]
        F1["AdaLN + CrossAttn"]
        F2["Huber + Gripper×5"]
        F3["Logit-Normal"]
        F4["SubgoalVAE"]
        F5["LatentFlowNet\nz空间双层扩散"]
    end

    B -->|+AdaLN +CrossAttn\n+Huber +CVAE| C
    C -->|+LatentFlowNet| F
```
