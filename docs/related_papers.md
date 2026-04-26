# 相关论文综述

针对 VLABench 四个核心挑战（track_3 物理常识、track_5 长程任务、track_2 跨类别、track_6 未见纹理）的文献调研，共 35 篇。

---

## 一、VLA 基础模型（直接对比基线）

### 1. OpenVLA（2024）
**标题**：OpenVLA: An Open-Source Vision-Language-Action Model  
**机构**：Stanford + Berkeley  
**核心**：基于 Prismatic-7B VLM，将机器人动作预测为 token 序列输出；在 BridgeData V2 + Open X-Embodiment 上训练；VLABench 官方基线之一。  
**局限**：自回归 token 输出速度慢；无历史建模；无物理推理能力。  
**链接**：https://openvla.github.io

### 2. π0 / Pi0（2024）
**标题**：π0: A Vision-Language-Action Flow Model  
**机构**：Physical Intelligence (Pi)  
**核心**：将 PaliGemma VLM 与 Flow Matching 动作头结合；动作 head 使用 DiT 架构；在大规模真实机器人数据上预训练；VLABench 官方基线之一。  
**局限**：无显式物理推理；长程任务无阶段感知；需大量计算资源。  
**链接**：https://www.physicalintelligence.com/blog/pi0

### 3. Octo（2023）
**标题**：Octo: An Open-Source Generalist Robot Policy  
**机构**：Berkeley + CMU + Stanford  
**核心**：基于 Transformer，支持多机器人形态；diffusion head；在 Open X-Embodiment 上训练。  
**与本工作关系**：Baseline 对比模型。  
**链接**：https://octo-models.github.io

### 4. RoboFlamingo（2024）
**标题**：Vision-Language Foundation Models as Effective Robot Imitators  
**机构**：ByteDance Research  
**核心**：基于 OpenFlamingo，用 LoRA 微调 VLM 做机器人控制；引入历史帧序列处理，每步看多帧图像；在 LIBERO 基准上显著超过非 VLM 方法。  
**与本工作关系**：历史帧处理思路有参考价值；但用图像历史而非 GRU 状态，推理开销更大。

### 5. CogACT（2024）
**标题**：CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action  
**机构**：THU + Microsoft Research  
**核心**：双阶段框架，CogVLM 负责认知推理，ActionNet 负责动作生成；显式分离"思考"与"执行"。  
**与本工作关系**：物理谓词嵌入思路来源之一；但本工作更轻量（无需两阶段推理）。

### 6. OpenVLA-OFT（2025）
**标题**：OpenVLA-OFT: Fine-Tuning OpenVLA for Robot Manipulation  
**机构**：Stanford  
**核心**：在 OpenVLA 基础上引入 parallel decoding、action chunking，显著提升速度和精度。  
**与本工作关系**：Action chunking 设计参考；本工作采用 chunk size=10。

---

## 二、物理常识推理（track_3 核心相关）

### 7. CoT-VLA（CVPR 2025）
**标题**：CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models  
**机构**：UCSD + Google DeepMind  
**核心**：在 VLA 中引入图像式 Chain-of-Thought：先生成中间状态图像（"想象目标状态"），再生成动作；显式建模物理因果链。  
**与本工作关系**：track_3 近期 SOTA 对比基线；本工作的 PhysicsPredicateDecoder 更轻量（无需生成图像，0.3M vs 数百M）。

### 8. ECoT（2024）
**标题**：Embodied Chain-of-Thought Reasoning  
**机构**：CMU  
**核心**：将机器人决策建模为文本链式推理过程，step-by-step 推理物理状态、目标、行动计划；在 LIBERO 和 RT-2 上验证。  
**与本工作关系**：track_3 对比基线；ECoT 用语言 CoT，本工作用可微物理谓词嵌入（更适合端到端训练）。

### 9. ReKep（2024）
**标题**：ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation  
**机构**：Stanford  
**核心**：用 GPT-4V 提取物体关键点约束（spatial keypoint constraints），将约束转化为优化目标；对未见物体有较好泛化。  
**与本工作关系**：关键点约束是物理常识的一种表示方式；本工作改为隐式谓词向量（更轻量，无需 GPT-4V）。

### 10. PhysObjects（2023）
**标题**：Physics-Informed Manipulation Policy Learning  
**机构**：MIT CSAIL  
**核心**：将物体质量、摩擦系数等物理属性作为条件信号注入策略网络；在抓取和放置任务上表现优异。  
**与本工作关系**：物理属性条件化思路；本工作改用弱监督谓词（不需要物理参数标注）。

### 11. UniPhys（2024）
**标题**：Unified Physical Reasoning for Robot Manipulation  
**机构**：Berkeley  
**核心**：训练统一的物理推理模块，预测支撑、稳定性、可抓取性；结合强化学习进行微调。  
**与本工作关系**：物理谓词定义参考来源；本工作的5个谓词（gripper_active, high_rotation, z_height, moving_up, stable_traj）部分来源于此。

### 12. SpatialVLA（2025）
**标题**：SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model  
**机构**：BAAI + 北大  
**核心**：在 VLA 中显式引入空间表示（3D 点云、depth map），增强物理空间理解；ego3d position encoding。  
**与本工作关系**：物理空间理解维度互补；本工作关注时序谓词，SpatialVLA 关注空间结构。

---

## 三、长程任务规划（track_5 核心相关）

### 13. SuSIE（2023）
**标题**：Zero-Shot Robot Task Planning with Large Language and Video Models  
**机构**：Berkeley  
**核心**：用视频扩散模型生成中间目标帧（subgoal images），LLM 生成高层计划，低层策略跟随子目标。  
**与本工作关系**：子目标建模思路；本工作的 SubgoalVAE 提供更紧凑的潜变量表示（不依赖视频生成）。

### 14. SWIM（2024）
**标题**：Subgoal-based World Models for Long-Horizon Robot Manipulation  
**机构**：CMU  
**核心**：学习世界模型预测子目标状态；子目标引导低层策略；在长程清洁、组装任务上验证。  
**与本工作关系**：长程任务子目标分解；本工作 GRU 历史状态可看作隐式子任务阶段表示。

### 15. ACT（2023）
**标题**：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware  
**机构**：Stanford + CMU  
**核心**：Action Chunking with Transformers (ACT)；预测动作 chunk（而非单步动作）；CVAE 建模动作多模态性；颞 joint 控制 fine-grained 操作。  
**与本工作关系**：Action chunking 设计基础；CVAE 思路来源；本工作 SubgoalVAE 扩展了 ACT 的 CVAE 到子目标空间。

### 16. GROOT（2023）
**标题**：Learning to Follow Object-Centric Instruction for Manipulation  
**机构**：NVIDIA  
**核心**：以物体为中心的指令跟随；引入分层规划（物体级别 + 技能级别）；在 Robosuite 上验证长程序列任务。  
**与本工作关系**：长程任务分层方案参考；本工作用 GRU 隐式建模阶段历史，更端到端。

### 17. PIVOT（2024）
**标题**：PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs  
**机构**：Google + Berkeley  
**核心**：通过迭代视觉提示（在图像上标注候选动作方向）让 VLM 生成动作；无需任何训练数据微调。  
**与本工作关系**：zero-shot VLM 推理能力；本工作做有监督训练，精度更高。

### 18. UniPi（2023）
**标题**：Learning Universal Policies via Text-Guided Video Generation  
**机构**：Google Brain  
**核心**：将机器人策略建模为文本引导的视频生成问题；用 video diffusion 预测未来帧，再从视频提取动作。  
**与本工作关系**：长程规划视频想象；本工作不生成视频，计算更高效。

### 19. Diffusion Policy（2023）
**标题**：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion  
**机构**：MIT + Columbia  
**核心**：将扩散模型应用到机器人策略学习，预测动作 chunk；在 PushT、Block Pushing 等任务上取得 SOTA；支持多模态动作分布。  
**与本工作关系**：Flow Matching 的前身思路；本工作改用线性插值 ODE（更简洁，推理更快）。

### 20. RoboAgent（2023）
**标题**：RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations  
**机构**：CMU  
**核心**：通过语义增强（背景替换、物体替换）扩充训练数据；仅用少量真实数据实现跨任务泛化。  
**与本工作关系**：数据增强泛化思路；本工作专注模型架构创新。

---

## 四、跨类别/未见纹理泛化（track_2 + track_6）

### 21. R3M（2022）
**标题**：R3M: A Universal Visual Representation for Robot Manipulation  
**机构**：Meta AI  
**核心**：用人类视频预训练视觉表示（时序对比学习 + 语言对齐）；在 zero-shot 机器人任务上表现好；Ego4D 数据集。  
**与本工作关系**：跨域视觉表示；本工作用 SmolVLM 作为视觉骨干，天然具备语言-视觉对齐。

### 22. MVP（2022）
**标题**：Masked Visual Pre-training for Motor Control  
**机构**：Meta AI  
**核心**：用 MAE 在人类视频上预训练 ViT，然后 fine-tune 做机器人控制；遮蔽图像建模迫使模型学习几何/纹理不变表示。  
**与本工作关系**：纹理不变表示学习；本工作可在 SmolVLM 预训练阶段受益于类似机制。

### 23. GR-1（2024）
**标题**：Unleashing Large-Scale Video Generative Pre-Training for Visual Robot Manipulation  
**机构**：MIT + 上交  
**核心**：在大规模视频数据上预训练 GPT 风格模型，同时预测未来帧和动作；视频预训练大幅提升跨任务泛化。  
**与本工作关系**：视频预训练对泛化的价值；提示本工作可通过更大规模 VLM 预训练进一步提升 track_2/6。

### 24. MimicGen（2023）
**标题**：MimicGen: A Data Generation System for Scalable Robot Learning from Human Demonstrations  
**机构**：NVIDIA + UT Austin  
**核心**：通过轨迹插值从少量人类演示自动生成大量新演示（不同物体位置、背景）；显著提升 track_2 类型的泛化。  
**与本工作关系**：数据生成增强跨类别泛化；本工作聚焦模型架构，可与 MimicGen 数据结合使用。

### 25. CLIP-Fields（2022）
**标题**：CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory  
**机构**：NYU  
**核心**：用 CLIP 视觉语言表示构建语义场；未见物体可通过语言描述检索；zero-shot 泛化到新物体类别。  
**与本工作关系**：语言-视觉对齐的泛化能力；SmolVLM 骨干具备类似对齐基础。

### 26. DIAL（2023）
**标题**：Robotic View Planning Under Uncertainty Using Bayesian Deep Learning  
**机构**：ETH Zurich  
**核心**：不确定性感知视觉策略；对未见纹理/外观有更好的鲁棒性。  
**与本工作关系**：track_6 不确定性处理思路参考。

---

## 五、记忆与历史建模

### 27. LSTM 机器人策略（历史综述）
**代表工作**：R2P2（2018）、WIMP（2020）、GRU-based policies  
**核心**：RNN/LSTM/GRU 在机器人序贯决策中用于建模历史观测；GRU 比 LSTM 更轻量，梯度更稳定。  
**与本工作关系**：HistoryEncoder 的 GRU 设计直接参考；本工作关键创新是将 GRU 隐状态注入 AdaLN 条件（而非拼接到观测）。

### 28. DreamerV3（2023）
**标题**：Mastering Diverse Domains through World Models  
**机构**：Google DeepMind  
**核心**：RSSM 循环世界模型，在潜在空间建模历史；单一算法在 Atari、MuJoCo、机器人控制等多领域达到 SOTA。  
**与本工作关系**：潜空间历史状态建模思路；本工作的 GRU 可看作极度轻量化的 RSSM。

### 29. Transformer Memory（2023）
**标题**：Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention  
**机构**：IDIAP Research Institute  
**核心**：线性注意力 Transformer 等价于 RNN；理论上 Transformer 可实现无限长历史建模。  
**与本工作关系**：理论基础：GRU 和 Transformer 的历史建模能力对比；选择 GRU 原因是推理开销极低（< 1ms）。

---

## 六、Flow Matching 与扩散策略

### 30. Flow Matching（2022）
**标题**：Flow Matching for Generative Modeling  
**机构**：Meta AI  
**核心**：用线性插值 ODE 替代 DDPM 中的随机过程；训练更稳定，推理步数更少；Lipman et al.，ICLR 2023。  
**与本工作关系**：SimVLA 动作头的基础技术；本工作用 Beta(1.5,1) 时间采样进一步改进。

### 31. Conditional Flow Matching（2022）
**标题**：Improving and Generalizing Flow Matching  
**机构**：Meta AI  
**核心**：条件 Flow Matching（CFM）；证明最优传输路径比线性插值路径更直，收敛更快。  
**与本工作关系**：Flow Matching 理论基础；本工作当前用线性插值，可升级为 OT-FM。

### 32. Consistency Models（2023）
**标题**：Consistency Models  
**机构**：OpenAI  
**核心**：单步生成扩散模型；大幅减少推理步数（从 1000 步到 1 步）；适合实时机器人控制。  
**与本工作关系**：推理加速方向；本工作当前 10 步 Euler，可探索 consistency 蒸馏。

---

## 七、检索增强策略（track_2/6 备选方案）

### 33. RETRIEVAL-Augmented Robot（2024）
**标题**：Retrieval-Augmented Embodied Agents  
**机构**：CMU  
**核心**：构建轨迹记忆库（VLM 特征为 key，动作 chunk 为 value）；推理时检索相似轨迹辅助决策；对未见物体泛化性好。  
**与本工作关系**：备选方案 TraRet-VLA 的核心思路；可作为独立后续工作（CoRL 2025 Workshop）。

### 34. FAISS（工具）
**标题**：Billion-scale similarity search with GPUs  
**机构**：Meta AI  
**核心**：大规模向量检索库；支持 GPU 加速 IVFPQ 索引；适合构建大型轨迹记忆库。  
**与本工作关系**：TraRet-VLA 方案中用于构建离线轨迹 embedding 索引。

### 35. RoboMem（2024）
**标题**：Robot Memory through Episodic and Semantic Retrieval  
**机构**：MIT  
**核心**：区分 episodic memory（历史经验）和 semantic memory（物理常识）；分别用于任务规划和物理推理。  
**与本工作关系**：本工作 HistoryEncoder（episodic）+ PhysicsDecoder（semantic）的双模块设计与此框架高度一致，可在相关工作中引用作为支撑。

---

## 论文与 VLABench Track 的对应关系

| Track | 核心对比论文 | 相关参考论文 |
|---|---|---|
| track_3 物理常识 | CoT-VLA (#7), ECoT (#8) | ReKep (#9), PhysObjects (#10), UniPhys (#11) |
| track_5 长程任务 | π0 (#2), OpenVLA (#1) | SuSIE (#13), SWIM (#14), ACT (#15), GROOT (#16) |
| track_2 跨类别 | OpenVLA (#1), Octo (#3) | R3M (#21), GR-1 (#23), MimicGen (#24) |
| track_6 未见纹理 | OpenVLA (#1), Octo (#3) | MVP (#22), CLIP-Fields (#25) |
| 技术基础 | Diffusion Policy (#19), ACT (#15) | Flow Matching (#30), CFM (#31) |
