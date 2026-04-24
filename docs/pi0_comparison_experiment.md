# π0 对比实验指南

本文档记录如何在 VLABench 数据集上 fine-tune π0，与 SimVLA 做公平对比实验。

---

## 一、背景

π0 是 Physical Intelligence 开源的 VLA 模型，架构与 SimVLA 同属 Flow Matching 路线，
但动作头直接嵌入 VLM transformer 层（action expert），而非独立 Transformer。

对比实验目标：**相同数据集、相同评估 track，π0 fine-tune vs SimVLA 训练，比较成功率。**

---

## 二、环境安装

```bash
conda create -n lerobot python=3.10 -y
conda activate lerobot

git clone https://github.com/huggingface/lerobot
cd lerobot
pip install -e ".[pi0]"
```

---

## 三、下载 π0 base 权重

```bash
# 需要 HuggingFace 账号（免费，无需申请）
huggingface-cli login

huggingface-cli download lerobot/pi0_base \
    --local-dir /datasets/models/pi0_base
```

可用的公开权重：
- `lerobot/pi0_base` — 通用 base，推荐用于 fine-tune
- `lerobot/pi0_libero_base` — 在 LIBERO 上 fine-tune 过的版本

---

## 四、数据格式转换（TODO：需要写转换脚本）

π0 通过 LeRobot 框架训练，要求数据为 `LeRobotDataset` 格式：

```
vlabench_lerobot/
├── meta/
│   └── info.json          # 数据集元信息（动作维度、摄像头名称等）
├── data/
│   └── chunk-000/
│       └── episode_*.parquet   # 每个 episode 的状态/动作数据
└── videos/
    └── chunk-000/
        ├── observation.front/
        │   └── episode_*.mp4
        └── observation.wrist/
            └── episode_*.mp4
```

VLABench 原始格式是 RLDS TFRecord（`/datasets/vlabench/data/1.0.0`），需要写转换脚本：

**转换脚本待写**：`scripts/convert_vlabench_to_lerobot.py`

核心字段映射：
| VLABench RLDS 字段 | LeRobot 字段 |
|---|---|
| `steps/action` [7] | `action` [7] |
| `steps/observation/ee_state` [7] | `observation.state` [7] |
| `steps/observation/front` (JPEG bytes) | `observation.front` (mp4 frame) |
| `steps/observation/wrist` (JPEG bytes) | `observation.wrist` (mp4 frame) |
| `steps/language_instruction` | `task` (meta/tasks.jsonl) |

`meta/info.json` 关键字段：
```json
{
  "fps": 10,
  "features": {
    "action": {"dtype": "float32", "shape": [7]},
    "observation.state": {"dtype": "float32", "shape": [7]},
    "observation.front": {"dtype": "video"},
    "observation.wrist": {"dtype": "video"}
  }
}
```

---

## 五、Fine-tune π0

```bash
conda activate lerobot
cd /path/to/lerobot

python lerobot/scripts/train.py \
    --policy.type=pi0 \
    --policy.pretrained_path=/datasets/models/pi0_base \
    --dataset.repo_id=/datasets/vlabench_lerobot \
    --dataset.local_files_only=true \
    --output_dir=/datasets/simvla_output/pi0_vlabench \
    --steps=100000 \
    --batch_size=32 \
    --learning_rate=1e-4 \
    --save_freq=10000 \
    --log_freq=100
```

单卡显存不足时加 LoRA：
```bash
    --policy.use_lora=true \
    --policy.lora_rank=32
```

---

## 六、启动推理服务器

```bash
conda activate lerobot

python lerobot/scripts/serve.py \
    --policy.path=/datasets/simvla_output/pi0_vlabench/checkpoints/100000 \
    --host 0.0.0.0 \
    --port 8001
```

服务器使用 WebSocket + msgpack 协议，与 VLABench 的 `OpenPiPolicy` 完全兼容。

---

## 七、VLABench 评估

```bash
conda activate vlabench
cd /datasets/code/VLABench

# 评估所有 track
for track in track_1_in_distribution track_2_cross_category track_3_common_sense track_4_semantic_instruction track_6_unseen_texture; do
    python scripts/evaluate_policy.py \
        --eval-track $track \
        --policy openpi \
        --host localhost \
        --port 8001 \
        --replanstep 4 \
        --n-episode 50 \
        --metrics success_rate intention_score progress_score \
        --save-dir /datasets/simvla_output/eval_results/pi0_vlabench
done
```

---

## 八、对比实验设置

公平对比的关键：

| 项目 | π0 | SimVLA |
|---|---|---|
| 训练数据 | VLABench（相同） | VLABench（相同） |
| 训练步数 | 100k | 100k / 200k |
| 评估 track | track_1~4, 6 | track_1~4, 6 |
| 每 task episode 数 | 50 | 50 |
| replan_steps | 4 | 4 |
| 图像输入 | front + wrist | front + wrist |

---

## 九、注意事项

1. **动作坐标系**：VLABench 的 `OpenPiPolicy`（`openpi.py` line 67-68）已做坐标偏移 `pos -= [0, -0.4, 0.78]`，π0 推理服务器输出的动作会自动经过这个转换，无需额外处理

2. **LeRobot 版本**：确认使用 `lerobot >= 0.5.0`，旧版本不支持 π0

3. **显存需求**：π0 base（PaliGemma-3B + action expert）全量 fine-tune 约需 40GB，单卡 A100 80GB 可跑；LoRA 约需 20GB

4. **数据量建议**：Physical Intelligence 官方建议 1~20 小时数据（约 3600~72000 步），VLABench 每任务 500 条轨迹应足够

---

## 十、参考资料

- [LeRobot π0 文档](https://huggingface.co/docs/lerobot/en/pi0)
- [π0 论文](https://arxiv.org/html/2410.24164v1)
- [openpi 官方博客](https://www.physicalintelligence.company/blog/openpi)
- [lerobot/pi0_base 权重](https://huggingface.co/lerobot/pi0_base)
