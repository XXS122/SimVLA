#!/usr/bin/env python
"""
计算 VLABench 数据集归一化统计量

数据格式（RLDS TFRecord）：
- steps/observation/ee_state: [T*7] float - 7维本体感知
- steps/action: [T*7] float - 7维动作

输出格式（与 LIBERO 一致，可直接传给 --norm_stats_path）：
{
  "norm_stats": {
    "state":   {"mean": [7], "std": [7], "q01": [7], "q99": [7]},
    "actions": {"mean": [7], "std": [7], "q01": [7], "q99": [7]}
  }
}

用法：
    python compute_vlabench_norm_stats.py \\
        --data_dir /data/kcl/zz/hyj/vlabench/data/1.0.0 \\
        --output ./norm_stats/vlabench_norm.json \\
        --max_shards 50
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

# 复用 LIBERO 脚本中的 RunningStats
from compute_libero_norm_stats import RunningStats


def compute_norm_stats(
    data_dir: str,
    output_path: str,
    max_shards: int = 0,
    split: str = "train",
) -> None:
    import tensorflow as tf

    pattern = os.path.join(data_dir, f"primitive-{split}.tfrecord-*")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"未找到 TFRecord 文件：{pattern}")

    if max_shards > 0:
        shard_paths = shard_paths[:max_shards]

    print(f"VLABench 归一化统计计算")
    print(f"  数据目录: {data_dir}")
    print(f"  Shard 数量: {len(shard_paths)}")
    print(f"  State 维度: 7 [xyz(3) + axis_angle(3) + gripper(1)]")
    print(f"  Action 维度: 7 [xyz(3) + axis_angle(3) + gripper(1)]")

    state_stats = RunningStats(dim=7)
    action_stats = RunningStats(dim=7)
    total_episodes = 0
    total_steps = 0

    for shard_path in tqdm(shard_paths, desc="处理 shard"):
        try:
            dataset = tf.data.TFRecordDataset(shard_path)
            for raw in dataset:
                ep = tf.train.Example()
                ep.ParseFromString(raw.numpy())
                feat = ep.features.feature

                T = len(feat["steps/is_first"].int64_list.value)
                if T < 2:
                    continue

                ee_flat = np.array(feat["steps/observation/ee_state"].float_list.value, dtype=np.float32)
                act_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)

                ee_state = ee_flat.reshape(T, 7)
                actions = act_flat.reshape(T, 7)

                state_stats.update(ee_state)
                action_stats.update(actions)
                total_episodes += 1
                total_steps += T

        except Exception as e:
            print(f"跳过损坏 shard {shard_path}: {e}")
            continue

    print(f"\n统计完成：{total_episodes} 个 episode，{total_steps} 步")

    s = state_stats.get_statistics()
    a = action_stats.get_statistics()

    labels = ["x", "y", "z", "ax", "ay", "az", "gripper"]
    print(f"\nState (7维)：")
    print(f"{'dim':<10} {'mean':>10} {'std':>10} {'q01':>10} {'q99':>10}")
    print("-" * 50)
    for i, lbl in enumerate(labels):
        print(f"{lbl:<10} {s['mean'][i]:>10.4f} {s['std'][i]:>10.4f} {s['q01'][i]:>10.4f} {s['q99'][i]:>10.4f}")

    print(f"\nActions (7维)：")
    print(f"{'dim':<10} {'mean':>10} {'std':>10} {'q01':>10} {'q99':>10}")
    print("-" * 50)
    for i, lbl in enumerate(labels):
        print(f"{lbl:<10} {a['mean'][i]:>10.4f} {a['std'][i]:>10.4f} {a['q01'][i]:>10.4f} {a['q99'][i]:>10.4f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "norm_stats": {
            "state": {
                "mean": s["mean"].tolist(),
                "std": s["std"].tolist(),
                "q01": s["q01"].tolist(),
                "q99": s["q99"].tolist(),
            },
            "actions": {
                "mean": a["mean"].tolist(),
                "std": a["std"].tolist(),
                "q01": a["q01"].tolist(),
                "q99": a["q99"].tolist(),
            },
        },
        "metadata": {
            "data_dir": data_dir,
            "num_shards": len(shard_paths),
            "num_episodes": total_episodes,
            "num_steps": total_steps,
            "state_dim": 7,
            "action_dim": 7,
            "labels": labels,
        },
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n已保存到: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/kcl/zz/hyj/vlabench/data/1.0.0")
    parser.add_argument("--output", type=str, default="./norm_stats/vlabench_norm.json")
    parser.add_argument("--max_shards", type=int, default=0, help="限制处理的 shard 数量，0 表示全部")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    compute_norm_stats(
        data_dir=args.data_dir,
        output_path=args.output,
        max_shards=args.max_shards,
        split=args.split,
    )
