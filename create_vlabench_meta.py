"""
生成 VLABench 数据集的训练元数据 JSON。

用法：
    python create_vlabench_meta.py \
        --data_dir /root/dataset/vlabench-data/1.0.0 \
        --output ./datasets/metas/vlabench_train.json \
        --obs_cameras front wrist
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def create_vlabench_meta(
    data_dir: str,
    output_path: str,
    obs_cameras: list[str] | None = None,
    split: str = "train",
) -> dict:
    if obs_cameras is None:
        obs_cameras = ["front", "wrist"]

    pattern = os.path.join(data_dir, f"primitive-{split}.tfrecord-*")
    shard_paths = sorted(glob.glob(pattern))

    if not shard_paths:
        raise FileNotFoundError(f"未找到 TFRecord 文件：{pattern}")

    meta = {
        "dataset_name": "vlabench",
        "data_dir": data_dir,
        "datalist": shard_paths,
        "obs_cameras": obs_cameras,
        "num_shards": len(shard_paths),
        "action_dim": 7,
        "proprio_dim": 7,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"已保存元数据到 {output_path}，共 {len(shard_paths)} 个 shard")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/root/dataset/vlabench-data/1.0.0")
    parser.add_argument("--output", type=str, default="./datasets/metas/vlabench_train.json")
    parser.add_argument("--obs_cameras", nargs="+", default=["front", "wrist"])
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    create_vlabench_meta(
        data_dir=args.data_dir,
        output_path=args.output,
        obs_cameras=args.obs_cameras,
        split=args.split,
    )
