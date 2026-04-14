#!/usr/bin/env python3
"""
SimVLA VLABench 评估脚本

在 vlabench 环境下运行：
    conda activate vlabench
    cd /data/kcl/zz/hyj/code/VLABench
    python /data/kcl/zz/hyj/code/SimVLA/evaluation/vlabench/evaluate_simvla.py \
        --eval-track track_5_long_horizon \
        --n-episode 10 \
        --port 8001 \
        --save-dir /datasets/simvla_output/eval_results

支持的 eval-track：
    track_1_in_distribution
    track_2_cross_category
    track_3_common_sense
    track_4_semantic_instruction
    track_5_long_horizon
    track_6_unseen_texture
"""

import argparse
import json
import os
import sys

os.environ.setdefault("VLABENCH_ROOT", os.path.join(os.path.dirname(__file__), "../../../../VLABench/VLABench"))
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.join(os.environ["VLABENCH_ROOT"], ".."))
from VLABench.tasks import *
from VLABench.robots import *
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openpi import OpenPiPolicy

VALID_TRACKS = [
    "track_1_in_distribution",
    "track_2_cross_category",
    "track_3_common_sense",
    "track_4_semantic_instruction",
    "track_5_long_horizon",
    "track_6_unseen_texture",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-track", type=str, required=True, choices=VALID_TRACKS)
    parser.add_argument("--n-episode", type=int, default=10)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--replan-steps", type=int, default=4,
                        help="每隔多少步重新推理一次动作 chunk")
    parser.add_argument("--save-dir", type=str, default="logs/simvla")
    parser.add_argument("--metrics", nargs="+",
                        default=["success_rate"],
                        choices=["success_rate", "intention_score", "progress_score"])
    parser.add_argument("--visualization", action="store_true")
    args = parser.parse_args()

    # 加载 episode config
    track_path = os.path.join(
        os.environ["VLABENCH_ROOT"],
        "configs/evaluation/tracks",
        f"{args.eval_track}.json"
    )
    if not os.path.exists(track_path):
        raise FileNotFoundError(
            f"找不到 {track_path}\n"
            f"如果是 track_5_long_horizon，请先运行：\n"
            f"  python /data/kcl/zz/hyj/code/VLABench/generate_track5_long_horizon.py"
        )

    with open(track_path) as f:
        episode_config = json.load(f)
    tasks = list(episode_config.keys())

    print(f"评估 track: {args.eval_track}")
    print(f"任务列表: {tasks}")
    print(f"每任务 episode 数: {args.n_episode}")

    save_dir = os.path.join(args.save_dir, args.eval_track)

    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_config,
        max_substeps=1,
        save_dir=save_dir,
        visulization=args.visualization,
        metrics=args.metrics,
    )

    # 复用 OpenPiPolicy 客户端（协议完全兼容）
    policy = OpenPiPolicy(
        host=args.host,
        port=args.port,
        replan_steps=args.replan_steps,
    )
    policy.name = "simvla"

    result = evaluator.evaluate(policy)

    out_dir = os.path.join(save_dir, "simvla")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evaluation_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n评估完成，结果保存到: {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
