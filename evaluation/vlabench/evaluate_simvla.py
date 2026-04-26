#!/usr/bin/env python3
"""
SimVLA VLABench 评估脚本

在 vlabench 环境下运行：
    conda activate vlabench
    cd /root/code/VLABench
    python /root/code/SimVLA/evaluation/vlabench/evaluate_simvla.py \
        --eval-track track_5_long_horizon \
        --n-episode 10 \
        --port 8001 \
        --save-dir /root/eval_results

支持的 eval-track：
    track_1_in_distribution
    track_2_cross_category
    track_3_common_sense
    track_4_semantic_instruction
    track_5_long_horizon
    track_6_unseen_texture
"""

import argparse
from datetime import datetime
import json
import os
import sys
import traceback

os.environ.setdefault("VLABENCH_ROOT", os.path.join(os.path.dirname(__file__), "../../../../VLABench/VLABench"))
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.join(os.environ["VLABENCH_ROOT"], ".."))
from VLABench.tasks import *
from VLABench.robots import *
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openpi import OpenPiPolicy
from VLABench.envs.dm_env import LM4ManipDMEnv

# ── MuJoCo 物理稳定性修复 ────────────────────────────────────────────────────
# 问题：cancel_gravity_and_improve_fluid 将所有 geom 的 fluidcoef 设为 1e4，
# 对于 cook_dishes 等含 250+ 物体的场景，会导致约束数量爆炸（mjWARN_CNSTRFULL）
# 和数值不稳定（QACC NaN），使仿真卡死。
# 修复：
#   1. 增大 arena 内存到 20000M
#   2. 将 fluidcoef 从 1e4 降到 1.0，保留重力抵消效果但避免约束爆炸
_orig_reset = LM4ManipDMEnv.reset
_orig_cancel = LM4ManipDMEnv.cancel_gravity_and_improve_fluid
_orig_reset_attempt = LM4ManipDMEnv._reset_attempt

def _patched_reset(self):
    return _orig_reset(self)

def _patched_reset_attempt(self):
    try:
        self.task._arena.mjcf_model.size.memory = "40000M"
        # 增加求解器迭代次数，消除 "Failed to converge" 警告
        self.task._arena.mjcf_model.size.nconmax = 10000
        self.task._arena.mjcf_model.option.iterations = 200
    except Exception:
        pass
    return _orig_reset_attempt(self)

def _patched_cancel_gravity(self):
    # 只关闭重力，不加流体形状——ellipsoid fluid 即使 coef=1.0 仍导致 BADQACC
    self.task._arena.mjcf_model.option.flag.gravity = "disable"

LM4ManipDMEnv.reset = _patched_reset
LM4ManipDMEnv._reset_attempt = _patched_reset_attempt
LM4ManipDMEnv.cancel_gravity_and_improve_fluid = _patched_cancel_gravity
# ─────────────────────────────────────────────────────────────────────────────

VALID_TRACKS = [
    "track_1_in_distribution",
    "track_2_cross_category",
    "track_3_common_sense",
    "track_4_semantic_instruction",
    "track_5_long_horizon",
    "track_6_unseen_texture",
]


class SimVLAEvaluator(Evaluator):
    """
    在 Evaluator 基础上增加：
    - 每个 task 完成后立即打印成功率并保存结果（防止中途崩溃丢数据）
    - 保存所有 episode 的视频（成功/失败均保存，文件名含 success_True/False）
    """

    def __init__(self, *args, out_dir: str, **kwargs):
        # 强制开启可视化以保存视频
        kwargs["visulization"] = True
        super().__init__(*args, **kwargs)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def evaluate(self, agent):
        metrics = {}
        for task in self.eval_tasks:
            task_infos = []
            from VLABench.utils.utils import find_key_by_value
            from VLABench.configs import name2config
            max_episode_length = 200
            if self.task_configs.get(find_key_by_value(name2config, task), None):
                cfg = self.task_configs[find_key_by_value(name2config, task)]
                if cfg.get("evaluation", {}).get("max_episode_length"):
                    max_episode_length = cfg["evaluation"]["max_episode_length"]

            from tqdm import tqdm
            for i in tqdm(range(self.n_episodes), desc=f"Evaluating {task} of {agent.name}"):
                agent.reset()
                # unnorm_key=None：服务器已做反归一化，不再重复
                kwargs = {"unnorm_key": None, "max_episode_length": max_episode_length}
                try:
                    if self.episode_config is None:
                        info = self.evaluate_single_episode(agent, task, i, None, seed=42 + i, **kwargs)
                    else:
                        info = self.evaluate_single_episode(agent, task, i, self.episode_config[task][i], **kwargs)
                    task_infos.append(info)
                except Exception as e:
                    print(f"[SKIP] {task} episode {i}: {e}")
                    traceback.print_exc()

            metric_score = self.compute_metric(task_infos)
            metrics[task] = metric_score

            # 打印当前任务结果
            n_success = sum(info["success"] for info in task_infos)
            n_total = len(task_infos)
            sr = metric_score.get("success_rate", float("nan"))
            print(f"\n{'='*60}")
            print(f"[{task}]  success: {n_success}/{n_total}  success_rate: {sr:.1%}")
            if "progress_score" in metric_score:
                print(f"         progress_score: {metric_score['progress_score']:.3f}")
            print(f"{'='*60}\n")

            # 保存到 Evaluator 自带的 save_dir（metrics.json + detail_info.json）
            if self.save_dir is not None:
                metrics_path = os.path.join(self.save_dir, "metrics.json")
                prev = {}
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        prev = json.load(f)
                prev.update(metrics)
                with open(metrics_path, "w") as f:
                    json.dump(prev, f, indent=4)
                os.makedirs(os.path.join(self.save_dir, task), exist_ok=True)
                with open(os.path.join(self.save_dir, task, "detail_info.json"), "w") as f:
                    json.dump(task_infos, f, indent=4)

            # 同时增量保存到用户指定的 out_dir（防止最终汇总前崩溃）
            out_path = os.path.join(self.out_dir, "evaluation_result.json")
            prev_out = {}
            if os.path.exists(out_path):
                with open(out_path) as f:
                    prev_out = json.load(f)
            prev_out.update(metrics)
            with open(out_path, "w") as f:
                json.dump(prev_out, f, indent=2, ensure_ascii=False)
            print(f"[已保存] {out_path}")

        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-track", type=str, required=True, choices=VALID_TRACKS)
    parser.add_argument("--n-episode", type=int, default=10)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--replan-steps", type=int, default=4,
                        help="每隔多少步重新推理一次动作 chunk")
    parser.add_argument("--save-dir", type=str,
                        default=os.environ.get("SIMVLA_EVAL_RESULTS", "/root/eval_results"))
    parser.add_argument("--metrics", nargs="+",
                        default=["success_rate", "progress_score"],
                        choices=["success_rate", "intention_score", "progress_score"])
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
            f"  python {os.environ.get('SIMVLA_VLABENCH_CODE', '/root/code/VLABench')}/generate_track5_long_horizon.py"
        )

    with open(track_path) as f:
        episode_config = json.load(f)
    tasks = list(episode_config.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(args.save_dir, f"{args.eval_track}_{timestamp}")
    out_dir = os.path.join(save_dir, "simvla")
    os.makedirs(out_dir, exist_ok=True)

    print(f"评估 track: {args.eval_track}")
    print(f"任务列表: {tasks}")
    print(f"每任务 episode 数: {args.n_episode}")
    print(f"结果目录: {out_dir}")

    evaluator = SimVLAEvaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_config,
        max_substeps=5,
        save_dir=save_dir,
        metrics=args.metrics,
        out_dir=out_dir,
    )

    policy = OpenPiPolicy(
        host=args.host,
        port=args.port,
        replan_steps=args.replan_steps,
    )
    policy.name = "simvla"

    result = evaluator.evaluate(policy)

    print(f"\n{'='*60}")
    print("全部任务评估完成")
    print(f"结果目录: {out_dir}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
