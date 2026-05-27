#!/usr/bin/env python3
"""
SimVLA LIBERO Evaluation Client

Observation format:
1. State: [eef_pos(3), axis_angle(3), gripper_qpos(2)] = 8D
2. Action: delta action (7D)
3. Default delta control mode
4. Images rotated 180 degrees
"""
from __future__ import annotations

import os
import sys

# Must be set before any mujoco/OpenGL import.
# Use osmesa for headless CPU rendering; fall back to egl if explicitly set.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import argparse
import collections
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import imageio
import json_numpy
import numpy as np
import requests
from tqdm import tqdm

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client
    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 800,
    "libero_object": 800,
    "libero_goal": 800,
    "libero_10": 900,
    "libero_90": 900,
}

NUM_STEPS_WAIT = 10

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _make_run_dir(base_dir: str, task_suite: str) -> Path:
    """Create timestamped output directory: {base_dir}/{task_suite}_{YYYYMMDD_HH}"""
    ts = datetime.now().strftime("%Y%m%d_%H")
    run_dir = Path(base_dir) / f"{task_suite}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    return run_dir


# -----------------------------------------------------------------------------
# Client Policy Classes
# -----------------------------------------------------------------------------
class WebSocketClient:
    def __init__(self, host: str, port: int, replan_steps: int = 5, resize_size: int = 224):
        if not HAS_WS_CLIENT:
            raise ImportError("openpi_client not installed. Run: pip install openpi-client")
        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], self.resize_size, self.resize_size)
            )
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
            }
            result = self.client.infer(element)
            action_chunk = result["actions"]
            if not isinstance(action_chunk, np.ndarray):
                action_chunk = np.array(action_chunk)
            assert len(action_chunk) >= self.replan_steps
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])
        return self.action_plan.popleft()


class HTTPClient:
    def __init__(self, host: str, port: int, replan_steps: int = 5):
        self.url = f"http://{host}:{port}/act"
        self.replan_steps = replan_steps
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()

    def infer(self, element: Dict) -> Dict:
        try:
            payload = {
                "image0": json_numpy.dumps(element["observation/image"]),
                "image1": json_numpy.dumps(element["observation/wrist_image"]),
                "proprio": json_numpy.dumps(element["observation/state"]),
                "language_instruction": element["prompt"],
                "steps": 10,
            }
            resp = requests.post(self.url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return {"actions": np.array(data["action"])}
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            element = {
                "observation/image": obs["image"],
                "observation/wrist_image": obs["wrist_image"],
                "observation/state": obs["state"],
                "prompt": goal,
            }
            result = self.infer(element)
            for action in result["actions"][:self.replan_steps]:
                self.action_plan.append(action)
        return self.action_plan.popleft()


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(
    client,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    run_dir: Path = None,
    save_video: bool = True,
    algo: str = "simvla",
) -> Dict:
    """
    Run LIBERO evaluation across all tasks in a suite.

    Returns a results dict matching the standard schema plus intermediate stats.
    """
    np.random.seed(seed)
    t_start = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)

    video_dir = run_dir / "videos" if run_dir else None
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task suite : {task_suite_name}")
    print(f"Tasks      : {num_tasks},  Trials per task: {num_trials}")
    print(f"Max steps  : {max_steps}")
    print(f"Output dir : {run_dir}")

    total_episodes, total_successes = 0, 0

    # Per-task summary (filled below)
    tasks_summary: List[Dict] = []

    # All episode records — saved separately as episodes_detail.json
    all_episodes: List[Dict] = []

    # For action_stats.npz: per-task lists of per-step action vectors (successful eps only)
    # Shape after collection: task → list of np.ndarray [steps, 7]
    task_action_seqs: Dict[int, List[np.ndarray]] = {}

    for task_id in tqdm(range(num_tasks - 1, -1, -1), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        task_successes = 0
        # Steps only for successful episodes (for best/avg/worst)
        success_steps: List[int] = []
        task_action_seqs[task_id] = []

        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])

            replay_images: List[np.ndarray] = []
            ep_actions: List[np.ndarray] = []  # all actions taken this episode
            gripper_prev = None
            gripper_changes = 0

            t = 0
            done = False
            failure_type = "timeout"

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    if save_video:
                        replay_images.append(img)

                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])

                    obs_dict = {"image": img, "wrist_image": wrist_img, "state": state}
                    action = client.step(obs_dict, task_description)
                    ep_actions.append(action.copy())

                    # Track gripper state changes (dim 6, threshold 0)
                    gripper_cmd = float(action[6])
                    if gripper_prev is not None and (gripper_cmd > 0) != (gripper_prev > 0):
                        gripper_changes += 1
                    gripper_prev = gripper_cmd

                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        failure_type = None
                        break

                    t += 1

                except Exception as e:
                    print(f"Error in rollout: {e}")
                    failure_type = "error"
                    break

            # Effective steps (excluding warm-up)
            effective_steps = max(0, t - NUM_STEPS_WAIT)
            total_episodes += 1

            if done:
                success_steps.append(effective_steps)
                if ep_actions:
                    task_action_seqs[task_id].append(np.stack(ep_actions))  # [T, 7]

            # ---------- episode record (for episodes_detail.json) ----------
            ep_actions_arr = np.stack(ep_actions) if ep_actions else np.zeros((0, 7))
            action_mag = float(np.mean(np.linalg.norm(ep_actions_arr, axis=1))) if len(ep_actions_arr) else 0.0
            action_std = float(np.std(np.linalg.norm(ep_actions_arr, axis=1))) if len(ep_actions_arr) else 0.0

            all_episodes.append({
                "task_id": task_id,
                "task_name": task_description,
                "trial_id": ep,
                "success": bool(done),
                "steps": effective_steps,
                "failure_type": failure_type,          # None / "timeout" / "error"
                "action_mean_mag": round(action_mag, 4),
                "action_std_mag": round(action_std, 4),
                "gripper_changes": gripper_changes,
            })
            # ----------------------------------------------------------------

            # Save video
            if replay_images and save_video and video_dir:
                suffix = "success" if done else "failure"
                tag = task_description.replace(" ", "_")[:40]
                video_path = video_dir / f"task{task_id:02d}_{tag}_ep{ep:03d}_{suffix}.mp4"
                imageio.mimwrite(str(video_path), replay_images, fps=10)

            status = "[OK]" if done else "[FAIL]"
            print(f"  {status} Task {task_id} Ep {ep}: steps={effective_steps}")

        env.close()
        sr = task_successes / num_trials
        print(f"  Task {task_id}: {task_successes}/{num_trials} ({sr*100:.1f}%)")

        # Per-task step statistics (successful episodes only)
        best_steps  = int(min(success_steps)) if success_steps else None
        avg_steps   = round(float(np.mean(success_steps)), 1) if success_steps else None
        worst_steps = int(max(success_steps)) if success_steps else None

        tasks_summary.append({
            "task_id": task_id,
            "task_name": task_description,
            "successes": task_successes,
            "trials": num_trials,
            "success_rate": round(sr, 4),
            "best_steps": best_steps,
            "avg_steps": avg_steps,
            "worst_steps": worst_steps,
        })

    duration = round(time.time() - t_start, 1)
    total_sr = round(total_successes / max(total_episodes, 1), 4)

    # Sort tasks by task_id descending (matches collection order)
    tasks_summary.sort(key=lambda x: x["task_id"], reverse=True)

    # -------------------------------------------------------------------------
    # Main results.json
    # -------------------------------------------------------------------------
    results = {
        "algo": algo,
        "task_suite": task_suite_name,
        "num_trials": num_trials,
        "seed": seed,
        "timestamp": timestamp_str,
        "duration_seconds": duration,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "total_success_rate": total_sr,
        "tasks": tasks_summary,
    }

    if run_dir:
        results_path = run_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved → {results_path}")

        # ---------------------------------------------------------------------
        # episodes_detail.json  (per-episode intermediate variables)
        # Fields: task_id, task_name, trial_id, success, steps, failure_type,
        #         action_mean_mag, action_std_mag, gripper_changes
        # ---------------------------------------------------------------------
        detail_path = run_dir / "episodes_detail.json"
        with open(detail_path, "w") as f:
            json.dump(all_episodes, f, indent=2, ensure_ascii=False)
        print(f"Episode detail saved → {detail_path}")

        # ---------------------------------------------------------------------
        # action_stats.npz  (per-task action distribution, successful eps only)
        # Keys: task{id}_actions  →  np.ndarray [N_steps_total, 7]
        #       task{id}_per_dim_mean, task{id}_per_dim_std
        # ---------------------------------------------------------------------
        npz_data = {}
        for tid, seqs in task_action_seqs.items():
            if not seqs:
                continue
            cat = np.concatenate(seqs, axis=0)  # [total_steps, 7]
            npz_data[f"task{tid:02d}_actions"]       = cat
            npz_data[f"task{tid:02d}_per_dim_mean"]  = cat.mean(axis=0)
            npz_data[f"task{tid:02d}_per_dim_std"]   = cat.std(axis=0)

        if npz_data:
            stats_path = run_dir / "action_stats.npz"
            np.savez_compressed(str(stats_path), **npz_data)
            print(f"Action stats saved → {stats_path}")

    print(f"\nTotal: {total_successes}/{total_episodes} ({total_sr*100:.1f}%)  [{duration}s]")
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--connection_info", type=str, default=None)
    parser.add_argument("--client_type", type=str, default="websocket",
                        choices=["websocket", "http"])
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Parent directory; a timestamped subdir is created inside")
    parser.add_argument("--algo", type=str, default="simvla",
                        help="Algorithm name recorded in results.json")
    parser.add_argument("--no_video", action="store_true",
                        help="Disable video recording for faster evaluation")

    args = parser.parse_args()

    if args.connection_info:
        print(f"Loading connection info from: {args.connection_info}")
        while not Path(args.connection_info).exists():
            sys.stdout.write("\rWaiting for server...")
            sys.stdout.flush()
            time.sleep(0.5)
        print()
        with open(args.connection_info) as f:
            info = json.load(f)
            args.host = info["host"]
            args.port = info["port"]

    protocol = "ws" if args.client_type == "websocket" else "http"
    print(f"Starting LIBERO evaluation client")
    print(f"   Client type : {args.client_type}")
    print(f"   Server      : {protocol}://{args.host}:{args.port}")
    print(f"   Task suite  : {args.task_suite}")
    print(f"   Replan steps: {args.replan_steps}")
    print()

    # Create timestamped run directory
    run_dir = _make_run_dir(args.output_dir, args.task_suite)

    if args.client_type == "websocket":
        client = WebSocketClient(args.host, args.port, replan_steps=args.replan_steps)
    else:
        client = HTTPClient(args.host, args.port, replan_steps=args.replan_steps)

    eval_libero(
        client=client,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        run_dir=run_dir,
        save_video=not args.no_video,
        algo=args.algo,
    )


if __name__ == "__main__":
    main()
