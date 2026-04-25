"""
VLABench RLDS TFRecord Data Handler

数据格式（RLDS TFRecord）：
- steps/action: [T*7] float - 7维动作 (xyz(3) + axis_angle(3) + gripper(1))
- steps/observation/ee_state: [T*7] float - 7维本体感知
- steps/observation/front: [T] bytes - 前置摄像头 JPEG
- steps/observation/wrist: [T] bytes - 腕部摄像头 JPEG
- steps/observation/image_0: [T] bytes - 摄像头0 JPEG
- steps/observation/image_1: [T] bytes - 摄像头1 JPEG
- steps/language_instruction: [T] bytes - 语言指令（每步相同）
- episode_metadata/file_path: [1] bytes - 原始路径

输出格式（vlabench_joint 模式）：
- proprio: 7维 [xyz(3) + axis_angle(3) + gripper(1)]
- actions: 7维 [xyz(3) + axis_angle(3) + gripper(1)]
"""

from __future__ import annotations

import io
import random
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from .base import DomainHandler


class VLABenchRLDSHandler(DomainHandler):
    """
    VLABench RLDS TFRecord 数据处理器。

    meta 格式：
    {
        "dataset_name": "vlabench",
        "data_dir": "/datasets/vlabench/data/1.0.0",
        "datalist": ["/path/to/shard0.tfrecord", ...],
        "obs_cameras": ["front", "wrist"]   # 可选，默认 front+wrist
    }
    """

    dataset_name = "vlabench"

    def __init__(self, meta: dict, num_views: int = 3) -> None:
        super().__init__(meta, num_views)
        self.shard_paths: List[str] = meta.get("datalist", [])
        self.obs_cameras: List[str] = meta.get("obs_cameras", ["front", "wrist"])

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "vlabench_joint",
        lang_aug_map: dict | None = None,
        history_seq_len: int = 1,
        **kwargs,
    ) -> Iterable[dict]:
        """遍历单个 shard 中的所有 episode。"""
        import tensorflow as tf

        shard_path = self.shard_paths[traj_idx]
        dataset = tf.data.TFRecordDataset(shard_path)

        for raw in dataset:
            try:
                yield from self._iter_episode_from_raw(
                    raw.numpy(),
                    num_actions=num_actions,
                    training=training,
                    image_aug=image_aug,
                    lang_aug_map=lang_aug_map,
                    history_seq_len=history_seq_len,
                )
            except Exception as e:
                print(f"[VLABenchRLDSHandler] 跳过损坏 episode in {shard_path}: {e}")
                continue

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _iter_episode_from_raw(
        self,
        raw_bytes: bytes,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        history_seq_len: int = 1,
    ) -> Iterable[dict]:
        import tensorflow as tf

        ep = tf.train.Example()
        ep.ParseFromString(raw_bytes)
        feat = ep.features.feature

        # 时间步数
        T = len(feat["steps/is_first"].int64_list.value)
        if T < 2:
            return

        # 动作 & 本体感知
        action_dim = 7
        actions_flat = np.array(feat["steps/action"].float_list.value, dtype=np.float32)
        ee_flat = np.array(feat["steps/observation/ee_state"].float_list.value, dtype=np.float32)
        actions = actions_flat.reshape(T, action_dim)   # [T, 7]
        proprio = ee_flat.reshape(T, action_dim)        # [T, 7]

        # 语言指令（取第0步）
        instruction = feat["steps/language_instruction"].bytes_list.value[0].decode("utf-8")
        if training and lang_aug_map and instruction in lang_aug_map:
            instruction = random.choice(lang_aug_map[instruction])

        # 图像 bytes 列表（每个摄像头 T 帧）
        cam_frames: List[List[bytes]] = []
        for cam in self.obs_cameras:
            key = f"steps/observation/{cam}"
            if key in feat:
                cam_frames.append(list(feat[key].bytes_list.value))
            else:
                cam_frames.append([None] * T)

        # image_mask
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        n_valid = min(len(self.obs_cameras), self.num_views)
        image_mask[:n_valid] = True

        # 采样起始帧
        max_start = T - num_actions
        if max_start <= 0:
            return
        indices = list(range(max_start))
        if training:
            random.shuffle(indices)

        for idx in indices:
            # 动作 chunk: [num_actions+1, 7]，索引0为当前状态，1..N为未来动作
            chunk = self._get_action_chunk(actions, idx, num_actions)

            # 处理图像
            imgs = []
            for v in range(n_valid):
                frame_bytes = cam_frames[v][idx]
                img = self._decode_image(frame_bytes)
                if image_aug is not None:
                    img = image_aug(img)
                imgs.append(img)

            # 补齐空视角
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            image_input = torch.stack(imgs, dim=0)  # [V, C, H, W]

            sample = {
                "language_instruction": instruction,
                "image_input": image_input,
                "image_mask": image_mask,
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),
                "abs_trajectory": torch.tensor(chunk, dtype=torch.float32),
            }

            # 历史本体感知序列 [K, 7]：以当前帧为终点往前 K 帧，不足时补零
            if history_seq_len > 1:
                K = history_seq_len
                hist = np.zeros((K, proprio.shape[1]), dtype=np.float32)
                for k in range(K):
                    src = idx - (K - 1 - k)
                    if src >= 0:
                        hist[k] = proprio[src]
                    # else: 保持零填充
                sample["proprio_sequence"] = torch.tensor(hist, dtype=torch.float32)

            # 弱监督物理谓词标签 [5]
            sample["physics_labels"] = self._compute_physics_labels(chunk, proprio[idx])

            # 弱监督 gripper 切换剩余步比例标签 [1]
            sample["switch_labels"] = self._compute_switch_label(actions, idx, num_actions)

            yield sample

    @staticmethod
    def _compute_physics_labels(chunk: np.ndarray, proprio_cur: np.ndarray) -> torch.Tensor:
        """
        从动作 chunk 和当前本体感知自动抽取 5 个物理谓词代理标签（弱监督）。

        谓词定义（均归一化到 [0, 1]）：
          0 gripper_active  : chunk[1:, 6].mean() > 0.05（gripper 正在关闭）
          1 high_rotation   : |chunk[1:, 3:6]|.mean() > 0.05（大幅旋转，潜在碰撞风险）
          2 z_height        : sigmoid(proprio[2] * 5)（末端 Z 坐标高度）
          3 moving_up       : chunk[1:, 2].mean() > 0.01（末端向上运动）
          4 stable_traj     : std(chunk[1:, :3]) < 0.02（轨迹平稳）
        """
        action_chunk = chunk[1:]  # [T, 7]，去掉索引0（当前状态）
        labels = np.array([
            float(action_chunk[:, 6].mean() > 0.05),
            float(np.abs(action_chunk[:, 3:6]).mean() > 0.05),
            float(1.0 / (1.0 + np.exp(-proprio_cur[2] * 5.0))),  # sigmoid(z*5)
            float(action_chunk[:, 2].mean() > 0.01),
            float(np.std(action_chunk[:, :3]) < 0.02),
        ], dtype=np.float32)
        return torch.tensor(labels, dtype=torch.float32)

    @staticmethod
    def _compute_switch_label(
        actions: np.ndarray, idx: int, num_actions: int
    ) -> torch.Tensor:
        """
        计算距离下次 gripper 状态切换的剩余步数比例（[0, 1]）。

        若 gripper 在未来 N 步内切换，比例 = 切换步 / num_actions；
        若不切换，标签为 1.0（很远）。
        """
        T = actions.shape[0]
        cur_gripper = actions[idx, 6]
        steps_until = num_actions  # 默认"不切换"
        for k in range(1, min(num_actions, T - idx)):
            if abs(actions[idx + k, 6] - cur_gripper) > 0.1:
                steps_until = k
                break
        ratio = float(steps_until) / float(num_actions)
        return torch.tensor([ratio], dtype=torch.float32)  # [1]，collate 后为 [B, 1]

    @staticmethod
    def _decode_image(frame_bytes: Optional[bytes]) -> Image.Image:
        if frame_bytes is None:
            return Image.new("RGB", (128, 128))
        return Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    @staticmethod
    def _get_action_chunk(actions: np.ndarray, start_idx: int, num_actions: int) -> np.ndarray:
        """返回 [num_actions+1, 7]，索引0为当前帧动作，1..N为未来动作。"""
        T, D = actions.shape
        chunk = np.zeros((num_actions + 1, D), dtype=np.float32)
        for i in range(num_actions + 1):
            t = min(start_idx + i, T - 1)
            chunk[i] = actions[t]
        return chunk
