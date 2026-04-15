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

            yield {
                "language_instruction": instruction,
                "image_input": image_input,
                "image_mask": image_mask,
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),
                "abs_trajectory": torch.tensor(chunk, dtype=torch.float32),
            }

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
