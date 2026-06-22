from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningMeanStd:
    """在线维护 observation 的均值和方差，用于输入归一化。"""

    shape: tuple[int, ...]
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(self.epsilon)

    def update(self, batch: np.ndarray) -> None:
        # batch shape: [batch_size, obs_dim]。单环境训练时 batch_size 通常是 1。
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        # 并行算法形式的 running mean/std 更新，避免保存所有历史 observation。
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        old_sum_squares = self.var * self.count
        batch_sum_squares = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (old_sum_squares + batch_sum_squares + correction) / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = float(total_count)

    def normalize(self, observation: np.ndarray, clip: float = 10.0) -> np.ndarray:
        # PPO 网络看到的是标准化后的 observation，clip 防止极端值冲击网络。
        normalized = (observation - self.mean) / np.sqrt(self.var + 1e-8)
        normalized = np.clip(normalized, -clip, clip)
        return normalized.astype(np.float32)

    def state_dict(self) -> dict[str, list[float] | float | tuple[int, ...]]:
        return {
            "shape": self.shape,
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shape = tuple(state["shape"])
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])
