from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class SurvivalBreaks:
    """简单封装离散时间区间断点。"""
    breaks: np.ndarray  # shape = (n_intervals+1,)

    @property
    def n_intervals(self) -> int:
        return len(self.breaks) - 1

    @staticmethod
    def from_uniform(max_time: float, n_intervals: int) -> "SurvivalBreaks":
        edges = np.linspace(0.0, max_time, num=n_intervals + 1, dtype=float)
        return SurvivalBreaks(edges)

def make_surv_targets(times: Iterable[float], events: Iterable[int], breaks: SurvivalBreaks) -> np.ndarray:
    """包装 nnet-survival 的 make_surv_array，生成训练所需标签矩阵。"""
    t = np.asarray(list(times), dtype=float)
    f = np.asarray(list(events), dtype=int)
    return _make_surv_array(t, f, breaks.breaks)


def _make_surv_array(t: np.ndarray, f: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """本地实现 nnet-survival 的 make_surv_array，避免外部依赖。"""
    n_samples = t.shape[0]
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_samples, n_intervals * 2), dtype=float)
    for i in range(n_samples):
        if f[i]:  # 发生事件
            y_train[i, 0:n_intervals] = (t[i] >= breaks[1:]).astype(float)
            if t[i] < breaks[-1]:
                idx = np.where(t[i] < breaks[1:])[0][0]
                y_train[i, n_intervals + idx] = 1.0
        else:  # 截尾
            y_train[i, 0:n_intervals] = (t[i] >= breaks_midpoint).astype(float)
    return y_train

def demo_fake_targets(batch: int, breaks: SurvivalBreaks) -> np.ndarray:
    """生成演示用随机标签（用于脚本 dry-run）。"""
    rng = np.random.default_rng(seed=42)
    times = rng.uniform(low=0.0, high=breaks.breaks[-1], size=batch)
    events = rng.integers(0, 2, size=batch)
    return make_surv_targets(times, events, breaks)

__all__ = ["SurvivalBreaks", "make_surv_targets", "demo_fake_targets"]
