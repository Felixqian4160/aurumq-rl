"""WaveHunter v10 四头融合进 PPO action。

旧 ``WaveHunterV10Policy`` 把 4 个 head 各自输出后，仅参与辅助损失，并不进入 PPO
action。本模块提供独立可测试的融合组件，将四个 head 的概率拼接成
``(B, ns, 4)``，再做几何/加权融合，返回 ``(B, ns)`` 的融合评分供 action_net 拼接或
直接构造 mean。

v10 的四头含义：
    A1 谷底附近
    A2 主升
    Peak 顶
    B1 筑底末期

此处不评估业务语义是否成立，只保证：
    1. 输出形状与输入形状匹配
    2. 不泄露任何未来信息（全部依赖上游 head logits）
    3. 与冻结的 v9 路径独立
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FourHeadFusionConfig:
    weight_a1: float = 0.40
    weight_a2: float = 0.40
    weight_b1: float = 0.20
    # Peak 在融合评分中起负向作用：顶部预测 → 减仓信号。
    weight_peak: float = 0.30
    use_softmax: bool = True


class FourHeadActionFusion:
    """Combine four head logits into a single per-stock fusion score."""

    def __init__(self, config: FourHeadFusionConfig | None = None) -> None:
        self.config = config or FourHeadFusionConfig()
        self._validate()

    def _validate(self) -> None:
        weights = (
            self.config.weight_a1,
            self.config.weight_a2,
            self.config.weight_b1,
            self.config.weight_peak,
        )
        if any(w < 0 for w in weights):
            raise ValueError("weights must be >= 0")
        if sum(weights) <= 0:
            raise ValueError("at least one weight must be > 0")

    def fuse(self, a1: np.ndarray, a2: np.ndarray, peak: np.ndarray, b1: np.ndarray) -> np.ndarray:
        for name, arr in (("a1", a1), ("a2", a2), ("peak", peak), ("b1", b1)):
            if arr.ndim != 2:
                raise ValueError(f"{name} must be 2D (B, ns); got shape {arr.shape}")
        if self.config.use_softmax:
            a1_p = _softmax(a1)
            a2_p = _softmax(a2)
            peak_p = _softmax(peak)
            b1_p = _softmax(b1)
        else:
            a1_p, a2_p, peak_p, b1_p = a1, a2, peak, b1
        score = (
            self.config.weight_a1 * a1_p
            + self.config.weight_a2 * a2_p
            + self.config.weight_b1 * b1_p
            - self.config.weight_peak * peak_p
        )
        return score

    def state(self) -> dict[str, float | int]:
        return {
            "weight_a1": float(self.config.weight_a1),
            "weight_a2": float(self.config.weight_a2),
            "weight_b1": float(self.config.weight_b1),
            "weight_peak": float(self.config.weight_peak),
            "use_softmax": int(self.config.use_softmax),
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)
