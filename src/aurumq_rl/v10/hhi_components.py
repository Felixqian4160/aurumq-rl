"""WaveHunter v10 portfolio concentration components."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HHIConfig:
    """Mild concentration penalty configuration."""

    weight: float = 0.5
    target_count: int = 10
    max_penalty: float = 0.5


class HHIConcentrationPenalty:
    """Herfindahl-Hirschman concentration penalty for portfolio weights.

    HHI is ``sum(weights**2)``.  It is normalized against the HHI of an
    equally-weighted target portfolio (``1 / target_count``), so the scale is
    interpretable across different portfolio sizes.  The penalty is bounded
    and never changes the weights themselves.
    """

    def __init__(self, config: HHIConfig | None = None) -> None:
        self.config = config or HHIConfig()
        if self.config.weight < 0:
            raise ValueError("weight must be >= 0")
        if self.config.target_count < 1:
            raise ValueError("target_count must be >= 1")
        if self.config.max_penalty <= 0:
            raise ValueError("max_penalty must be > 0")

    def _normalized_weights(self, weights: np.ndarray | list[float]) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w = np.nan_to_num(np.clip(w, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(w))
        return w / total if total > 1e-12 else np.zeros_like(w)

    def compute(self, weights: np.ndarray | list[float]) -> float:
        w = self._normalized_weights(weights)
        hhi = float(np.sum(w * w))
        target_hhi = 1.0 / float(self.config.target_count)
        # 以目标持仓数为参考：达到或超过目标分散度不罚，少于目标持仓数才惩罚。
        excess = max(0.0, hhi - target_hhi)
        relative = float(np.clip(excess / max(1.0 - target_hhi, 1e-12), 0.0, 1.0))
        return float(-min(self.config.max_penalty, self.config.weight * relative))

    def state(self, weights: np.ndarray | list[float]) -> dict[str, float | int]:
        w = self._normalized_weights(weights)
        hhi = float(np.sum(w * w))
        return {
            "hhi": hhi,
            "effective_count": float(1.0 / max(hhi, 1e-12)) if hhi > 0 else 0.0,
            "active_count": int(np.sum(w > 1e-8)),
            "total_weight": float(np.sum(w)),
            "target_count": int(self.config.target_count),
        }
