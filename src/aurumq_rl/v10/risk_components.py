"""WaveHunter v10 risk components.

Independent risk components for the v10 reward path.  The CVaR component
uses only drawdown observations already seen by the environment; it never
uses future observations or labels.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CVaRConfig:
    """Configuration for the online drawdown-tail penalty."""

    window: int = 252
    alpha: float = 0.95
    warmup: int = 30
    normal_scale: float = 0.5
    tail_scale: float = 1.0
    max_penalty: float = 2.0


class CVaRDrawdownPenalty:
    """Online CVaR penalty for portfolio drawdown.

    Drawdowns are non-negative losses.  During warmup the component returns a
    bounded linear penalty.  After warmup it estimates VaR and CVaR from the
    drawdowns observed so far.  Normal drawdowns are penalized gently; the
    tail beyond VaR receives a stronger but bounded penalty.
    """

    def __init__(self, config: CVaRConfig | None = None) -> None:
        self.config = config or CVaRConfig()
        if self.config.window < 2:
            raise ValueError("window must be >= 2")
        if not 0.5 <= self.config.alpha < 1.0:
            raise ValueError("alpha must be in [0.5, 1.0)")
        if self.config.warmup < 1:
            raise ValueError("warmup must be >= 1")
        if self.config.normal_scale < 0 or self.config.tail_scale < 0:
            raise ValueError("penalty scales must be >= 0")
        if self.config.max_penalty <= 0:
            raise ValueError("max_penalty must be > 0")
        self.history: deque[float] = deque(maxlen=self.config.window)
        self.reset()

    def reset(self) -> None:
        self.history.clear()

    def update(self, drawdown: float) -> float:
        """Record one drawdown and return a bounded penalty in [-max_penalty, 0]."""
        dd = float(drawdown)
        if not np.isfinite(dd):
            return 0.0
        dd = float(np.clip(dd, 0.0, 1.0))
        self.history.append(dd)
        if len(self.history) < self.config.warmup:
            return float(-self.config.normal_scale * min(dd / 0.20, 1.0))

        values = np.asarray(self.history, dtype=np.float64)
        var = float(np.quantile(values, self.config.alpha))
        tail = values[values >= var]
        cvar = float(np.mean(tail)) if tail.size else var
        if dd <= var:
            raw = -self.config.normal_scale * dd / max(var, 1e-8)
        else:
            tail_width = max(cvar - var, 1e-8)
            raw = -self.config.tail_scale * (1.0 + (dd - var) / tail_width)
        return float(np.clip(raw, -self.config.max_penalty, 0.0))

    def state(self) -> dict[str, float | int]:
        """Return auditable rolling risk state."""
        values = np.asarray(self.history, dtype=np.float64)
        if values.size == 0:
            return {"count": 0, "var": 0.0, "cvar": 0.0, "last_drawdown": 0.0}
        var = float(np.quantile(values, self.config.alpha))
        tail = values[values >= var]
        return {
            "count": int(values.size),
            "var": var,
            "cvar": float(np.mean(tail)) if tail.size else var,
            "last_drawdown": float(values[-1]),
        }
