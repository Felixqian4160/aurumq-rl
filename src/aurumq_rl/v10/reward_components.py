"""WaveHunter v10 Reward components.

This module is independent from the frozen v9 reward path.  It provides a
small, deterministic Bayesian volatility estimator using a discounted
Inverse-Gamma posterior over return variance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BayesianVolConfig:
    """Hyperparameters for the discounted variance posterior."""

    alpha0: float = 2.0
    beta0: float = 1e-4
    decay: float = 0.98
    uncertainty_weight: float = 0.50
    min_vol: float = 0.005


class BayesianVolEstimator:
    """Discounted Inverse-Gamma posterior estimator for return volatility.

    The posterior state is ``(alpha, beta)`` for variance.  Each observation
    discounts old evidence and adds the Normal/Inverse-Gamma sufficient
    statistic ``0.5 * return**2``.  ``variance_mean`` and ``volatility`` are
    kept separate so callers cannot accidentally use variance as volatility.
    """

    def __init__(self, config: BayesianVolConfig | None = None) -> None:
        self.config = config or BayesianVolConfig()
        if self.config.alpha0 <= 1.0:
            raise ValueError("alpha0 must be > 1 so posterior variance mean exists")
        if self.config.beta0 <= 0.0:
            raise ValueError("beta0 must be > 0")
        if not 0.0 < self.config.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if self.config.uncertainty_weight < 0.0:
            raise ValueError("uncertainty_weight must be >= 0")
        if self.config.min_vol <= 0.0:
            raise ValueError("min_vol must be > 0")
        self.reset()

    def reset(self) -> None:
        """Reset to the configured prior."""
        self.alpha = float(self.config.alpha0)
        self.beta = float(self.config.beta0)

    def update(self, return_value: float) -> float:
        """Update posterior with one return and return effective volatility."""
        r = float(return_value)
        if not math.isfinite(r):
            return self.volatility()
        self.alpha = self.config.decay * self.alpha + 0.5
        self.beta = self.config.decay * self.beta + 0.5 * r * r
        return self.volatility()

    def variance_mean(self) -> float:
        """Posterior mean of variance."""
        return float(self.beta / max(self.alpha - 1.0, np.finfo(float).eps))

    def variance_std(self) -> float:
        """Posterior standard deviation of variance."""
        if self.alpha <= 2.0:
            return self.variance_mean()
        variance = self.beta * self.beta / (
            (self.alpha - 1.0) ** 2 * (self.alpha - 2.0)
        )
        return float(math.sqrt(max(variance, 0.0)))

    def volatility(self) -> float:
        """Posterior volatility with uncertainty buffer and lower bound."""
        variance = max(self.variance_mean(), 0.0)
        uncertainty = max(self.variance_std(), 0.0)
        # Uncertainty is in variance units; convert the buffered variance to σ.
        buffered_variance = max(variance + self.config.uncertainty_weight * uncertainty, 0.0)
        return float(max(math.sqrt(buffered_variance), self.config.min_vol))

    def state(self) -> dict[str, float]:
        """Return auditable posterior state for logging/metadata."""
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "variance_mean": self.variance_mean(),
            "variance_std": self.variance_std(),
            "volatility": self.volatility(),
        }
