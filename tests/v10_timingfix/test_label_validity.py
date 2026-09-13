import numpy as np
import pytest

from scripts.v10_timingfix.label_validity import mann_whitney_validity


def test_returns_clearly_different_distributions():
    rng = np.random.default_rng(0)
    pos = rng.normal(0.02, 0.01, size=400)
    neg = rng.normal(-0.005, 0.01, size=400)
    returns = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(400), np.zeros(400)])
    result = mann_whitney_validity(returns, labels)
    assert result["passed"]
    assert result["p_value"] < 1e-3
    assert result["mean_future_return_main_wave"] > result["mean_future_return_not_main_wave"]


def test_returns_indistinguishable_distributions_fails():
    rng = np.random.default_rng(1)
    pos = rng.normal(0.0, 0.01, size=300)
    neg = rng.normal(0.0, 0.01, size=300)
    returns = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(300), np.zeros(300)])
    result = mann_whitney_validity(returns, labels)
    assert not result["passed"]


def test_insufficient_samples_returns_failed_with_reason():
    result = mann_whitney_validity([0.01] * 4, [1] * 4)
    assert not result["passed"]
    assert result["reason"] == "insufficient samples"
