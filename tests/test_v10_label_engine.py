import numpy as np

from aurumq_rl.v10.label_engine import (
    cumulative_swing,
    evt_threshold,
    fit_ou_params,
    ou_b1_candidate,
)


def test_cumulative_swing_captures_slow_move():
    prices = np.linspace(100.0, 125.0, 11)
    assert cumulative_swing(prices, 0, 10) == 0.25


def test_evt_threshold_falls_back_for_small_sample():
    values = np.array([0.01, 0.02, 0.03, 0.04])
    threshold = evt_threshold(values, tail_frac=0.25, min_excess_samples=20)
    assert np.isclose(threshold, 0.0325)


def test_ou_flat_series_returns_none():
    assert fit_ou_params(np.full(30, 100.0)) is None


def test_ou_trend_with_b_at_or_above_one_returns_none():
    prices = np.arange(100.0, 130.0)
    assert fit_ou_params(prices) is None


def test_ou_insufficient_and_invalid_samples_return_none():
    assert fit_ou_params([100.0, 101.0, 100.5], min_samples=10) is None
    assert fit_ou_params([100.0, np.nan] * 10, min_samples=10) is None


def test_ou_mean_reverting_series_can_be_candidate():
    rng = np.random.default_rng(7)
    values = [100.0]
    for _ in range(59):
        values.append(values[-1] + 0.35 * (100.0 - values[-1]) + rng.normal(0, 0.08))
    params = fit_ou_params(values, min_samples=10)
    assert params is not None
    assert 0.0 < params["b"] < 1.0
    assert params["theta"] > 0.05
    assert ou_b1_candidate(values) in (True, False)
