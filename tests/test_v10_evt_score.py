import numpy as np

from aurumq_rl.v10.label_engine import forward_return_quality, score_pivot_candidates


def test_score_uses_geometric_mean_and_stock_local_candidate_pool():
    prices = np.array([100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100, 101, 102], dtype=float)
    pivots = np.array([5, 10, 15, 20, 25])
    peak = np.zeros(prices.size, dtype=bool); peak[[5, 15, 25]] = True
    valley = np.zeros(prices.size, dtype=bool); valley[[10, 20]] = True
    result = score_pivot_candidates(prices, pivots, peak, valley, trend_window=4, target_density=0.5)
    assert result["combined_score"].size > 0
    assert np.all(result["combined_score"] >= 0)
    assert result["selected"].sum() >= 1
    assert set(np.flatnonzero(result["selected"])).issubset(set(pivots))


def test_forward_quality_has_expected_direction():
    prices = np.array([100, 101, 99, 97, 95, 94, 93, 92, 91, 90, 89, 88], dtype=float)
    selected = np.array([False, False, True] + [False] * 9)
    peak = forward_return_quality(prices, selected, horizons=(5,), direction="peak")
    valley = forward_return_quality(prices, selected, horizons=(5,), direction="valley")
    assert peak["5"]["mean_return"] < 0
    assert valley["5"]["mean_return"] < 0
    assert peak["5"]["expected_direction_hit_rate"] == 1.0
    assert valley["5"]["expected_direction_hit_rate"] == 0.0
