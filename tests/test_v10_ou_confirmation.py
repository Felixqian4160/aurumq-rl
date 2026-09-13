import numpy as np

from aurumq_rl.v10.label_engine import post_ou_confirmation_features, stop_falling_confirmation_score


def test_confirmation_score_is_weighted_and_bounded():
    score = stop_falling_confirmation_score({"no_new_low": 1.0, "slope_turn": 0.5, "vol_contraction": 0.0})
    assert 0.0 < score < 1.0


def test_post_ou_confirmation_uses_future_window_for_label_validation():
    prices = np.array([100, 98, 96, 95, 95, 96, 98, 100, 101], dtype=float)
    features = post_ou_confirmation_features(prices, ou_end=3, window=5, pre_window=3)
    assert features["no_new_low"] == 1.0
    assert features["slope_turn"] > 0.5
    assert 0.0 <= features["vol_contraction"] <= 1.0


def test_post_confirmation_insufficient_future_is_neutral():
    features = post_ou_confirmation_features([100, 99, 98], ou_end=2)
    assert features == {"no_new_low": 0.0, "slope_turn": 0.0, "vol_contraction": 0.0}
