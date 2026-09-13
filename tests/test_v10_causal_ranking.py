import numpy as np
from aurumq_rl.v10.label_engine import causal_combined_score, causal_percentile


def test_causal_percentile_excludes_future_by_contract():
    value, status = causal_percentile(5.0, np.arange(30.0), min_history=30)
    assert status == "ready"
    assert value == 0.2
    value2, status2 = causal_percentile(5.0, np.arange(29.0), min_history=30)
    assert status2 == "cold_start"
    assert np.isnan(value2)


def test_causal_combined_score_requires_both_historical_distributions():
    score, status = causal_combined_score(5.0, 2.0, np.arange(30.0), np.arange(30.0), min_history=30)
    assert status == "ready"
    assert 0.0 <= score <= 1.0
    score2, status2 = causal_combined_score(5.0, 2.0, np.arange(29.0), np.arange(30.0), min_history=30)
    assert status2 == "cold_start"
    assert np.isnan(score2)
