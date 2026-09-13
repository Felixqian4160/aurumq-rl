import numpy as np
import pytest
from aurumq_rl.v10.hhi_components import HHIConfig, HHIConcentrationPenalty


def test_equal_weight_target_has_mild_penalty():
    p = HHIConcentrationPenalty(HHIConfig(weight=0.5, target_count=10, max_penalty=0.5))
    value = p.compute(np.full(10, 0.1))
    assert np.isclose(value, 0.0)
    assert np.isclose(p.compute(np.array([0.02] + [0.0] * 9)), -0.5)


def test_concentrated_portfolio_is_more_penalized_and_bounded():
    p = HHIConcentrationPenalty(HHIConfig(weight=0.5, target_count=10, max_penalty=0.5))
    diversified = p.compute(np.full(10, 0.1))
    concentrated = p.compute(np.array([1.0]))
    assert concentrated < diversified
    assert concentrated >= -0.5


def test_empty_nan_and_negative_weights_are_safe():
    p = HHIConcentrationPenalty()
    assert p.compute([]) == 0.0
    assert np.isfinite(p.compute([np.nan, -1.0, np.inf]))


@pytest.mark.parametrize("kwargs", [{"weight": -1.0}, {"target_count": 0}, {"max_penalty": 0.0}])
def test_invalid_hhi_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HHIConcentrationPenalty(HHIConfig(**kwargs))
