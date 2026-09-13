import math
import numpy as np
import pytest

from aurumq_rl.v10.risk_components import CVaRConfig, CVaRDrawdownPenalty


def test_warmup_penalty_is_bounded():
    p = CVaRDrawdownPenalty(CVaRConfig(warmup=3, max_penalty=2.0))
    assert p.update(0.10) < 0
    assert -2.0 <= p.update(0.20) <= 0.0


def test_tail_drawdown_gets_stronger_penalty_but_is_bounded():
    p = CVaRDrawdownPenalty(CVaRConfig(window=20, warmup=5, alpha=0.8, max_penalty=2.0))
    for value in [0.01, 0.02, 0.03, 0.04, 0.05]:
        p.update(value)
    normal = p.update(0.04)
    tail = p.update(0.40)
    assert tail < normal
    assert -2.0 <= tail <= 0.0
    assert p.state()["cvar"] >= p.state()["var"]


def test_nan_and_inf_do_not_corrupt_history():
    p = CVaRDrawdownPenalty()
    assert p.update(float("nan")) == 0.0
    assert p.update(float("inf")) == 0.0
    assert p.state()["count"] == 0


@pytest.mark.parametrize("kwargs", [
    {"window": 1}, {"alpha": 0.4}, {"alpha": 1.0},
    {"warmup": 0}, {"normal_scale": -1.0}, {"max_penalty": 0.0},
])
def test_invalid_cvar_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CVaRDrawdownPenalty(CVaRConfig(**kwargs))
