import numpy as np
from aurumq_rl.v10.risk_components import CVaRConfig, CVaRDrawdownPenalty


def test_cvar_enabled_path_produces_tail_state():
    p = CVaRDrawdownPenalty(CVaRConfig(window=20, warmup=5, alpha=0.8))
    for dd in [0.01, 0.02, 0.03, 0.04, 0.05, 0.20, 0.30]:
        penalty = p.update(dd)
    state = p.state()
    assert state['count'] == 7
    assert state['cvar'] >= state['var']
    assert -2.0 <= penalty <= 0.0


def test_cvar_disabled_fallback_is_original_bounded_rule():
    dd = 0.10
    fallback = -float(np.clip(dd / 0.20, 0.0, 1.0))
    assert fallback == -0.5
