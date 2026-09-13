import numpy as np

from aurumq_rl.v10.risk_components import CVaRConfig, CVaRDrawdownPenalty


def test_var_cvar_equal_and_current_drawdown_near_var_is_finite_and_bounded():
    """覆盖尾部值相等时 cvar-var≈0 的隐蔽退化场景。"""
    penalty = CVaRDrawdownPenalty(
        CVaRConfig(window=20, warmup=5, alpha=0.95, max_penalty=2.0)
    )
    # 所有历史回撤相同，使 VaR == CVaR；当前值贴近该 VaR。
    for _ in range(30):
        penalty.update(0.10)
    near_var = penalty.update(0.100000001)
    assert np.isfinite(near_var)
    assert -2.0 <= near_var <= 0.0
    state = penalty.state()
    assert np.isclose(state["var"], state["cvar"])
    assert np.isclose(state["last_drawdown"], state["var"])
