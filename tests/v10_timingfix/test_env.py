from datetime import date
import numpy as np

from aurumq_rl.v10_timingfix.env import TimingFixEnvConfig, TimingFixTradingEnv
from aurumq_rl.v10_timingfix.market_data import ExecutionPricePanel


def make_env():
    factors = np.zeros((5, 2, 3), dtype=np.float32)
    prices = ExecutionPricePanel(
        dates=tuple(date(2025, 1, i + 1) for i in range(5)),
        stock_codes=("A", "B"),
        open_array=np.array([[10, 20], [11, 19], [12, 18], [13, 17], [14, 16]], dtype=float),
        close_array=np.array([[10, 20], [12, 18], [13, 17], [14, 16], [15, 15]], dtype=float),
    )
    return TimingFixTradingEnv(
        factors, prices,
        TimingFixEnvConfig(window=2, max_position_pct=0.5, top_k=2, cost_bps=0, slippage_bps=0),
    )


def test_daily_step_executes_next_open_and_marks_next_close():
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (2 * 2 * 3,)
    _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
    assert not terminated
    assert info["step"] == 2
    assert env._state is not None
    assert env._state.trades[0]["date"] == "2025-01-03"
    assert env._state.trades[0]["execution_price"] == 12.0
    assert env._state.trades[0]["shares"] > 0
    assert reward != 0.0


def test_env_uses_shared_state_machine():
    env = make_env()
    env.reset()
    env.step(np.array([1.0, 0.0]))
    assert env._state.__class__.__module__.endswith("trading_state_machine")
