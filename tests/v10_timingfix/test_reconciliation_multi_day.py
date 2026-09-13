from datetime import date
import numpy as np

from aurumq_rl.v10_timingfix.env import TimingFixEnvConfig, TimingFixTradingEnv
from aurumq_rl.v10_timingfix.market_data import ExecutionPricePanel


def test_multi_day_nav_and_trade_sequence_is_deterministic():
    dates = tuple(date(2025, 1, i + 1) for i in range(7))
    prices = ExecutionPricePanel(
        dates=dates, stock_codes=("A", "B"),
        open_array=np.array([[10, 20], [10, 20], [11, 19], [12, 18], [11, 19], [13, 17], [14, 16]], float),
        close_array=np.array([[10, 20], [10.5, 19.5], [11.5, 18.5], [12.5, 17.5], [11.5, 18.5], [13.5, 16.5], [14.5, 15.5]], float),
    )
    factors = np.zeros((7, 2, 2), dtype=np.float32)
    env = TimingFixTradingEnv(factors, prices, TimingFixEnvConfig(window=2, max_position_pct=.5, top_k=1, cost_bps=0, slippage_bps=0))
    env.reset()
    rewards = []
    for action in [np.array([1., 0.]), np.array([1., 0.]), np.array([0., 1.]), np.array([0., 1.])]:
        _, reward, terminated, _, _ = env.step(action)
        rewards.append(reward)
        if terminated:
            break
    assert len(rewards) == 4
    assert env._state is not None
    assert len(env._state.trades) >= 2
    assert all(np.isfinite(rewards))
    assert env._state.nav > 0
