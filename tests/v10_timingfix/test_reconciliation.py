from datetime import date
import numpy as np

from aurumq_rl.v10_timingfix.env import TimingFixEnvConfig, TimingFixTradingEnv
from aurumq_rl.v10_timingfix.market_data import ExecutionPricePanel
from aurumq_rl.v10_timingfix.trading_state_machine import TradingConfig, TradingStateMachine


def _fixture():
    prices = ExecutionPricePanel(
        dates=tuple(date(2025, 1, i + 1) for i in range(6)),
        stock_codes=("A", "B"),
        open_array=np.array([[10, 20], [11, 19], [12, 18], [13, 17], [14, 16], [15, 15]], dtype=float),
        close_array=np.array([[10, 20], [12, 18], [13, 17], [14, 16], [15, 15], [16, 14]], dtype=float),
    )
    factors = np.zeros((6, 2, 2), dtype=np.float32)
    return factors, prices


def test_env_and_shared_state_machine_match_one_transition():
    factors, prices = _fixture()
    cfg = TimingFixEnvConfig(window=2, max_position_pct=0.5, top_k=2, cost_bps=15, slippage_bps=10)
    env = TimingFixTradingEnv(factors, prices, cfg)
    env.reset()
    action = np.array([1.0, 0.0], dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    direct = TradingStateMachine(TradingConfig(cost_bps=15, slippage_bps=10))
    direct.execute_open(
        date="2025-01-03",
        open_prices={"A": 12.0, "B": 18.0},
        close_prices={"A": 13.0, "B": 17.0},
        target_weights={"A": 0.5},
        forced_sells=set(),
        tradable={"A", "B"},
    )
    assert np.isclose(info["nav"], direct.nav)
    assert np.isclose(reward, (direct.nav / 100000.0 - 1.0) * 10.0)
    assert env._state is not None
    assert env._state.positions.keys() == direct.positions.keys()
    for code in direct.positions:
        assert env._state.positions[code].shares == direct.positions[code].shares
        assert np.isclose(env._state.positions[code].entry_price, direct.positions[code].entry_price)
    assert len(env._state.trades) == len(direct.trades)
