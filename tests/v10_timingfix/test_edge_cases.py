from datetime import date
import numpy as np

from aurumq_rl.v10_timingfix.env import TimingFixEnvConfig, TimingFixTradingEnv
from aurumq_rl.v10_timingfix.market_data import ExecutionPricePanel
from aurumq_rl.v10_timingfix.trading_state_machine import TradingConfig, TradingStateMachine


def test_stop_loss_has_priority_and_cooldown_blocks_reentry():
    sm = TradingStateMachine(TradingConfig(cost_bps=0, slippage_bps=0, stop_loss_pct=10, cooldown_days=5))
    sm.execute_open("2025-01-02", {"A": 10}, {"A": 10}, {"A": 0.5}, tradable={"A"})
    sm.execute_open("2025-01-03", {"A": 8.9}, {"A": 8.9}, {"A": 0.5}, tradable={"A"})
    assert sm.trades[-1]["reason"] == "stop_loss"
    assert "A" not in sm.positions
    # Same stock is still blocked immediately after stop loss.
    sm.execute_open("2025-01-04", {"A": 9}, {"A": 9}, {"A": 0.5}, tradable={"A"})
    assert "A" not in sm.positions


def test_missing_open_price_does_not_create_fake_trade():
    sm = TradingStateMachine(TradingConfig(cost_bps=0, slippage_bps=0))
    sm.execute_open("2025-01-02", {"A": 10}, {"A": 10}, {"A": 0.5}, tradable={"A"})
    before = len(sm.trades)
    result = sm.execute_open("2025-01-03", {}, {}, forced_sells={"A"}, tradable=set())
    assert len(sm.trades) == before
    assert "A" in sm.positions
    assert result["nav"] > 0


def test_event_sell_cash_can_be_redeployed_on_next_open():
    sm = TradingStateMachine(TradingConfig(cost_bps=0, slippage_bps=0))
    sm.execute_open("2025-01-02", {"A": 10, "B": 10}, {"A": 10, "B": 10}, {"A": 0.5}, tradable={"A", "B"})
    sm.execute_open("2025-01-03", {"A": 11, "B": 10}, {"A": 11, "B": 10}, {"B": 0.5}, forced_sells={"A"}, tradable={"A", "B"})
    assert "A" not in sm.positions
    # A's released cash is available in this same open batch for B.
    assert "B" in sm.positions
    assert any(t["side"] == "sell" for t in sm.trades)
    assert any(t["side"] == "buy" and t["stock_code"] == "B" for t in sm.trades)


def test_end_of_period_force_close_is_explicit():
    sm = TradingStateMachine(TradingConfig(cost_bps=0, slippage_bps=0))
    sm.execute_open("2025-01-02", {"A": 10}, {"A": 10}, {"A": 0.5}, tradable={"A"})
    sm.force_close("2025-01-03", {"A": 11})
    assert not sm.positions
    assert sm.trades[-1]["side"] == "sell"
    # The shared state machine records the caller's forced-close event; the
    # runner will map this final operation to end_of_period in its ledger.
    assert sm.trades[-1]["reason"] == "end_of_period"


def test_future_suffix_zero_does_not_change_prior_observations():
    dates = tuple(date(2025, 1, i + 1) for i in range(8))
    prices = ExecutionPricePanel(
        dates=dates, stock_codes=("A",),
        open_array=np.ones((8, 1)), close_array=np.ones((8, 1)),
    )
    original = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
    mutated = original.copy(); mutated[5:] = 0.0
    env_a = TimingFixTradingEnv(original, prices, TimingFixEnvConfig(window=3))
    env_b = TimingFixTradingEnv(mutated, prices, TimingFixEnvConfig(window=3))
    for t in range(5):
        assert np.array_equal(env_a._observation(t), env_b._observation(t))
    assert not np.array_equal(env_a._observation(5), env_b._observation(5))
