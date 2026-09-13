import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aurumq_rl.v10_timingfix.trading_state_machine import TradingConfig, TradingStateMachine


def test_known_trade_cost_and_slippage():
    sm = TradingStateMachine(TradingConfig(initial_capital=100_000, cost_bps=15, slippage_bps=10))
    first = sm.execute_open("2025-01-02", {"A": 10}, {"A": 10}, {"A": 0.10}, tradable={"A"})
    assert first["positions"] == {"A": 900}
    buy = sm.trades[0]
    assert math.isclose(buy["execution_price"], 10.01, rel_tol=0, abs_tol=1e-9)
    second = sm.execute_open("2025-01-03", {"A": 11}, {"A": 11}, forced_sells={"A"})
    sell = sm.trades[1]
    expected_sell = 900 * 11 * (1 - 10 / 10000)
    expected_fee = expected_sell * 15 / 10000
    assert math.isclose(sell["execution_price"], 10.989, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(sell["pnl"], expected_sell - expected_fee - buy["cost_basis"], rel_tol=0, abs_tol=1e-8)
    assert second["positions"] == {}


def test_sell_before_buy_and_stop_loss_cooldown():
    sm = TradingStateMachine(TradingConfig(initial_capital=100_000, cost_bps=0, slippage_bps=0, stop_loss_pct=10, cooldown_days=5))
    sm.execute_open("2025-01-02", {"A": 10, "B": 10}, {"A": 10, "B": 10}, {"A": 0.5}, tradable={"A", "B"})
    assert "A" in sm.positions
    result = sm.execute_open("2025-01-03", {"A": 8.9, "B": 10}, {"A": 8.9, "B": 10}, {"A": 0.5, "B": 0.5}, tradable={"A", "B"})
    assert "A" not in sm.positions
    assert "B" in sm.positions
    assert sm.trades[1]["reason"] == "stop_loss"
    assert sm.cooldown_until["A"] == 6


def test_buy_and_hold_anchor():
    sm = TradingStateMachine(TradingConfig(initial_capital=100_000, cost_bps=15, slippage_bps=10))
    sm.execute_open("2025-01-02", {"A": 10}, {"A": 10}, {"A": 0.10}, tradable={"A"})
    buy = sm.trades[0]
    marked = sm.execute_open("2025-01-03", {"A": 12}, {"A": 12})
    expected_nav = sm.cash + buy["shares"] * 12
    assert math.isclose(marked["nav"], expected_nav, rel_tol=0, abs_tol=1e-8)
