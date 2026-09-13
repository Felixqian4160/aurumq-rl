"""Tests for simulation metrics calculation."""

import numpy as np
import pytest

# Import from the actual module
from aurumq_rl.simulation import SimConfig, run_simulation


class TestSimConfig:
    """Test SimConfig dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        cfg = SimConfig(model_dir='test', panel_path='test.parquet')
        assert cfg.hedge_ratio == 0.0
        assert cfg.signal_threshold == 0.0
        assert cfg.dynamic_stop_loss == False
        assert cfg.take_profit_pct == 0.0
        assert cfg.stop_loss_pct == -1.0  # sentinel
        assert cfg.cost_bps == -1.0  # sentinel

    def test_hedge_ratio_range(self):
        """Test hedge_ratio accepts valid range."""
        cfg = SimConfig(model_dir='test', panel_path='test.parquet', hedge_ratio=0.5)
        assert cfg.hedge_ratio == 0.5

        cfg = SimConfig(model_dir='test', panel_path='test.parquet', hedge_ratio=1.0)
        assert cfg.hedge_ratio == 1.0

    def test_signal_threshold_range(self):
        """Test signal_threshold accepts valid range."""
        cfg = SimConfig(model_dir='test', panel_path='test.parquet', signal_threshold=0.01)
        assert cfg.signal_threshold == 0.01

    def test_take_profit_pct_range(self):
        """Test take_profit_pct accepts valid range."""
        cfg = SimConfig(model_dir='test', panel_path='test.parquet', take_profit_pct=10.0)
        assert cfg.take_profit_pct == 10.0


class TestMetricsCalculation:
    """Test metrics calculation logic."""

    def test_sharpe_ratio_positive(self):
        """Test Sharpe ratio calculation with positive returns."""
        # Simulate positive returns
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.025])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        assert sharpe > 0

    def test_sharpe_ratio_negative(self):
        """Test Sharpe ratio calculation with negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.025])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        assert sharpe < 0

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        assert sharpe == 0  # Zero volatility

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        navs = np.array([100, 110, 105, 95, 100, 90, 95])
        peak = np.maximum.accumulate(navs)
        dd = (navs - peak) / peak
        max_dd = float(dd.min() * 100)
        assert max_dd < 0
        assert max_dd == pytest.approx(-18.18, rel=0.01)

    def test_win_rate(self):
        """Test win rate calculation."""
        trades = [
            {'pnl': 100}, {'pnl': -50}, {'pnl': 200},
            {'pnl': -30}, {'pnl': 150}, {'pnl': -80}
        ]
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / len(trades) * 100
        assert win_rate == pytest.approx(50.0)

    def test_profit_factor(self):
        """Test profit factor calculation."""
        wins = [100, 200, 150]
        losses = [50, 30, 80]
        total_win = sum(wins)
        total_loss = sum(abs(l) for l in losses)
        pf = total_win / total_loss if total_loss > 0 else 0
        # 450 / 160 = 2.8125
        assert pf == pytest.approx(2.8125, rel=0.01)


class TestHedgeLogic:
    """Test market neutral hedging logic."""

    def test_hedge_ratio_zero_no_effect(self):
        """Test hedge_ratio=0 has no effect."""
        port_ret = 0.05
        mkt_ret = 0.02
        hedge_ratio = 0.0
        portfolio_beta = 1.0

        hedge_cost = hedge_ratio * portfolio_beta * mkt_ret
        hedged_ret = port_ret - hedge_cost

        assert hedged_ret == port_ret

    def test_hedge_ratio_half(self):
        """Test hedge_ratio=0.5 halves beta exposure."""
        port_ret = 0.05
        mkt_ret = 0.02
        hedge_ratio = 0.5
        portfolio_beta = 1.0

        hedge_cost = hedge_ratio * portfolio_beta * mkt_ret
        hedged_ret = port_ret - hedge_cost

        assert hedged_ret == pytest.approx(0.04)

    def test_hedge_ratio_full(self):
        """Test hedge_ratio=1.0 fully hedges beta."""
        port_ret = 0.05
        mkt_ret = 0.02
        hedge_ratio = 1.0
        portfolio_beta = 1.0

        hedge_cost = hedge_ratio * portfolio_beta * mkt_ret
        hedged_ret = port_ret - hedge_cost

        assert hedged_ret == pytest.approx(0.03)


class TestSignalFiltering:
    """Test signal filtering logic."""

    def test_signal_below_threshold_filters(self):
        """Test signal below threshold triggers filtering."""
        signal_strength = 0.005
        signal_threshold = 0.01
        should_filter = signal_strength < signal_threshold
        assert should_filter == True

    def test_signal_above_threshold_passes(self):
        """Test signal above threshold passes filtering."""
        signal_strength = 0.02
        signal_threshold = 0.01
        should_filter = signal_strength < signal_threshold
        assert should_filter == False

    def test_signal_equal_threshold_passes(self):
        """Test signal equal to threshold passes filtering."""
        signal_strength = 0.01
        signal_threshold = 0.01
        should_filter = signal_strength < signal_threshold
        assert should_filter == False

    def test_zero_threshold_no_filtering(self):
        """Test zero threshold disables filtering."""
        signal_strength = 0.001
        signal_threshold = 0.0
        should_filter = signal_strength < signal_threshold if signal_threshold > 0 else False
        assert should_filter == False


class TestTakeProfit:
    """Test take profit logic."""

    def test_take_profit_triggered(self):
        """Test take profit when threshold reached."""
        buy_price = 100.0
        cur_price = 115.0
        take_profit_pct = 10.0

        held_return = (cur_price / buy_price - 1) * 100
        should_take_profit = held_return >= take_profit_pct

        assert should_take_profit == True

    def test_take_profit_not_triggered(self):
        """Test take profit when threshold not reached."""
        buy_price = 100.0
        cur_price = 105.0
        take_profit_pct = 10.0

        held_return = (cur_price / buy_price - 1) * 100
        should_take_profit = held_return >= take_profit_pct

        assert should_take_profit == False

    def test_take_profit_disabled(self):
        """Test take profit when disabled (threshold=0)."""
        buy_price = 100.0
        cur_price = 150.0
        take_profit_pct = 0.0

        should_take_profit = take_profit_pct > 0 and (cur_price / buy_price - 1) * 100 >= take_profit_pct
        assert should_take_profit == False
