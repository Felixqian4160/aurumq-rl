"""v10_timingfix training environment.

Daily action -> next-open execution -> same-day close mark-to-market.  The
execution and accounting implementation is shared with the simulator through
TradingStateMachine; this module only prepares observations and signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from aurumq_rl.v10_timingfix.market_data import ExecutionPricePanel
from aurumq_rl.v10_timingfix.trading_state_machine import TradingConfig, TradingStateMachine


@dataclass(frozen=True)
class TimingFixEnvConfig:
    window: int = 20
    max_position_pct: float = 0.05
    top_k: int = 20
    cost_bps: float = 15.0
    slippage_bps: float = 10.0
    stop_loss_pct: float = 0.0
    cooldown_days: int = 5
    reward_scale: float = 10.0


class TimingFixTradingEnv(gym.Env):
    """Daily PPO environment backed by the shared trading state machine."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        factor_panel: np.ndarray,
        execution_prices: ExecutionPricePanel,
        config: TimingFixEnvConfig = TimingFixEnvConfig(),
        tradeable_mask: np.ndarray | None = None,
        evt_peak_labels: np.ndarray | None = None,
        evt_valley_labels: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if factor_panel.ndim != 3:
            raise ValueError("factor_panel must have shape (dates, stocks, factors)")
        t, s, f = factor_panel.shape
        if execution_prices.open_array.shape != (t, s):
            raise ValueError("execution price shape must match factor_panel dates/stocks")
        if t <= config.window:
            raise ValueError("factor_panel must contain more dates than window")
        self.factor_panel = np.nan_to_num(factor_panel.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.execution_prices = execution_prices
        self.config = config
        self.n_dates, self.n_stocks, self.n_factors = t, s, f
        self.window = config.window
        self.tradeable_mask = (
            tradeable_mask.astype(bool)
            if tradeable_mask is not None else np.ones((t, s), dtype=bool)
        )
        if self.tradeable_mask.shape != (t, s):
            raise ValueError("tradeable_mask shape must match factor_panel")
        self.evt_peak_labels = (np.asarray(evt_peak_labels, dtype=np.int64)
                                if evt_peak_labels is not None else np.full((t, s), -1, dtype=np.int64))
        self.evt_valley_labels = (np.asarray(evt_valley_labels, dtype=np.int64)
                                  if evt_valley_labels is not None else np.full((t, s), -1, dtype=np.int64))
        if self.evt_peak_labels.shape != (t, s) or self.evt_valley_labels.shape != (t, s):
            raise ValueError("EVT label shapes must match factor_panel")
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(s * config.window * f,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(s,), dtype=np.float32)
        self._state: TradingStateMachine | None = None
        self._current_step = self.window - 1
        self._previous_nav = 0.0

    def _observation(self, t: int) -> np.ndarray:
        lo = max(0, t - self.window + 1)
        x = self.factor_panel[lo:t + 1].transpose(1, 0, 2)
        if x.shape[1] < self.window:
            x = np.pad(x, ((0, 0), (self.window - x.shape[1], 0), (0, 0)))
        return x.reshape(-1).astype(np.float32)

    def _target_weights(self, action: np.ndarray, t: int) -> dict[str, float]:
        raw = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        raw[~self.tradeable_mask[t]] = 0.0
        if self.config.top_k > 0 and np.count_nonzero(raw) > self.config.top_k:
            idx = np.argpartition(raw, -self.config.top_k)[-self.config.top_k:]
            keep = np.zeros(self.n_stocks, dtype=bool)
            keep[idx] = True
            raw[~keep] = 0.0
        weights = np.minimum(raw, self.config.max_position_pct)
        total = weights.sum()
        if total > 1.0:
            weights /= total
        return {code: float(weights[i]) for i, code in enumerate(self.execution_prices.stock_codes) if weights[i] > 0}

    def _price_maps(self, t: int) -> tuple[dict[str, float], dict[str, float]]:
        opens = self.execution_prices.open_array[t]
        closes = self.execution_prices.close_array[t]
        open_map = {c: float(opens[i]) for i, c in enumerate(self.execution_prices.stock_codes) if np.isfinite(opens[i]) and opens[i] > 0}
        close_map = {c: float(closes[i]) for i, c in enumerate(self.execution_prices.stock_codes) if np.isfinite(closes[i]) and closes[i] > 0}
        return open_map, close_map

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._state = TradingStateMachine(TradingConfig(
            initial_capital=100_000.0,
            cost_bps=self.config.cost_bps,
            slippage_bps=self.config.slippage_bps,
            stop_loss_pct=self.config.stop_loss_pct,
            cooldown_days=self.config.cooldown_days,
        ))
        self._current_step = self.window - 1
        self._previous_nav = self._state.nav
        return self._observation(self._current_step), {"step": self._current_step, "nav": self._state.nav,
            "evt_peak": self.evt_peak_labels[self._current_step],
            "evt_valley": self.evt_valley_labels[self._current_step]}

    def step(self, action: np.ndarray):
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        t = self._current_step
        if t >= self.n_dates - 1:
            return self._observation(t), 0.0, True, False, {"step": t, "nav": self._state.nav}
        target = self._target_weights(action, t)
        held_codes = set(self._state.positions)
        forced_sells = held_codes - set(target)
        open_map, close_map = self._price_maps(t + 1)
        result = self._state.execute_open(
            date=str(self.execution_prices.dates[t + 1]),
            open_prices=open_map,
            close_prices=close_map,
            target_weights=target,
            forced_sells=forced_sells,
            tradable=open_map.keys(),
            signal_date=str(self.execution_prices.dates[t]),
            )
        reward = (result["nav"] / self._previous_nav - 1.0) * self.config.reward_scale if self._previous_nav else 0.0
        self._previous_nav = result["nav"]
        self._current_step += 1
        terminated = self._current_step >= self.n_dates - 1
        info = {"step": self._current_step, "nav": result["nav"], "trades": self._state.trades[-len(result["sold"]) - len(target):],
            "evt_peak": self.evt_peak_labels[self._current_step],
            "evt_valley": self.evt_valley_labels[self._current_step]}
        return self._observation(self._current_step), float(reward), terminated, False, info

    def close(self):
        self._state = None
