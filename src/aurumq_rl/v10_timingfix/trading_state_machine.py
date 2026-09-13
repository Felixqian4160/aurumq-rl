"""Shared event-driven trading state machine for v10_timingfix.

Both training and simulation must import this module. Signals are produced after
session T and passed to ``execute_open`` for the next session's open.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping
import math


@dataclass(frozen=True)
class TradingConfig:
    initial_capital: float = 100_000.0
    cost_bps: float = 15.0
    slippage_bps: float = 10.0
    stop_loss_pct: float = 0.0
    cooldown_days: int = 5

    def __post_init__(self) -> None:
        for name in ("cost_bps", "slippage_bps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be in [0, 100], got {value}")
        if self.initial_capital <= 0 or self.cooldown_days < 0:
            raise ValueError("initial_capital must be positive and cooldown_days non-negative")
        if self.stop_loss_pct < 0 or self.stop_loss_pct >= 100:
            raise ValueError("stop_loss_pct must be in [0, 100)")


@dataclass
class Position:
    shares: int
    entry_price: float
    entry_date: str
    cost_basis: float


@dataclass
class TradingStateMachine:
    """Deterministic cash/positions/fees/slippage state machine.

    ``target_weights`` are desired portfolio weights at the next open. Forced
    event sells are processed first; cash released by those sells is available
    for buys in the same open batch. A zero target means no new position, while
    an explicit forced sell is required for event-driven exits of held positions.
    """

    config: TradingConfig
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    cooldown_until: dict[str, int] = field(default_factory=dict, init=False)
    nav: float = field(init=False)
    day_index: int = field(default=0, init=False)
    trades: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.config.initial_capital)
        self.nav = self.cash

    @property
    def fee_rate(self) -> float:
        return self.config.cost_bps / 10000.0

    @property
    def slip_rate(self) -> float:
        return self.config.slippage_bps / 10000.0

    def _valid_price(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid execution price: {value}")
        return value

    def execute_open(
        self,
        date: str,
        open_prices: Mapping[str, float],
        close_prices: Mapping[str, float],
        target_weights: Mapping[str, float] | None = None,
        forced_sells: Iterable[str] = (),
        tradable: Iterable[str] | None = None,
        forced_reason: str = "event_sell",
        signal_date: str | None = None,
    ) -> dict:
        """Execute next-open orders and mark NAV at that day's close."""
        target_weights = target_weights or {}
        tradable_set = set(tradable) if tradable is not None else set(open_prices)
        forced = set(forced_sells)
        sold: set[str] = set()

        # Explicit event sells and stop losses always precede buys.
        for code in list(self.positions):
            if code not in open_prices:
                continue
            market = self._valid_price(open_prices[code])
            pos = self.positions[code]
            stop = self.config.stop_loss_pct > 0 and market <= pos.entry_price * (1 - self.config.stop_loss_pct / 100)
            if code in forced or stop:
                exec_price = market * (1 - self.slip_rate)
                amount = pos.shares * exec_price
                fee = amount * self.fee_rate
                self.cash += amount - fee
                reason = "stop_loss" if stop and code not in forced else forced_reason
                self.trades.append({"date": date, "signal_date": signal_date or date, "execution_date": date, "stock_code": code, "side": "sell",
                                    "market_price": market, "execution_price": exec_price,
                                    "shares": pos.shares, "fee": fee, "pnl": amount - fee - pos.cost_basis,
                                    "reason": reason})
                del self.positions[code]
                sold.add(code)
                if reason == "stop_loss":
                    self.cooldown_until[code] = self.day_index + self.config.cooldown_days

        # Allocate target capital after sells. Buy orders are in the same batch.
        available_nav = self.cash + sum(
            p.shares * float(close_prices.get(code, p.entry_price))
            for code, p in self.positions.items()
        )
        for code, weight in sorted(target_weights.items()):
            weight = float(weight)
            if weight <= 0 or code in self.positions or code not in tradable_set or code not in open_prices:
                continue
            if self.cooldown_until.get(code, -1) > self.day_index:
                continue
            market = self._valid_price(open_prices[code])
            exec_price = market * (1 + self.slip_rate)
            budget = available_nav * weight
            shares = int(budget / exec_price / 100) * 100
            cost = shares * exec_price
            fee = cost * self.fee_rate
            if shares <= 0 or cost + fee > self.cash:
                continue
            self.cash -= cost + fee
            self.positions[code] = Position(shares, exec_price, date, cost + fee)
            self.trades.append({"date": date, "signal_date": signal_date or date, "execution_date": date, "stock_code": code, "side": "buy",
                                "market_price": market, "execution_price": exec_price,
                                "shares": shares, "amount": cost, "fee": fee, "cost_basis": cost + fee, "pnl": 0.0,
                                "reason": "target_buy"})

        close_value = sum(
            p.shares * self._valid_price(close_prices[code])
            for code, p in self.positions.items() if code in close_prices
        )
        self.nav = self.cash + close_value
        result = {"date": date, "nav": self.nav, "cash": self.cash,
                  "positions": {k: v.shares for k, v in self.positions.items()},
                  "executed_trades": len(self.trades), "sold": sorted(sold)}
        self.day_index += 1
        return result

    def force_close(self, date: str, open_prices: Mapping[str, float]) -> dict:
        """Close all remaining positions at the supplied final executable open."""
        return self.execute_open(date, open_prices, open_prices,
                                 target_weights={}, forced_sells=list(self.positions),
                                 tradable=open_prices, forced_reason="end_of_period")
