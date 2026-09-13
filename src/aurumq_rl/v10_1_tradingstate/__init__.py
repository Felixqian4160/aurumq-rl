"""v10.1-tradingstate shared execution bridge.

This module re-exports the frozen ``TradingStateMachine`` so v10.1 simulation
and future v10.1 training paths share one execution contract.  It does not
modify the frozen module; it only imports it under a version-specific alias.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aurumq_rl.v10_timingfix.trading_state_machine import (
    Position,
    TradingConfig,
    TradingStateMachine,
)


V101_TRADING_STATE_MACHINE_VERSION = "v10.1-tradingstate-bridge-v1"


@dataclass(frozen=True)
class V101TradingConfig:
    """v10.1 simulation parameters mirrored to TradingConfig.

    The defaults match the previously verified v3 ledger so that switching the
    runner from ``run_sim_v10_1_v3.py`` to ``run_sim_v10_1_v4.py`` preserves the
    NAV/trade ledger bit-by-bit on the CSI500 2024-01-01~2025-12-31 fixture.
    """

    initial_capital: float = 100_000.0
    cost_bps: float = 15.0
    slippage_bps: float = 10.0
    stop_loss_pct: float = 0.0
    cooldown_days: int = 5


def build_trading_config(
    *,
    initial_capital: float,
    cost_bps: float,
    slippage_bps: float,
    stop_loss_pct: float = 0.0,
    cooldown_days: int = 5,
) -> TradingConfig:
    cfg = V101TradingConfig(
        initial_capital=initial_capital,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        stop_loss_pct=stop_loss_pct,
        cooldown_days=cooldown_days,
    )
    return TradingConfig(
        initial_capital=cfg.initial_capital,
        cost_bps=cfg.cost_bps,
        slippage_bps=cfg.slippage_bps,
        stop_loss_pct=cfg.stop_loss_pct,
        cooldown_days=cfg.cooldown_days,
    )


def create_trading_state_machine(
    *,
    initial_capital: float,
    cost_bps: float,
    slippage_bps: float,
    stop_loss_pct: float = 0.0,
    cooldown_days: int = 5,
) -> TradingStateMachine:
    cfg = build_trading_config(
        initial_capital=initial_capital,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        stop_loss_pct=stop_loss_pct,
        cooldown_days=cooldown_days,
    )
    return TradingStateMachine(cfg)


__all__ = [
    "Position",
    "TradingConfig",
    "TradingStateMachine",
    "V101TradingConfig",
    "V101_TRADING_STATE_MACHINE_VERSION",
    "build_trading_config",
    "create_trading_state_machine",
]
