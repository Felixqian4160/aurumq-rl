"""Standalone v10_timingfix simulator using the shared TradingStateMachine."""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter, build_tradeable_mask, align_panel_to_training_universe
from aurumq_rl.v10.head_mapping import HEAD_MAPPING, threshold_for
from aurumq_rl.v10.inference import MultiHeadInference
from aurumq_rl.v10.signal_engine import BUY_HEAD, BUY_KEY, SELL_HEAD, SELL_KEY, scores_from_heads
from aurumq_rl.v10_timingfix.market_data import load_execution_prices
from aurumq_rl.v10_timingfix.trading_state_machine import TradingConfig, TradingStateMachine


def run_timingfix_simulation(
    model_dir: str | Path,
    panel_path: str | Path,
    start_date: date,
    end_date: date,
    *,
    initial_capital: float = 100_000.0,
    top_k: int = 20,
    cost_bps: float = 15.0,
    slippage_bps: float = 10.0,
    stop_loss_pct: float = 0.0,
    cooldown_days: int = 5,
    thresholds: dict[str, float] | None = None,
    ledger_path: str | Path | None = None,
    review_days: int = 1,
) -> dict:
    """Run signals at T and execute them at T+1 open through one shared state machine.

    ``review_days`` controls scheduled portfolio review frequency. Event sell
    signals remain checked daily; on non-review days no new target positions
    are opened or rebalanced.
    """
    model_dir, panel_path = Path(model_dir), Path(panel_path)
    if not isinstance(review_days, int) or review_days < 1:
        raise ValueError("review_days must be a positive integer")
    meta = json.loads((model_dir / "metadata.json").read_text())
    thresholds = thresholds or {name: threshold_for(name) for name in HEAD_MAPPING
                                 if HEAD_MAPPING[name].threshold is not None}
    loader = FactorPanelLoader(str(panel_path))
    panel = loader.load_panel(start_date, end_date, n_factors=int(meta["n_factors"]),
                              forward_period=1, universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
                              factor_names=meta["factor_names"])
    panel, universe_stats = align_panel_to_training_universe(panel, meta["stock_codes"])
    prices = load_execution_prices(panel_path, panel.dates, panel.stock_codes)
    mask = build_tradeable_mask(panel)
    agent = MultiHeadInference(model_dir)
    state = TradingStateMachine(TradingConfig(initial_capital=initial_capital,
                                               cost_bps=cost_bps,
                                               slippage_bps=slippage_bps,
                                               stop_loss_pct=stop_loss_pct,
                                               cooldown_days=cooldown_days))
    nav_curve = []
    window = int(meta.get("hyperparams", {}).get("window", 20))
    for t in range(len(panel.dates) - 1):
        lo = max(0, t - window + 1)
        x = panel.factor_array[lo:t + 1].transpose(1, 0, 2)
        if x.shape[1] < window:
            x = np.pad(x, ((0, 0), (window - x.shape[1], 0), (0, 0)))
        out = agent.predict(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1))
        score = scores_from_heads(out, thresholds[BUY_HEAD], None, thresholds[SELL_HEAD], None)
        if score.size == 0:
            score = np.zeros(len(panel.stock_codes), dtype=np.float32)
        candidates = np.where((score > 0) & mask[t])[0]
        if len(candidates) > top_k:
            candidates = candidates[np.argsort(score[candidates])[::-1][:top_k]]
        target = {panel.stock_codes[i]: 1.0 / max(len(candidates), 1) for i in candidates}
        if t % review_days != 0:
            target = {code: 1.0 / max(len(state.positions), 1) for code in state.positions}
        forced = {code for i, code in enumerate(panel.stock_codes) if code in state.positions
                  and (float(out[SELL_KEY][i]) >= thresholds[SELL_HEAD]
                       or (t % review_days == 0 and score[i] <= 0))}
        next_t = t + 1
        opens = prices.open_array[next_t]; closes = prices.close_array[next_t]
        open_map = {c: float(opens[i]) for i, c in enumerate(panel.stock_codes) if np.isfinite(opens[i]) and opens[i] > 0}
        close_map = {c: float(closes[i]) for i, c in enumerate(panel.stock_codes) if np.isfinite(closes[i]) and closes[i] > 0}
        result = state.execute_open(str(panel.dates[next_t]), open_map, close_map,
                                    target_weights=target, forced_sells=forced,
                                    tradable=[panel.stock_codes[i] for i in np.where(mask[next_t])[0]],
                                    signal_date=str(panel.dates[t]))
        nav_curve.append(result)
    if state.positions:
        last = len(panel.dates) - 1
        opens = {c: float(prices.open_array[last, i]) for i, c in enumerate(panel.stock_codes)
                 if np.isfinite(prices.open_array[last, i]) and prices.open_array[last, i] > 0}
        # The liquidation is itself the terminal NAV observation. Persist it in
        # nav_curve; otherwise state.nav reflects the forced close while the
        # ledger curve ends one valuation earlier, producing inconsistent final
        # NAV, return, Sharpe, and drawdown calculations.
        terminal = state.force_close(str(panel.dates[last]), opens)
        nav_curve.append(terminal)
    ledger = {"version": "v10_timingfix", "model": str(model_dir), "panel": str(panel_path),
              "config": {"initial_capital": initial_capital, "top_k": top_k,
                         "cost_bps": cost_bps, "slippage_bps": slippage_bps,
                         "stop_loss_pct": stop_loss_pct, "cooldown_days": cooldown_days,
                         "review_days": review_days},
              "nav_curve": nav_curve, "trades": state.trades, "final_nav": state.nav}
    if ledger_path:
        Path(ledger_path).write_text(json.dumps(ledger, ensure_ascii=False, indent=2))
    return ledger


if __name__ == "__main__":
    config_path, state_path = Path(sys.argv[1]), Path(sys.argv[2])
    config = json.loads(config_path.read_text())
    try:
        ledger_path = Path("data/ledgers") / (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_timingfix.json"
        )
        result = run_timingfix_simulation(
            model_dir=config["model_dir"], panel_path=config["panel_path"],
            start_date=date.fromisoformat(config["start_date"]),
            end_date=date.fromisoformat(config["end_date"]),
            initial_capital=config["initial_capital"], top_k=config["top_k"],
            cost_bps=config["cost_bps"], slippage_bps=config["slippage_bps"],
            stop_loss_pct=config["stop_loss_pct"], cooldown_days=config["cooldown_days"],
            review_days=int(config.get("review_days", 1)),
            thresholds={k: float(v) for k, v in {
                BUY_HEAD: config.get("buy_threshold"),
                SELL_HEAD: config.get("sell_threshold"),
            }.items() if v is not None},
            ledger_path=ledger_path,
        )
        state = json.loads(state_path.read_text())
        state.update(status="finished", running=False, returncode=0,
                     ledger_file=str(ledger_path), final_nav=result["final_nav"])
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as exc:
        state = json.loads(state_path.read_text())
        state.update(status="failed", running=False, returncode=1, error=str(exc))
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        raise