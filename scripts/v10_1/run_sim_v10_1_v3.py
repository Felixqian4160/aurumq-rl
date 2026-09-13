#!/usr/bin/env python3
"""v10.1 simulation-v3: v2 behavior plus auditable portfolio snapshots.

This runner is intentionally additive.  It preserves v2's execution logic and
adds daily per-stock market values, weights, cash ratio, and portfolio HHI to
the ledger.  It does not impose a maximum holding period and does not alter the
frozen v10.1 model, calibration, or v2 runner.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_training_universe,
    build_tradeable_mask,
)
from aurumq_rl.v10_1.calibration import load_or_calibrate
from aurumq_rl.v10_1.inference import MultiHeadV10_1Inference
from aurumq_rl.v10_1.signal_engine import decision_to_signal
from aurumq_rl.v10_1.contract import enforce_simulation_contract, ContractViolationError


ROOT = Path(os.environ.get("AURUMQ_RL_ROOT", Path(__file__).resolve().parents[2]))


def write_state(path: Path, **updates) -> None:
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(updates)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def portfolio_snapshot(
    *,
    date: str,
    nav: float,
    cash: float,
    holdings: dict[int, dict],
    close_px: np.ndarray,
    codes: list[str],
) -> dict:
    """Create one daily, point-in-time portfolio exposure snapshot.

    Market value uses the current raw close.  Invalid/missing closes are not
    silently treated as zero: they are recorded with zero market value and
    excluded from HHI, while the aggregate NAV remains the runner's value.
    """
    positions = []
    market_values = []
    for i, holding in sorted(holdings.items()):
        close = float(close_px[i])
        market_value = (
            float(holding["shares"]) * close
            if np.isfinite(close) and close > 0
            else 0.0
        )
        market_values.append(market_value)
        positions.append(
            {
                "stock_code": codes[i],
                "shares": int(holding["shares"]),
                "entry_price": round(float(holding["price"]), 8),
                "entry_date": holding["entry_date"],
                "market_price": round(close, 8) if np.isfinite(close) else None,
                "market_value": round(market_value, 2),
                "weight": round(market_value / nav, 8) if nav > 0 else 0.0,
            }
        )
    total_marked = float(sum(market_values))
    weights = np.asarray(
        [m / nav for m in market_values if nav > 0 and m > 0], dtype=np.float64
    )
    return {
        "date": date,
        "nav": round(float(nav), 2),
        "cash": round(float(cash), 2),
        "marked_positions_value": round(total_marked, 2),
        "holding_value": round(float(nav - cash), 2),
        "cash_weight": round(float(cash / nav), 8) if nav > 0 else 0.0,
        "invested_weight": round(float(total_marked / nav), 8) if nav > 0 else 0.0,
        "marked_value_plus_cash": round(float(cash + total_marked), 2),
        "nav_reconciliation_diff": round(float(cash + total_marked - nav), 2),
        "n_positions": len(holdings),
        "hhi": round(float(np.square(weights).sum()), 8),
        "positions": positions,
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_sim_v10_1_v3.py CONFIG STATE")
    cfg_path, state_path = map(Path, sys.argv[1:3])
    cfg = json.loads(cfg_path.read_text())
    state = json.loads(state_path.read_text())
    log_path = Path(state["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        print(message, flush=True)

    try:
        model = Path(cfg["model_dir"])
        panel_path = Path(cfg["panel_path"])
        meta = json.loads((model / "metadata.json").read_text())
        if meta.get("version") != "v10.1":
            raise ValueError("metadata.version 不是 v10.1")
        if "wavehunter_v10_1_" not in str(meta.get("panel", "")):
            raise ValueError("metadata.panel 不是 v10.1 面板")
        # ── P0-3 contract check (2026-09-13): 训练/模拟参数一致性强制校验 ──
        # 任何不匹配直接拒绝跑，避免"训练 0.02 / 模拟 0.05"这类 bug 第四次出现
        contract_audit = enforce_simulation_contract(meta, cfg)

        start = datetime.date.fromisoformat(cfg["start_date"])
        end = datetime.date.fromisoformat(cfg["end_date"])
        panel = FactorPanelLoader(panel_path).load_panel(
            start,
            end,
            n_factors=int(meta["n_factors"]),
            factor_names=list(meta.get("factor_names") or []) or None,
            forward_period=1,
            universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
        )
        panel, alignment = align_panel_to_training_universe(panel, meta["stock_codes"])
        n_dates, n_stocks, n_factors = panel.factor_array.shape
        if [n_stocks, n_factors] != [int(meta["n_stocks"]), int(meta["n_factors"])]:
            raise ValueError(f"shape mismatch: {(n_stocks, n_factors)}")
        if panel.close_array is None:
            raise ValueError("v10.1 simulation needs raw close")

        import polars as pl

        raw = (
            pl.scan_parquet(str(panel_path))
            .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
            .select(["trade_date", "ts_code", "open"])
            .collect()
        )
        dates = list(panel.dates)
        codes = list(panel.stock_codes)
        date_index = {d: i for i, d in enumerate(dates)}
        code_index = {c: i for i, c in enumerate(codes)}
        opens = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        for row in raw.iter_rows(named=True):
            i = date_index.get(row["trade_date"])
            j = code_index.get(row["ts_code"])
            if i is not None and j is not None and row["open"] is not None:
                opens[i, j] = float(row["open"])

        mask = build_tradeable_mask(panel)
        closes = panel.close_array
        hyperparams = meta.get("hyperparams") if isinstance(meta.get("hyperparams"), dict) else {}
        window = int(hyperparams.get("window", 20))
        agent = MultiHeadV10_1Inference(model)
        calibration_path = Path(
            cfg.get("threshold_file", "")
            or (ROOT / "data" / "threshold_calibration" / f"{model.name}_v10_1.json")
        )
        calibration = load_or_calibrate(
            model,
            panel_path,
            meta,
            start,
            datetime.date.fromisoformat(cfg.get("calibration_start_date", "2023-01-01")),
            datetime.date.fromisoformat(cfg.get("calibration_end_date", "2023-12-29")),
            calibration_path,
        )
        thresholds = calibration["thresholds"]
        a1_threshold = float(thresholds["a1"])
        a2_threshold = float(thresholds["a2"])
        peak_threshold = float(thresholds["peak"])
        b1_threshold = float(thresholds["b1"])

        capital = float(cfg.get("initial_capital", 100000.0))
        cash = capital
        holdings: dict[int, dict] = {}
        pending: list[dict] = []
        trades: list[dict] = []
        nav_curve: list[dict] = []
        portfolio_curve: list[dict] = []
        prev_nav = capital
        signal_counts = {"buy": 0, "hold": 0, "sell": 0, "wait": 0}
        candidate_count = 0
        cost_bps = float(cfg.get("cost_bps", 15.0))
        slippage = float(cfg.get("slippage_bps", 10.0)) / 10000.0
        top_k = int(cfg.get("top_k", 20))
        rebalance_days = max(1, int(cfg.get("rebalance_days", 20)))
        max_position_pct = float(cfg.get("max_position_pct", 0.05))
        if not 0 <= cost_bps <= 100 or not 0 <= slippage * 10000 <= 100:
            raise ValueError("cost/slippage out of range")

        log(
            f"📐 v10.1 simulation-v3: {n_dates}天 × {n_stocks}股 × {n_factors}因子 "
            f"| aligned={alignment} | thresholds={thresholds}"
        )

        for t, date in enumerate(dates):
            signal_date = date.isoformat()
            close_px = np.asarray(closes[t], dtype=float)
            open_px = np.asarray(opens[t], dtype=float)

            # Execute pending orders generated after the preceding close.
            for order in pending:
                i = order["i"]
                market_px = open_px[i]
                if not np.isfinite(market_px) or market_px <= 0 or not mask[t, i]:
                    continue
                if order["side"] == "sell" and i in holdings:
                    h = holdings.pop(i)
                    exec_px = market_px * (1.0 - slippage)
                    gross = h["shares"] * exec_px
                    fee = gross * cost_bps / 10000.0
                    pnl = gross - fee - h["cost"]
                    cash += gross - fee
                    trades.append(
                        {
                            "signal_date": order["signal_date"],
                            "execution_date": signal_date,
                            "stock_code": codes[i],
                            "side": "sell",
                            "market_price": float(market_px),
                            "sell_price": round(float(exec_px), 6),
                            "buy_price": h["price"],
                            "shares": h["shares"],
                            "pnl": round(float(pnl), 2),
                            "fee": round(float(fee), 2),
                            "reason": order["reason"],
                            "execution_price_type": "raw_open",
                        }
                    )
                elif order["side"] == "buy" and i not in holdings:
                    exec_px = market_px * (1.0 + slippage)
                    amount = min(order["amount"], max_position_pct * capital)
                    if not np.isfinite(exec_px) or exec_px <= 0 or not np.isfinite(amount) or amount <= 0:
                        continue
                    shares = int(amount / exec_px / 100) * 100
                    gross = shares * exec_px
                    fee = gross * cost_bps / 10000.0
                    if shares > 0 and gross + fee <= cash:
                        cash -= gross + fee
                        holdings[i] = {
                            "shares": shares,
                            "price": exec_px,
                            "cost": gross + fee,
                            "entry_date": signal_date,
                        }
                        trades.append(
                            {
                                "signal_date": order["signal_date"],
                                "execution_date": signal_date,
                                "stock_code": codes[i],
                                "side": "buy",
                                "market_price": float(market_px),
                                "buy_price": round(float(exec_px), 6),
                                "shares": shares,
                                "amount": round(float(gross), 2),
                                "fee": round(float(fee), 2),
                                "pnl": 0,
                                "reason": order["reason"],
                                "execution_price_type": "raw_open",
                            }
                        )
            pending = []

            nav_value = float(cash + sum(
                h["shares"] * close_px[i]
                for i, h in holdings.items()
                if np.isfinite(close_px[i]) and close_px[i] > 0
            ))
            nav_curve.append(
                {
                    "date": signal_date,
                    "nav": round(nav_value, 2),
                    "daily_return": round((nav_value / prev_nav - 1) * 100, 4) if prev_nav else 0,
                }
            )
            portfolio_curve.append(
                portfolio_snapshot(
                    date=signal_date,
                    nav=nav_value,
                    cash=float(cash),
                    holdings=holdings,
                    close_px=close_px,
                    codes=codes,
                )
            )
            prev_nav = nav_value
            if t >= n_dates - 1:
                continue

            lo = max(0, t - window + 1)
            observation = panel.factor_array[lo : t + 1].transpose(1, 0, 2)
            if observation.shape[1] < window:
                observation = np.pad(
                    observation,
                    ((0, 0), (window - observation.shape[1], 0), (0, 0)),
                )
            probs = agent.predict(observation.reshape(-1))
            decisions = []
            for i in range(n_stocks):
                decision = decision_to_signal(
                    {
                        "a1": probs["a1"][i],
                        "a2": probs["a2"][i],
                        "peak": probs["peak"][i],
                        "b1": probs["b1"][i],
                    },
                    a1_threshold,
                    a2_threshold,
                    peak_threshold,
                    b1_threshold,
                )
                decisions.append(decision.action)
                signal_counts[decision.action] += 1

            buy_signal = np.asarray([x == "buy" for x in decisions], dtype=bool)
            buy_score = 0.6 * probs["a1"] + 0.4 * probs["a2"]
            eligible = buy_signal & mask[t] & np.isfinite(close_px) & (close_px > 0)
            candidate_count += int(eligible.sum())
            for i in list(holdings):
                if decisions[i] == "sell":
                    pending.append(
                        {
                            "i": i,
                            "side": "sell",
                            "signal_date": signal_date,
                            "reason": "event_sell",
                        }
                    )

            if t == 0 or t % rebalance_days == 0:
                candidates = np.where(eligible)[0]
                if len(candidates) > top_k:
                    candidates = candidates[
                        np.argsort(buy_score[candidates])[::-1][:top_k]
                    ]
                target_amount = (
                    cash
                    + sum(h["shares"] * close_px[i] for i, h in holdings.items())
                ) / max(len(candidates), 1)
                for i in candidates:
                    if i not in holdings:
                        pending.append(
                            {
                                "i": int(i),
                                "side": "buy",
                                "signal_date": signal_date,
                                "amount": target_amount,
                                "reason": "a1_or_a2",
                            }
                        )

        # End-of-period liquidation is a real terminal execution event.  Keep
        # the pre-liquidation mark for audit, then make the final NAV the
        # post-fee cash value so NAV, return, drawdown, and ledger agree.
        terminal_date = dates[-1].isoformat()
        final_px = np.asarray(closes[-1], dtype=float)
        pre_liquidation_nav = nav_curve[-1]["nav"] if nav_curve else capital
        if holdings:
            for i, h in list(holdings.items()):
                if not np.isfinite(final_px[i]) or final_px[i] <= 0:
                    raise RuntimeError(
                        f"cannot liquidate {codes[i]} at terminal date {terminal_date}: invalid close"
                    )
                gross = h["shares"] * final_px[i]
                fee = gross * cost_bps / 10000.0
                pnl = gross - fee - h["cost"]
                cash += gross - fee
                trades.append(
                    {
                        "signal_date": terminal_date,
                        "execution_date": terminal_date,
                        "stock_code": codes[i],
                        "side": "sell",
                        "market_price": float(final_px[i]),
                        "sell_price": float(final_px[i]),
                        "buy_price": h["price"],
                        "shares": h["shares"],
                        "pnl": round(float(pnl), 2),
                        "fee": round(float(fee), 2),
                        "reason": "end_of_period",
                        "execution_price_type": "raw_close",
                    }
                )
            holdings.clear()

        terminal_value = float(cash)
        if nav_curve and nav_curve[-1]["date"] == terminal_date:
            nav_curve[-1]["nav"] = round(terminal_value, 2)
            nav_curve[-1]["daily_return"] = (
                round((terminal_value / prev_nav - 1) * 100, 4) if prev_nav else 0.0
            )
            portfolio_curve[-1] = portfolio_snapshot(
                date=terminal_date,
                nav=terminal_value,
                cash=terminal_value,
                holdings={},
                close_px=final_px,
                codes=codes,
            )

        returns = np.asarray([x["daily_return"] / 100 for x in nav_curve], dtype=float)
        navs = np.asarray([x["nav"] for x in nav_curve], dtype=float)
        peaks = np.maximum.accumulate(navs)
        drawdown = (navs - peaks) / peaks
        sells = [x for x in trades if x["side"] == "sell"]
        wins = [x for x in sells if x["pnl"] > 0]
        loss = sum(x["pnl"] for x in sells if x["pnl"] < 0)
        gain = sum(x["pnl"] for x in wins)
        eop_fee_total = sum(
            float(x.get("fee") or 0.0)
            for x in trades
            if x["side"] == "sell" and x["reason"] == "end_of_period"
        )
        metrics = {
            "final_nav": round(float(cash), 2),
            "nav_curve_last": round(float(navs[-1]), 2) if len(navs) else capital,
            "pre_liquidation_nav": round(float(pre_liquidation_nav), 2),
            "terminal_liquidation_fee_total": round(float(eop_fee_total), 2),
            "total_return_pct": round((float(cash) / capital - 1) * 100, 4),
            "annualized_return_pct": round(((float(cash) / capital) ** (252 / max(len(navs), 1)) - 1) * 100, 4) if cash > 0 else 0.0,
            "sharpe": round(float(returns.mean() / returns.std() * math.sqrt(252)), 4) if len(returns) > 1 and returns.std() > 0 else 0.0,
            "max_drawdown_pct": round(float(drawdown.min() * 100), 4) if len(drawdown) else 0.0,
            "win_rate_pct": round(len(wins) / len(sells) * 100, 4) if sells else 0.0,
            "profit_factor": round(gain / abs(loss), 4) if loss < 0 else 0.0,
            "total_trades": len(trades),
            "buy_trades": sum(x["side"] == "buy" for x in trades),
            "sell_trades": len(sells),
            "signal_counts": signal_counts,
            "candidate_count": candidate_count,
            "aligned": alignment,
        }
        if metrics["final_nav"] != metrics["nav_curve_last"]:
            raise RuntimeError(
                f"terminal NAV mismatch: cash={metrics['final_nav']} nav_curve_last={metrics['nav_curve_last']}"
            )

        ledger_dir = ROOT / "data" / "ledgers"
        ledger_dir.mkdir(exist_ok=True)
        ledger_id = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + model.name
            + "_v10_1_simulation_v3_t1.json"
        )
        ledger_path = ledger_dir / ledger_id
        ledger = {
            "version": "v10.1-sim-v3",
            "model": str(model),
            "panel": str(panel_path),
            "status": "finished",
            "config": cfg,
            "contract_audit": contract_audit,
            "thresholds": thresholds,
            "calibration": calibration,
            "metrics": metrics,
            "trades": trades,
            "nav_curve": nav_curve,
            "portfolio_curve": portfolio_curve,
            "labels_used": False,
            "execution_contract": "T close signal -> T+1 raw open execution -> T+1 close NAV; hold/wait preserve holdings; end_of_period final close",
            "raw_open_nonnull_pct": round(float(np.isfinite(opens).mean() * 100), 4),
            "audit": {
                "portfolio_snapshot_fields": [
                    "cash",
                    "cash_weight",
                    "marked_positions_value",
                    "marked_value_plus_cash",
                    "n_positions",
                    "hhi",
                    "positions[].market_value",
                    "positions[].weight",
                ],
                "holding_period_limit": None,
                "intent": "long_main_uptrend_portfolio",
            },
        }
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2))
        write_state(
            state_path,
            status="finished",
            running=False,
            returncode=0,
            ledger_file=str(ledger_path),
            metrics=metrics,
            finished_at=time.time(),
        )
        log(f"📒 v10.1 simulation-v3 ledger: {ledger_path}")
        log(f"✅ v10.1 simulation-v3 finished: trades={len(trades)} return={metrics['total_return_pct']:.2f}%")
    except ContractViolationError as exc:
        write_state(
            state_path,
            status="contract_violation",
            running=False,
            returncode=1,
            error=str(exc),
            finished_at=time.time(),
        )
        log(f"❌ v10.1 simulation-v3 contract violation: {exc}")
        raise
    except Exception as exc:
        write_state(
            state_path,
            status="failed",
            running=False,
            returncode=1,
            error=str(exc),
            finished_at=time.time(),
        )
        log(f"❌ v10.1 simulation-v3 failed: {exc}")
        raise


if __name__ == "__main__":
    main()
