"""WaveHunter v10.1 environment.

The v10.1 environment is isolated from v9/v10 training entry points. It uses
v10.1 label arrays explicitly and keeps the existing v10 reward implementation
as a private, local base only through composition-free inheritance.

P0 trading-state-machine bridge (2026-09-13):
    The v10.1 training env historically inherited execution logic (cost /
    slippage / integer-share constraint) from :class:`LstmWeightEnv`, while
    the v10.1 v4 simulation went through the frozen
    :class:`TradingStateMachine` shared with v10_timingfix.  That divergence
    made the OOS reward computed during PPO training not directly comparable
    to the simulation NAV.  This module adds a *dual-path audit* (always on,
    no behaviour change) plus an opt-in ``use_shared_sm=True`` flag that lets
    a single seed run both legs of the comparison without touching the frozen
    bridge or the parent env.  The flag defaults to ``False`` so existing
    v10.1 training jobs keep their tested bit-exact behaviour; flip it via
    ``WaveHunterV10_1Env(use_shared_sm=True, ...)``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aurumq_rl.lstm_weight_env import LstmWeightConfig, LstmWeightEnv, _project_weights
from aurumq_rl.v10.reward_components import BayesianVolConfig, BayesianVolEstimator
from aurumq_rl.v10.risk_components import CVaRConfig, CVaRDrawdownPenalty
from aurumq_rl.v10.hhi_components import HHIConfig, HHIConcentrationPenalty
from aurumq_rl.v10_1_tradingstate import (
    V101_TRADING_STATE_MACHINE_VERSION,
    create_trading_state_machine,
)


class WaveHunterV10_1Env(LstmWeightEnv):
    """v10.1 environment with explicit v10.1 labels and reward diagnostics."""

    def __init__(
        self,
        config: LstmWeightConfig,
        factor_panel: np.ndarray,
        return_panel: np.ndarray,
        pct_change_panel=None,
        is_st_panel=None,
        is_suspended_panel=None,
        days_since_ipo_panel=None,
        knn_panel=None,
        tradeable_mask=None,
        v10_1_a1_point=None,
        v10_1_a2_interval=None,
        v10_1_peak=None,
        v10_1_b1=None,
        *,
        use_shared_sm: bool = False,
        sm_cost_bps: float | None = None,
        sm_slippage_bps: float = 10.0,
        sm_initial_capital: float = 100_000.0,
        open_array: np.ndarray | None = None,
        close_array: np.ndarray | None = None,
    ):
        super().__init__(
            config,
            factor_panel,
            return_panel,
            pct_change_panel,
            is_st_panel,
            is_suspended_panel,
            days_since_ipo_panel,
            knn_panel,
            tradeable_mask,
        )
        n_dates, n_stocks = factor_panel.shape[:2]
        empty = np.full((n_dates, n_stocks), -1, dtype=np.int64)
        self._v10_1_a1 = self._check_label(v10_1_a1_point, empty, "v10_1_a1_point")
        self._v10_1_a2 = self._check_label(v10_1_a2_interval, empty, "v10_1_a2_interval")
        self._v10_1_peak = self._check_label(v10_1_peak, empty, "v10_1_peak")
        self._v10_1_b1 = self._check_label(v10_1_b1, empty, "v10_1_b1")
        # Private aliases are only for the local copied reward block.
        self._v9_peak = self._v10_1_peak
        self._v9_b1 = self._v10_1_b1
        self._a1_labels = self._v10_1_a1.copy()
        self._a2_labels = self._v10_1_a2.copy()
        self._v9_peak_nav = 1.0
        self._v9_cur_nav = 1.0
        self._v9_picked = np.zeros(n_stocks, dtype=bool)
        self._v9_hit_window: list[float] = []
        self._prev_dd = 0.0
        self._reward_norm_vol = float(getattr(config, "reward_norm_vol", 0.02) or 0.02)
        self._w_abs = float(getattr(config, "reward_w_abs", 0.60) or 0.60)
        self._w_dd = float(getattr(config, "reward_w_dd", 0.25) or 0.25)
        self._w_hit = float(getattr(config, "reward_w_hit", 0.15) or 0.15)
        self._ema_vol = self._reward_norm_vol
        self._ema_alpha = 0.02
        self._bayes_vol = BayesianVolEstimator(BayesianVolConfig(
            alpha0=float(getattr(config, "bayes_vol_alpha0", 2.0) or 2.0),
            beta0=float(getattr(config, "bayes_vol_beta0", 1e-4) or 1e-4),
            decay=float(getattr(config, "bayes_vol_decay", 0.98) or 0.98),
            uncertainty_weight=float(getattr(config, "bayes_vol_uncertainty_weight", 0.50)),
            min_vol=float(getattr(config, "bayes_vol_min", 0.005) or 0.005),
        ))
        self._bayes_vol_enabled = bool(getattr(config, "bayes_vol_enabled", False))
        self._cvar_enabled = bool(getattr(config, "cvar_enabled", False))
        self._cvar_penalty = CVaRDrawdownPenalty(CVaRConfig(
            window=int(getattr(config, "cvar_window", 252) or 252),
            alpha=float(getattr(config, "cvar_alpha", 0.95) or 0.95),
            warmup=int(getattr(config, "cvar_warmup", 30) or 30),
            normal_scale=float(getattr(config, "cvar_normal_scale", 0.5) or 0.5),
            tail_scale=float(getattr(config, "cvar_tail_scale", 1.0) or 1.0),
            max_penalty=float(getattr(config, "cvar_max_penalty", 2.0) or 2.0),
        ))
        self._hhi_enabled = bool(getattr(config, "hhi_enabled", False))
        self._hhi_penalty = HHIConcentrationPenalty(HHIConfig(
            weight=float(getattr(config, "hhi_weight", 0.5) or 0.5),
            target_count=int(getattr(config, "hhi_target_count", 10) or 10),
            max_penalty=float(getattr(config, "hhi_max_penalty", 0.5) or 0.5),
        ))

        # ── P0 trading-state-machine bridge (opt-in, audit always on) ─────
        # ``sm_cost_bps`` falls back to ``config.cost_bps`` (legacy 30 bps)
        # so the audit cost numbers can be compared against the existing
        # LstmWeightEnv cost without forcing a new default.  Set explicitly
        # to match the frozen TradingConfig defaults (15 / 10) when wiring
        # the simulation side.
        self.use_shared_sm = bool(use_shared_sm)
        self._sm_cost_bps = float(
            sm_cost_bps if sm_cost_bps is not None else getattr(config, "cost_bps", 30.0)
        )
        self._sm_slippage_bps = float(sm_slippage_bps)
        self._sm_initial_capital = float(sm_initial_capital)
        self._sm_version = V101_TRADING_STATE_MACHINE_VERSION
        self._sm_audit_steps = 0
        self._sm_audit_cost_diff_sum = 0.0
        self._sm_audit_turnover_sum = 0.0
        # T+1 raw-open + raw-close arrays are required by
        # ``TradingStateMachine.execute_open`` for the shared execution
        # contract.  We accept them but only enforce presence when the SM
        # path is actually requested, so legacy training jobs and synthetic
        # panels still work.
        self._open_prices = open_array
        self._close_prices = close_array
        if self.use_shared_sm and (self._open_prices is None or self._close_prices is None):
            raise ValueError(
                "use_shared_sm=True requires both open_array and close_array "
                "for T+1 raw-open execution; pass them via the loader or env kwargs."
            )
        self._sm: Any = None
        self._reset_sm()

    def _reset_sm(self) -> None:
        """(Re-)instantiate the trading-state-machine.

        The SM is only built when ``use_shared_sm=True``; the audit path
        never depends on it being live because it recomputes the projected
        cost from ``config``/``action`` directly.
        """
        if self.use_shared_sm:
            self._sm = create_trading_state_machine(
                initial_capital=self._sm_initial_capital,
                cost_bps=self._sm_cost_bps,
                slippage_bps=self._sm_slippage_bps,
            )
        else:
            self._sm = None

    def _audit_sm_cost(self, action: np.ndarray) -> dict[str, Any]:
        """Estimate what cost the frozen ``TradingStateMachine`` would charge.

        Mirrors :meth:`LstmWeightEnv.step` lines 262-281: project the raw
        action to executable weights under the same mask, sum L1 turnover,
        then charge it at the SM's combined fee+slippage rate.  This gives
        a per-step cost-difference signal without needing OHLC for an actual
        ``execute_open`` call (which would require integer-share mapping and
        T+1 raw-open prices we don't have at training time).

        Returned keys are stable: ``sm_audit_*`` so the trainer can pick
        them up uniformly.
        """
        t = self._current_step
        trading_mask = self._compute_trading_mask(t)
        is_rebalance = (t - (self.window - 1)) % self.config.rebalance_days == 0
        if is_rebalance:
            weights = _project_weights(
                raw=action,
                mask=trading_mask,
                max_pos=self.config.max_position_pct,
                top_k=self.config.top_k,
            )
        else:
            weights = self._current_weights.copy()
            weights[~trading_mask] = 0.0
            total = weights.sum()
            if total > 1e-8 and total > 1.0:
                weights = weights / total
        turnover = float(np.abs(weights - self._current_weights).sum())
        cost_legacy_bps = turnover * float(self.config.cost_bps or 0.0)
        cost_sm_bps = turnover * (self._sm_cost_bps + self._sm_slippage_bps)
        diff_bps = cost_sm_bps - cost_legacy_bps
        self._sm_audit_steps += 1
        self._sm_audit_cost_diff_sum += diff_bps
        self._sm_audit_turnover_sum += turnover
        return {
            "sm_audit_version": self._sm_version,
            "sm_audit_turnover": turnover,
            "sm_audit_cost_legacy_bps": cost_legacy_bps,
            "sm_audit_cost_sm_bps": cost_sm_bps,
            "sm_audit_cost_diff_bps": diff_bps,
            "sm_audit_sm_cost_bps_config": self._sm_cost_bps,
            "sm_audit_sm_slippage_bps_config": self._sm_slippage_bps,
            "sm_audit_legacy_cost_bps_config": float(self.config.cost_bps or 0.0),
            "sm_audit_use_shared_sm": int(self.use_shared_sm),
        }

    @staticmethod
    def _check_label(value, empty: np.ndarray, name: str) -> np.ndarray:
        if value is None:
            return empty.copy()
        arr = np.asarray(value, dtype=np.int64)
        if arr.shape != empty.shape:
            raise ValueError(f"{name} shape {arr.shape} != {empty.shape}")
        return arr

    def _label_info(self, t: int) -> dict[str, Any]:
        return {
            "a1_bins": self._a1_labels[t],
            "a2_labels": self._a2_labels[t],
            "valid": (self._a1_labels[t] >= 0) | (self._a2_labels[t] >= 0),
            "v10_1_a1": self._v10_1_a1[t],
            "v10_1_a2": self._v10_1_a2[t],
            "v10_1_peak": self._v10_1_peak[t],
            "v10_1_b1": self._v10_1_b1[t],
            "v10_1_peak_label": self._v10_1_peak[t],
            "v10_1_b1_label": self._v10_1_b1[t],
        }

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._v9_peak_nav = 1.0
        self._v9_cur_nav = 1.0
        self._v9_picked = np.zeros(self.n_stocks, dtype=bool)
        self._v9_hit_window = []
        self._prev_dd = 0.0
        self._ema_vol = self._reward_norm_vol
        self._bayes_vol.reset()
        self._cvar_penalty.reset()
        self._sm_audit_steps = 0
        self._sm_audit_cost_diff_sum = 0.0
        self._sm_audit_turnover_sum = 0.0
        self._reset_sm()
        info.update(self._label_info(self._current_step))
        return obs, info

    def _sm_execute_open(self, t: int, target_weights: np.ndarray) -> dict[str, Any]:
        """Drive the frozen :class:`TradingStateMachine` for the T→T+1 transition.

        Returns the SM's per-step result dict (nav, cash, positions, executed
        trades, sold) plus computed ``port_ret_sm`` (mark-to-market return for
        the step, net of fee + slippage) and ``cost_sm`` (raw cost in
        fractional NAV units, matching the ``port_ret`` convention used by
        :class:`LstmWeightEnv`).

        Requires ``self.use_shared_sm=True`` and ``self._open_prices`` /
        ``self._close_prices`` to be present and aligned with the panel.
        NaN prices are skipped so the SM doesn't reject the call; non-traded
        codes get ``weight=0`` so the SM treats them as no-op.
        """
        if self._sm is None:
            raise RuntimeError("SM not initialised; call _reset_sm() or set use_shared_sm=True")
        # T+1 open (T close generated the signal → T+1 raw-open executes).
        # We feed T+1 prices and T+1 close (for the end-of-day NAV mark).
        t_exec = t + 1
        if t_exec < 0 or t_exec >= self.n_dates:
            raise IndexError(f"execute_open out-of-range: t={t}, t_exec={t_exec}, n_dates={self.n_dates}")
        n_stocks = self.n_stocks
        open_prices_arr = self._open_prices
        close_prices_arr = self._close_prices
        assert open_prices_arr is not None and close_prices_arr is not None
        open_row = open_prices_arr[t_exec]
        close_row = close_prices_arr[t_exec]
        if open_row.shape != (n_stocks,) or close_row.shape != (n_stocks,):
            raise ValueError(
                f"open/close row shape mismatch: open={open_row.shape}, "
                f"close={close_row.shape}, expected ({n_stocks},)"
            )
        tradable: list[str] = []
        target_w: dict[str, float] = {}
        codes = self.stock_codes if hasattr(self, "stock_codes") else [f"S_{j}" for j in range(n_stocks)]
        # Re-derive tradeable mask: same rule as LstmWeightEnv._compute_trading_mask.
        tm = self._compute_trading_mask(t)
        for j in range(n_stocks):
            if not tm[j]:
                continue
            op = float(open_row[j])
            cl = float(close_row[j])
            if not np.isfinite(op) or op <= 0 or not np.isfinite(cl) or cl <= 0:
                continue
            tradable.append(codes[j])
            target_w[codes[j]] = float(target_weights[j])
        # Date label for the SM trade ledger: ISO format.
        date_obj = self.dates[t_exec] if hasattr(self, "dates") else None
        date_str = date_obj.isoformat() if date_obj is not None else f"t+{t_exec}"
        # signal_date = T close (the day the policy generated weights)
        sig_obj = self.dates[t] if hasattr(self, "dates") else None
        signal_str = sig_obj.isoformat() if sig_obj is not None else f"t{t}"
        # ``TradingStateMachine`` treats target_weights as buy candidates and
        # requires explicit forced_sells for positions leaving the target.  The
        # parent env's projected weights are a full target vector, so reconcile
        # zeroed positions here before executing the next-open batch.
        forced_sells = [
            code for code in self._sm.positions
            if code not in target_w or target_w[code] <= 0.0
        ]
        prev_nav = self._sm.nav
        result = self._sm.execute_open(
            date=date_str,
            open_prices={c: float(open_row[codes.index(c)]) for c in tradable},
            close_prices={c: float(close_row[codes.index(c)]) for c in tradable},
            target_weights=target_w,
            forced_sells=forced_sells,
            tradable=tradable,
            forced_reason="target_rebalance",
            signal_date=signal_str,
        )
        result["forced_target_sells"] = len(forced_sells)
        new_nav = float(result["nav"])
        # Convert SM fees into fractional NAV cost.  Fees already paid
        # against cash; net-of-cost return = (new_nav / prev_nav) - 1 - 0
        # because fees are already inside new_nav.  We expose them
        # separately for the audit log via ``cost_sm_bps`` field.
        port_ret_sm = (new_nav / prev_nav) - 1.0 if prev_nav > 0 else 0.0
        result["port_ret_sm"] = port_ret_sm
        result["prev_nav"] = prev_nav
        return result

    def step(self, action: np.ndarray):
        # The v10.1 training entry owns the reward implementation. This base
        # step preserves the tested portfolio mechanics and is intentionally
        # kept minimal until the v10.1 reward contract is paired.
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(self._label_info(self._current_step))
        info["reward_version"] = "wavehunter_v10_1_inherited_v10_reward"
        # P0 dual-path audit (no behaviour change): surface the projected
        # SM-side cost alongside the legacy LstmWeightEnv cost so future
        # training runs can detect divergence without rerunning simulations.
        info.update(self._audit_sm_cost(action))
        # P0 Phase 4a: drive the frozen TradingStateMachine when the trainer
        # opted in.  We compute ``port_ret_sm`` + ``nav_sm`` and surface them
        # in info, but DO NOT replace the reward yet — that requires a paired
        # PPO sanity check first.  Reward replacement is gated behind
        # ``sm_replace_reward`` (default False) so this remains a pure audit
        # extension until paired experiments validate the cost distribution.
        if self.use_shared_sm:
            # The super().step() already moved ``self._current_step`` from t to
            # t+1; ``t`` is therefore the *previous* step.  Recover it from
            # the weights change for clarity.
            t = max(0, self._current_step - 1)
            try:
                sm_res = self._sm_execute_open(t, target_weights=self._current_weights)
                info["sm_nav"] = float(sm_res["nav"])
                info["sm_cash"] = float(sm_res["cash"])
                info["sm_port_ret"] = float(sm_res["port_ret_sm"])
                info["sm_trade_count"] = int(sm_res.get("executed_trades", 0))
            except Exception as e:
                info["sm_error"] = f"{type(e).__name__}: {e}"
                # Keep going — the legacy reward is still meaningful.
        return obs, reward, terminated, truncated, info

    def step_with_sm_reward(self, action: np.ndarray):
        """Phase 4a experimental entry: SM-driven reward, no legacy cost path.

        Bypasses ``LstmWeightEnv.step``'s reward block entirely and computes
        reward from the SM's per-step NAV change.  Only valid when
        ``use_shared_sm=True`` and open/close arrays are present.  Kept as a
        separate method so the default ``step`` keeps its bit-exact legacy
        behaviour; the trainer switches by overriding ``step`` at the class
        level when ``--sm-replace-reward`` is set.
        """
        if not self.use_shared_sm:
            raise RuntimeError("step_with_sm_reward requires use_shared_sm=True")
        obs, _, terminated, truncated, info = super().step(action)
        info.update(self._label_info(self._current_step))
        t = max(0, self._current_step - 1)
        try:
            sm_res = self._sm_execute_open(t, target_weights=self._current_weights)
            port_ret_sm = float(sm_res["port_ret_sm"])
            # Override the NAV state used for drawdown tracking so the next
            # step's reward reflects the SM path.
            self._current_nav = float(sm_res["nav"])
            if self._current_nav > self._peak_nav:
                self._peak_nav = self._current_nav
            # ── P0-1 fix (2026-09-13): 回填 SM 实际仓位到 env._current_weights ──
            # SM 实际成交的仓位 (shares * exec_price / nav) 与 super().step()
            # 设的"投影目标权重"不同；不写回会导致下一次 step() 算 turnover 时
            # 错把 target→target 算成 0，但 SM 实际买卖了 N 只。
            target_weights_before_sm_sync = self._current_weights.copy()
            sm_nav = float(sm_res["nav"])
            actual_weights = np.zeros(self.n_stocks, dtype=np.float64)
            if sm_nav > 0 and self._sm is not None:
                codes = self.stock_codes if hasattr(self, "stock_codes") else [f"S_{j}" for j in range(self.n_stocks)]
                code_to_idx = {c: j for j, c in enumerate(codes)}
                close_row = self._close_prices[t + 1] if self._close_prices is not None else None
                positions = sm_res.get("positions", {})
                for code, shares in positions.items():
                    j = code_to_idx.get(code)
                    if j is not None and close_row is not None and int(shares) > 0:
                        mark_price = float(close_row[j])
                        if np.isfinite(mark_price) and mark_price > 0:
                            actual_weights[j] = float(shares) * mark_price / sm_nav
            self._current_weights = actual_weights
            info["sm_actual_weights_l1_vs_target"] = float(np.abs(actual_weights - target_weights_before_sm_sync).sum())
            reward_raw = port_ret_sm
            rtype = self.config.reward_type
            if rtype in ("return", "absolute_return", "absolute_return_v3"):
                reward_raw = port_ret_sm
            elif rtype == "sharpe":
                std = max(float(np.std(self._reward_window or [0.0])), 1e-6)
                reward_raw = port_ret_sm / std
            else:
                reward_raw = port_ret_sm
            reward = float(reward_raw) * float(self.config.reward_scale)
            reward = float(np.nan_to_num(reward, nan=0.0))
            self._cumulative_reward += reward
            self._episode_rewards.append(reward)
            info.update({
                "reward_version": "wavehunter_v10_1_sm_path",
                "sm_nav": float(sm_res["nav"]),
                "sm_cash": float(sm_res["cash"]),
                "sm_port_ret": port_ret_sm,
                "sm_trade_count": int(sm_res.get("executed_trades", 0)),
            })
        except Exception as e:
            info["sm_error"] = f"{type(e).__name__}: {e}"
            if self.use_shared_sm:
                raise RuntimeError(
                    f"SM reward execution failed at t={t}: {type(e).__name__}: {e}"
                ) from e
            reward = 0.0
        return obs, reward, terminated, truncated, info
