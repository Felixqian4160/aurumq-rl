#!/usr/bin/env python3
"""Train v10_timingfix Baseline or New Stack under the same execution contract.

``--new-stack`` is the canonical switch between modes. Only when enabled do
we pull in ``JointWaveHunterV10Policy`` and require the EVT panel columns.
The EVT labels themselves are kept out of the observation; they are only
referenced through metadata so the runner can audit them, never as inputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import math
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter, build_tradeable_mask
from aurumq_rl.v10.policy import WaveHunterV10Policy
from aurumq_rl.v10.joint_four_head_policy import JointWaveHunterV10Policy
from aurumq_rl.v10_timingfix.env import TimingFixEnvConfig, TimingFixTradingEnv
from aurumq_rl.v10_timingfix.market_data import load_execution_prices

import importlib.util as _ilu
_TL_PATH = Path(__file__).resolve().parent / "train_loop.py"
_spec = _ilu.spec_from_file_location("tf_train_loop", _TL_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"failed to load TimingFix train_loop from {_TL_PATH}")
_tf_loop = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tf_loop)
TimingFixPPO = _tf_loop.TimingFixPPO


# Hard-coded legacy New Stack hyperparameters from the v10 design contract.
LEGACY_NEWSTACK = {
    "aux_lambda": 0.1,
    "a1_lambda": 1.0,
    "a2_lambda": 1.0,
    "peak_lambda": 1.5,
    "b1_lambda": 0.3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v10_timingfix Baseline or New Stack PPO")
    p.add_argument("--panel", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--total-timesteps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-position-pct", type=float, default=0.05)
    p.add_argument("--cost-bps", type=float, default=15.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--stop-loss-pct", type=float, default=0.0)
    p.add_argument("--cooldown-days", type=int, default=5)
    p.add_argument("--lstm-hidden", type=int, default=32)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    # Legacy New Stack reward coefficients from the v10 design contract.
    # They MUST be present on the New Stack model and MUST match preflight.
    p.add_argument("--aux-lambda", type=float, default=LEGACY_NEWSTACK["aux_lambda"])
    p.add_argument("--a1-lambda", type=float, default=LEGACY_NEWSTACK["a1_lambda"])
    p.add_argument("--a2-lambda", type=float, default=LEGACY_NEWSTACK["a2_lambda"])
    p.add_argument("--peak-lambda", type=float, default=LEGACY_NEWSTACK["peak_lambda"])
    p.add_argument("--b1-lambda", type=float, default=LEGACY_NEWSTACK["b1_lambda"])
    p.add_argument("--new-stack", action="store_true",
                   help="Switch to New Stack: JointWaveHunterV10Policy + EVT panel columns")
    p.add_argument("--joint-logit-scale", type=float, default=1.0)
    p.add_argument("--evt-peak-column", default="v10_evt_peak")
    p.add_argument("--evt-valley-column", default="v10_evt_valley")
    return p.parse_args()


def _count_params(policy: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in policy.parameters())


def _resolve_policy_class(new_stack: bool) -> type[torch.nn.Module]:
    if new_stack:
        return JointWaveHunterV10Policy
    return WaveHunterV10Policy


def _validate_evt_panel(parquet_path: str, peak_column: str, valley_column: str) -> None:
    raw_columns = pl.read_parquet(parquet_path, columns=[peak_column, valley_column]).columns
    missing = [c for c in (peak_column, valley_column) if c not in raw_columns]
    if missing:
        raise ValueError(f"EVT panel missing required columns: {missing}")


def _load_evt_targets(parquet_path: str, panel, new_stack: bool,
                      peak_column: str, valley_column: str):
    t, s = panel.factor_array.shape[:2]
    full = np.full((t, s), -1, dtype=np.int64)
    if not new_stack:
        return full, full
    df = pl.read_parquet(parquet_path, columns=["trade_date", "ts_code", peak_column, valley_column])
    codes = panel.stock_codes
    code_idx = {c: i for i, c in enumerate(codes)}
    peak = full.copy(); valley = full.copy()
    date_str_to_idx = getattr(panel, "_date_str_to_idx", None)
    if date_str_to_idx is None:
        date_str_to_idx = {(d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d)[:10].replace("-", ""))
                           : i for i, d in enumerate(panel.dates)}
    for r in df.iter_rows(named=True):
        d = r["trade_date"]
        key = d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d)[:10].replace("-", "")
        j = code_idx.get(r["ts_code"])
        if j is None:
            continue
        i = date_str_to_idx.get(key)
        if i is None:
            continue
        pv = r[peak_column]; vv = r[valley_column]
        peak[i, j] = int(pv) if pv is not None else -1
        valley[i, j] = int(vv) if vv is not None else -1
    return peak, valley


class TimingFixMetricsCallback(BaseCallback):
    """Persist a small JSONL training curve for the WebUI chart contract."""

    def __init__(self, path: Path, log_freq: int = 1000):
        super().__init__(verbose=0)
        self.path = Path(path)
        self.log_freq = max(1, int(log_freq))
        self._rewards: list[float] = []
        self._started = time.monotonic()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        try:
            self._rewards.extend(float(x) for x in np.asarray(rewards).reshape(-1))
        except Exception:
            pass
        if self.num_timesteps <= 0 or self.num_timesteps % self.log_freq != 0:
            return True
        values = {}
        try:
            values = dict(self.logger.name_to_value)
        except Exception:
            pass
        def val(*keys, default=0.0):
            for key in keys:
                if key in values and values[key] is not None:
                    try: return float(values[key])
                    except (TypeError, ValueError): pass
            return default
        elapsed = max(time.monotonic() - self._started, 1e-9)
        reward = (sum(self._rewards) / len(self._rewards)) if self._rewards else val('rollout/ep_rew_mean')
        self._rewards.clear()
        record = {
            'timestep': int(self.num_timesteps),
            'episode_reward_mean': reward,
            'policy_loss': val('train/policy_gradient_loss', 'train/loss'),
            'value_loss': val('train/value_loss'),
            'entropy': -val('train/entropy_loss'),
            'explained_variance': val('train/explained_variance'),
            'learning_rate': val('train/learning_rate', default=1e-9),
            'fps': int(self.num_timesteps / elapsed),
            'algorithm': type(self.model).__name__ if self.model is not None else 'PPO',
            'extra': values,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=lambda x: float(x)) + '\n')
        return True


def main() -> None:
    args = parse_args()
    start = dt.date.fromisoformat(args.start_date)
    end = dt.date.fromisoformat(args.end_date)
    if start >= end or end > dt.date.today():
        raise ValueError("invalid training date range")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    loader = FactorPanelLoader(args.panel)
    panel = loader.load_panel(start, end, n_factors=None, forward_period=1,
                              universe_filter=UniverseFilter.MAIN_BOARD_NON_ST)
    prices = load_execution_prices(args.panel, panel.dates, panel.stock_codes)
    mask = build_tradeable_mask(panel)
    t, stocks, factors = panel.factor_array.shape
    policy_class = _resolve_policy_class(args.new_stack)
    if args.new_stack:
        _validate_evt_panel(args.panel, args.evt_peak_column, args.evt_valley_column)
    print(f"[timingfix] panel={t} days x {stocks} stocks x {factors} factors", flush=True)
    print(f"[timingfix] execution=T+1 open, mark=T+1 close, mask={mask.mean():.4f}", flush=True)
    print(f"[timingfix] mode={'new_stack' if args.new_stack else 'baseline'} policy={policy_class.__name__}", flush=True)

    cfg = TimingFixEnvConfig(window=args.window, max_position_pct=args.max_position_pct,
                             top_k=args.top_k, cost_bps=args.cost_bps,
                             slippage_bps=args.slippage_bps, stop_loss_pct=args.stop_loss_pct,
                             cooldown_days=args.cooldown_days, reward_scale=10.0)
    evt_peak_labels, evt_valley_labels = _load_evt_targets(args.panel, panel, args.new_stack,
                                                            args.evt_peak_column, args.evt_valley_column)
    def make_env():
        return TimingFixTradingEnv(panel.factor_array, prices, cfg, mask,
                                    evt_peak_labels=evt_peak_labels,
                                    evt_valley_labels=evt_valley_labels)
    vec_env = DummyVecEnv([make_env])

    policy_kwargs = {"lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
                     "_n_stocks": stocks, "_window": args.window, "_n_factors": factors}
    if args.new_stack:
        policy_kwargs.update({
            "aux_lambda": args.aux_lambda,
            "a1_lambda": args.a1_lambda,
            "a2_lambda": args.a2_lambda,
            "peak_lambda": args.peak_lambda,
            "b1_lambda": args.b1_lambda,
            "joint_logit_scale": args.joint_logit_scale,
            "joint_use_softmax": True,
        })
    PPOClass = TimingFixPPO if args.new_stack else PPO
    metrics_callback = TimingFixMetricsCallback(out_dir / "training_metrics.jsonl", log_freq=max(1000, args.n_steps))
    ppo_kwargs = dict(policy=policy_class, env=vec_env, learning_rate=args.learning_rate,
                      n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
                      gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=args.seed,
                      verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=str(out_dir / "tb_logs"))
    if args.new_stack:
        ppo_kwargs["aux_lambda"] = args.aux_lambda
        model = TimingFixPPO(**ppo_kwargs)
    else:
        model = PPO(**ppo_kwargs)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=metrics_callback)
    model.save(str(out_dir / "ppo_final.zip"))

    parameter_count = _count_params(model.policy)
    policy = model.policy.eval().cpu()
    dummy = torch.zeros(1, stocks * args.window * factors, dtype=torch.float32)
    class Export(torch.nn.Module):
        def __init__(self, source): super().__init__(); self.source = source
        def forward(self, observation): return self.source.forward_with_heads(observation)
    try:
        torch.onnx.export(Export(policy), dummy, str(out_dir / "policy.onnx"), opset_version=14,
                          input_names=["observation"], output_names=["action", "a1", "a2", "peak", "b1"],
                          dynamic_axes={x: {0: "batch"} for x in ["observation", "action", "a1", "a2", "peak", "b1"]},
                          dynamo=False)
    except Exception as exc:
        print(f"[timingfix] ONNX export failed: {exc}", flush=True)

    observation_names = list(panel.factor_names)
    hyperparams = vars(args)
    hyperparams["policy_class"] = policy_class.__name__
    if args.new_stack:
        hyperparams["joint_four_head"] = True
        hyperparams["use_evt_labels"] = True
        hyperparams["observation_names"] = observation_names
        hyperparams["evt_columns"] = {"peak": args.evt_peak_column, "valley": args.evt_valley_column}
    else:
        hyperparams["joint_four_head"] = False
        hyperparams["use_evt_labels"] = False
        hyperparams["observation_names"] = observation_names

    evt_target_count = int((evt_peak_labels == 1).sum() + (evt_valley_labels == 1).sum())

    metadata = {
        "algorithm": "PPO_V10_TIMINGFIX", "model": "wavehunter_v10_timingfix_daily",
        "policy_class": policy_class.__name__,
        "joint_four_head": args.new_stack, "use_evt_labels": args.new_stack,
        "evt_columns": {"peak": args.evt_peak_column, "valley": args.evt_valley_column} if args.new_stack else {},
        "panel": args.panel, "start_date": args.start_date, "end_date": args.end_date,
        "n_stocks": stocks, "n_factors": factors, "obs_shape": [stocks * args.window * factors],
        "action_shape": [stocks], "stock_codes": list(panel.stock_codes),
        "factor_names": observation_names, "observation_names": observation_names,
        "execution_contract": {
            "signal": "T_close", "execution": "T+1_open", "mark": "T+1_close",
            "cost_bps": args.cost_bps, "slippage_bps": args.slippage_bps,
            "stop_loss_pct": args.stop_loss_pct, "cooldown_days": args.cooldown_days,
        },
        "parameter_count": parameter_count,
        "metrics_file": str(out_dir / "training_metrics.jsonl"),
        "tensorboard_dir": str(out_dir / "tb_logs"),
        "evt_target_loaded": bool(args.new_stack and evt_target_count > 0),
        "evt_target_count": evt_target_count,
        "hyperparams": hyperparams,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (out_dir / "training_summary.json").write_text(json.dumps({
        "algorithm": "PPO_V10_TIMINGFIX", "total_timesteps": args.total_timesteps,
        "n_stocks": stocks, "n_factors": factors, "parameter_count": parameter_count,
        "policy_class": policy_class.__name__, "mode": "new_stack" if args.new_stack else "baseline",
        "metrics_file": str(out_dir / "training_metrics.jsonl"),
        "tensorboard_dir": str(out_dir / "tb_logs"),
        "evt_target_loaded": bool(args.new_stack and evt_target_count > 0),
        "evt_target_count": evt_target_count,
        "aux_stats": getattr(model, "aux_stats", {}) if args.new_stack else {},
    }, ensure_ascii=False, indent=2))
    print(f"[timingfix] complete: {out_dir} evt_target_count={evt_target_count}", flush=True)


if __name__ == "__main__":
    main()
