"""WaveHunter v10 joint four-head smoke entry.

独立于 baseline 入口 train_wavehunter_v10.py，专门用于联合训练 smoke。
构造 SB3 PPO 时强制用 JointWaveHunterV10Policy，并通过 policy_factory
把已经实例化的 frozen WaveHunterV10Policy 注入 wrapper。
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import polars as pl
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter, build_tradeable_mask
from aurumq_rl.lstm_weight_env import LstmWeightConfig
from aurumq_rl.v10.env import WaveHunterV10Env
from aurumq_rl.v10.joint_four_head import JointFourHeadActionAdapter, JointFourHeadConfig
from aurumq_rl.v10.joint_four_head_policy import JointWaveHunterV10Policy
from aurumq_rl.v10.four_head_fusion import FourHeadActionFusion, FourHeadFusionConfig
from aurumq_rl.v10.metric_flattener import flatten_state

sys.path.insert(0, str(ROOT / "scripts" / "v10"))
import train_wavehunter_v10 as base_train


class WrapPolicyHook(BaseCallback):
    """Wrap the SB3-constructed frozen policy with the joint wrapper after setup_model."""

    def __init__(self, logit_scale: float = 1.0) -> None:
        super().__init__()
        self.logit_scale = logit_scale

    def _on_training_start(self) -> None:
        factory = self.model
        if hasattr(factory, "_frozen"):
            return
        wrapped = JointWaveHunterV10Policy(
            factory.policy.observation_space,
            factory.policy.action_space,
            getattr(factory.policy, "lr_schedule", None),
            use_joint_four_head=True,
            joint_logit_scale=self.logit_scale,
            joint_use_softmax=True,
            frozen_policy=factory.policy,
        )
        wrapped.optimizer = factory.policy.optimizer
        wrapped.scheduler = factory.policy.scheduler
        factory.policy = wrapped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/wavehunter_v10_hs300_20040102_20260804.parquet")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2023-12-31")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--total-timesteps", type=int, default=10000)
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lstm-hidden", type=int, default=32)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--joint-logit-scale", type=float, default=1.0)
    args = p.parse_args()

    panel_path = ROOT / args.panel
    loader = FactorPanelLoader(parquet_path=str(panel_path))
    panel = loader.load_panel(
        datetime.date.fromisoformat(args.start_date),
        datetime.date.fromisoformat(args.end_date),
        universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
        forward_period=20,
    )
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[joint] 面板: {n_dates}天 × {n_stocks}股 × {n_factors}因子")

    df_all = pl.read_parquet(str(panel_path))
    stock_codes = panel.stock_codes
    dates = panel.dates
    precomputed_mask = build_tradeable_mask(panel)

    def _pivot_col(df, col, dates, stock_codes):
        if col not in df.columns: return None
        sub = df.select(["trade_date", "ts_code", col])
        try: sub = sub.with_columns(pl.col("trade_date").dt.strftime("%Y%m%d"))
        except Exception: sub = sub.with_columns(pl.col("trade_date").cast(pl.Utf8).str.slice(0, 8))
        sub = sub.group_by(["trade_date", "ts_code"]).agg(pl.col(col).mean())
        piv = sub.pivot(values=col, index="trade_date", on="ts_code").sort("trade_date")
        out = np.full((len(dates), len(stock_codes)), np.nan, dtype=np.float32)
        s_idx = {s: i for i, s in enumerate(stock_codes)}
        d_idx = {}
        for i, d in enumerate(dates):
            d_idx[d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d)[:8]] = i
        for r in piv.iter_rows(named=True):
            td = str(r["trade_date"])[:8]
            t = d_idx.get(td)
            if t is None: continue
            for code, j in s_idx.items():
                v = r.get(code)
                if v is not None: out[t, j] = v
        return out

    a1 = _pivot_col(df_all, "v10_a1_point", dates, stock_codes)
    a2 = _pivot_col(df_all, "v10_a2_interval", dates, stock_codes)
    peak_raw = _pivot_col(df_all, "v10_zig_peak", dates, stock_codes)
    b1_raw = _pivot_col(df_all, "v10_b1_interval", dates, stock_codes)
    if any(x is None for x in (a1, a2, peak_raw, b1_raw)):
        raise ValueError("v10 panel missing labels")
    a1 = np.where(np.isfinite(a1), a1.astype(np.int32), -1)
    a2 = np.where(np.isfinite(a2), a2.astype(np.int32), -1)
    peak = np.where(np.isfinite(peak_raw), peak_raw.astype(np.int32), -1)
    b1 = np.where(np.isfinite(b1_raw), b1_raw.astype(np.int32), -1)

    def _make_env(rank):
        cfg = LstmWeightConfig(
            start_date=datetime.date.fromisoformat(args.start_date),
            end_date=datetime.date.fromisoformat(args.end_date),
            n_factors=n_factors, window=args.window, forward_period=20,
            reward_type="absolute_return_v3", reward_scale=10,
            rebalance_days=20, top_k=50,
        )
        return WaveHunterV10Env(
            config=cfg, factor_panel=panel.factor_array,
            return_panel=panel.return_array,
            pct_change_panel=panel.pct_change_array,
            is_st_panel=panel.is_st_array,
            is_suspended_panel=panel.is_suspended_array,
            days_since_ipo_panel=panel.days_since_ipo_array,
            tradeable_mask=precomputed_mask,
            v9_a1_point=a1, v9_a2_interval=a2, v9_peak=peak, v9_b1=b1,
        )

    vec_env = DummyVecEnv([lambda i=i: _make_env(i) for i in range(args.n_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "reward_metrics.jsonl"
    metrics_path.write_text("")

    from train_loop_v10 import V10PPO, V10Monitor

    from aurumq_rl.v10.policy import WaveHunterV10Policy
    model = V10PPO(
        policy=WaveHunterV10Policy, env=vec_env, learning_rate=args.learning_rate,
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        max_grad_norm=0.5, target_kl=0.02, seed=args.seed, verbose=1,
        policy_kwargs={"lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
                       "_n_stocks": n_stocks, "_window": args.window, "_n_factors": n_factors,
                       "aux_lambda": 0.1, "a1_lambda": 1.0, "a2_lambda": 1.0,
                       "peak_lambda": 1.5, "b1_lambda": 0.3,
                       "a1_pos_weight": 20.0, "a2_pos_weight": 40.0,
                       "peak_pos_weight": 80.0, "b1_pos_weight": 10.0},
    )

    # Replace SB3's frozen policy with our wrapper.
    wrapped_policy.optimizer = model.policy.optimizer
    model.policy = wrapped_policy

    # Force placeholder frozen policy onto the same device as the env tensor
    # so the LSTM parameters match the rollout obs device (CPU vs CUDA).
    device = model.device if hasattr(model, 'device') else 'cpu'
    if str(device) != 'cpu':
        wrapped_policy._frozen.to(device)

    monitor = _HookedMonitor(model, metrics_path=metrics_path)

    print(f"[joint] 开始训练 {args.total_timesteps} 步 (独立 train_v10_joint_smoke.py)")
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=monitor)
    print(f"[joint] 训练完成")


if __name__ == "__main__":
    main()
