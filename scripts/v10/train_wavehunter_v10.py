#!/usr/bin/env python3
"""train_wavehunter_v10.py — v10 独立多头输出训练入口

继承 frozen train_wavehunter_v9.py 全部功能，差异:
  - 用 WaveHunterV10Policy 替代 WaveHunterPolicy (新文件)
  - ONNX 导出用 forward_with_heads() 输出 5 个 head
  - 模型目录 models/whv10_*

零依赖修改 frozen 文件。
"""
from __future__ import annotations
import os, sys
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # v10/ for v10_train_adapter

import argparse, datetime, json, random, time
import numpy as np
import polars as pl

from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter, build_tradeable_mask
from aurumq_rl.v10.env import WaveHunterV10Env
from aurumq_rl.lstm_weight_env import LstmWeightConfig
from aurumq_rl.v10.policy import WaveHunterV10Policy
import importlib.util


def parse_args():
    p = argparse.ArgumentParser(description="Train WaveHunter v10 (multi-head ONNX)")
    p.add_argument("--panel", required=True)
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2023-12-31")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--universe-filter", default="main_board_non_st")
    p.add_argument("--n-factors", type=int, default=0)
    p.add_argument("--forward-period", type=int, default=20)
    p.add_argument("--max-position-pct", type=float, default=0.02)
    p.add_argument("--rebalance-days", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=15.0)
    p.add_argument("--total-timesteps", type=int, default=300000)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--lstm-hidden", type=int, default=64)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--reward-type", default="absolute_return_v3")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vec-normalize", action="store_true")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--aux-lambda", type=float, default=0.1)
    p.add_argument("--a1-lambda", type=float, default=1.0)
    p.add_argument("--a2-lambda", type=float, default=1.0)
    p.add_argument("--peak-lambda", type=float, default=1.5)
    p.add_argument("--b1-lambda", type=float, default=0.3)
    p.add_argument("--a1-pos-weight", type=float, default=20.0)
    p.add_argument("--a2-pos-weight", type=float, default=40.0)
    p.add_argument("--peak-pos-weight", type=float, default=80.0)
    p.add_argument("--b1-pos-weight", type=float, default=10.0)
    p.add_argument("--label-window", type=int, default=20)
    p.add_argument("--label-entry-days", type=int, default=1)
    p.add_argument("--label-n-bins", type=int, default=5)
    p.add_argument("--label-a2-threshold", type=float, default=0.10)
    p.add_argument("--label-a2-window", type=int, default=3)
    p.add_argument("--label-min-amount-pct", type=float, default=0.05)
    p.add_argument("--hit-rate-bonus-weight", type=float, default=0.0)
    p.add_argument("--continuation-bonus-weight", type=float, default=0.0)
    p.add_argument("--drawdown-penalty", type=float, default=0.0)
    p.add_argument("--excess-weight", type=float, default=0.0)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--hedge-ratio", type=float, default=0.0)
    p.add_argument("--signal-threshold", type=float, default=0.0)
    p.add_argument("--dynamic-stop-loss", action="store_true")
    p.add_argument("--use-evt-labels", action="store_true",
                   help="Append EVT causal labels to metadata/audit; does NOT replace ZigZag.")
    p.add_argument("--evt-peak-column", default="v10_evt_peak")
    p.add_argument("--evt-valley-column", default="v10_evt_valley")
    p.add_argument("--take-profit-pct", type=float, default=0.0)
    p.add_argument("--reward-w-abs", type=float, default=0.60)
    p.add_argument("--reward-w-dd", type=float, default=0.25)
    p.add_argument("--reward-w-hit", type=float, default=0.15)
    p.add_argument("--reward-norm-vol", type=float, default=0.02)
    p.add_argument("--bayes-vol-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bayes-vol-alpha0", type=float, default=2.0)
    p.add_argument("--bayes-vol-beta0", type=float, default=1e-4)
    p.add_argument("--bayes-vol-decay", type=float, default=0.98)
    p.add_argument("--bayes-vol-uncertainty-weight", type=float, default=0.50)
    p.add_argument("--bayes-vol-min", type=float, default=0.005)
    p.add_argument("--cvar-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cvar-window", type=int, default=252)
    p.add_argument("--cvar-alpha", type=float, default=0.95)
    p.add_argument("--cvar-warmup", type=int, default=30)
    p.add_argument("--cvar-normal-scale", type=float, default=0.5)
    p.add_argument("--cvar-tail-scale", type=float, default=1.0)
    p.add_argument("--cvar-max-penalty", type=float, default=2.0)
    p.add_argument("--hhi-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--hhi-weight", type=float, default=0.5)
    p.add_argument("--hhi-target-count", type=int, default=10)
    p.add_argument("--hhi-max-penalty", type=float, default=0.5)
    p.add_argument("--four-head-fusion-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--joint-four-head", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--joint-logit-scale", type=float, default=1.0)
    p.add_argument("--knn-enabled", action="store_true")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--label-version", default="v9", choices=["v9", "v2", "auto"])
    return p.parse_args()


def _pivot_col(df_all, col, dates, stock_codes):
    """复用 frozen 的 _pivot_col 逻辑 (独立副本)。"""
    if col not in df_all.columns:
        return None
    sub = df_all.select(["trade_date", "ts_code", col])
    try:
        sub = sub.with_columns(pl.col("trade_date").dt.strftime("%Y%m%d"))
    except Exception:
        sub = sub.with_columns(pl.col("trade_date").cast(pl.Utf8).str.slice(0, 8))
    sub = sub.group_by(["trade_date", "ts_code"]).agg(pl.col(col).mean())
    piv = sub.pivot(values=col, index="trade_date", on="ts_code").sort("trade_date")
    out = np.full((len(dates), len(stock_codes)), np.nan, dtype=np.float32)
    stock_idx = {s: i for i, s in enumerate(stock_codes)}
    date_idx = {}
    for i, d in enumerate(dates):
        date_idx[d.strftime("%Y%m%d") if hasattr(d, "strftime") else str(d)[:8]] = i
    for r in piv.iter_rows(named=True):
        td = str(r["trade_date"])[:8]
        t = date_idx.get(td)
        if t is None:
            continue
        for s_code, j in stock_idx.items():
            v = r.get(s_code)
            if v is not None:
                out[t, j] = float(v)
    return out


def main():
    args = parse_args()
    start_dt = datetime.date.fromisoformat(args.start_date)
    end_dt = datetime.date.fromisoformat(args.end_date)
    if start_dt >= end_dt:
        raise ValueError("start>=end")
    if end_dt > datetime.date.today():
        raise ValueError("end must not be in the future")
    if args.n_steps > 64:
        args.n_steps = 64
    if args.batch_size > 64:
        args.batch_size = 64
    if args.lstm_hidden > 64:
        args.lstm_hidden = 64
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _uf_map = {"main_board_non_st": UniverseFilter.MAIN_BOARD_NON_ST, "all_a": UniverseFilter.ALL_A,
               "hs300": UniverseFilter.HS300, "zz500": UniverseFilter.ZZ500}
    loader = FactorPanelLoader(parquet_path=args.panel)
    panel = loader.load_panel(start_dt, end_dt,
                              n_factors=args.n_factors if args.n_factors and args.n_factors > 0 else None,
                              forward_period=args.forward_period,
                              universe_filter=_uf_map.get(args.universe_filter, UniverseFilter.MAIN_BOARD_NON_ST))
    n_dates, n_stocks, n_factors = panel.factor_array.shape
    print(f"[wh9v2] 面板: {n_dates}天 × {n_stocks}股 × {n_factors}因子")

    df_all = pl.read_parquet(args.panel)
    stock_codes = panel.stock_codes
    dates = panel.dates
    required = ("v10_a1_point", "v10_a2_interval", "v10_zig_peak", "v10_b1_interval")
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(f"v10 面板缺少标签列: {missing}; 请使用 wavehunter_v10_*.parquet")
    print("[v10] 使用 v10 标签 (A1/A2/Peak/B1)")
    a1 = _pivot_col(df_all, "v10_a1_point", dates, stock_codes)
    a2 = _pivot_col(df_all, "v10_a2_interval", dates, stock_codes)
    peak_raw = _pivot_col(df_all, "v10_zig_peak", dates, stock_codes)
    b1_raw = _pivot_col(df_all, "v10_b1_interval", dates, stock_codes)
    if any(x is None for x in (a1, a2, peak_raw, b1_raw)):
        raise ValueError("v10 标签透视失败")
    a1_labels = np.where(np.isfinite(a1), a1.astype(np.int32), -1)
    a2_labels = np.where(np.isfinite(a2), a2.astype(np.int32), -1)
    peak_labels = np.where(np.isfinite(peak_raw), peak_raw.astype(np.int32), -1)
    b1_labels = np.where(np.isfinite(b1_raw), b1_raw.astype(np.int32), -1)
    if args.use_evt_labels:
        evt_peak_raw = _pivot_col(df_all, args.evt_peak_column, dates, stock_codes)
        evt_valley_raw = _pivot_col(df_all, args.evt_valley_column, dates, stock_codes)
        if evt_peak_raw is None or evt_valley_raw is None:
            evt_peak_labels = np.full((len(dates), len(stock_codes)), -1, dtype=np.int32)
            evt_valley_labels = np.full((len(dates), len(stock_codes)), -1, dtype=np.int32)
        else:
            evt_peak_labels = np.where(np.isfinite(evt_peak_raw), evt_peak_raw.astype(np.int32), -1)
            evt_valley_labels = np.where(np.isfinite(evt_valley_raw), evt_valley_raw.astype(np.int32), -1)
    else:
        evt_peak_labels = np.full((len(dates), len(stock_codes)), -1, dtype=np.int32)
        evt_valley_labels = np.full((len(dates), len(stock_codes)), -1, dtype=np.int32)
    print(f"[v10] Peak==1: {(peak_labels == 1).sum() / max((peak_labels >= 0).sum(), 1) * 100:.2f}%")
    print(f"[v10] B1==1: {(b1_labels == 1).sum() / max((b1_labels >= 0).sum(), 1) * 100:.2f}%")
    print(f"[v10] A1==1: {(a1_labels == 1).sum() / max((a1_labels >= 0).sum(), 1) * 100:.2f}%")
    print(f"[v10] A2==1: {(a2_labels == 1).sum() / max((a2_labels >= 0).sum(), 1) * 100:.2f}%")

    precomputed_mask = build_tradeable_mask(panel)
    print(f"[wh9v2] 可交易比例: {precomputed_mask.mean() * 100:.1f}%")

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    def _make_env(_rank):
        def _init():
            cfg = LstmWeightConfig(start_date=start_dt, end_date=end_dt, n_factors=n_factors,
                                   window=args.window, reward_type=args.reward_type, cost_bps=args.cost_bps,
                                   max_position_pct=args.max_position_pct, top_k=args.top_k,
                                   rebalance_days=args.rebalance_days,
                                   hit_rate_bonus_weight=args.hit_rate_bonus_weight,
                                   continuation_bonus_weight=args.continuation_bonus_weight,
                                   drawdown_penalty_weight=args.drawdown_penalty,
                                   excess_weight=args.excess_weight, reward_scale=args.reward_scale,
                                   hedge_ratio=args.hedge_ratio, signal_threshold=args.signal_threshold)
            cfg.reward_w_abs = args.reward_w_abs
            cfg.reward_w_dd = args.reward_w_dd
            cfg.reward_w_hit = args.reward_w_hit
            cfg.reward_norm_vol = args.reward_norm_vol
            cfg.__dict__.update({
                'bayes_vol_enabled': args.bayes_vol_enabled,
                'bayes_vol_alpha0': args.bayes_vol_alpha0,
                'bayes_vol_beta0': args.bayes_vol_beta0,
                'bayes_vol_decay': args.bayes_vol_decay,
                'bayes_vol_uncertainty_weight': args.bayes_vol_uncertainty_weight,
                'bayes_vol_min': args.bayes_vol_min,
                'cvar_enabled': args.cvar_enabled,
                'cvar_window': args.cvar_window,
                'cvar_alpha': args.cvar_alpha,
                'cvar_warmup': args.cvar_warmup,
                'cvar_normal_scale': args.cvar_normal_scale,
                'cvar_tail_scale': args.cvar_tail_scale,
                'cvar_max_penalty': args.cvar_max_penalty,
                'hhi_enabled': args.hhi_enabled,
                'hhi_weight': args.hhi_weight,
                'hhi_target_count': args.hhi_target_count,
                'hhi_max_penalty': args.hhi_max_penalty,
                })
            return WaveHunterV10Env(config=cfg, factor_panel=panel.factor_array,
                                   return_panel=panel.return_array,
                                   pct_change_panel=panel.pct_change_array,
                                   is_st_panel=panel.is_st_array,
                                   is_suspended_panel=panel.is_suspended_array,
                                   days_since_ipo_panel=panel.days_since_ipo_array,
                                   tradeable_mask=precomputed_mask,
                                   v9_a1_point=a1_labels, v9_a2_interval=a2_labels,
                                   v9_peak=peak_labels, v9_b1=b1_labels,
                                   evt_peak=evt_peak_labels, evt_valley=evt_valley_labels)
        return _init
    env_fns = [_make_env(i) for i in range(max(1, args.n_envs))]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn") if args.n_envs > 1 else DummyVecEnv(env_fns)
    if args.vec_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, clip_reward=10.0)

    # v10 使用独立训练循环，不导入 frozen scripts/train_loop.py
    from train_loop_v10 import V10PPO, V10Monitor
    if args.joint_four_head:
        from aurumq_rl.v10.joint_four_head_policy import JointWaveHunterV10Policy
        policy_cls = JointWaveHunterV10Policy
    else:
        policy_cls = WaveHunterV10Policy
    policy_kwargs = {"lstm_hidden": args.lstm_hidden, "lstm_layers": args.lstm_layers,
                     "_n_stocks": n_stocks, "_window": args.window, "_n_factors": n_factors,
                     "aux_lambda": args.aux_lambda, "a1_lambda": args.a1_lambda,
                     "a2_lambda": args.a2_lambda, "peak_lambda": args.peak_lambda,
                     "b1_lambda": args.b1_lambda,
                     "a1_pos_weight": args.a1_pos_weight, "a2_pos_weight": args.a2_pos_weight,
                     "peak_pos_weight": args.peak_pos_weight, "b1_pos_weight": args.b1_pos_weight}
    if args.joint_four_head:
        policy_kwargs.update({"joint_logit_scale": args.joint_logit_scale, "joint_use_softmax": True})
    model = V10PPO(policy=policy_cls, env=vec_env, learning_rate=args.learning_rate,
                   aux_lambda=args.aux_lambda, n_steps=args.n_steps, batch_size=args.batch_size,
                   n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                   max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                   seed=args.seed, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=V10Monitor(model, metrics_path=out_dir / "reward_metrics.jsonl"))
    ppo_path = out_dir / "ppo_final.zip"
    model.save(str(ppo_path))
    print(f"[wh9v2] 保存 {ppo_path}")

    # 多头 ONNX 导出
    try:
        import torch as _t
        _t.set_num_threads(1)
        policy = model.policy
        policy.eval(); policy.cpu()
        obs_shape = (n_stocks * args.window * n_factors,)
        dummy = _t.zeros(1, *obs_shape, dtype=_t.float32)
        onnx_out = out_dir / "policy.onnx"
        # 用 wrapper 显式调用 forward_with_heads；直接 export(policy) 只会调用
        # PPO 标准 forward 的 3 个输出 (action/value/log_prob)，无法导出 5 头。
        class _V10ExportWrapper(_t.nn.Module):
            def __init__(self, source):
                super().__init__()
                self.source = source

            def forward(self, observation):
                return self.source.forward_with_heads(observation)

        export_policy = _V10ExportWrapper(policy).eval()
        _t.onnx.export(export_policy, dummy, str(onnx_out), opset_version=14,
                       input_names=["observation"],
                       output_names=["action", "a1", "a2", "peak", "b1"],
                       dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"},
                                     "a1": {0: "batch"}, "a2": {0: "batch"},
                                     "peak": {0: "batch"}, "b1": {0: "batch"}},
                       export_params=True, dynamo=False)
        print(f"[wh9v2] 多头 ONNX {onnx_out} ({onnx_out.stat().st_size // 1024}KB)")
    except Exception as e:
        print(f"[wh9v2] ONNX 失败: {e}")
        import traceback; traceback.print_exc()

    metadata = {"algorithm": "PPO_V10", "model": "wavehunter_v10_4heads",
                "panel": str(args.panel), "reward_type": args.reward_type,
                "obs_shape": obs_shape, "action_shape": (n_stocks,),
                "reward_composite": {"w_abs": args.reward_w_abs, "w_dd": args.reward_w_dd,
                                    "w_hit": args.reward_w_hit, "norm_vol": args.reward_norm_vol},
                "bayesian_volatility": {"enabled": args.bayes_vol_enabled, "alpha0": args.bayes_vol_alpha0,
                                       "beta0": args.bayes_vol_beta0, "decay": args.bayes_vol_decay,
                                       "uncertainty_weight": args.bayes_vol_uncertainty_weight, "min_vol": args.bayes_vol_min},
                "cvar": {"enabled": args.cvar_enabled, "window": args.cvar_window, "alpha": args.cvar_alpha,
                         "warmup": args.cvar_warmup, "normal_scale": args.cvar_normal_scale,
                         "tail_scale": args.cvar_tail_scale, "max_penalty": args.cvar_max_penalty},
                "hhi": {"enabled": args.hhi_enabled, "weight": args.hhi_weight, "target_count": args.hhi_target_count,
                        "max_penalty": args.hhi_max_penalty},
                "evt_labels": {
                    "enabled": bool(args.use_evt_labels),
                    "peak_column": args.evt_peak_column,
                    "valley_column": args.evt_valley_column,
                } if args.use_evt_labels else {"enabled": False},
                "start_date": args.start_date, "end_date": args.end_date,
                "training_timesteps": args.total_timesteps,
                "n_stocks": n_stocks, "n_factors": n_factors,
                "stock_codes": list(panel.stock_codes),
                "factor_names": list(panel.factor_names),
                "hyperparams": vars(args),
                "v10_outputs": ["action", "a1", "a2", "peak", "b1"]}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (out_dir / "training_summary.json").write_text(json.dumps({
        "model": "wavehunter_v10_4heads", "panel": args.panel,
        "total_timesteps": args.total_timesteps, "n_stocks": n_stocks,
        "n_factors": n_factors,
        "aux_stats": getattr(model, "aux_stats", {})}, indent=2, ensure_ascii=False))
    print(f"[wh9v2] 完成: {out_dir}")


if __name__ == "__main__":
    main()
