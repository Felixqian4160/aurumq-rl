#!/usr/bin/env python3
"""v10_1 ZigZag 标签 (zig_peak/valley/A1/A2/B1) 脱敏窗口检验。

参考 scripts/v10_timingfix/label_validity_de_sensitized.py 的逻辑：
- 标签构造用了 day 1-N 信息（peak_zone/valley_zone 最多 3 天 + forward_period=20 监督窗口）
- 检验只测 day 21-40 / 21-60 这些"标签构造信息之外"的窗口
- 每个股票聚合为一个均值，再做 Mann-Whitney U 检验 + bootstrap 95% CI
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu

P_THRESHOLD = 0.01
BOOTSTRAP_N = 5000


def stock_test(main: np.ndarray, control: np.ndarray, seed: int) -> dict:
    """单只股票：标签组 vs 对照组的脱敏窗口平均收益 Mann-Whitney U + bootstrap CI."""
    u, p = mannwhitneyu(main, control, alternative="two-sided")
    rng = np.random.default_rng(seed)
    diffs = np.empty(BOOTSTRAP_N)
    for i in range(BOOTSTRAP_N):
        diffs[i] = rng.choice(main, len(main), replace=True).mean() - rng.choice(
            control, len(control), replace=True).mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    n1, n2 = len(main), len(control)
    return {
        "n_main": n1, "n_control": n2,
        "u": float(u), "p": float(p), "passed_p_0_01": bool(p < P_THRESHOLD),
        "median_main": float(np.median(main)) if n1 > 0 else None,
        "median_control": float(np.median(control)) if n2 > 0 else None,
        "mean_main": float(main.mean()) if n1 > 0 else None,
        "mean_control": float(control.mean()) if n2 > 0 else None,
        "mean_difference": float(main.mean() - control.mean()) if n1 > 0 and n2 > 0 else None,
        "rank_biserial": float(2.0 * u / (n1 * n2) - 1.0) if n1 > 0 and n2 > 0 else None,
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_ci95_lo": float(lo), "bootstrap_ci95_hi": float(hi),
        "bootstrap_ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def forward_window_returns(df: pl.DataFrame, lo: int, hi: int, col_ret: str) -> pl.DataFrame:
    """对每只股票在指定 [lo, hi) 区间内的累计收益取一个均值。

    Returns a DataFrame with columns: ts_code, window, mean_ret, hit_count.
    """
    df = df.sort(["ts_code", "trade_date"])
    # group-wise 滚动累计收益
    df = df.with_columns([
        (pl.col(col_ret).cum_sum().over(["ts_code"])).alias("cum_ret_to_now"),
    ])
    # 截取区间
    return df.select([
        pl.col("ts_code"),
        pl.col("cum_ret_to_now").alias("cum_ret"),
    ]).group_by("ts_code").agg([])


def build_label_groups(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
    """把每个 ts_code 上 label==1 的日期集合返回成 group, 然后对每只股票算 label-group vs control-group 的窗口收益."""
    # 用 day_offset (since first label date) 切窗口
    df = df.sort(["ts_code", "trade_date"])
    df = df.with_columns([
        pl.col("trade_date").rank().over(["ts_code"]).cast(pl.Int32).alias("day_idx"),
    ])
    # 找出每只股票第一个 label==1 的 day_idx
    label_days = (
        df.filter(pl.col(label_col) == 1)
        .group_by("ts_code")
        .agg(pl.col("day_idx").min().alias("label_day"))
    )
    df = df.join(label_days, on="ts_code", how="left")
    df = df.with_columns([
        (pl.col("day_idx") - pl.col("label_day")).alias("days_since_label"),
    ])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, help="v10_1 panel parquet")
    ap.add_argument("--label", required=True,
                    choices=["v10_1_zig_peak", "v10_1_zig_valley",
                             "v10_1_a1_point", "v10_1_a2_interval", "v10_1_b1_interval",
                             "v10_1_peak_zone", "v10_1_valley_zone"])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--ret-col", default="pct_chg",
                    help="日收益列 (默认 pct_chg)")
    ap.add_argument("--windows", default="21_40,21_60",
                    help="脱敏窗口 (days_since_label 区间, 半开)")
    ap.add_argument("--seed", type=int, default=20260913)
    args = ap.parse_args()

    df = pl.read_parquet(args.panel).with_columns([
        pl.col(args.ret_col).cast(pl.Float64).fill_null(0.0),
        pl.col(args.label).cast(pl.Int8).fill_null(0),
    ]).sort(["ts_code", "trade_date"])

    # day_idx
    df = df.with_columns([
        pl.col("trade_date").rank().over(["ts_code"]).cast(pl.Int32).alias("day_idx"),
    ])

    # 找出每只股票第一个 label==1 的 day_idx
    label_days = (
        df.filter(pl.col(args.label) == 1)
        .group_by("ts_code")
        .agg(pl.col("day_idx").min().alias("label_day"))
    )
    df = df.join(label_days, on="ts_code", how="left")
    df = df.with_columns([
        (pl.col("day_idx") - pl.col("label_day")).alias("days_since_label"),
    ])

    # 每只股票在每个窗口内的累计收益
    windows = {}
    for w in args.windows.split(","):
        lo, hi = map(int, w.split("_"))
        windows[f"day_{lo}_{hi}"] = (lo, hi)

    # stock_test 用 numpy；先把窗口切好
    label_panel = df.filter(pl.col(args.label) == 1).select(["ts_code", "days_since_label", args.ret_col])
    non_label_panel = df.filter(pl.col(args.label) != 1).select(["ts_code", "days_since_label", args.ret_col])

    per_window_results = {}
    overall_summary = {}

    for win_name, (lo, hi) in windows.items():
        main_per_stock = (
            df.filter((pl.col(args.label) == 1) &
                      (pl.col("days_since_label") >= lo) &
                      (pl.col("days_since_label") < hi))
            .group_by("ts_code")
            .agg(pl.col(args.ret_col).mean().alias("mean_ret"))
        )
        control_per_stock = (
            df.filter((pl.col(args.label) != 1) &
                      (pl.col("days_since_label") >= lo) &
                      (pl.col("days_since_label") < hi))
            .group_by("ts_code")
            .agg(pl.col(args.ret_col).mean().alias("mean_ret"))
        )

        main_arr = main_per_stock["mean_ret"].to_numpy()
        ctrl_arr = control_per_stock["mean_ret"].to_numpy()

        if len(main_arr) == 0 or len(ctrl_arr) == 0:
            per_window_results[win_name] = {"error": "empty group"}
            continue

        result = stock_test(main_arr, ctrl_arr, seed=args.seed + hash(win_name) % 10000)
        result["window"] = win_name
        result["label"] = args.label
        per_window_results[win_name] = result

        if result["passed_p_0_01"] and result["bootstrap_ci_excludes_zero"]:
            overall_summary.setdefault("passed_windows", []).append(win_name)
        else:
            overall_summary.setdefault("failed_windows", []).append(win_name)

    out = {
        "panel": args.panel,
        "label": args.label,
        "design": "stock_level_cluster_mean_de_sensitized_window_v10_1",
        "label_used_window": "day_0_to_3 (zone) + day_1_to_20 (forward_period监督)",
        "tested_windows": list(windows),
        "statistical_unit": "one stock, one mean pct_chg per label group per window",
        "p_threshold": P_THRESHOLD,
        "bootstrap_n": BOOTSTRAP_N,
        "overall": overall_summary,
        "per_window": per_window_results,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved → {args.out_json}")
    print()
    print("=" * 60)
    print(f"Label = {args.label}")
    print("=" * 60)
    for win_name, res in per_window_results.items():
        if "error" in res:
            print(f"  {win_name}: {res['error']}")
            continue
        passed = "✅" if res["passed_p_0_01"] else "❌"
        ci_ok = "✅" if res["bootstrap_ci_excludes_zero"] else "❌"
        print(f"  {win_name}: p={res['p']:.4f} {passed}  "
              f"CI=[{res['bootstrap_ci95_lo']:+.4f}, {res['bootstrap_ci95_hi']:+.4f}] {ci_ok}  "
              f"diff={res['mean_difference']:+.4f}  "
              f"n_main={res['n_main']} n_ctrl={res['n_control']}")


if __name__ == "__main__":
    main()
