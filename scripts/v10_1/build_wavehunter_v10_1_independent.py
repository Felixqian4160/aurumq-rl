#!/usr/bin/env python3
"""WaveHunter v10.1 独立高低点面板构建器。

与 v10 独立面板一样从原始缓存构建；不读取 v8/v9/v10 面板，不覆盖旧产物。
输出: data/wavehunter_v10_1_{pool}_{start}_{end}.parquet
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_factor_panel as bfp  # noqa: E402
import build_wavehunter_panel as base  # noqa: E402
from aurumq_rl.v10_1.turning_points import TurningPointConfig, compute_v10_1_labels  # noqa: E402

DATA_DIR = ROOT / "data"


def _pivot(df: pl.DataFrame, column: str, dates: list, codes: list[str]) -> np.ndarray:
    wide = df.sort(["trade_date", "ts_code"]).pivot(
        index="trade_date", on="ts_code", values=column, aggregate_function="first"
    ).sort("trade_date")
    present = [c for c in wide.columns if c != "trade_date"]
    arr = wide.drop("trade_date").to_numpy().astype(np.float64)
    if present == codes:
        return arr
    out = np.full((len(dates), len(codes)), np.nan, dtype=np.float64)
    for j, code in enumerate(codes):
        if code in present:
            out[:, j] = arr[:, present.index(code)]
    return out


def _build_labels(df: pl.DataFrame) -> dict[str, np.ndarray]:
    dates = df["trade_date"].unique().sort().to_list()
    codes = df["ts_code"].unique().sort().to_list()
    ordered = df.sort(["trade_date", "ts_code"])
    return compute_v10_1_labels(
        _pivot(ordered, "adj_close", dates, codes),
        high=_pivot(ordered, "high", dates, codes),
        low=_pivot(ordered, "low", dates, codes),
        amount=_pivot(ordered, "amount", dates, codes),
        cfg=TurningPointConfig(
            wave_threshold=0.06,
            min_pullback_days=3,
            neighbor_days=1,
            adjacent_move_threshold=0.03,
            a2_min_return=0.05,
            a2_max_drawdown=0.12,
            a2_min_r2=0.30,
            min_amount_pct=0.05,
            b1_narrow_pct=0.20,
        ),
    )


def build_v10_1(pool: str, start_date: str, end_date: str, include_fundamental: bool = True) -> dict:
    started = time.time()
    bfp.add_log_callback(print)
    print(f"WaveHunter v10.1 独立面板构建: pool={pool} {start_date}~{end_date}")
    print("[1/4] 从原始缓存构建基础面板（不读取 v8/v9/v10）...")
    base_stats = bfp.build_pool(
        pool=pool, start_date=start_date, end_date=end_date,
        max_stocks=0, include_fundamental=include_fundamental,
    )
    base_path = Path(base_stats["path"])
    if not base_path.exists():
        raise FileNotFoundError(f"基础面板未生成: {base_path}")
    df = pl.read_parquet(str(base_path))
    required = {"trade_date", "ts_code", "close", "high", "low", "amount", "vol"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"v10.1 基础面板缺少字段: {missing}")
    df = base.ensure_ohlc_amount(df)
    df = base.add_adj_close(df)
    df = base.merge_turnover(df)
    df = base.merge_index_strength(df)
    required = {"trade_date", "ts_code", "adj_close", "high", "low", "amount"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"v10.1 后处理缺少字段: {missing}")
    print(f"[2/4] 基础面板: {len(df):,} 行 × {len(df.columns)} 列")
    print("[3/4] 计算 v10.1 峰谷标签（±1 日，前一日方向性3%过滤）...")
    labels = _build_labels(df)
    dates = df["trade_date"].unique().sort().to_list()
    codes = df["ts_code"].unique().sort().to_list()
    # 记录“前一日方向性3%过滤”的确切数量，供面板审计，不混入标签列。
    adj = _pivot(df.sort(["trade_date", "ts_code"]), "adj_close", dates, codes)
    raw_close = _pivot(df.sort(["trade_date", "ts_code"]), "close", dates, codes)
    peak_center = labels["zig_peak"] == 1
    valley_center = labels["zig_valley"] == 1
    valley_prev_drop = np.zeros_like(valley_center, dtype=bool)
    peak_prev_rise = np.zeros_like(peak_center, dtype=bool)
    if adj.shape[0] >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            move_prev = raw_close[1:] / raw_close[:-1] - 1.0
        # move_prev[k] = close[k+1]/close[k]-1；中心 t 的前一日波动是 move_prev[t-2]。
        valley_prev_drop[2:] = valley_center[2:] & (move_prev[:-1] <= -0.03)
        peak_prev_rise[2:] = peak_center[2:] & (move_prev[:-1] >= 0.03)
    flat = lambda key: labels[key].reshape(-1).astype(np.int8)
    new_cols = {
        "v10_1_zig_peak": flat("zig_peak"),
        "v10_1_zig_valley": flat("zig_valley"),
        "v10_1_peak_zone": flat("peak_zone"),
        "v10_1_valley_zone": flat("valley_zone"),
        "v10_1_a1_point": flat("a1_point"),
        "v10_1_a2_interval": flat("a2_interval"),
        "v10_1_a2_start": flat("a2_start"),
        "v10_1_b1_interval": flat("b1_interval"),
        "v10_1_b1_start": flat("b1_start"),
        "v10_1_down_interval": flat("down_interval"),
        "v10_1_down_start": flat("down_start"),
    }
    label_df = pl.DataFrame({
        "trade_date": np.repeat(np.array(dates, dtype="datetime64[ns]"), len(codes)),
        "ts_code": np.tile(np.array(codes), len(dates)),
        **new_cols,
    })
    print(
        f"  Peak中心={int((new_cols['v10_1_zig_peak'] == 1).sum())} "
        f"谷中心={int((new_cols['v10_1_zig_valley'] == 1).sum())} "
        f"峰区间={int((new_cols['v10_1_peak_zone'] == 1).sum())} "
        f"谷区间/A1={int((new_cols['v10_1_a1_point'] == 1).sum())}"
    )
    print(
        f"  前一日过滤: 谷中心排除={int(valley_prev_drop.sum())} "
        f"峰中心排除={int(peak_prev_rise.sum())}"
    )
    if np.array_equal(labels["b1_interval"], labels["down_interval"]):
        raise RuntimeError("v10.1 标签错误：B1 与 Down 完全相同")
    print("[4/4] 回填 v10.1 标签并保存...")
    df = df.join(label_df, on=["trade_date", "ts_code"], how="left")
    out = DATA_DIR / f"wavehunter_v10_1_{pool}_{start_date}_{end_date}.parquet"
    df.write_parquet(str(out), compression="zstd", statistics=True)
    stats = {
        "status": "ok", "version": "v10.1_independent", "pool": pool,
        "path": str(out), "filename": out.name, "rows": len(df),
        "columns": len(df.columns), "stock_count": df["ts_code"].n_unique(),
        "date_from": str(df["trade_date"].min()), "date_to": str(df["trade_date"].max()),
        "size_gb": round(out.stat().st_size / 1024**3, 3),
        "labels": {k: int((v == 1).sum()) for k, v in new_cols.items()},
        "label_config": {
            "neighbor_days": 1,
            "adjacent_move_threshold": 0.03,
            "valley_prior_day_drop_rule": "<= -3% excludes valley-1",
            "peak_prior_day_rise_rule": ">= +3% excludes peak-1",
            "valley_prior_day_excluded_centers": int(valley_prev_drop.sum()),
            "peak_prior_day_excluded_centers": int(peak_prev_rise.sum()),
            "semantic": "peak/valley center ±1 day with directional previous-day filter",
        },
        "elapsed_seconds": round(time.time() - started),
    }
    (DATA_DIR / f"build_stats_v10_1_{pool}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ v10.1 独立面板完成: {out} ({stats['elapsed_seconds']}s)")
    return stats


if __name__ == "__main__":
    pool = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("V10_1_POOL", "hs300")
    start = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("V10_1_START", "20040102")
    end = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("V10_1_END", "20260804")
    build_v10_1(pool, start, end, include_fundamental=True)
