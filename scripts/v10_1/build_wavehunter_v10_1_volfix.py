#!/usr/bin/env python3
"""Build a v10.1 panel with the vol/volume alias fixed.

This wrapper intentionally writes a new namespace and does not overwrite the
existing v10.1 panel. The underlying builder normalizes vol before vwap/factor
calculation: vol = coalesce(vol, volume), while amount remains turnover amount.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import build_factor_panel as bfp
import build_wavehunter_panel as base
import polars as pl


def build(pool: str, start: str, end: str) -> dict:
    # build_pool now uses the fixed add_derived_columns implementation and writes
    # only the base enhanced panel; the v10.1 labels are added below.
    stats = bfp.build_pool(
        pool=pool,
        start_date=start,
        end_date=end,
        max_stocks=0,
        include_fundamental=True,
    )
    base_path = Path(stats["path"])
    df = pl.read_parquet(str(base_path))
    df = base.ensure_ohlc_amount(df)
    df = base.add_adj_close(df)
    df = base.merge_turnover(df)
    df = base.merge_index_strength(df)
    required = {"trade_date", "ts_code", "close", "high", "low", "amount", "vol", "adj_close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"volfix base panel missing fields: {missing}")

    # Reuse the existing independent v10.1 label builder; labels are unchanged.
    from v10_1.build_wavehunter_v10_1_independent import _build_labels, _pivot
    import numpy as np

    dates = df["trade_date"].unique().sort().to_list()
    codes = df["ts_code"].unique().sort().to_list()
    labels = _build_labels(df)
    flat = lambda key: labels[key].reshape(-1).astype(np.int8)
    label_df = pl.DataFrame(
        {
            "trade_date": np.repeat(np.array(dates, dtype="datetime64[ns]"), len(codes)),
            "ts_code": np.tile(np.array(codes), len(dates)),
            **{
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
            },
        }
    )
    df = df.join(label_df, on=["trade_date", "ts_code"], how="left")
    # 输出只保留项目标准成交量字段 vol；volume 仅是上游别名，不进入最终面板。
    if "volume" in df.columns:
        df = df.drop("volume")
    out = ROOT / "data" / f"wavehunter_v10_1_volfix_{pool}_{start}_{end}.parquet"
    df.write_parquet(str(out), compression="zstd", statistics=True)
    result = {
        "status": "ok",
        "version": "v10.1_volfix",
        "path": str(out),
        "rows": df.height,
        "columns": len(df.columns),
        "stock_count": df["ts_code"].n_unique(),
        "date_from": str(df["trade_date"].min()),
        "date_to": str(df["trade_date"].max()),
        "vol_null_count": df["vol"].null_count(),
        "vol_nonnull_rate": df["vol"].is_not_null().mean(),
        "amount_null_count": df["amount"].null_count(),
        "amount_semantics": "成交额（千元）",
        "vol_semantics": "成交量（手）",
        "labels_unchanged_contract": True,
    }
    stats_path = ROOT / "data" / f"build_stats_v10_1_volfix_{pool}.json"
    stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    pool = sys.argv[1] if len(sys.argv) > 1 else "csi500"
    start = sys.argv[2] if len(sys.argv) > 2 else "20040102"
    end = sys.argv[3] if len(sys.argv) > 3 else "20260909"
    build(pool, start, end)
