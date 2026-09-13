#!/usr/bin/env python3
"""Build causal-label Walk-Forward training panels from the frozen volfix panel.

For each fold, labels are recomputed using only rows through train_end. The last
20 trading dates are purged because their future-confirmed labels are not mature.
Factor values and raw fields are copied only for the requested training window.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SOURCE = ROOT / "data/wavehunter_v10_1_volfix_csi500_20040102_20260909.parquet"
LABELS = [
    "v10_1_zig_peak", "v10_1_zig_valley", "v10_1_peak_zone",
    "v10_1_valley_zone", "v10_1_a1_point", "v10_1_a2_interval",
    "v10_1_a2_start", "v10_1_b1_interval", "v10_1_b1_start",
    "v10_1_down_interval", "v10_1_down_start",
]
RAW_LABEL_INPUTS = ["trade_date", "ts_code", "adj_close", "high", "low", "amount"]


def _pivot(df: pl.DataFrame, column: str, dates: list, codes: list[str]) -> np.ndarray:
    wide = df.select(["trade_date", "ts_code", column]).pivot(
        index="trade_date", on="ts_code", values=column, aggregate_function="first"
    ).sort("trade_date")
    present = [c for c in wide.columns if c != "trade_date"]
    values = wide.drop("trade_date").to_numpy().astype(np.float64)
    out = np.full((len(dates), len(codes)), np.nan, dtype=np.float64)
    col_idx = {c: i for i, c in enumerate(present)}
    for j, code in enumerate(codes):
        if code in col_idx:
            out[:, j] = values[:, col_idx[code]]
    return out


def _purge_label_tail(labels: dict[str, np.ndarray], n_dates: int, purge_days: int) -> None:
    if purge_days <= 0:
        return
    start = max(0, n_dates - purge_days)
    # compute_v10_1_labels returns internal keys without the v10_1_ prefix;
    # the prefix is added only when writing the fold parquet.
    for key in (
        "zig_peak", "zig_valley", "peak_zone", "valley_zone", "a1_point",
        "a2_interval", "a2_start", "b1_interval", "b1_start",
        "down_interval", "down_start",
    ):
        if key in labels:
            labels[key][start:, :] = -1


def build_fold(fold_id: str, train_start: str, train_end: str, purge_days: int = 20) -> dict:
    train_start_d = dt.date.fromisoformat(train_start)
    train_end_d = dt.date.fromisoformat(train_end)
    if train_start_d >= train_end_d:
        raise ValueError("train_start must be before train_end")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    schema_sample = pl.scan_parquet(str(SOURCE)).head(1).collect()
    from aurumq_rl.data_loader import discover_factor_columns
    factor_names = discover_factor_columns(schema_sample)
    if len(factor_names) != 328:
        raise ValueError(f"expected 328 factors, got {len(factor_names)}")
    if "volume" in schema_sample.columns:
        raise ValueError("source volfix panel must not contain volume alias")

    # Prefix through train_end is the only input to retrospective label creation.
    prefix = (
        pl.scan_parquet(str(SOURCE))
        .filter(pl.col("trade_date") <= pl.lit(dt.datetime.combine(train_end_d, dt.time())))
        .select(RAW_LABEL_INPUTS)
        .collect()
    )
    all_dates = prefix["trade_date"].unique().sort().to_list()
    all_codes = prefix["ts_code"].unique().sort().to_list()
    if not all_dates or not all_codes:
        raise ValueError("empty causal prefix")
    labels_input = {
        "adj_close": _pivot(prefix, "adj_close", all_dates, all_codes),
        "high": _pivot(prefix, "high", all_dates, all_codes),
        "low": _pivot(prefix, "low", all_dates, all_codes),
        "amount": _pivot(prefix, "amount", all_dates, all_codes),
    }
    sys.path.insert(0, str(ROOT / "src"))
    from aurumq_rl.v10_1.turning_points import TurningPointConfig, compute_v10_1_labels
    labels = compute_v10_1_labels(
        labels_input["adj_close"],
        high=labels_input["high"],
        low=labels_input["low"],
        amount=labels_input["amount"],
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
    train_dates = [x for x in all_dates if train_start_d <= x.date() <= train_end_d]
    if len(train_dates) <= purge_days:
        raise ValueError("training window too short for purge")
    _purge_label_tail(labels, len(all_dates), purge_days)

    # Copy only the train-window factor/raw rows from the frozen source panel.
    output_base = list(dict.fromkeys(
        ["ts_code", "trade_date", "close", "pct_chg", "vol", "amount", "open", "high", "low", "adj_factor", "adj_close"]
        + factor_names
    ))
    train = (
        pl.scan_parquet(str(SOURCE))
        .filter(
            (pl.col("trade_date") >= pl.lit(dt.datetime.combine(train_start_d, dt.time())))
            & (pl.col("trade_date") <= pl.lit(dt.datetime.combine(train_end_d, dt.time())))
        )
        .select(output_base)
        .collect()
    )
    train_codes = train["ts_code"].unique().sort().to_list()
    date_index = {d: i for i, d in enumerate(all_dates)}
    code_index = {c: i for i, c in enumerate(all_codes)}
    date_idx = [date_index[d] for d in train["trade_date"].unique().sort().to_list()]
    code_idx = [code_index[c] for c in train_codes]
    label_dates = train["trade_date"].unique().sort().to_list()
    label_data = {
        "trade_date": np.repeat(np.array(label_dates, dtype="datetime64[ns]"), len(train_codes)),
        "ts_code": np.tile(np.array(train_codes), len(label_dates)),
    }
    label_key_map = {
        "v10_1_zig_peak": "zig_peak",
        "v10_1_zig_valley": "zig_valley",
        "v10_1_peak_zone": "peak_zone",
        "v10_1_valley_zone": "valley_zone",
        "v10_1_a1_point": "a1_point",
        "v10_1_a2_interval": "a2_interval",
        "v10_1_a2_start": "a2_start",
        "v10_1_b1_interval": "b1_interval",
        "v10_1_b1_start": "b1_start",
        "v10_1_down_interval": "down_interval",
        "v10_1_down_start": "down_start",
    }
    for output_key, source_key in label_key_map.items():
        selected = labels[source_key][np.ix_(date_idx, code_idx)]
        label_data[output_key] = selected.reshape(-1).astype(np.int8)
    label_df = pl.DataFrame(label_data)
    output = train.drop([c for c in LABELS if c in train.columns]).join(
        label_df, on=["trade_date", "ts_code"], how="left"
    )
    out_path = ROOT / "data" / f"wavehunter_v10_1_volfix_{fold_id}_{train_start.replace('-', '')}_{train_end.replace('-', '')}.parquet"
    output.write_parquet(str(out_path), compression="zstd", statistics=True)

    # Hard checks: no post-cutoff source rows, canonical volume, and purged tail.
    if output["vol"].null_count() != 0:
        raise AssertionError("volfix output contains null vol")
    if "volume" in output.columns:
        raise AssertionError("volume alias leaked into fold panel")
    tail_dates = output["trade_date"].unique().sort().to_list()[-purge_days:]
    tail = output.filter(pl.col("trade_date").is_in(tail_dates))
    tail_bad = sum(int((tail[c] != -1).sum()) for c in LABELS)
    if tail_bad != 0:
        raise AssertionError(f"purge failed: {tail_bad} non--1 labels")
    result = {
        "status": "ok",
        "fold_id": fold_id,
        "source_panel": str(SOURCE.relative_to(ROOT)),
        "source_panel_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "path": str(out_path.relative_to(ROOT)),
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "train_start": train_start,
        "train_end": train_end,
        "label_prefix_end": train_end,
        "purge_days": purge_days,
        "rows": output.height,
        "dates": output["trade_date"].n_unique(),
        "stocks": output["ts_code"].n_unique(),
        "factors": len(factor_names),
        "factor_names": factor_names,
        "label_positive_counts": {c: int((output[c] == 1).sum()) for c in LABELS},
        "purged_tail_label_nonminus1": tail_bad,
        "vol_null_count": output["vol"].null_count(),
        "has_volume_alias": "volume" in output.columns,
    }
    stats_path = ROOT / "data" / f"build_stats_v10_1_volfix_{fold_id}.json"
    stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: build_walkforward_fold_panel.py FOLD_ID TRAIN_START TRAIN_END")
    print(json.dumps(build_fold(sys.argv[1], sys.argv[2], sys.argv[3]), ensure_ascii=False, indent=2))
