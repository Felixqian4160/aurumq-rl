#!/usr/bin/env python3
"""ZigZag label validity check via Mann-Whitney U test.

Compares the distribution of future-N-day returns for samples labeled as
``main_wave`` versus ``not_main_wave`` by ZigZag. Returns effect size and
p-value so that an upstream researcher can decide whether the label itself
has discriminative power before stacking EVT on top.

The script accepts two CSV or Parquet files of precomputed future returns
(per-row aligned with labels), so it stays independent of the runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_returns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"label column missing in {path}")
    if "future_return" not in df.columns:
        raise ValueError(f"future_return column missing in {path}")
    return df["future_return"].to_numpy(dtype=float), df["label"].to_numpy(dtype=int)


def _effect_size(u: float, n1: int, n2: int) -> float:
    denom = n1 * n2
    if denom == 0:
        return float("nan")
    return 1.0 - (2.0 * u) / denom


def mann_whitney_validity(
    returns: Iterable[float], labels: Iterable[int]
) -> dict:
    r = np.asarray(list(returns)); y = np.asarray(list(labels))
    pos = r[y == 1]; neg = r[y == 0]
    if len(pos) < 5 or len(neg) < 5:
        return {"passed": False, "reason": "insufficient samples", "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return {
        "passed": p < 0.01,
        "u_statistic": float(u),
        "p_value": float(p),
        "effect_size_1_minus_2u_over_n1n2": _effect_size(u, len(pos), len(neg)),
        "n_main_wave": int(len(pos)),
        "n_not_main_wave": int(len(neg)),
        "mean_future_return_main_wave": float(pos.mean()),
        "mean_future_return_not_main_wave": float(neg.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns", required=True, help="CSV/Parquet with future_return and label columns")
    parser.add_argument("--future-days", type=int, default=20)
    args = parser.parse_args()
    returns, labels = _load_returns(Path(args.returns))
    result = mann_whitney_validity(returns, labels)
    result["future_days"] = args.future_days
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
