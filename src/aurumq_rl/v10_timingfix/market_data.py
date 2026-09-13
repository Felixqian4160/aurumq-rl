"""Execution-price adapter for v10_timingfix.

Keeps raw open/close prices aligned to the already-loaded FactorPanel universe
without modifying the legacy data loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ExecutionPricePanel:
    dates: tuple[date, ...]
    stock_codes: tuple[str, ...]
    open_array: np.ndarray
    close_array: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.dates), len(self.stock_codes))
        if self.open_array.shape != expected or self.close_array.shape != expected:
            raise ValueError(
                f"price arrays must have shape {expected}, got "
                f"open={self.open_array.shape}, close={self.close_array.shape}"
            )


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def load_execution_prices(
    parquet_path: str | Path,
    dates: Sequence[date | datetime],
    stock_codes: Sequence[str],
) -> ExecutionPricePanel:
    """Load raw OHLC execution prices in the exact model panel order."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if not dates or not stock_codes:
        raise ValueError("dates and stock_codes must be non-empty")

    target_dates = tuple(_as_date(x) for x in dates)
    target_stocks = tuple(stock_codes)
    date_index = {d: i for i, d in enumerate(target_dates)}
    stock_index = {code: i for i, code in enumerate(target_stocks)}
    df = pl.read_parquet(
        path,
        columns=["trade_date", "ts_code", "open", "close"],
    )
    opens = np.full((len(target_dates), len(target_stocks)), np.nan, dtype=np.float64)
    closes = np.full_like(opens, np.nan)
    for row in df.iter_rows(named=True):
        raw_date = row["trade_date"]
        row_date = _as_date(raw_date)
        ti = date_index.get(row_date)
        si = stock_index.get(row["ts_code"])
        if ti is None or si is None:
            continue
        if row["open"] is not None:
            opens[ti, si] = float(row["open"])
        if row["close"] is not None:
            closes[ti, si] = float(row["close"])
    return ExecutionPricePanel(target_dates, target_stocks, opens, closes)
