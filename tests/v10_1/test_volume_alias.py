from __future__ import annotations

import datetime

import polars as pl

from build_factor_panel import add_derived_columns


def test_partial_vol_is_coalesced_from_volume() -> None:
    frame = pl.DataFrame(
        {
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "date": [
                datetime.datetime(2026, 1, 5),
                datetime.datetime(2026, 1, 6),
                datetime.datetime(2026, 1, 7),
            ],
            "close": [10.0, 10.2, 10.1],
            "amount": [100_000.0, 110_000.0, 105_000.0],
            "vol": [None, 2_000.0, None],
            "volume": [1_000.0, 2_000.0, 1_500.0],
        }
    )

    result = add_derived_columns(frame)

    assert result["vol"].to_list() == [1_000.0, 2_000.0, 1_500.0]
    assert result["volume"].to_list() == [1_000.0, 2_000.0, 1_500.0]
    assert result["vol"].null_count() == 0
    assert result["vwap"].to_list() == [1_000.0, 550.0, 700.0]
    assert "volume" in result.columns  # raw alias may remain in intermediate data
    assert result["amount"].to_list() == [100_000.0, 110_000.0, 105_000.0]
