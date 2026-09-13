from datetime import date
import polars as pl

from aurumq_rl.v10_timingfix.market_data import load_execution_prices


def test_load_execution_prices_preserves_requested_order(tmp_path):
    path = tmp_path / "panel.parquet"
    pl.DataFrame({
        "trade_date": [date(2025, 1, 2), date(2025, 1, 2), date(2025, 1, 3)],
        "ts_code": ["B", "A", "A"],
        "open": [20.0, 10.0, 11.0],
        "close": [21.0, 10.5, 11.5],
    }).write_parquet(path)
    result = load_execution_prices(path, [date(2025, 1, 2), date(2025, 1, 3)], ["A", "B"])
    assert result.open_array[0, 0] == 10.0
    assert result.open_array[0, 1] == 20.0
    assert result.open_array[1, 0] == 11.0
    assert result.open_array[1, 1] != result.open_array[1, 1]
    assert result.close_array[0, 0] == 10.5
    assert result.open_array.shape == (2, 2)
