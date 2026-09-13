"""模拟交易基准加载模块。"""
from __future__ import annotations

import os
from pathlib import Path


def load_csi300_benchmark(panel_dates: list, start_date: str, end_date: str) -> dict:
    """Load a CSI300 benchmark from an explicitly configured local path.

    The library never assumes a developer-specific filesystem layout. Set
    ``CSI300_BENCHMARK_PATH`` or adapt this loader to your own data pipeline.
    """
    import pandas as pd

    configured = os.environ.get("CSI300_BENCHMARK_PATH")
    candidates = [Path(configured)] if configured else []

    for p in candidates:
        if Path(p).exists():
            try:
                df = pd.read_parquet(p)
                date_col = 'trade_date' if 'trade_date' in df.columns else 'date' if 'date' in df.columns else None
                if date_col and 'close' in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
                    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].sort_values(date_col)
                    # 用 close 价格直接算日收益率，避免 pct_chg 精度问题
                    closes = df['close'].values.astype(float)
                    dates = df[date_col].values
                    result = {}
                    for i in range(1, len(closes)):
                        if closes[i-1] > 0:
                            result[str(dates[i])] = (closes[i] / closes[i-1]) - 1.0
                    return result
            except Exception:
                pass
    return {}
