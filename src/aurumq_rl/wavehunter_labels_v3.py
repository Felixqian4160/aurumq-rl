"""WaveHunter v3 双头标签 — 左侧埋伏 rebound + 右侧延续 continue.

与 v2 (a1_reversal_pre_signal / a2_start) 的区别:
  - A1_rebound: 未来20日 max_return ≥ wave_threshold, 且未来5日不再新低, 不含"今天就是底" hindsight
  - A2_continue: A1 之后 ≥ min_pullback_days 才出现, 且后续10日累计收益 ≥ a2_threshold

输出: (T, S) int32 array, 1=正样本 / 0=负样本 / -1=无效
"""
from __future__ import annotations

import numpy as np
import polars as pl


def compute_v3_labels(panel, df_all: pl.DataFrame,
                       wave_threshold: float = 0.10,
                       min_pullback_days: int = 5,
                       a2_threshold: float = 0.03,
                       forward_window: int = 20,
                       a2_holding_window: int = 10,
                       confirm_window: int = 5):
    """计算 v3 双头标签.

    Args:
        panel: FactorPanel (panel.return_array = (T, S))
        df_all: 原始 parquet (用于补 adj_close / vol)
        wave_threshold: A1 rebound 阈值 (未来20日 max_return ≥ 此值)
        min_pullback_days: A1 与 A2 之间最小天数 (强制时间分离)
        a2_threshold: A2 continue 阈值 (持仓期累计收益 ≥ 此值)
        forward_window: A1 预测窗口 (20天)
        a2_holding_window: A2 持有期窗口 (10天)
        confirm_window: A1 后确认窗口 (5日不新低)

    Returns:
        (a1_labels, a2_labels) 都是 (T, S) int32, 1=正样本 0=负样本 -1=无效
    """
    T, S = panel.return_array.shape
    dates = panel.dates
    stock_codes = panel.stock_codes

    # 用 adj_close 算未来收益 (更稳)
    if "adj_factor" in df_all.columns and panel.close_array is not None:
        adj_factor = _pivot_col(df_all, "adj_factor", dates, stock_codes)
        adj_close = panel.close_array * adj_factor
        adj_close = np.where(
            np.isnan(panel.close_array) | np.isnan(adj_factor), np.nan, adj_close
        )
    else:
        adj_close = panel.close_array

    # ── A1 rebound: 未来 20 日 max_return ≥ wave_threshold
    # ── 且未来 5 日不再新低 (防止假反弹)
    a1_labels = np.full((T, S), -1, dtype=np.int32)
    a2_labels = np.full((T, S), -1, dtype=np.int32)

    # 逐股票处理
    for s in range(S):
        c = adj_close[:, s]
        valid = np.isfinite(c)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < forward_window + confirm_window:
            continue

        # 未来 forward_window 日累计最大收益: max(c[t+1:t+1+forward_window]) / c[t] - 1
        fwd_max = np.full(T, np.nan)
        for t in range(T - forward_window):
            if not valid[t]:
                continue
            window = c[t + 1:t + 1 + forward_window]
            if not np.isfinite(window).all():
                continue
            m = np.nanmax(window)
            if m > 0 and np.isfinite(m):
                fwd_max[t] = m / c[t] - 1.0

        # 未来 confirm_window 日不再新低: min(c[t+1:t+1+confirm_window]) >= c[t] * (1 - buffer)
        fwd_min5 = np.full(T, np.nan)
        for t in range(T - confirm_window):
            if not valid[t]:
                continue
            window = c[t + 1:t + 1 + confirm_window]
            if not np.isfinite(window).all():
                continue
            mn = np.nanmin(window)
            if mn > 0 and np.isfinite(mn):
                fwd_min5[t] = mn / c[t] - 1.0

        # A1 = 1: max_ret >= wave_threshold AND 不新低
        rebound_mask = np.isfinite(fwd_max) & (fwd_max >= wave_threshold) \
                       & np.isfinite(fwd_min5) & (fwd_min5 >= -0.05)
        a1_labels[valid, s] = rebound_mask[valid].astype(np.int32)

        # ── A2 continue: A1 之后 ≥ min_pullback_days 且后续持有期累计 ≥ a2_threshold
        a1_indices = np.where(a1_labels[:, s] == 1)[0]
        for t_a1 in a1_indices:
            t_a2_start = t_a1 + min_pullback_days
            t_a2_end = t_a2_start + a2_holding_window
            if t_a2_end > T:
                continue
            if not valid[t_a2_start]:
                continue
            # 持有期累计收益: sum of return
            holding = panel.return_array[t_a2_start:t_a2_end, s]
            valid_h = np.isfinite(holding).all()
            if not valid_h:
                continue
            cum_ret = float(np.prod(1 + holding) - 1)
            if cum_ret >= a2_threshold:
                # 在 t_a2_start 时刻打 1 (而非 t_a1)
                a2_labels[t_a2_start, s] = 1

    # A2 负样本: 在 (a1_labels==1 后 min_pullback_days) 但不满足 continue 的位置填 0
    for s in range(S):
        a1_idx = np.where(a1_labels[:, s] == 1)[0]
        for t_a1 in a1_idx:
            t_a2_start = t_a1 + min_pullback_days
            if t_a2_start >= T:
                continue
            if a2_labels[t_a2_start, s] == -1:  # 还是无效
                continue
            # 如果 A1 触发了但 continue 没标 1 → 标 0 (负样本)
            if a2_labels[t_a2_start, s] != 1:
                a2_labels[t_a2_start, s] = 0

    return a1_labels, a2_labels


def _pivot_col(df_all: pl.DataFrame, col: str, dates, stock_codes) -> np.ndarray | None:
    if col not in df_all.columns:
        return None
    sub = df_all.select(["trade_date", "ts_code", col])
    sub = sub.with_columns(pl.col("trade_date").dt.strftime("%Y%m%d"))
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