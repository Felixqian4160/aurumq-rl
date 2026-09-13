"""v10.1 高低点标签引擎。

设计合同：
- ZigZag 峰/谷仍由现有、已验证的 `wavehunter_zigzag` 识别；
- A1/Peak 是中心点 ±1 个交易日的候选区间（最多 3 天）；
- 谷前一日若相对前一日跌幅 <= -3%，不把该日扩展为谷标签；
- 峰前一日若相对前一日涨幅 >= +3%，不把该日扩展为峰标签；
- 中心点永远保留；过滤只影响扩展边界，不改变 ZigZag 中心点；
- 旧 v9/v10 标签及面板不覆盖。

这些标签是事后监督目标，不是实时交易信号。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from aurumq_rl.wavehunter_zigzag import (
    ZigZagConfig,
    _detect_zigzag_single,
    _compute_r2,
    _max_drawdown_in_range,
)


@dataclass(frozen=True)
class TurningPointConfig:
    """v10.1 标签参数；参数固定后再构建正式面板。"""

    wave_threshold: float = 0.06
    min_pullback_days: int = 3
    neighbor_days: int = 1
    adjacent_move_threshold: float = 0.03
    a2_min_return: float = 0.05
    a2_max_drawdown: float = 0.12
    a2_min_r2: float = 0.30
    min_amount_pct: float = 0.05
    b1_narrow_pct: float = 0.20


def _valid_price(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & (x > 0)


def _expand_point(
    prices: np.ndarray,
    centers: np.ndarray,
    *,
    direction: str,
    neighbor_days: int,
    adjacent_move_threshold: float,
) -> np.ndarray:
    """扩展中心点 ±N；前一日出现方向性大波动时保留为0。

    注意：这里仅做标签边界约束，不以未来数据决定中心点是否存在。
    `direction='valley'` 对应前一日大跌排除，`peak` 对应前一日大涨排除。
    """
    n = len(prices)
    out = np.zeros(n, dtype=np.int8)
    valid = _valid_price(prices)
    for center in np.flatnonzero(centers):
        c = int(center)
        if not valid[c]:
            continue
        lo = max(0, c - int(neighbor_days))
        hi = min(n, c + int(neighbor_days) + 1)
        out[c] = 1
        for j in range(lo, hi):
            if j == c or not valid[j]:
                continue
            # 只排除中心点前一交易日；中心日和后一天不因该条件删除。
            if j == c - 1 and c >= 2 and valid[c - 2]:
                move = prices[j] / prices[j - 1] - 1.0
                if direction == "valley" and move <= -adjacent_move_threshold:
                    continue
                if direction == "peak" and move >= adjacent_move_threshold:
                    continue
            out[j] = 1
    return out


def compute_v10_1_labels(
    adj_close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    amount: Optional[np.ndarray] = None,
    cfg: Optional[TurningPointConfig] = None,
) -> dict[str, np.ndarray]:
    """计算 v10.1 独立峰谷/A1/A2/B1 标签。

    输入形状为 (T, S)。输出中的标签列均为 (T, S)，且未完成/无效位置为 -1。
    `zig_peak` 与 `zig_valley` 是中心点；`peak_zone`/`valley_zone` 是最多三日
    展开区间。A1 使用 valley_zone，但 A1 仍代表谷附近监督区域。
    """
    if cfg is None:
        cfg = TurningPointConfig()
    if adj_close.ndim != 2:
        raise ValueError("adj_close 必须是 (T,S) 二维数组")
    T, S = adj_close.shape
    if high is not None and high.shape != adj_close.shape:
        raise ValueError("high shape 必须与 adj_close 一致")
    if low is not None and low.shape != adj_close.shape:
        raise ValueError("low shape 必须与 adj_close 一致")
    if amount is not None and amount.shape != adj_close.shape:
        raise ValueError("amount shape 必须与 adj_close 一致")
    if cfg.neighbor_days < 0 or cfg.neighbor_days > 1:
        raise ValueError("v10.1 neighbor_days 固定允许 0 或 1")
    if cfg.adjacent_move_threshold <= 0:
        raise ValueError("adjacent_move_threshold 必须为正")

    zig_peak = np.zeros((T, S), dtype=np.int8)
    zig_valley = np.zeros((T, S), dtype=np.int8)
    peak_zone = np.full((T, S), -1, dtype=np.int8)
    valley_zone = np.full((T, S), -1, dtype=np.int8)
    a1_point = np.full((T, S), -1, dtype=np.int8)
    a2_interval = np.full((T, S), -1, dtype=np.int8)
    a2_start = np.zeros((T, S), dtype=np.int8)
    down_interval = np.full((T, S), -1, dtype=np.int8)
    down_start = np.zeros((T, S), dtype=np.int8)
    b1_interval = np.full((T, S), -1, dtype=np.int8)
    b1_start = np.zeros((T, S), dtype=np.int8)

    low_volume = None
    if amount is not None:
        with np.errstate(all="ignore"):
            avg = np.nanmean(amount, axis=0, keepdims=True)
            low_volume = (amount < avg * cfg.min_amount_pct) & np.isfinite(amount)

    for s in range(S):
        prices = adj_close[:, s].astype(float)
        valid = _valid_price(prices)
        if int(valid.sum()) < 10:
            continue
        peaks, valleys = _detect_zigzag_single(
            prices,
            high=high[:, s] if high is not None else None,
            low=low[:, s] if low is not None else None,
            wave_threshold=cfg.wave_threshold,
            min_pullback_days=cfg.min_pullback_days,
        )
        peak_i = np.asarray(peaks, dtype=bool)
        valley_i = np.asarray(valleys, dtype=bool)
        zig_peak[peak_i, s] = 1
        zig_valley[valley_i, s] = 1

        peak_z = _expand_point(
            prices, peak_i, direction="peak", neighbor_days=cfg.neighbor_days,
            adjacent_move_threshold=cfg.adjacent_move_threshold,
        )
        valley_z = _expand_point(
            prices, valley_i, direction="valley", neighbor_days=cfg.neighbor_days,
            adjacent_move_threshold=cfg.adjacent_move_threshold,
        )
        if low_volume is not None:
            peak_z[low_volume[:, s]] = 0
            valley_z[low_volume[:, s]] = 0
        peak_zone[:, s] = np.where(valid, peak_z, -1)
        valley_zone[:, s] = np.where(valid, valley_z, -1)
        a1_point[:, s] = valley_zone[:, s]

        peak_idx = np.flatnonzero(peak_i)
        valley_idx = np.flatnonzero(valley_i)
        for vi in valley_idx:
            if valley_z[vi] != 1:
                continue
            future = peak_idx[peak_idx > vi]
            if len(future) == 0:
                continue
            pj = int(future[0])
            if (_compute_r2(prices, int(vi), int(pj)) < cfg.a2_min_r2
                    or _max_drawdown_in_range(prices, int(vi), int(pj)) > cfg.a2_max_drawdown
                    or prices[pj] / prices[vi] - 1 < cfg.a2_min_return):
                continue
            a2_interval[vi:pj + 1, s] = 1
            a2_start[vi, s] = 1

        for pj in peak_idx:
            future = valley_idx[valley_idx > pj]
            if len(future) == 0:
                continue
            vj = int(future[0])
            down_interval[pj + 1:vj + 1, s] = 1
            down_start[pj, s] = 1
            span = vj - pj
            if span >= 5:
                narrow = max(3, int(span * cfg.b1_narrow_pct))
                b1_lo = vj - narrow + 1
                b1_interval[b1_lo:vj + 1, s] = 1
                b1_start[b1_lo, s] = 1

        active = valid & ~(low_volume[:, s] if low_volume is not None else False)
        a2_interval[active & (a2_interval[:, s] < 0), s] = 0
        down_interval[active & (down_interval[:, s] < 0), s] = 0
        b1_interval[active & (b1_interval[:, s] < 0), s] = 0

    return {
        "zig_peak": zig_peak,
        "zig_valley": zig_valley,
        "peak_zone": peak_zone,
        "valley_zone": valley_zone,
        "a1_point": a1_point,
        "a2_interval": a2_interval,
        "a2_start": a2_start,
        "down_interval": down_interval,
        "down_start": down_start,
        "b1_interval": b1_interval,
        "b1_start": b1_start,
    }
