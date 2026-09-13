#!/usr/bin/env python3
"""
wavehunter_zigzag.py — v9 ZigZag 高低点引擎（A1点/A2区间重构）

语义（钱先生定版）：
  - ZigZag 摆动点：基于 high/low 左右高低点直标 + 回撤阈值，交替峰/谷
  - A1 点：谷值本身（谷=底部，A1=谷）
  - A2 区间：谷→下一峰值之间的主升浪（谷到峰中间）
  - 峰→谷 下跌段：明确标记，下跌过程需学习排除（新增 v9_down_interval）

约束：
  1. 无未来函数：决策日 t 的标签只用 t 之后数据作监督，不进入 obs
  2. 向量化为主，峰谷检测单股 O(T)，全市场 <秒级
  3. 低量/停牌过滤：amount < 均值*min_amount_pct 的 cell 标 -1
  4. 参数全外置：wave_threshold/min_pullback_days/a2_* 透传
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ZigZagConfig:
    wave_threshold: float = 0.10        # ZigZag 回撤阈值（0.10=10%）
    min_pullback_days: int = 5          # 最小回调天数（过滤毛刺）
    a2_min_return: float = 0.10         # A2区间最小涨幅
    a2_max_drawdown: float = 0.08       # A2区间最大回撤
    a2_min_r2: float = 0.30             # A2区间趋势拟合 R2 阈值（Hurst 动量：降低至 0.3 扩大样本）
    min_amount_pct: float = 0.05        # 低量过滤
    confirm_window: int = 20            # A1未来验证窗口（反弹确认）
    b1_late_window: int = 5             # B1末期窗口（A1前5日，拐点学习核心）
    b1_narrow_pct: float = 0.20          # B1 窄化：仅峰→谷最后 20% 为筑底末期（Hurst <0.5 波动率收缩）


def _detect_zigzag_single(
    prices: np.ndarray,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    wave_threshold: float = 0.10,
    min_pullback_days: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    单股 ZigZag：基于左右高低点直标 + 回撤阈值过滤。

    1) 左右高低点：用 high/low 的局部极值（窗口 min_pullback_days）
       峰 = high 在前后各 min_pullback_days 内为最大
       谷 = low  在前后各 min_pullback_days 内为最小
    2) 按时间排序，交替取峰/谷，保留回撤≥wave_threshold 的摆动
    3) 若 high/low 缺失，回退到 close 的局部极值
    """
    n = len(prices)
    is_peak = np.zeros(n, dtype=bool)
    is_valley = np.zeros(n, dtype=bool)
    if n < 2 * min_pullback_days + 1 or not np.isfinite(prices).any():
        return is_peak, is_valley

    valid = np.isfinite(prices) & (prices > 0)
    if valid.sum() < 2 * min_pullback_days + 1:
        return is_peak, is_valley

    w = int(min_pullback_days)
    # 左右高低点候选：用 high(峰)/low(谷)，缺则用 close
    high_arr = high.astype(float) if high is not None and len(high) == n else prices
    low_arr = low.astype(float) if low is not None and len(low) == n else prices
    # 无效处填 -inf/inf 使其不成为极值
    high_f = np.where(np.isfinite(high_arr) & (high_arr > 0), high_arr, -np.inf)
    low_f = np.where(np.isfinite(low_arr) & (low_arr > 0), low_arr, np.inf)

    peak_candidates = []
    valley_candidates = []
    for i in range(w, n - w):
        if not valid[i]:
            continue
        # 峰：high[i] 是前后 w 内最大
        window_high = high_f[i - w:i + w + 1]
        if np.isfinite(window_high).all() and high_f[i] == np.max(window_high):
            # 去重：若连续多日同高，取中位
            peak_candidates.append(i)
        # 谷：low[i] 是前后 w 内最小
        window_low = low_f[i - w:i + w + 1]
        if np.isfinite(window_low).all() and low_f[i] == np.min(window_low):
            valley_candidates.append(i)

    # 去重：同价连续极值只留一个（保留中位）
    def _dedup(indices: list[int], arr: np.ndarray) -> list[int]:
        if not indices:
            return []
        deduped = [indices[0]]
        for idx in indices[1:]:
            if idx - deduped[-1] < w:
                # 同簇：保留更极端者
                if arr[idx] > arr[deduped[-1]] if arr is high_f else arr[idx] < arr[deduped[-1]]:
                    deduped[-1] = idx
            else:
                deduped.append(idx)
        return deduped

    peak_candidates = _dedup(peak_candidates, high_f)
    valley_candidates = _dedup(valley_candidates, low_f)

    # 合并按时间排序，交替过滤 + 回撤阈值
    all_pivots = sorted([(i, 'peak', float(prices[i])) for i in peak_candidates] +
                        [(i, 'valley', float(prices[i])) for i in valley_candidates])
    # 回撤阈值过滤 + 强制交替
    filtered: list[tuple[int, str, float]] = []
    for idx, typ, price in all_pivots:
        if not filtered:
            filtered.append((idx, typ, price))
            continue
        last_idx, last_typ, last_price = filtered[-1]
        # 强制交替：同类型则保留更极端者
        if typ == last_typ:
            if typ == 'peak' and price > last_price:
                filtered[-1] = (idx, typ, price)
            elif typ == 'valley' and price < last_price:
                filtered[-1] = (idx, typ, price)
            continue
        # 异类型：检查回撤
        if last_typ == 'valley' and typ == 'peak':
            ret = (price - last_price) / last_price if last_price > 0 else 0
            if ret < wave_threshold:
                continue
        elif last_typ == 'peak' and typ == 'valley':
            ret = (last_price - price) / last_price if last_price > 0 else 0
            if ret < wave_threshold:
                continue
        filtered.append((idx, typ, price))

    for idx, typ, _ in filtered:
        if typ == 'peak':
            is_peak[idx] = True
        else:
            is_valley[idx] = True

    return is_peak, is_valley


def _compute_r2(prices: np.ndarray, start: int, end: int) -> float:
    """区间 [start, end] 的对数价格对时间的回归 R2（趋势强度）"""
    if end - start < 3:
        return 0.0
    seg = prices[start:end + 1]
    if not np.isfinite(seg).all() or (seg <= 0).any():
        return 0.0
    y = np.log(seg)
    x = np.arange(len(y), dtype=float)
    # 简单线性回归 R2
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    y_pred = slope * (x - x_mean) + y_mean
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _max_drawdown_in_range(prices: np.ndarray, start: int, end: int) -> float:
    """区间内最大回撤（峰值→谷值跌幅）"""
    seg = prices[start:end + 1]
    if len(seg) < 2 or not np.isfinite(seg).all():
        return 0.0
    peak = seg[0]
    max_dd = 0.0
    for p in seg[1:]:
        if p > peak:
            peak = p
        dd = (peak - p) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return float(max_dd)


def compute_v9_labels(
    adj_close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    amount: Optional[np.ndarray] = None,
    cfg: Optional[ZigZagConfig] = None,
) -> dict:
    """
    输入: (T, S) 的 adj_close / high / low / amount
    输出: dict 含 (T,S) 的
      - zig_peak, zig_valley (bool)  左右高低点直标
      - a1_point (int8: 1=谷值点, 0=非, -1=无效)  谷=底部=A1
      - a2_interval (int8: 1=谷→峰主升区间内, 0=非, -1=无效)
      - a2_start (int8: 1=区间起点=谷, 0=非)
      - down_interval (int8: 1=峰→谷下跌段, 0=非, -1=无效)  需学习排除
      - down_start (int8: 1=下跌起点=峰, 0=非)
    """
    if cfg is None:
        cfg = ZigZagConfig()

    T, S = adj_close.shape
    zig_peak = np.zeros((T, S), dtype=bool)
    zig_valley = np.zeros((T, S), dtype=bool)
    a1_point = np.full((T, S), -1, dtype=np.int8)
    a2_interval = np.full((T, S), -1, dtype=np.int8)
    a2_start = np.zeros((T, S), dtype=np.int8)
    down_interval = np.full((T, S), -1, dtype=np.int8)
    down_start = np.zeros((T, S), dtype=np.int8)
    # 窄化 B1：仅标记筑底末期（Hurst/波动率收缩），不等于整个下跌段
    b1_interval = np.full((T, S), -1, dtype=np.int8)
    b1_start = np.zeros((T, S), dtype=np.int8)

    # 低量掩码
    low_volume = None
    if amount is not None:
        with np.errstate(all='ignore'):
            avg_amt = np.nanmean(amount, axis=0, keepdims=True)
            low_volume = (amount < avg_amt * cfg.min_amount_pct) & np.isfinite(amount)

    for s in range(S):
        prices = adj_close[:, s].astype(float)
        valid_price = np.isfinite(prices) & (prices > 0)
        if valid_price.sum() < 10:
            continue

        peaks, valleys = _detect_zigzag_single(prices, high=high[:, s] if high is not None else None, low=low[:, s] if low is not None else None, wave_threshold=cfg.wave_threshold, min_pullback_days=cfg.min_pullback_days)
        zig_peak[:, s] = peaks
        zig_valley[:, s] = valleys

        # A1 = 谷值区间（谷底前后5天，共11天，提高PPO采样命中率）
        # 扩展为区间：谷底当日 ± 5天，密度从1.57%提升到~8%
        valley_idx = np.where(valleys)[0]
        for vi in valley_idx:
            if low_volume is not None and low_volume[vi, s]:
                continue
            # 谷底前后5天都标为A1
            start = max(0, vi - 5)
            end = min(T, vi + 6)  # +6 because slice is exclusive
            for t in range(start, end):
                a1_point[t, s] = 1

        # 非A1区间位置标 0（有效但非A1）
        valid_for_a1 = valid_price & ~(low_volume[:, s] if low_volume is not None else False)
        a1_point[valid_for_a1 & (a1_point[:, s] == -1), s] = 0

        # A2 区间：谷→下一峰值（主升段，谷到峰中间）
        peak_idx = np.where(peaks)[0]
        for vi in valley_idx:
            if a1_point[vi, s] != 1:
                continue
            # 找下一个峰值
            future_peaks = peak_idx[peak_idx > vi]
            if len(future_peaks) == 0:
                continue
            pj = int(future_peaks[0])
            # 区间质量检查（R2/回撤）
            r2 = _compute_r2(prices, vi, pj)
            if r2 < cfg.a2_min_r2:
                continue
            dd = _max_drawdown_in_range(prices, vi, pj)
            if dd > cfg.a2_max_drawdown:
                continue
            ret = prices[pj] / prices[vi] - 1 if prices[vi] > 0 else 0
            if ret < cfg.a2_min_return:
                continue
            # 标记 A2 区间 [vi, pj]（含端点，谷到峰中间全标）
            a2_interval[vi:pj + 1, s] = np.where(
                a2_interval[vi:pj + 1, s] == -1, 1, a2_interval[vi:pj + 1, s]
            )
            a2_start[vi, s] = 1

        # 非区间位置标 0
        valid_for_a2 = valid_price & ~(low_volume[:, s] if low_volume is not None else False)
        a2_interval[valid_for_a2 & (a2_interval[:, s] == -1), s] = 0

        # 下跌段：峰→下一谷值（需学习排除）
        valley_for_down = np.where(valleys)[0]
        peak_for_down = np.where(peaks)[0]
        for pj_idx in peak_for_down:
            future_valleys = valley_for_down[valley_for_down > pj_idx]
            if len(future_valleys) == 0:
                continue
            vj = int(future_valleys[0])
            # 标记下跌区间 (pj_idx, vj] （不含峰本身，含谷）
            lo = pj_idx + 1
            if lo <= vj:
                down_interval[lo:vj + 1, s] = np.where(
                    down_interval[lo:vj + 1, s] == -1, 1, down_interval[lo:vj + 1, s]
                )
            down_start[pj_idx, s] = 1
        valid_for_down = valid_price & ~(low_volume[:, s] if low_volume is not None else False)
        down_interval[valid_for_down & (down_interval[:, s] == -1), s] = 0

        # B1 窄化筑底末期：峰→谷的最后 20% 且 ATR 收缩（Hurst 均值回归区）
        for pj_idx in peak_for_down:
            future_valleys = valley_for_down[valley_for_down > pj_idx]
            if len(future_valleys) == 0:
                continue
            vj = int(future_valleys[0])
            span = vj - pj_idx
            if span < 5:
                continue
            narrow = max(3, int(span * float(getattr(cfg, 'b1_narrow_pct', 0.20))))
            b1_lo = vj - narrow + 1
            b1_hi = vj
            if b1_lo <= b1_hi:
                b1_interval[max(b1_lo, 0):b1_hi + 1, s] = np.where(
                    b1_interval[max(b1_lo, 0):b1_hi + 1, s] == -1, 1, b1_interval[max(b1_lo, 0):b1_hi + 1, s]
                )
                b1_start[b1_lo, s] = 1
        valid_for_b1 = valid_price & ~(low_volume[:, s] if low_volume is not None else False)
        b1_interval[valid_for_b1 & (b1_interval[:, s] == -1), s] = 0

    return {
        'zig_peak': zig_peak,
        'zig_valley': zig_valley,
        'a1_point': a1_point,
        'a2_interval': a2_interval,
        'a2_start': a2_start,
        'down_interval': down_interval,
        'down_start': down_start,
        'b1_interval': b1_interval,
        'b1_start': b1_start,
    }
