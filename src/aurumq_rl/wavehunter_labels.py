#!/usr/bin/env python3
"""
wavehunter_labels.py — WaveHunter A1/A2 标签生成器
====================================================
目标: 把"未来 N 天上涨"定义成监督标签，教 LSTM encoder 识别主升浪前兆。

A1 主升浪左侧 (5 档 CE):
    future_ret[t] = adj_close[t+entry+window] / adj_close[t+entry] - 1
    按决策日 t 当天全市场截面分 5 档 (quantile 20/40/60/80)
    档4 = 未来最强 (种子) / 档0 = 未来最弱
    → cross-entropy 监督头学到"哪些股未来最可能大涨"

A2 主升浪右侧 (BCE 启动):
    启动信号 = 放量突破: pct_chg>bump_pct AND close>MA_ma AND vol>vol_mult*MA_vol
    未来 [t+entry, t+entry+confirm_window) 内出现 = A2=1
    → 二元头学到"启动前模式"（A1 左侧前兆 → A2 右侧确认）

设计约束（适配 autoresearch 自动寻优）:
  1. 全参数化 — 一切可调，不硬编码
  2. 向量化 — 单次重算 354股×20年 < 秒级，矩阵才扫得动
  3. 无未来函数 — 决策日 t 的 obs 只用 t 及以前; 标签用 t 之后(监督目标)
  4. 训练/验证隔离 — 标签只允许在训练期优化, 验证期一票否决
  5. 低量过滤 — amount < 个股均值×min_amount_pct 的 cell 标 label_invalid
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ════════════════════════════════════════════════════════════
# 配置（全参数化，autoresearch 可扫）
# ════════════════════════════════════════════════════════════
@dataclass
class WaveHunterLabelConfig:
    # ---- 通用 ----
    window: int = 20                 # 未来 N 交易日 (主升浪探测窗)
    entry_offset: int = 1            # 决策日 t → 入场 t+1 (防前视, 成交在次日)
    n_bins: int = 5                  # A1 分档数
    bins_mode: str = "quantile"      # 'quantile'(截面分位) | 'absolute'(固定阈值)
    min_amount_pct: float = 0.05     # 低量过滤: amount < 个股均值×5% → 无效

    # ---- A1 超额 vs 绝对收益方向 ----
    return_mode: str = "excess"      # 'excess'(相对000300指数) | 'absolute'(自身收益)

    # ---- A2 启动确认 ----
    bump_pct: float = 0.05           # 放量突破幅度阈值 (5%)
    ma_window: int = 10              # 突破 MA10
    vol_mult: float = 1.5            # 量 > 1.5×MA20 量
    vol_ma_window: int = 20
    confirm_window: int = 3          # 未来窗内启动后 N 天内确认

    # ---- 回归细化: 档间阈值 (quantile 自动算) ----
    abs_bins: tuple = (0.0, 0.05, 0.10, 0.20)  # bins_mode='absolute' 时用

    # ---- 输出控制 ----
    dtype: np.dtype = np.float32


# ════════════════════════════════════════════════════════════
# 向量化滚动均值 (行 = 时间)
# 算法: 前缀和差分, O(T*S) 一次 cumsum, 无逐行 Python 循环
# ════════════════════════════════════════════════════════════
def _rolling_mean_2d(a: np.ndarray, w: int) -> np.ndarray:
    T, S = a.shape
    cs = np.cumsum(np.nan_to_num(a, nan=0.0), axis=0, dtype=np.float64)
    cs_pad = np.vstack([np.zeros((1, S), dtype=np.float64), cs])
    # 窗口和: out[t] = sum(a[t-w+1..t])  (t>=w-1)
    win_sum = np.full((T, S), np.nan, dtype=np.float64)
    win_sum[w - 1:] = cs_pad[w:] - cs_pad[:-w]
    # 实际观测数 (部分窗口在前 w-1 行)
    counts = np.minimum(np.arange(1, T + 1), w).astype(np.float64)
    # 前 w-1 行: 前缀和 / 实际天数
    partial = cs / counts[:, None]
    out = np.where(np.isnan(win_sum), partial, win_sum / w)
    return out.astype(np.float32)


# ════════════════════════════════════════════════════════════
# A1 标签: 未来 window 天收益分档
# ════════════════════════════════════════════════════════════
def compute_a1_bins(
    adj_close: np.ndarray,
    entry: int,
    window: int,
    n_bins: int,
    idx_close: Optional[np.ndarray] = None,
    return_mode: str = "excess",
    bins_mode: str = "quantile",
    abs_bins: tuple = (0.0, 0.05, 0.10, 0.20),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 A1 分档标签。

    Parameters
    ----------
    adj_close : (T, S) 复权价
    entry, window : 入场偏移 & 探测窗
    idx_close : (T,) 000300 指数收盘 (return_mode='excess' 时需要)
    return_mode : 'excess' → 相对指数超额 / 'absolute' → 自身收益
    bins_mode : 'quantile' 截面分位 | 'absolute' 固定阈值
    abs_bins : bins_mode='absolute' 时的档间阈值

    Returns
    -------
    future_ret : (T, S) 决策日 t 处 = adj[t+entry+window]/adj[t+entry]-1
    bins        : (T, S) int 档位 0..n_bins-1 (无效处 = -1)
    label_valid : (T, S) bool 是否可评估
    """
    T, S = adj_close.shape
    a = adj_close.astype(np.float64)
    future_ret = np.full((T, S), np.nan, dtype=np.float64)

    # 决策日 t 有效范围: t+entry+window <= T  →  t <= T-entry-window
    n_valid = T - entry - window
    if n_valid > 0:
        # num[t] = a[t+entry+window], den[t] = a[t+entry]  (t in [0, n_valid))
        with np.errstate(divide='ignore', invalid='ignore'):
            num = a[entry + window: T]       # 长度 n_valid
            den = a[entry: T - window]       # 长度 n_valid
            future_ret[:n_valid] = num / den - 1.0

    # 标签有效: 未来窗完整 + 复权价 非0/非NaN (den>0 且 num>0)
    label_valid = ~np.isnan(future_ret)
    label_valid &= (a > 0)               # 决策日价格 > 0
    # num 侧 (未来价格) 也必须 > 0: 停牌填充 0 → future_ret=-1 是假信号
    if n_valid > 0:
        num_ok = np.full((T, S), False, dtype=bool)
        num_ok[:n_valid] = a[entry + window: T] > 0
        label_valid &= num_ok

    # 超额收益方向: 减去同期指数收益 (同样对齐)
    if return_mode == "excess" and idx_close is not None and len(idx_close) == T:
        idx_ret = np.full(T, np.nan)
        if n_valid > 0:
            idx_num = idx_close[entry + window: T]
            idx_den = idx_close[entry: T - window]
            idx_ret[:n_valid] = idx_num / idx_den - 1.0
        base = np.broadcast_to(idx_ret[:, None], future_ret.shape)
        future_ret = future_ret - base

    # ── 分档 ──
    bins = np.full((T, S), -1, dtype=np.int32)
    if n_bins <= 1:
        bins[label_valid] = 0
        return future_ret.astype(np.float32), bins, label_valid

    if bins_mode == "quantile":
        # 截面分位: 每决策日 t, 有效股票按收益切分位
        # 向量化: 用 np.nanquantile 逐行(行数 T, 每次 S 个值, T~5000 次 < 0.5s)
        qs_grid = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # 切 n_bins-1 刀
        # 逐行处理 (有效行才做)
        for t in range(T):
            row = future_ret[t]
            rv = label_valid[t]
            if not rv.any():
                continue
            qs = np.nanquantile(row[rv], qs_grid)
            # 避免重复阈值导致空档 (quantile 相同 → 档合并, 仍有效)
            bins[t, rv] = np.clip(np.digitize(row[rv], qs), 0, n_bins - 1)
    elif bins_mode == "absolute":
        # 固定阈值分档: 收益 <= abs_bins[0] → 0; > abs_bins[-1] → n_bins-1
        thresholds = np.array(abs_bins, dtype=np.float64)
        for t in range(T):
            rv = label_valid[t]
            if not rv.any():
                continue
            row = future_ret[t, rv]
            bins[t, rv] = np.clip(np.digitize(row, thresholds), 0, n_bins - 1)

    return future_ret.astype(np.float32), bins, label_valid


# ════════════════════════════════════════════════════════════
# A2 标签: 启动确认 (放量突破)
# ════════════════════════════════════════════════════════════
def compute_a2_start(
    pct_chg: np.ndarray,
    close: np.ndarray,
    vol: np.ndarray,
    entry: int,
    window: int,
    bump_pct: float,
    ma_window: int,
    vol_mult: float,
    vol_ma_window: int,
    confirm_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 A2 启动确认标签。

    A2[t] = 未来 [t+entry, t+entry+confirm_window) 内是否有启动事件。

    Returns
    -------
    a2_label : (T, S) int {0,1}; 0 = 未来窗内无启动
    a2_valid  : (T, S) bool 未来窗存在
    """
    T, S = close.shape
    ma_close = _rolling_mean_2d(close, ma_window)
    ma_vol = _rolling_mean_2d(vol, vol_ma_window)

    # 启动事件 (逐帧): 放量 + 突破 MA + 涨幅
    event = (pct_chg > bump_pct) & (close > ma_close) & (vol > vol_mult * ma_vol)

    # A2[t] = OR over event[t+entry : t+entry+confirm_window]
    # 向量化: 用滑动 OR (与 rolling mean 同理, 但用 max)
    a2 = np.zeros((T, S), dtype=np.int32)
    a2_valid = np.full((T, S), False, dtype=bool)

    # 未来窗 OR: 反向滚动 max (窗口 confirm_window, 中心 = t+entry)
    # 等价于: win_max[t] = max(event[t+entry : t+entry+confirm_window])
    # 用前缀技巧不可行(OR 无逆), 改为逐 t 但只对 event 稀疏行优化:
    # 直接用滑动窗口 max (numpy stride_tricks 或循环 T 次, T~5000*S~354 ≈ 1.8M 布尔 OR)
    # 简单可靠: 逐 t 切片 OR (T 次, 每次 confirm_window×S, 总 ~5k×3×354 ≈ 5.3M, <0.1s)
    # 只标记完整确认窗；尾部不足 confirm_window 天不能作为负样本
    ev = event.astype(np.int8)
    max_t = T - entry - confirm_window
    for t in range(max(0, max_t + 1)):
        lo = t + entry
        hi = lo + confirm_window
        a2[t] = ev[lo:hi].max(axis=0)
        a2_valid[t] = True
    return a2, a2_valid


# ════════════════════════════════════════════════════════════
# 主入口: 从面板张量 + 配置计算完整标签包
# ════════════════════════════════════════════════════════════
@dataclass
class WaveHunterLabels:
    future_ret: np.ndarray      # (T,S) 未来收益 (复权)
    a1_bins: np.ndarray         # (T,S) int 0..n_bins-1; -1 无效
    a1_valid: np.ndarray        # (T,S) bool
    a2_label: np.ndarray        # (T,S) int {0,1}; -1 无效
    a2_valid: np.ndarray        # (T,S) bool
    label_valid: np.ndarray     # (T,S) 通用有效性 (可训练/评估)
    config: WaveHunterLabelConfig


def compute_wavehunter_labels(
    adj_close: np.ndarray,
    pct_chg: np.ndarray,
    close: np.ndarray,
    vol: np.ndarray,
    cfg: WaveHunterLabelConfig,
    amount: Optional[np.ndarray] = None,
    idx_close: Optional[np.ndarray] = None,
) -> WaveHunterLabels:
    """从面板张量计算完整 A1/A2 标签包。

    Parameters
    ----------
    adj_close, pct_chg, close, vol : (T, S)
    cfg : 配置
    amount : (T,S) 可选, 低量过滤
    idx_close : (T,) 可选, 000300 指数 (超额收益方向)
    """
    T, S = adj_close.shape
    entry = cfg.entry_offset
    window = cfg.window

    # 未来收益 & A1 分档
    future_ret, a1_bins, _ = compute_a1_bins(
        adj_close, entry, window, cfg.n_bins,
        idx_close, cfg.return_mode, cfg.bins_mode, cfg.abs_bins,
    )

    # A2 启动
    a2_label, a2_valid = compute_a2_start(
        pct_chg, close, vol, entry, window,
        cfg.bump_pct, cfg.ma_window, cfg.vol_mult, cfg.vol_ma_window,
        cfg.confirm_window,
    )

    # 通用有效性: 决策日能评估 (未来窗完整 + 价格健康)
    label_valid = (~np.isnan(future_ret)) & (adj_close > 0)

    # 低量过滤: 与 data_loader.build_tradeable_mask 同源语义（铁律：训练/模拟一致）
    # 个股当日成交额 < 该股全历史均值×min_amount_pct → 极端缩量，不可成交
    if amount is not None:
        with np.errstate(all='ignore'):
            avg_amt = np.nanmean(amount, axis=0, keepdims=True)  # (1,S) 每只股票全历史均值
            low_volume = (amount < avg_amt * cfg.min_amount_pct) & np.isfinite(amount)
        label_valid &= ~low_volume

    # 无效 cell: A1/A2 标 -1
    a1_bins_out = np.where(label_valid, a1_bins, -1)
    a2_out = np.where(label_valid, a2_label, -1)

    return WaveHunterLabels(
        future_ret=future_ret.astype(np.float32),
        a1_bins=a1_bins_out,
        a1_valid=label_valid,
        a2_label=a2_out,
        a2_valid=a2_valid & label_valid,
        label_valid=label_valid,
        config=cfg,
    )
