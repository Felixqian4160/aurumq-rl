"""v10 模拟交易 — 四段周期策略

新链路:
  p_a1>τ → 买入
  p_a2>τ → 持有(不加仓)
  p_peak>τ → 卖出
  p_b1>τ → 观望(不动作)

兼容: 若 ONNX 缺 4 头 → 降级为 sigmoid(weights) top_k 选股 (等同 v9)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .inference import MultiHeadInference, RlAgentV10Inference

# 四段周期阈值 (默认基于训练集 v9_v2 后的经验值, 0.55 略高于随机的 0.5)
DEFAULT_THRESHOLDS = {
    'a1_buy':   0.55,  # A1 谷底概率 > 此值 → 买入
    'a2_hold':  0.55,  # A2 主升概率 > 此值 → 持有
    'peak_sell':0.50,  # Peak 顶部概率 > 此值 → 卖出 (敏感性高)
    'b1_wait':  0.55,  # B1 筑底概率 > 此值 → 观望
}


@dataclass
class V10Config:
    """v10 模拟配置 — 复用 wavehunter_sim_core 主体, 仅替换选股/换仓逻辑。"""
    a1_threshold: float = 0.55
    a2_threshold: float = 0.55
    peak_threshold: float = 0.50
    b1_threshold: float = 0.55
    # 行为: 强弱信号权重
    buy_strength_weight: float = 0.6    # A1 信号权重
    hold_strength_weight: float = 0.3   # A2 信号权重
    peak_strength_weight: float = 0.8   # Peak 卖出权重
    # 风控
    min_buy_score: float = 0.4         # 综合买入评分阈值 (避免弱信号)
    top_k_fallback: int = 20             # 缺 4 头时降级用


def four_cycle_decision(probs: dict, cfg: V10Config) -> dict:
    """根据 4 头概率给出交易意图。
    Returns: {'action': 'buy'|'hold'|'sell'|'wait', 'strength': float, 'stocks': ...}
    probs: {a1, a2, peak, b1}
    """
    p_a1   = float(probs.get('a1',   0.0))
    p_a2   = float(probs.get('a2',   0.0))
    p_peak = float(probs.get('peak', 0.0))
    p_b1   = float(probs.get('b1',   0.0))

    # 优先级: Peak > B1(wait) > A1/A2
    if p_peak >= cfg.peak_threshold:
        return {'action': 'sell', 'strength': p_peak * cfg.peak_strength_weight,
                'reason': f'Peak P={p_peak:.2f} >= τ={cfg.peak_threshold}'}
    if p_b1 >= cfg.b1_threshold and p_a1 < cfg.a1_threshold and p_a2 < cfg.a2_threshold:
        return {'action': 'wait', 'strength': p_b1,
                'reason': f'B1 P={p_b1:.2f} (筑底观望)'}
    if p_a1 >= cfg.a1_threshold:
        score = p_a1 * cfg.buy_strength_weight + p_a2 * cfg.hold_strength_weight
        return {'action': 'buy', 'strength': score, 'score': score,
                'reason': f'A1 P={p_a1:.2f} (谷底买入) A2={p_a2:.2f}'}
    if p_a2 >= cfg.a2_threshold:
        return {'action': 'hold', 'strength': p_a2,
                'reason': f'A2 P={p_a2:.2f} (主升持有)'}
    return {'action': 'wait', 'strength': max(p_a1, p_a2, p_b1),
            'reason': '信号弱 — 观望'}


def select_stocks_v10(probs: dict, action: str, cfg: V10Config,
                      tradeable_mask: np.ndarray, top_k: int = 20) -> np.ndarray:
    """根据 action + probs 选出股票 (返回 idx 数组)。
    - buy:  A1 概率 + A2 概率加权 (可选强谷底+主升)
    - hold: A2 概率排名
    - sell: 返回空 (全部卖出)
    - wait: 返回当前持仓 (无新进)
    """
    p_a1   = probs.get('a1',   np.zeros(len(tradeable_mask)))
    p_a2   = probs.get('a2',   np.zeros(len(tradeable_mask)))
    p_peak = probs.get('peak', np.zeros(len(tradeable_mask)))
    p_b1   = probs.get('b1',   np.zeros(len(tradeable_mask)))

    S = len(p_a1)
    mask = tradeable_mask.copy().astype(bool)
    valid_idx = np.where(mask)[0]

    if action == 'sell':
        return np.array([], dtype=np.int64)

    if action == 'wait':
        return np.array([], dtype=np.int64)

    # buy / hold: 综合评分 (排除 Peak/B1 高的)
    if action == 'buy':
        score = p_a1 * cfg.buy_strength_weight + p_a2 * cfg.hold_strength_weight
    else:  # hold
        score = p_a2

    # 排除 Peak/B1 高的 (避免买入即将反转)
    exclude = (p_peak >= cfg.peak_threshold) | (p_b1 >= cfg.b1_threshold)
    score = np.where(exclude, -np.inf, score)
    # 只在 valid mask 内
    score = np.where(mask, score, -np.inf)

    n_top = min(top_k, len(valid_idx))
    if n_top == 0:
        return np.array([], dtype=np.int64)

    # 阈值过滤
    buy_mask = score >= cfg.min_buy_score
    if buy_mask.sum() == 0:
        # 退而求其次: 取分数最高的 top_k (低阈值)
        idx_sorted = np.argsort(score)[::-1]
        return idx_sorted[idx_sorted < S][:n_top]

    idx_sorted = np.argsort(score)[::-1]
    top_idx = idx_sorted[idx_sorted < S][:n_top]
    return top_idx


def compute_scores_v10(probs: dict) -> np.ndarray:
    """统一评分: A1 买入意图 → A2 持有意图 → 0 (卖出/观望)
    返回 shape (S,) 的综合评分, 用于 simulation_core 的 top_k 选股
    """
    p_a1 = probs.get('a1', np.zeros(1))
    p_a2 = probs.get('a2', np.zeros(1))
    # buy 意图: A1 主, A2 次; hold 意图: A2
    return 0.6 * p_a1 + 0.4 * p_a2
