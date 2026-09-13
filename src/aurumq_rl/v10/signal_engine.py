"""v10 模拟核心：只输出最终 buy/sell 信号，不改旧 simulation_core。

Head ↔ decision 映射通过 head_mapping.HEAD_MAPPING 单一真源维护；
新增/调整 head 必须先修改该模块，本文件不接受硬编码覆盖。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .head_mapping import HEAD_MAPPING, threshold_for

MASKED = tuple(name for name, spec in HEAD_MAPPING.items() if spec.target is None)
BUY_HEAD = next(name for name, spec in HEAD_MAPPING.items() if spec.decision == "buy")
SELL_HEAD = next(name for name, spec in HEAD_MAPPING.items() if spec.decision == "sell")

# Inference dict keys strip the "_head" suffix to remain compatible with
# MultiHeadInference.predict() which returns {a1, a2, peak, b1}.
def _to_key(head_name: str) -> str:
    return head_name.replace("_head", "")


BUY_KEY = _to_key(BUY_HEAD)
SELL_KEY = _to_key(SELL_HEAD)


@dataclass(frozen=True)
class V10Signal:
    buy: bool
    sell: bool
    reason: str


def decision_to_signal(probs: dict, buy_threshold=None, hold_threshold=None,
                       peak_threshold=None, b1_threshold=None) -> V10Signal:
    """Head 概率 → buy/sell/wait。

    由 head_mapping.HEAD_MAPPING 决定：
      * SELL_HEAD 概率 ≥ threshold → 卖出
      * BUY_HEAD  概率 ≥ threshold → 买入
      * 其余情形 → wait
    """
    if buy_threshold is None:
        buy_threshold = threshold_for(BUY_HEAD)
    if peak_threshold is None:
        peak_threshold = threshold_for(SELL_HEAD)
    buy_p = float(probs.get(BUY_KEY, 0.0))
    sell_p = float(probs.get(SELL_KEY, 0.0))
    if sell_p >= peak_threshold:
        return V10Signal(False, True, f'{SELL_HEAD}={sell_p:.3f}')
    if buy_p >= buy_threshold:
        return V10Signal(True, False, f'{BUY_HEAD}={buy_p:.3f},buy')
    return V10Signal(False, False, 'weak,wait')


def scores_from_heads(heads: dict, buy_threshold=None, hold_threshold=None,
                      peak_threshold=None, b1_threshold=None) -> np.ndarray:
    """按股票生成最终买入评分；SELL_HEAD 概率高的股票评分置零。"""
    if buy_threshold is None:
        buy_threshold = threshold_for(BUY_HEAD)
    if peak_threshold is None:
        peak_threshold = threshold_for(SELL_HEAD)
    buy_p = np.asarray(heads.get(BUY_KEY, np.zeros(0)), dtype=np.float32)
    sell_p = np.asarray(heads.get(SELL_KEY, np.zeros(0)), dtype=np.float32)
    if buy_p.size == 0:
        return buy_p
    score = buy_p.copy()
    score[sell_p >= peak_threshold] = 0.0
    score[buy_p < buy_threshold] = 0.0
    return score
