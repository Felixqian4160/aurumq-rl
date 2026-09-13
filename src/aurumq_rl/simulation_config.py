"""模拟交易配置模块 — SimConfig/SimResult 数据类定义。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class SimConfig:
    """模拟交易配置。"""
    model_dir: str
    panel_path: str
    initial_capital: float = 100_000.0
    start_date: str = '2024-01-01'
    end_date: str = ''
    top_k: int = -1  # sentinel: -1 = metadata fallback
    cost_bps: float = -1.0  # sentinel: -1 = metadata fallback
    slippage_bps: float = -1.0  # sentinel: -1 = metadata fallback (0 = no slippage)
    rebalance_days: int = -1  # sentinel: -1 = metadata fallback
    stop_loss_pct: float = -1.0  # sentinel: -1 = metadata fallback (0 = no stop loss)
    max_position_pct: float = -1.0  # sentinel: -1 = metadata fallback
    max_holding_days: int = -1  # sentinel: -1 = metadata fallback (0 = no limit)
    # ── 市场中性对冲 ──
    hedge_ratio: float = 0.0  # 0=不对冲, 1.0=完全对冲beta
    # ── 信号过滤 ──
    signal_threshold: float = 0.0  # 0=不过滤, >0=信号强度低于此值时不调仓
    # ── 动态止损 ──
    dynamic_stop_loss: bool = False  # 是否根据波动率动态调整止损
    # ── 止盈 ──
    take_profit_pct: float = 0.0  # 0=不止盈, >0=持仓收益超过此阈值时止盈
    # ── v9 四段周期策略 ──
    v9_enabled: bool = False  # 是否启用 v9 四段周期策略
    v9_a1_threshold: float = 0.5  # A1 买入阈值
    v9_a2_threshold: float = 0.5  # A2 持有阈值
    v9_peak_threshold: float = 0.5  # Peak 卖出阈值
    v9_b1_threshold: float = 0.5  # B1 观望阈值


@dataclass
class SimResult:
    """模拟交易结果。"""
    status: str = 'idle'
    ledger_id: str = ''
    config: dict = field(default_factory=dict)
    nav_curve: list = field(default_factory=list)
    weekly_nav: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    holdings: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    progress: str = ''


# ── 全局状态 ──
_sim_result: SimResult = SimResult()
_sim_stop: bool = False


def get_sim_result() -> dict:
    """获取当前模拟结果。"""
    raw = asdict(_sim_result)
    return _sanitize(raw)


def _sanitize(obj: Any) -> Any:
    """清理 numpy 类型，确保 JSON 可序列化。"""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def stop_sim() -> None:
    """停止模拟。"""
    global _sim_stop
    _sim_stop = True


def _reset_sim() -> None:
    """重置模拟状态（内部使用）。"""
    global _sim_result, _sim_stop
    _sim_stop = False
    _sim_result = SimResult()
