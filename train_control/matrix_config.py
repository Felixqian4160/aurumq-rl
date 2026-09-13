"""矩阵配置模块 — FIXED_TRAIN/FIXED_SIM/POOL_TRAIN_PROFILES 常量定义。"""
from __future__ import annotations

from pathlib import Path


_AURUMQ_ROOT = Path(__file__).resolve().parent.parent

POOLS = {
    "hs300": "wavehunter_v8_hs300_20040102_20260804.parquet",
    "csi500": "wavehunter_v8_csi500_20040102_20260827.parquet",
    "cs800": "wavehunter_v8_cs800_20040102_20260827.parquet",
}

SPANS = {
    "12m": ("2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    "18m": ("2022-07-01", "2023-12-31", "2024-01-01", "2025-06-30"),
    "24m": ("2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
}

STEPS = [100_000, 200_000, 300_000, 500_000, 700_000, 1_000_000]
SMOKE_STEPS = [10_000, 50_000, 100_000]
FULL_STEPS = [300_000, 500_000, 1_000_000]
PHASE_TIPS = {
    "smoke": "冒烟阶段 9 run (3跨度 × 3步数)，先看 Sharpe/超额趋势再决定是否开 full",
    "full": "正式阶段 9 run (3跨度 × 3步数)，耗时更长，结果需结合 smoke 解读",
}

FIXED_TRAIN = dict(
    window=20, top_k=20, universe_filter="main_board_non_st", n_factors=0,
    forward_period=20, max_position_pct=0.02, rebalance_days=20, cost_bps=15.0,
    learning_rate=1e-4, lstm_hidden=128, lstm_layers=1, reward_type="absolute_return_v3",
    n_envs=1, seed=42, max_grad_norm=1.0, target_kl=0.02, batch_size=64,
    n_steps=64, aux_lambda=0.1, a1_lambda=1.0, a2_lambda=1.0,
    a1_pos_weight=3.0, a2_pos_weight=3.0, label_window=20, label_entry_days=1, label_n_bins=5,
    label_a2_threshold=0.10, label_a2_window=3, label_min_amount_pct=0.05,
    wave_threshold=0.10, min_pullback_days=5,
    # v4 reward: 绝对收益为主, 弱超额辅助, 信号奖励, 分段回撤惩罚
    hit_rate_bonus_weight=0.3, continuation_bonus_weight=0.2, drawdown_penalty=0.1, excess_weight=0.5,
    # ── 市场中性对冲 ──
    hedge_ratio=0.0,
    # ── 信号过滤 ──
    signal_threshold=0.0,
    # ── 动态止损 ──
    dynamic_stop_loss=False,
    # ── 止盈 ──
    take_profit_pct=0.0,
)

FIXED_SIM = dict(
    initial_capital=100_000.0, top_k=20, cost_bps=15.0, slippage_bps=10.0,
    rebalance_days=20, stop_loss_pct=8.0, max_position_pct=0.02,
    max_holding_days=120,
)

# 每个股票池独立的显存安全起始配置；正式参数由10k smoke test逐级确定。
POOL_TRAIN_PROFILES = {
    "hs300": dict(FIXED_TRAIN, lstm_hidden=64, batch_size=32, n_steps=32, memory_profile="hs300_v2_safe"),
    "csi500": dict(FIXED_TRAIN, n_steps=16, batch_size=16, lstm_hidden=64, memory_profile="csi500_safe_start"),
    "cs800": dict(FIXED_TRAIN, n_steps=8, batch_size=8, lstm_hidden=64, memory_profile="cs800_safe_start"),
}

_VALID_POOLS = tuple(POOLS.keys())


def _phase_steps(phase: str) -> list[int]:
    """返回指定阶段的训练步数列表。"""
    return list(SMOKE_STEPS if phase == "smoke" else FULL_STEPS)


def _build_runs_for_phase(phase: str, pool: str = "hs300", version: str = "v3") -> list[dict]:
    """按当前 phase + pool + version 构造 run 列表。"""
    steps = _phase_steps(phase)
    runs = []
    panel = POOLS.get(pool, POOLS["hs300"])
    for span, (ts, te, vs, ve) in SPANS.items():
        for s in steps:
            rid = f"{pool}_{span}_{s // 1000}k_seed42"
            runs.append({
                "run_id": rid, "pool": pool, "panel": panel, "span": span,
                "version": version,
                "train_start": ts, "train_end": te, "oos_start": vs,
                "oos_end": ve, "steps": s, "status": "pending",
                "train": dict(POOL_TRAIN_PROFILES[pool], total_timesteps=s),
                "simulation": dict(FIXED_SIM), "artifacts": {}, "metrics": {},
                "gates": {}, "error": "",
            })
    return runs
