"""WaveHunter v2 Env — 主升浪猎手环境（绝对收益 reward）。

与 LstmWeightEnv 的差异:
  1. info 中暴露当前时点的 A1(5档主升浪左侧信号) / A2(启动右侧信号) 标签
     → train_wavehunter.py 在 PPO rollout 后从 info 收集标签, 算辅助 CE/BCE 损失
  2. obs/action/reward/turnover/mask 机制完全复用父类 (零改动)

设计原则: 训练时无需重写 step — 标签是"观测外监督信号", 不影响 env 决策,
只作为额外监督目标注入 PPO 的总损失。env 只负责把标签放进 info。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aurumq_rl.lstm_weight_env import LstmWeightEnv, LstmWeightConfig


class WaveHunterEnv(LstmWeightEnv):
    def __init__(
        self,
        config: LstmWeightConfig,
        factor_panel: np.ndarray,
        return_panel: np.ndarray,
        pct_change_panel: np.ndarray | None = None,
        is_st_panel: np.ndarray | None = None,
        is_suspended_panel: np.ndarray | None = None,
        days_since_ipo_panel: np.ndarray | None = None,
        knn_panel: np.ndarray | None = None,
        tradeable_mask: np.ndarray | None = None,
        # ── WaveHunter 额外输入: 标签面板 ──
        a1_labels: np.ndarray | None = None,   # (n_dates, n_stocks) int 0..4, -1=无效
        a2_labels: np.ndarray | None = None,   # (n_dates, n_stocks) int {0,1}, -1=无效
    ) -> None:
        super().__init__(
            config=config,
            factor_panel=factor_panel,
            return_panel=return_panel,
            pct_change_panel=pct_change_panel,
            is_st_panel=is_st_panel,
            is_suspended_panel=is_suspended_panel,
            days_since_ipo_panel=days_since_ipo_panel,
            knn_panel=knn_panel,
            tradeable_mask=tradeable_mask,
        )
        n_dates, n_stocks = factor_panel.shape[0], factor_panel.shape[1]
        self._a1_labels = (
            a1_labels.astype(np.int64)
            if a1_labels is not None
            else np.full((n_dates, n_stocks), -1, dtype=np.int64)
        )
        self._a2_labels = (
            a2_labels.astype(np.int64)
            if a2_labels is not None
            else np.full((n_dates, n_stocks), -1, dtype=np.int64)
        )

    def _label_info(self, t: int) -> dict[str, Any]:
        """返回当前时点 (t) 的标签 + 有效掩码。

        时间对齐: 训练时 PPO 在 t 观测 → 决策, 标签用 t 日的 A1/A2 值
        (前瞻: 标签基于未来窗口算好, t 日已知; 无未来函数 — 标签只监督,
        不进入 obs, 不会泄漏到决策).
        """
        return {
            "a1_bins": self._a1_labels[t],          # (n_stocks,) int
            "a2_labels": self._a2_labels[t],        # (n_stocks,) int
            "valid": (self._a1_labels[t] >= 0) | (self._a2_labels[t] >= 0),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        # P0 修复: 跨 episode 清零 EMA，避免上一 episode 的收益分布泄漏到下一 episode
        # 之前 reset 未清，导致 sharpe_z 在 episode 边界用旧 mean/var，且初始 var 1e-4 在前 60 步放大 ~30×
        self._port_return_ema_mean = 0.0
        self._port_return_ema_var = 1e-4
        self._port_return_ema_n = 0
        t = self._current_step
        info.update(self._label_info(t))
        return obs, info

    def step(self, action: np.ndarray):
        # 先复用 v1 的权重投影、交易 mask、换手成本和状态推进。
        # v1 的 reward 是 excess；v2 在同一状态转移上改为绝对组合收益。
        obs, _v1_reward, terminated, truncated, info = super().step(action)
        # v1 的 info[port_return] 已经是组合收益扣除换手成本后的结果：
        #   sum(weights * next_ret) - turnover * cost_bps / 10000
        # 这里不能再次扣 cost，否则会重复扣费。
        absolute_return = float(info.get("port_return", 0.0))
        # ── Reward 设计 v2.1 (审计 P0: turnover 已在 port_ret 扣过 cost, 不再二次惩罚) ──
        # 原始 absolute_return 仅在 22k 步就达到峰值, 之后无改善,
        # 容易过拟合训练期高换手样本, 200k 步样本外崩盘。
        # 改进:
        #   1. 维护 EMA mean/std (60 步窗口), 加 sharpe-like z-score 项
        #      → 奖励"稳定"正收益, 抑制高波动短促爆赚
        #   2. 不加 turnover 惩罚: 已在 port_return 中扣 cost_bps
        #   3. reward_scale 由训练脚本决定 (默认 1.0 兼容旧配置)
        if not hasattr(self, "_port_return_ema_mean"):
            self._port_return_ema_mean = 0.0
            self._port_return_ema_var = 1e-4
            self._port_return_ema_n = 0
        # EMA 更新 (decay=0.05 ≈ 60 步有效窗口)
        alpha = 0.05
        self._port_return_ema_n += 1
        self._port_return_ema_mean = (1 - alpha) * self._port_return_ema_mean + alpha * absolute_return
        self._port_return_ema_var = (1 - alpha) * self._port_return_ema_var + alpha * (absolute_return - self._port_return_ema_mean) ** 2
        std = float(np.sqrt(max(self._port_return_ema_var, 1e-8)))
        sharpe_z = (absolute_return - self._port_return_ema_mean) / std
        # 主项: 绝对收益 (1.0) + sharpe z-score (0.3)
        # 不再二次惩罚 turnover (cost 已在 port_return 中扣过)
        # 总信号量级与原 absolute_return 一致
        reward = 1.0 * absolute_return + 0.3 * sharpe_z
        reward = float(self.config.reward_scale) * reward
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        info["absolute_return"] = absolute_return
        info["sharpe_z"] = float(sharpe_z)
        info["reward_version"] = "wavehunter_v2_sharpe_v2.1"
        t = self._current_step
        info.update(self._label_info(t))
        return obs, reward, terminated, truncated, info