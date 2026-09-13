"""WaveHunter Env — 主升浪猎手环境（继承 LstmWeightEnv，复用全部机制）。

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
        t = self._current_step
        info.update(self._label_info(t))
        return obs, info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)
        t = self._current_step
        info.update(self._label_info(t))
        return obs, reward, terminated, truncated, info