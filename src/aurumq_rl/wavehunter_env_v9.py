"""WaveHunter v9 Env — Reward 综合计分 (绝对收益 + 回撤 + 胜率) + v9 ZigZag 标签。

继承 LstmWeightEnv，新增：
  1. Reward 综合分：w1*port_ret/0.02 + w2*(-DD²/0.01) + w3*hit_rate  → *reward_scale
  2. Tier 感知：暴露 P(A1)/P(A2) 预测所需的标签语义（v9_a1_point / v9_a2_interval）
  3. 复用全部 obs/action/mask/turnover 机制（不动现有内容）

v8/v2/v3 不动，v9 独立演进。
"""
from __future__ import annotations
from typing import Any
import numpy as np
from aurumq_rl.lstm_weight_env import LstmWeightEnv, LstmWeightConfig


class WaveHunterV9Env(LstmWeightEnv):
    """v9 综合 Reward Env。"""

    def __init__(self, config: LstmWeightConfig, factor_panel: np.ndarray,
                 return_panel: np.ndarray, pct_change_panel=None,
                 is_st_panel=None, is_suspended_panel=None,
                 days_since_ipo_panel=None, knn_panel=None,
                 tradeable_mask=None,
                 a1_labels=None, a2_labels=None,
                 v9_a1_point=None, v9_a2_interval=None,
                 v9_peak=None, v9_b1=None):
        super().__init__(config, factor_panel, return_panel,
                         pct_change_panel, is_st_panel, is_suspended_panel,
                         days_since_ipo_panel, knn_panel, tradeable_mask)
        # v9 标签：优先用 v9 列
        n_dates, n_stocks = factor_panel.shape[0], factor_panel.shape[1]
        self._v9_a1 = v9_a1_point if v9_a1_point is not None else (a1_labels if a1_labels is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64))
        self._v9_a2 = v9_a2_interval if v9_a2_interval is not None else (a2_labels if a2_labels is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64))
        self._v9_peak = v9_peak if v9_peak is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64)
        self._v9_b1 = v9_b1 if v9_b1 is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64)
        # 复用父类的 _a1_labels/_a2_labels 供旧 compute_aux_loss 兼容
        self._a1_labels = self._v9_a1.astype(np.int64)
        self._a2_labels = self._v9_a2.astype(np.int64)
        # Reward 综合分权重（从 config 读，无则默认）
        self._w_abs = float(getattr(config, 'reward_w_abs', 0.60) or 0.60)
        self._w_dd = float(getattr(config, 'reward_w_dd', 0.25) or 0.25)
        self._w_hit = float(getattr(config, 'reward_w_hit', 0.15) or 0.15)
        # Langevin/OU 自适应：EMA 波动率归一，避免固定 0.02 导致回报方差爆炸
        self._reward_norm_vol = float(getattr(config, 'reward_norm_vol', 0.02) or 0.02)
        self._adaptive_vol = True
        self._ema_vol = float(getattr(config, 'reward_norm_vol', 0.02) or 0.02)
        self._ema_alpha = 0.02  # EMA decay ~50步半衰期
        # 回撤追踪
        self._v9_peak_nav = 1.0
        self._v9_cur_nav = 1.0
        self._v9_picked = np.zeros(n_stocks, dtype=bool)
        self._v9_hit_window = []
        self._prev_dd = 0.0

    def _label_info(self, t: int) -> dict[str, Any]:
        return {"a1_bins": self._a1_labels[t], "a2_labels": self._a2_labels[t],
                "valid": (self._a1_labels[t] >= 0) | (self._a2_labels[t] >= 0),
                "v9_a1": self._v9_a1[t], "v9_a2": self._v9_a2[t],
                "v9_peak": self._v9_peak[t], "v9_b1": self._v9_b1[t]}

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._v9_peak_nav = 1.0
        self._v9_cur_nav = 1.0
        self._v9_picked = np.zeros(self.n_stocks, dtype=bool)
        self._v9_hit_window = []
        self._prev_dd = 0.0
        self._ema_vol = float(self._reward_norm_vol)
        # 清 EMA（复用父类）
        self._port_return_ema_mean = 0.0
        self._port_return_ema_var = 1e-4
        self._port_return_ema_n = 0
        t = self._current_step
        info.update(self._label_info(t))
        return obs, info

    def step(self, action: np.ndarray):
        # 复用父类 step 的 reward 计算，但替换为综合计分
        # 1) 先走父类拿到原始 reward_raw 的 port_ret / excess 等 info
        # 简化：直接复用父类逻辑，然后重算 reward
        t_before = self._current_step
        is_rebalance = (t_before - (self.window - 1)) % self.config.rebalance_days == 0
        obs, _old_reward, terminated, truncated, info = super().step(action)
        # super().step 已更新 _current_step 和 _current_weights
        port_ret = float(info.get("port_return", 0.0))
        # 综合 Reward：w1*port_ret/EMA_vol + w2*(-DD²) + w3*hit_rate  (EMA_vol 自适应，Langevin/OU)
        w_abs, w_dd, w_hit = self._w_abs, self._w_dd, self._w_hit
        # 绝对收益归一 — EMA 波动率归一 + 裁剪，避免 0.02 固定归一导致 value 爆炸
        self._ema_vol = (1 - self._ema_alpha) * self._ema_vol + self._ema_alpha * abs(port_ret)
        norm_vol = max(self._ema_vol, 0.005, self._reward_norm_vol * 0.25)
        r_abs = float(np.clip(port_ret / max(norm_vol, 1e-6), -3.0, 3.0))
        # 回撤项
        self._v9_cur_nav *= (1.0 + port_ret)
        self._v9_peak_nav = max(self._v9_peak_nav, self._v9_cur_nav)
        dd = (self._v9_peak_nav - self._v9_cur_nav) / self._v9_peak_nav if self._v9_peak_nav > 0 else 0.0
        r_dd = 0.0
        if dd > 0.05:
            r_dd = - (dd - 0.05) ** 2 / 0.01  # 归一：1% 超阈 = -1
        # 胜率项（调仓日）
        r_hit = 0.0
        if is_rebalance and self._v9_hit_window:
            r_hit = float(np.mean(self._v9_hit_window)) if self._v9_hit_window else 0.0
            self._v9_hit_window = []
        # 累积 hit 样本
        if port_ret > 0:
            self._v9_hit_window.append(1.0)
        elif port_ret < 0:
            self._v9_hit_window.append(0.0)

        # ── 四段周期 Reward (基于理论框架) ──
        # A1 谷底: +1.0 if 预测正确且买入盈利, -0.5 if 预测错误
        # A2 主升: +0.5 if 持有且盈利, -0.3 if 持有但亏损
        # Peak 顶部: +1.0 if 正确卖出避免回撤, -0.5 if 错过卖出
        # B1 筑底: +0.1 if 正确观望, -0.1 if 错误交易

        r_a1 = 0.0
        r_a2 = 0.0
        r_peak = 0.0
        r_b1 = 0.0

        # 获取当前周期标签
        cur_a1 = info.get('v9_a1', 0)
        cur_a2 = info.get('v9_a2', 0)

        # A1 谷底奖励: 如果持仓盈利且处于 A1 区间
        if cur_a1 == 1 and port_ret > 0:
            r_a1 = 1.0  # 正确预测 A1 且盈利
        elif cur_a1 == 1 and port_ret <= 0:
            r_a1 = -0.5  # 预测 A1 但亏损

        # A2 主升奖励: 如果持仓盈利且处于 A2 区间
        if cur_a2 == 1 and port_ret > 0:
            r_a2 = 0.5  # 正确持有 A2 且盈利
        elif cur_a2 == 1 and port_ret <= 0:
            r_a2 = -0.3  # 持有 A2 但亏损

        # Peak 顶部奖励: 如果及时卖出避免回撤
        if hasattr(self, '_prev_dd') and dd > self._prev_dd + 0.05:
            r_peak = 1.0  # 正确卖出避免回撤
        elif hasattr(self, '_prev_dd') and dd < self._prev_dd - 0.05:
            r_peak = -0.5  # 错过卖出时机

        # B1 筑底奖励: 如果正确观望（中性）
        if cur_a1 == 0 and cur_a2 == 0 and abs(port_ret) < 0.01:
            r_b1 = 0.1  # 正确观望
        elif cur_a1 == 0 and cur_a2 == 0 and abs(port_ret) >= 0.01:
            r_b1 = -0.1  # 错误交易

        self._prev_dd = dd

        # 综合四段周期 Reward
        w_a1, w_a2, w_peak, w_b1 = 0.4, 0.3, 0.2, 0.1
        reward_cycle = w_a1 * r_a1 + w_a2 * r_a2 + w_peak * r_peak + w_b1 * r_b1

        # 结合原有 reward 和四段周期 reward
        reward_raw = float(np.clip(0.5 * (w_abs * r_abs + w_dd * r_dd + w_hit * r_hit) + 0.5 * reward_cycle, -3.0, 3.0))
        reward = float(self.config.reward_scale) * reward_raw
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        info["reward_raw"] = float(reward_raw)
        info["r_abs"] = float(r_abs)
        info["r_dd"] = float(r_dd)
        info["r_hit"] = float(r_hit)
        info["r_cycle"] = float(reward_cycle)
        info["ema_vol"] = float(self._ema_vol)
        info["dd"] = float(dd)
        info["reward_version"] = "wavehunter_v9_composite_060_025_015"
        t = self._current_step
        info.update(self._label_info(t))
        return obs, reward, terminated, truncated, info
