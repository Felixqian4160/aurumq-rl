"""WaveHunter v10 Env — Reward 综合计分 (绝对收益 + 回撤 + 胜率) + v9 ZigZag 标签。

继承 LstmWeightEnv，新增：
  1. Reward 综合分：w1*port_ret/0.02 + w2*(-DD²/0.01) + w3*hit_rate  → *reward_scale
  2. Tier 感知：暴露 P(A1)/P(A2) 预测所需的标签语义（v9_a1_point / v9_a2_interval）
  3. 复用全部 obs/action/mask/turnover 机制（不动现有内容）

v8/v2/v3 不动，v10 独立演进。
"""
from __future__ import annotations
from typing import Any
import numpy as np
from aurumq_rl.lstm_weight_env import LstmWeightEnv, LstmWeightConfig
from aurumq_rl.v10.reward_components import BayesianVolConfig, BayesianVolEstimator
from aurumq_rl.v10.risk_components import CVaRConfig, CVaRDrawdownPenalty
from aurumq_rl.v10.hhi_components import HHIConfig, HHIConcentrationPenalty


class WaveHunterV10Env(LstmWeightEnv):
    """v9 综合 Reward Env。"""

    def __init__(self, config: LstmWeightConfig, factor_panel: np.ndarray,
                 return_panel: np.ndarray, pct_change_panel=None,
                 is_st_panel=None, is_suspended_panel=None,
                 days_since_ipo_panel=None, knn_panel=None,
                 tradeable_mask=None,
                 a1_labels=None, a2_labels=None,
                 v9_a1_point=None, v9_a2_interval=None,
                 v9_peak=None, v9_b1=None,
                 evt_peak=None, evt_valley=None):
        super().__init__(config, factor_panel, return_panel,
                         pct_change_panel, is_st_panel, is_suspended_panel,
                         days_since_ipo_panel, knn_panel, tradeable_mask)
        # v9 标签：优先用 v9 列
        n_dates, n_stocks = factor_panel.shape[0], factor_panel.shape[1]
        self._v9_a1 = v9_a1_point if v9_a1_point is not None else (a1_labels if a1_labels is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64))
        self._v9_a2 = v9_a2_interval if v9_a2_interval is not None else (a2_labels if a2_labels is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64))
        self._v9_peak = v9_peak if v9_peak is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64)
        self._v9_b1 = v9_b1 if v9_b1 is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64)
        n_dates, n_stocks = factor_panel.shape[0], factor_panel.shape[1]
        self._evt_peak = evt_peak if evt_peak is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64)
        self._evt_valley = evt_valley if evt_valley is not None else np.full((n_dates, n_stocks), -1, dtype=np.int64)
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
        self._bayes_vol = BayesianVolEstimator(BayesianVolConfig(
            alpha0=float(getattr(config, 'bayes_vol_alpha0', 2.0) or 2.0),
            beta0=float(getattr(config, 'bayes_vol_beta0', 1e-4) or 1e-4),
            decay=float(getattr(config, 'bayes_vol_decay', 0.98) or 0.98),
            uncertainty_weight=float(getattr(config, 'bayes_vol_uncertainty_weight', 0.50)),
            min_vol=float(getattr(config, 'bayes_vol_min', 0.005) or 0.005),
        ))
        self._bayes_vol_enabled = bool(getattr(config, 'bayes_vol_enabled', True))
        self._cvar_enabled = bool(getattr(config, 'cvar_enabled', False))
        self._cvar_penalty = CVaRDrawdownPenalty(CVaRConfig(
            window=int(getattr(config, 'cvar_window', 252) or 252),
            alpha=float(getattr(config, 'cvar_alpha', 0.95) or 0.95),
            warmup=int(getattr(config, 'cvar_warmup', 30) or 30),
            normal_scale=float(getattr(config, 'cvar_normal_scale', 0.5) or 0.5),
            tail_scale=float(getattr(config, 'cvar_tail_scale', 1.0) or 1.0),
            max_penalty=float(getattr(config, 'cvar_max_penalty', 2.0) or 2.0),
        ))
        self._hhi_enabled = bool(getattr(config, 'hhi_enabled', False))
        self._hhi_penalty = HHIConcentrationPenalty(HHIConfig(
            weight=float(getattr(config, 'hhi_weight', 0.5) or 0.5),
            target_count=int(getattr(config, 'hhi_target_count', 10) or 10),
            max_penalty=float(getattr(config, 'hhi_max_penalty', 0.5) or 0.5),
        ))
        # 回撤追踪
        self._v9_peak_nav = 1.0
        self._v9_cur_nav = 1.0
        self._v9_picked = np.zeros(n_stocks, dtype=bool)
        self._v9_hit_window = []
        self._prev_dd = 0.0

    def _label_info(self, t: int) -> dict[str, Any]:
        info = {
            "a1_bins": self._a1_labels[t], "a2_labels": self._a2_labels[t],
            "valid": (self._a1_labels[t] >= 0) | (self._a2_labels[t] >= 0),
            "v9_a1": self._v9_a1[t], "v9_a2": self._v9_a2[t],
            "v9_peak": self._v9_peak[t], "v9_b1": self._v9_b1[t]}
        if self._evt_peak is not None:
            ep = self._evt_peak[t].flatten()
            ev = self._evt_valley[t].flatten()
            info["v10_evt_peak"] = int(ep[0])
            info["v10_evt_valley"] = int(ev[0])
        return info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._v9_peak_nav = 1.0
        self._v9_cur_nav = 1.0
        self._v9_picked = np.zeros(self.n_stocks, dtype=bool)
        self._v9_hit_window = []
        self._prev_dd = 0.0
        self._ema_vol = float(self._reward_norm_vol)
        self._bayes_vol.reset()
        self._cvar_penalty.reset()
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
        # v10 Reward：平滑有界，不把绝大多数样本硬裁剪到同一 -30。
        # 绝对收益用 tanh 标准化；回撤使用当前风险状态；胜率只奖励当期方向。
        # 这样保留排序信息，PPO 才能区分不同动作。
        w_abs, w_dd, w_hit = self._w_abs, self._w_dd, self._w_hit
        self._ema_vol = (1 - self._ema_alpha) * self._ema_vol + self._ema_alpha * abs(port_ret)
        bayes_vol = self._bayes_vol.update(port_ret)
        norm_vol = bayes_vol if self._bayes_vol_enabled else max(self._ema_vol, 0.005, self._reward_norm_vol * 0.25)
        r_abs = float(np.tanh(port_ret / norm_vol))
        # 回撤项
        self._v9_cur_nav *= (1.0 + port_ret)
        self._v9_peak_nav = max(self._v9_peak_nav, self._v9_cur_nav)
        dd = (self._v9_peak_nav - self._v9_cur_nav) / self._v9_peak_nav if self._v9_peak_nav > 0 else 0.0
        # 回撤项平滑到 [-1, 0]，不产生平方惩罚爆炸
        r_dd = self._cvar_penalty.update(dd) if self._cvar_enabled else -float(np.clip(dd / 0.20, 0.0, 1.0))
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
        # HHI 仅作为独立 Reward 分量，不改变执行权重。
        r_div = self._hhi_penalty.compute(self._current_weights) if self._hhi_enabled else 0.0
        reward_raw = 0.5 * (w_abs * r_abs + w_dd * r_dd + w_hit * r_hit) + 0.5 * reward_cycle + r_div
        reward_raw = float(np.clip(reward_raw, -1.0, 1.0))
        reward = float(self.config.reward_scale) * reward_raw
        reward = float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
        info["reward_raw"] = float(reward_raw)
        info["r_abs"] = float(r_abs)
        info["r_dd"] = float(r_dd)
        info["r_hit"] = float(r_hit)
        info["r_cycle"] = float(reward_cycle)
        info["ema_vol"] = float(self._ema_vol)
        info["bayes_vol"] = float(bayes_vol)
        info["bayes_vol_state"] = self._bayes_vol.state()
        info["cvar_enabled"] = self._cvar_enabled
        info["cvar_state"] = self._cvar_penalty.state()
        info["hhi_enabled"] = self._hhi_enabled
        info["r_div"] = float(r_div)
        info["hhi_state"] = self._hhi_penalty.state(self._current_weights)
        # Frozen four-head fusion diagnostic signals; do NOT influence action net.
        labels_now = {
            'a1': float(info.get('a1_bins', 0) == 1),
            'a2': float(info.get('a2_labels', 0) == 1),
            'peak': float(info.get('v9_peak', 0) == 1),
            'b1': float(info.get('v9_b1', 0) == 1),
        }
        scores = {
            'a1': labels_now['a1'],
            'a2': labels_now['a2'],
            'peak': labels_now['peak'],
            'b1': labels_now['b1'],
        }
        info['fused_score'] = float(0.40 * scores['a1'] + 0.40 * scores['a2']
                                     + 0.20 * scores['b1'] - 0.30 * scores['peak'])
        info['head_weights'] = {f'{k}_w': float(v) for k, v in scores.items()}
        info["dd"] = float(dd)
        info["reward_version"] = "wavehunter_v10_smooth_060_025_015"
        t = self._current_step
        info.update(self._label_info(t))
        return obs, reward, terminated, truncated, info
