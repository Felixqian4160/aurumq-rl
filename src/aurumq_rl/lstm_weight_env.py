"""LSTM+PPO 联合决策环境 — 一个网络直接输出仓位权重（选股+择时一体）。

与 PortfolioWeightEnv 的区别：
  - obs 是窗口历史 (n_stocks, window, n_factors)，网络用 LSTM 编码时序，
    直接学到"什么时候该买/该空仓"（择时内化在权重里，无需单独温度模型）
  - action 是连续权重 (n_stocks,) ∈ [0,1]，0=空仓，1=满仓
  - reward 是组合收益（含成本），PPO 端到端训练

这就是"两个模型合二为一"：LSTM 的时序建模 + PPO 的决策能力
在一个网络里端到端训练，没有中间温度/融合层。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

NEW_STOCK_PROTECT_DAYS = 60


@dataclass
class LstmWeightConfig:
    start_date: Any = None
    end_date: Any = None
    n_factors: int = 64
    window: int = 20              # 时序窗口
    forward_period: int = 5       # 持仓期
    reward_type: str = 'return'   # return / sharpe / sortino / win_rate / absolute_return_v3
    risk_aversion: float = 1.0
    cost_bps: float = 30.0
    max_position_pct: float = 0.05
    max_industry_pct: float = 0.30
    reward_scale: float = 100.0
    top_k: int = 0                # 0=全市场权重分配; >0=选权重前K只股票
    rebalance_days: int = 1       # 调仓周期（天），1=每日调仓，20=每半月
    # reward 组件权重 (absolute_return_v4 时生效)
    hit_rate_bonus_weight: float = 0.0
    continuation_bonus_weight: float = 0.0
    drawdown_penalty_weight: float = 0.0
    excess_weight: float = 0.0
    # ── 市场中性对冲 ──
    hedge_ratio: float = 0.0      # 0=不对冲, 1.0=完全对冲beta
    # ── 信号过滤 ──
    signal_threshold: float = 0.0 # 0=不过滤, >0=信号强度阈值（低于阈值不调仓）
    # ── v9 综合 Reward 权重 ──
    reward_w_abs: float = 0.60
    reward_w_dd: float = 0.25
    reward_w_hit: float = 0.15
    reward_norm_vol: float = 0.02


def _project_weights(raw: np.ndarray, mask: np.ndarray, max_pos: float, top_k: int = 0) -> np.ndarray:
    """clip → top_k 过滤 → per-stock cap → mask → normalize 到总仓位 ≤ 1（现金 = 择时）

    B3 优化：top_k 过滤从 O(n log n) 排序 → O(n) argpartition

    max_pos 设计原则（全局修复）：
    - 用户传的 max_pos 是【最终生效的单股上限】，不再是 raw 输入
    - top_k 模式下不再做 max_pos 强制覆盖，因为：
      1. 用户已经显式表达了风险偏好（max_pos）
      2. 训练/模拟必须严格一致，调参才有意义
      3. 单股上限 < 1/top_k*1.5 意味着允许"少选+集中持仓"或"少选+较低仓位"
    - 极端情况：若 max_pos * top_k < 1，总仓位可能 < 1，会形成现金 = 择时信号
    """
    w = np.clip(raw, 0.0, 1.0)
    # top_k 过滤: 选权重前 K 只股票，其余归零
    if top_k > 0:
        valid_mask = mask & (w > 0)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) > top_k:
            # B3 优化：np.argpartition 一次搞定，无需全排序
            valid_weights = w[valid_idx]
            # partition 后 top_k 大的在最前面（无序），置零非 top_k
            partition_idx = np.argpartition(valid_weights, -top_k)[-top_k:]
            keep = set(valid_idx[partition_idx])
            for i in valid_idx:
                if i not in keep:
                    w[i] = 0.0
        # 全局修复：max_pos 直接生效，不再被 1/top_k*1.5 覆盖
        # 训练和模拟必须严格使用同一个公式，否则调参无效（B6 修复）
    # per-stock cap: 单股不超过 max_pos
    w = np.minimum(w, max_pos)
    w[~mask] = 0.0
    total = w.sum()
    if total > 1.0:
        w = w / total  # 归一化：总仓位 = 1 表示满仓，<1 表示现金（择时）
    elif total <= 1e-8:
        w = np.zeros_like(w)  # 全空仓
    return w


class LstmWeightEnv(gym.Env):
    """窗口历史 obs + 连续权重 action 的联合决策环境。

    Observation: Box(-inf, inf, shape=(n_stocks, window, n_factors))
    Action:      Box(0.0, 1.0, shape=(n_stocks,))
    Reward:      组合收益 × reward_scale（含换手成本）
    """

    metadata: dict[str, Any] = {"render_modes": []}

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
    ) -> None:
        super().__init__()

        self.config = config
        self.factor_panel = factor_panel.astype(np.float32)
        self.return_panel = return_panel.astype(np.float32)

        n_dates, n_stocks, n_factors = factor_panel.shape
        self.n_dates = n_dates
        self.n_stocks = n_stocks
        self.n_factors = n_factors
        self.window = config.window

        # ── KNN 特征通道（融合 KNN+LSTM+PPO）──
        # knn_panel: (n_dates, n_stocks, n_knn) — knn_fwd_ret + knn_dist
        # 拼接到因子面板 → LSTM 输入维度 = n_factors + n_knn
        self.knn_panel = (
            knn_panel.astype(np.float32)
            if knn_panel is not None
            else np.zeros((n_dates, n_stocks, 0), dtype=np.float32)
        )
        self.n_knn = self.knn_panel.shape[2]
        self._n_input = n_factors + self.n_knn

        self._pct_change_panel = (
            pct_change_panel.astype(np.float32)
            if pct_change_panel is not None
            else np.zeros((n_dates, n_stocks), dtype=np.float32)
        )
        self._is_st_panel = (
            is_st_panel.astype(np.bool_)
            if is_st_panel is not None
            else np.zeros((n_dates, n_stocks), dtype=np.bool_)
        )
        self._is_suspended_panel = (
            is_suspended_panel.astype(np.bool_)
            if is_suspended_panel is not None
            else np.zeros((n_dates, n_stocks), dtype=np.bool_)
        )
        self._days_since_ipo_panel = (
            days_since_ipo_panel.astype(np.float32)
            if days_since_ipo_panel is not None
            else np.full((n_dates, n_stocks), NEW_STOCK_PROTECT_DAYS * 2, dtype=np.float32)
        )
        # ── 预计算的可交易 mask (T, S) bool — 与模拟 build_tradeable_mask 单一真源 ──
        # 含 ST/停牌/新股/涨停/跌停/低成交量 过滤；None → fallback 到简化 mask
        self._tradeable_mask = (
            tradeable_mask.astype(np.bool_)
            if tradeable_mask is not None
            else None
        )

        # obs: 1D flattened (n_stocks * window * n_input,)
        # 在 policy 内部 reshape 为 3D (n_stocks, window, n_input)
        # n_input = n_factors + n_knn（因子 + KNN 特征通道）
        self._obs_3d_shape = (n_stocks, self.window, self._n_input)
        obs_dim = n_stocks * self.window * self._n_input
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        # action: (n_stocks,) 连续权重 [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n_stocks,), dtype=np.float32)

        self._current_step: int = 0
        self._current_weights: np.ndarray = np.zeros(n_stocks, dtype=np.float64)
        self._cumulative_reward: float = 0.0
        self._episode_rewards: list[float] = []
        # ── reward_type 滚动统计 ──
        self._reward_window: list[float] = []  # 滑动窗口（最近 20 期超额收益）
        self._reward_window_size: int = 20
        # ── v3/v4 reward 组件追踪 ──
        # 有任一组件权重 > 0 或 reward_type 含 v3 时启用
        self._is_v3_reward = (
            config.reward_type == "absolute_return_v3"
            or config.hit_rate_bonus_weight > 0
            or config.continuation_bonus_weight > 0
            or config.drawdown_penalty_weight > 0
            or config.excess_weight > 0
        )
        if self._is_v3_reward:
            self._peak_nav: float = 1.0       # 组合净值峰值 (初始 ¥1)
            self._current_nav: float = 1.0    # 当前净值
            self._picked_stocks: np.ndarray = np.zeros(n_stocks, dtype=bool)   # 当前持仓股
            self._hold_days: np.ndarray = np.zeros(n_stocks, dtype=np.int32)   # 每只股持仓天数
            self._hold_cumret: np.ndarray = np.zeros(n_stocks, dtype=np.float64)  # 每只股持仓累计收益
            self._last_rebalance_weights: np.ndarray = np.zeros(n_stocks, dtype=np.float64)

    def _compute_trading_mask(self, t: int) -> np.ndarray:
        """(n_stocks,) bool — 可交易（非ST/非停牌/非新股/非涨停/非跌停/非低量）。

        优先使用 build_tradeable_mask 预计算的完整 mask（与模拟严格一致），
        fallback 到简化版本（仅 ST/停牌/新股）。
        """
        if self._tradeable_mask is not None:
            return self._tradeable_mask[t]
        return (
            (~self._is_st_panel[t])
            & (~self._is_suspended_panel[t])
            & (self._days_since_ipo_panel[t] >= NEW_STOCK_PROTECT_DAYS)
        )

    def _get_obs(self, t: int) -> np.ndarray:
        """取 [t-window+1, t] 的窗口历史（因子 + KNN 特征），前面补零，返回 1D flatten"""
        if t + 1 >= self.window:
            obs3d = self.factor_panel[t - self.window + 1: t + 1].transpose(1, 0, 2).astype(np.float32)
            if self.n_knn > 0:
                knn3d = self.knn_panel[t - self.window + 1: t + 1].transpose(1, 0, 2).astype(np.float32)
                obs3d = np.concatenate([obs3d, knn3d], axis=2)
        else:
            pad = self.window - (t + 1)
            past = self.factor_panel[: t + 1].transpose(1, 0, 2)
            zeros = np.zeros((self.n_stocks, pad, self.n_factors), dtype=np.float32)
            obs3d = np.concatenate([zeros, past], axis=1).astype(np.float32)
            if self.n_knn > 0:
                knn_past = self.knn_panel[: t + 1].transpose(1, 0, 2).astype(np.float32)
                knn_zeros = np.zeros((self.n_stocks, pad, self.n_knn), dtype=np.float32)
                knn3d = np.concatenate([knn_zeros, knn_past], axis=1).astype(np.float32)
                obs3d = np.concatenate([obs3d, knn3d], axis=2)
        return obs3d.reshape(-1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._current_step = self.window - 1  # 从窗口满开始
        self._current_weights = np.zeros(self.n_stocks, dtype=np.float64)
        self._cumulative_reward = 0.0
        self._episode_rewards = []
        obs = self._get_obs(self._current_step)
        info = {"step": self._current_step, "n_stocks": self.n_stocks,
                "n_factors": self.n_factors, "window": self.window}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        t = self._current_step
        if t >= self.n_dates - 1:
            # 结束
            obs = self._get_obs(t)
            return obs, 0.0, True, False, {"step": t}

        trading_mask = self._compute_trading_mask(t)
        # 调仓日才更新权重，非调仓日保持不变（与模拟引擎 rebalance_days 对齐）
        is_rebalance = (t - (self.window - 1)) % self.config.rebalance_days == 0
        if is_rebalance:
            weights = _project_weights(
                raw=action,
                mask=trading_mask,
                max_pos=self.config.max_position_pct,
                top_k=self.config.top_k,
            )
        else:
            weights = self._current_weights.copy()
            # 非调仓日仍需 mask 掉停牌/ST 股票
            weights[~trading_mask] = 0.0
            total = weights.sum()
            if total > 1e-8 and total > 1.0:
                weights = weights / total
        # 换手成本
        turnover = np.abs(weights - self._current_weights).sum()
        cost = turnover * self.config.cost_bps / 10000.0

        # 组合收益（下一期）
        next_ret = self.return_panel[t + 1]  # (n_stocks,)
        # NaN 安全: 用 0 替代 NaN 收益，防止 NaN 传播到 buffer
        next_ret = np.nan_to_num(next_ret, nan=0.0)
        port_ret = float(np.sum(weights * next_ret)) - cost

        # 等权市场收益（可交易池均值）— 用于超额收益，防止靠 beta 赚钱
        mkt_ret = float(np.mean(next_ret[trading_mask])) if trading_mask.any() else 0.0

        # ── hedge_ratio: 训练时应用 ──
        if self.config.hedge_ratio > 0:
            portfolio_beta = 1.0
            hedge_cost = self.config.hedge_ratio * portfolio_beta * mkt_ret
            port_ret -= hedge_cost

        # ── reward: 根据 reward_type 计算 ──
        excess = port_ret - mkt_ret
        self._reward_window.append(excess)
        if len(self._reward_window) > self._reward_window_size:
            self._reward_window.pop(0)

        rtype = self.config.reward_type
        if rtype in ('return', 'absolute_return', 'absolute_return_v3'):
            # ── v4 核心: 绝对收益为主 ──
            # 主项: 组合净收益（扣除交易成本）
            reward_raw = port_ret
            # 辅助项: 弱超额收益 (excess_weight × 超额, 默认0=不启用)
            reward_raw += self.config.excess_weight * excess
            # ── drawdown_penalty: 分段回撤惩罚 (5%以下不罚, 5%以上平方惩罚) ──
            if self._is_v3_reward and self.config.drawdown_penalty_weight > 0:
                self._current_nav *= (1.0 + port_ret)
                self._peak_nav = max(self._peak_nav, self._current_nav)
                drawdown = (self._peak_nav - self._current_nav) / self._peak_nav if self._peak_nav > 0 else 0.0
                if drawdown > 0.05:
                    reward_raw -= self.config.drawdown_penalty_weight * (drawdown - 0.05) ** 2
            # ── hit_rate_bonus + continuation_bonus: 调仓日结算 ──
            if self._is_v3_reward and is_rebalance:
                new_picked = weights > 1e-6
                if t > 0:
                    for s in range(self.n_stocks):
                        if self._picked_stocks[s]:
                            self._hold_days[s] += 1
                            self._hold_cumret[s] += float(self.return_panel[t, s]) if np.isfinite(self.return_panel[t, s]) else 0.0
                old_picked = self._picked_stocks.copy()
                if old_picked.any() and self._hold_days[old_picked].max() > 0:
                    held = old_picked & (self._hold_days > 0)
                    if held.any():
                        hit_rate = float((self._hold_cumret[held] > 0).mean())
                        reward_raw += self.config.hit_rate_bonus_weight * hit_rate
                if old_picked.any():
                    cont = old_picked & (self._hold_days >= 3) & (self._hold_cumret > 0)
                    if cont.any():
                        avg_cont_ret = float(self._hold_cumret[cont].mean())
                        reward_raw += self.config.continuation_bonus_weight * min(avg_cont_ret, 0.10)
                self._picked_stocks = new_picked.copy()
                self._hold_days[~new_picked] = 0
                self._hold_cumret[~new_picked] = 0.0
                self._hold_days[new_picked] = np.maximum(self._hold_days[new_picked], 0)
                self._hold_cumret[new_picked] = np.maximum(self._hold_cumret[new_picked], 0.0)
                self._last_rebalance_weights = weights.copy()
        elif rtype == 'sharpe':
            # Sharpe: excess / rolling_std（std≈0 时退化为 excess）
            if len(self._reward_window) >= 5:
                std = float(np.std(self._reward_window))
                reward_raw = excess / max(std, 1e-6)
            else:
                reward_raw = excess
        elif rtype == 'sortino':
            # Sortino: excess / downside_std（只惩罚负收益波动）
            if len(self._reward_window) >= 5:
                neg = [r for r in self._reward_window if r < 0]
                downside_std = float(np.std(neg)) if len(neg) >= 2 else 1e-6
                reward_raw = excess / max(downside_std, 1e-6)
            else:
                reward_raw = excess
        elif rtype == 'win_rate':
            # 胜率奖励: 正收益天数占比 × 超额收益
            if len(self._reward_window) >= 5:
                win_ratio = float(np.mean([1.0 if r > 0 else 0.0 for r in self._reward_window]))
                reward_raw = win_ratio * excess
            else:
                reward_raw = excess
        elif rtype == 'mean_variance':
            # Mean-Variance: excess - risk_aversion * variance
            if len(self._reward_window) >= 5:
                var = float(np.var(self._reward_window))
                reward_raw = excess - self.config.risk_aversion * var
            else:
                reward_raw = excess
        else:
            reward_raw = excess

        reward = reward_raw * self.config.reward_scale
        # NaN 安全: 防止 reward 污染 buffer
        reward = float(np.nan_to_num(reward, nan=0.0))
        self._cumulative_reward += reward
        self._episode_rewards.append(reward)

        self._current_weights = weights
        self._current_step = t + 1

        obs = self._get_obs(t + 1)
        info = {
            "step": t + 1,
            "port_return": port_ret,
            "turnover": float(turnover),
            "total_weight": float(weights.sum()),
            "excess": float(excess),
        }
        return obs, float(reward), False, False, info
