"""WaveHunter v3 Policy — 共享编码器 + PPO 头 + 双头 (左侧埋伏 A1 + 右侧延续 A2).

架构继承自 WaveHunterPolicy (v2):
  obs (n_stocks, window, n_factors)
    └─> LSTM encoder (共享, 与 v2 完全相同)
          ├─> per-stock features pf (n_stocks, h)
          │     ├─> PPO action_net    → 仓位权重 (n_stocks,)
          │     ├─> A1 rebound head   → 左侧埋伏 (未来20日反弹概率)
          │     └─> A2 continue head  → 右侧延续 (持仓期持续上涨概率)
          └─> market vector mv → value_net → state value

与 v2 的差异 (这是主升浪猎手的关键改进):
  - A1/A2 强制时间分离 (min_pullback_days >= 5)
  - 标签定义改写: A1 不含 "今天就是底" 的完美 hindsight
  - reward 设计: base_return + hit_rate_bonus + continuation_days_bonus - drawdown_penalty
    → 真正激励 "高成功率通道" 而不是偶尔的暴利

与 v2 的关系:
  - 完全继承 LSTM/PPO 共享编码器 (复用 v2 设计)
  - 不依赖 v2 模块 (从 0 写一份, 但与 v2 不共享 Python 对象)
  - 不修改 v2 任何代码 (独立演进)
"""
from __future__ import annotations
from functools import partial
from typing import Any

import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class WaveHunterV3Policy(ActorCriticPolicy):
    """双头 + 胜率加权 reward 的 WaveHunter v3 策略."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 *args: Any,
                 lstm_hidden: int = 64, lstm_layers: int = 1,
                 _n_stocks: int = 0, _window: int = 20, _n_factors: int = 692,
                 aux_lambda: float = 0.1,
                 a1_lambda: float = 1.0,
                 a2_lambda: float = 1.0,
                 a1_pos_weight: float = 3.0,
                 a2_pos_weight: float = 3.0,
                 **kwargs: Any):
        self._lstm_hidden = lstm_hidden
        self._lstm_layers = lstm_layers
        self._n_stocks = _n_stocks
        self._window = _window
        self._n_factors = _n_factors
        self._aux_lambda = aux_lambda
        self._a1_lambda = a1_lambda
        self._a2_lambda = a2_lambda
        self._a1_pos_weight = a1_pos_weight
        self._a2_pos_weight = a2_pos_weight
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        h = self._lstm_hidden
        # 共享编码器 (与 v2 相同)
        self.lstm = nn.LSTM(self._n_factors, h, self._lstm_layers, batch_first=True)
        self.lstm_fc = nn.Linear(h, h)
        # PPO 头
        self.action_net = nn.Sequential(nn.Linear(h * 3, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_net = nn.Sequential(nn.Linear(h * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.action_dist = DiagGaussianDistribution(self._n_stocks)
        self.log_std = nn.Parameter(torch.clamp(torch.full((self._n_stocks,), -2.0), -5.0, 2.0))

        # ── A1 左侧埋伏头 (二分类 BCE, 未来20日反弹概率) ──
        self.a1_head = nn.Sequential(
            nn.Linear(h * 3, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        # ── A2 右侧延续头 (二分类 BCE, 持仓期延续概率) ──
        self.a2_head = nn.Sequential(
            nn.Linear(h * 3, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        if getattr(self, "ortho_init", False):
            self.action_net.apply(partial(self.init_weights, gain=1.0))
            self.value_net.apply(partial(self.init_weights, gain=1.0))
            self.a1_head.apply(partial(self.init_weights, gain=1.0))
            self.a2_head.apply(partial(self.init_weights, gain=1.0))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _encode(self, obs):
        ns, w, nf, h = self._n_stocks, self._window, self._n_factors, self._lstm_hidden
        x = obs.reshape(-1, ns, w, nf).reshape(-1, w, nf)
        lstm_out, _ = self.lstm(x)
        ps = torch.relu(self.lstm_fc(lstm_out[:, -1, :])).reshape(-1, ns, h)
        mv = torch.cat([ps.mean(1), ps.std(1) + 1e-6], dim=-1)
        pf = torch.cat([ps, mv.unsqueeze(1).expand(-1, ns, h * 2)], dim=-1)
        return pf, mv

    def forward(self, obs, deterministic=False):
        pf, mv = self._encode(obs)
        mean = torch.sigmoid(self.action_net(pf).squeeze(-1))
        vals = self.value_net(mv)
        clamped_log_std = torch.clamp(self.log_std, -5.0, 2.0)
        dist = self.action_dist.proba_distribution(mean, clamped_log_std)
        acts = dist.get_actions(deterministic=deterministic)
        return acts, vals, dist.log_prob(acts)

    def _predict(self, observation, deterministic=False):
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions):
        pf, mv = self._encode(obs)
        mean = torch.sigmoid(self.action_net(pf).squeeze(-1))
        vals = self.value_net(mv)
        clamped_log_std = torch.clamp(self.log_std, -5.0, 2.0)
        dist = self.action_dist.proba_distribution(mean, clamped_log_std)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return vals, log_prob, entropy

    def predict_values(self, obs):
        _, mv = self._encode(obs)
        return self.value_net(mv)

    def extra_aux_loss(self, infos, labels):
        """v3 双头 BCE 辅助损失 — 与 v2 接口兼容.

        labels: dict with keys
          a1_rebound_start : (T, S) float32, A1 左侧埋伏标签
          a2_continue_start: (T, S) float32, A2 右侧延续标签
        """
        device = next(self.parameters()).device
        if infos is None or not infos:
            return torch.zeros((), device=device)
        last = infos[-1] if isinstance(infos, list) else infos
        pf = last.get("pf")
        if pf is None:
            return torch.zeros((), device=device)

        a1_t = labels.get("a1_rebound_start")
        a2_t = labels.get("a2_continue_start")
        if a1_t is None or a2_t is None:
            return torch.zeros((), device=device)

        a1_logit = self.a1_head(pf).squeeze(-1)
        a2_logit = self.a2_head(pf).squeeze(-1)

        # A1 BCE
        a1_valid = torch.isfinite(a1_t)
        if a1_valid.any():
            logit_v = a1_logit[a1_valid]
            target_v = a1_t[a1_valid]
            pos_weight = torch.tensor(self._a1_pos_weight, device=device, dtype=torch.float32)
            a1_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)
        else:
            a1_loss = torch.zeros((), device=device)

        # A2 BCE
        a2_valid = torch.isfinite(a2_t)
        if a2_valid.any():
            logit_v = a2_logit[a2_valid]
            target_v = a2_t[a2_valid]
            pos_weight = torch.tensor(self._a2_pos_weight, device=device, dtype=torch.float32)
            a2_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)
        else:
            a2_loss = torch.zeros((), device=device)

        return self._aux_lambda * (self._a1_lambda * a1_loss + self._a2_lambda * a2_loss)

    def compute_aux_loss(self, obs, a1_bins, a2_label, valid_mask, peak_label=None, b1_label=None):
        """兼容通用 WaveHunterPPO 训练循环，返回 15 元组。

        v3 只有 A1/A2 两个监督头；Peak/B1 为兼容位，指标置零。
        """
        pf, _ = self._encode(obs)
        B, ns, h = pf.shape
        pf_flat = pf.reshape(B * ns, h)

        a1_logit = self.a1_head(pf_flat).reshape(B, ns)
        a1_t = torch.as_tensor(a1_bins, device=pf.device, dtype=torch.float32)
        a1_valid = (a1_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        a1_loss = torch.tensor(0.0, device=pf.device)
        a1_acc = torch.tensor(0.0, device=pf.device)
        a1_precision = torch.tensor(0.0, device=pf.device)
        a1_recall = torch.tensor(0.0, device=pf.device)
        if a1_valid.any():
            logit_v = a1_logit[a1_valid]
            target_v = a1_t[a1_valid]
            pos_weight = torch.tensor(self._a1_pos_weight, device=pf.device, dtype=torch.float32)
            a1_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)
            pred_v = torch.sigmoid(logit_v) > 0.5
            true_v = target_v > 0.5
            a1_acc = (pred_v == true_v).float().mean()
            tp = (pred_v & true_v).sum().float()
            a1_precision = tp / pred_v.sum().clamp_min(1.0)
            a1_recall = tp / true_v.sum().clamp_min(1.0)

        a2_logit = self.a2_head(pf_flat).reshape(B, ns)
        a2_t = torch.as_tensor(a2_label, device=pf.device, dtype=torch.float32)
        a2_valid = (a2_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        a2_loss = torch.tensor(0.0, device=pf.device)
        a2_acc = torch.tensor(0.0, device=pf.device)
        a2_precision = torch.tensor(0.0, device=pf.device)
        a2_recall = torch.tensor(0.0, device=pf.device)
        if a2_valid.any():
            logit_v = a2_logit[a2_valid]
            target_v = a2_t[a2_valid]
            pos_weight = torch.tensor(self._a2_pos_weight, device=pf.device, dtype=torch.float32)
            a2_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)
            pred_v = torch.sigmoid(logit_v) > 0.5
            true_v = target_v > 0.5
            a2_acc = (pred_v == true_v).float().mean()
            tp = (pred_v & true_v).sum().float()
            a2_precision = tp / pred_v.sum().clamp_min(1.0)
            a2_recall = tp / true_v.sum().clamp_min(1.0)

        total_aux = self._a1_lambda * a1_loss + self._a2_lambda * a2_loss
        zero = torch.zeros((), device=pf.device)
        return (total_aux, a1_loss, a2_loss, a1_acc, a2_acc,
                a1_precision, a1_recall, a2_precision, a2_recall,
                zero, zero, zero, zero, zero, zero)