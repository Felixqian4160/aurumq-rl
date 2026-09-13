"""WaveHunter LSTM+PPO Policy — 共享编码器 + PPO 头 + A1(5档CE) + A2(启动BCE).

架构:
    obs (n_stocks, window, n_factors)
      └─> LSTM encoder (共享)
            ├─> per-stock features pf (n_stocks, h)
            │     ├─> PPO action_net  → 仓位权重 (n_stocks,)
            │     ├─> A1 head (CE 5档) → 未来20天主升浪分档 logits
            │     └─> A2 head (BCE)    → 启动确认 logit
            └─> market vector mv → value_net → state value

与 LstmJointPolicy 的关系:
  - 编码器/PPO 头完全复用其设计 (LSTM + lstm_fc + action/value net)
  - 新增 A1/A2 监督头, 训练时 extra_aux_loss 注入 PPO loss
  - 不修改 lstm_joint_policy.py (不动现有内容铁律)
"""
from __future__ import annotations
from functools import partial
from typing import Any

import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class WaveHunterPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 *args: Any,
                 lstm_hidden: int = 128, lstm_layers: int = 1,
                 _n_stocks: int = 0, _window: int = 20, _n_factors: int = 323,
                 aux_lambda: float = 0.1,  # aux loss 权重 (A1+A2)
                 a1_lambda: float = 1.0,   # A1 CE 相对权重
                 a2_lambda: float = 1.0,   # A2 BCE 相对权重
                 a2_pos_weight: float = 25.0,  # A2 正样本加权 (0.4% 稀有 → 权重 25:1)
                 **kwargs: Any):
        self._lstm_hidden = lstm_hidden
        self._lstm_layers = lstm_layers
        self._n_stocks = _n_stocks
        self._window = _window
        self._n_factors = _n_factors
        self._aux_lambda = aux_lambda
        self._a1_lambda = a1_lambda
        self._a2_lambda = a2_lambda
        self._a2_pos_weight = a2_pos_weight
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        h = self._lstm_hidden
        # 共享编码器 (与 LstmJointPolicy 相同)
        self.lstm = nn.LSTM(self._n_factors, h, self._lstm_layers, batch_first=True)
        self.lstm_fc = nn.Linear(h, h)
        # PPO 头
        self.action_net = nn.Sequential(nn.Linear(h * 3, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_net = nn.Sequential(nn.Linear(h * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.action_dist = DiagGaussianDistribution(self._n_stocks)
        self.log_std = nn.Parameter(torch.clamp(torch.full((self._n_stocks,), -2.0), -5.0, 2.0))

        # ── A1 主升浪左侧: 5 档 CE 头 ──
        # 输入: pf per-stock (B, ns, 3h) → flatten (B*ns, 3h) → 5 档 logits
        self.a1_head = nn.Sequential(
            nn.Linear(h * 3, 64), nn.ReLU(), nn.Linear(64, 5)
        )
        # ── A2 主升浪右侧: 启动 BCE 头 ──
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

    def _predict(self, obs, deterministic=False):
        pf, _ = self._encode(obs)
        mean = torch.sigmoid(self.action_net(pf).squeeze(-1))
        clamped_log_std = torch.clamp(self.log_std, -5.0, 2.0)
        return self.action_dist.proba_distribution(mean, clamped_log_std).get_actions(deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        pf, mv = self._encode(obs)
        mean = torch.sigmoid(self.action_net(pf).squeeze(-1))
        vals = self.value_net(mv)
        clamped_log_std = torch.clamp(self.log_std, -5.0, 2.0)
        dist = self.action_dist.proba_distribution(mean, clamped_log_std)
        return vals, dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        _, mv = self._encode(obs)
        return self.value_net(mv)

    # ════════════════════════════════════════════════════════
    # A1/A2 监督头 (训练时由 train_wavehunter.py 调用)
    # ════════════════════════════════════════════════════════
    def compute_aux_loss(self, obs, a1_bins, a2_label, valid_mask):
        """计算 A1 CE + A2 BCE 辅助损失。

        Parameters
        ----------
        obs : (B, ns*w*nf) 批量观测
        a1_bins : (B, ns) int 0..4 (目标档, -1=无效)
        a2_label : (B, ns) int {0,1} (目标, -1=无效)
        valid_mask : (B, ns) bool 有效掩码

        Returns
        -------
        (a1_loss, a2_loss, a1_acc, a2_acc) 标量
        """
        pf, _ = self._encode(obs)  # (B, ns, h)
        B, ns, h = pf.shape
        pf_flat = pf.reshape(B * ns, h)

        # A1: CE on valid bins
        a1_logits = self.a1_head(pf_flat).reshape(B, ns, 5)  # (B, ns, 5)
        a1_bins_t = torch.as_tensor(a1_bins, device=pf.device, dtype=torch.long)
        a1_valid = (a1_bins_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        a1_loss = torch.tensor(0.0, device=pf.device)
        a1_acc = torch.tensor(0.0, device=pf.device)
        if a1_valid.any():
            logits_v = a1_logits[a1_valid]
            targets_v = a1_bins_t[a1_valid]
            a1_loss = nn.functional.cross_entropy(logits_v, targets_v)
            a1_acc = (logits_v.argmax(-1) == targets_v).float().mean()

        # A2: BCE on valid labels (审计 A-4: 正样本 0.4% 极稀有 → pos_weight 加权)
        a2_logit = self.a2_head(pf_flat).reshape(B, ns)  # (B, ns)
        a2_t = torch.as_tensor(a2_label, device=pf.device, dtype=torch.float32)
        a2_valid = (a2_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        a2_loss = torch.tensor(0.0, device=pf.device)
        a2_acc = torch.tensor(0.0, device=pf.device)
        if a2_valid.any():
            logit_v = a2_logit[a2_valid]
            target_v = a2_t[a2_valid]
            # pos_weight 放大正样本梯度 (BCE: loss = -[w·y·log σ(x) + (1-y)·log(1-σ(x))])
            pos_weight = torch.tensor(
                self._a2_pos_weight, device=pf.device, dtype=torch.float32
            )
            a2_loss = nn.functional.binary_cross_entropy_with_logits(
                logit_v, target_v, pos_weight=pos_weight
            )
            a2_acc = ((torch.sigmoid(logit_v) > 0.5) == (target_v > 0.5)).float().mean()

        return a1_loss, a2_loss, a1_acc, a2_acc
