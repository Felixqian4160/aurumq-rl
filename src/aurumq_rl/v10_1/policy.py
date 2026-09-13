"""v10.1 独立四头策略。

这是 v10 policy 的独立副本命名空间，只修改模块/类名称，不修改 v10。
"""
from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class WaveHunterV10_1Policy(ActorCriticPolicy):
    """v10.1 Policy：PPO action + A1/A2/Peak/B1 四头 logits。"""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args: Any,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        _n_stocks: int = 0,
        _window: int = 20,
        _n_factors: int = 323,
        aux_lambda: float = 0.1,
        a1_lambda: float = 1.0,
        a2_lambda: float = 1.0,
        peak_lambda: float = 1.5,
        b1_lambda: float = 0.3,
        a1_pos_weight: float = 20.0,
        a2_pos_weight: float = 40.0,
        peak_pos_weight: float = 80.0,
        b1_pos_weight: float = 10.0,
        **kwargs: Any,
    ):
        self._lstm_hidden = lstm_hidden
        self._lstm_layers = lstm_layers
        self._n_stocks = _n_stocks
        self._window = _window
        self._n_factors = _n_factors
        self._aux_lambda = aux_lambda
        self._a1_lambda, self._a2_lambda = a1_lambda, a2_lambda
        self._peak_lambda, self._b1_lambda = peak_lambda, b1_lambda
        self._a1_pos_weight, self._a2_pos_weight = a1_pos_weight, a2_pos_weight
        self._peak_pos_weight, self._b1_pos_weight = peak_pos_weight, b1_pos_weight
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        h = self._lstm_hidden
        self.lstm = nn.LSTM(self._n_factors, h, self._lstm_layers, batch_first=True)
        self.lstm_fc = nn.Linear(h, h)
        self.action_net = nn.Sequential(nn.Linear(h * 3, h), nn.ReLU(), nn.Linear(h, 1))
        self.value_net = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))
        self.a1_head = nn.Sequential(nn.Linear(h * 3, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a2_head = nn.Sequential(nn.Linear(h * 3, 32), nn.ReLU(), nn.Linear(32, 1))
        self.peak_head = nn.Sequential(nn.Linear(h * 3, 32), nn.ReLU(), nn.Linear(32, 1))
        self.b1_head = nn.Sequential(nn.Linear(h * 3, 32), nn.ReLU(), nn.Linear(32, 1))
        if getattr(self, "ortho_init", False):
            for net in (self.action_net, self.a1_head, self.a2_head, self.peak_head, self.b1_head):
                net.apply(partial(self.init_weights, gain=1.0))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _encode(self, obs):
        ns, window, nf, h = self._n_stocks, self._window, self._n_factors, self._lstm_hidden
        x = obs.reshape(-1, ns, window, nf).reshape(-1, window, nf)
        out, _ = self.lstm(x)
        ps = torch.relu(self.lstm_fc(out[:, -1, :])).reshape(-1, ns, h)
        mv = torch.cat([ps.mean(1), ps.std(1) + 1e-6], dim=-1)
        pf = torch.cat([ps, mv.unsqueeze(1).expand(-1, ns, h * 2)], dim=-1)
        return pf, mv

    def _policy_forward(self, obs):
        pf, mv = self._encode(obs)
        batch, ns, h3 = pf.shape
        mean_actions = self.action_net(pf.reshape(batch * ns, h3)).reshape(batch, ns)
        values = self.value_net(mv).squeeze(-1)
        return mean_actions, values

    def forward(self, obs, deterministic=False):
        mean_actions, values = self._policy_forward(obs)
        dist = self.action_dist.proba_distribution(mean_actions, torch.clamp(self.log_std, -5.0, 2.0))
        actions = dist.get_actions(deterministic=deterministic)
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        mean_actions, values = self._policy_forward(obs)
        dist = self.action_dist.proba_distribution(mean_actions, torch.clamp(self.log_std, -5.0, 2.0))
        return values, dist.log_prob(actions), dist.entropy()

    def predict_values(self, obs):
        _, mv = self._encode(obs)
        return self.value_net(mv).squeeze(-1)

    def get_distribution(self, obs):
        mean_actions, _ = self._policy_forward(obs)
        return self.action_dist.proba_distribution(mean_actions, torch.clamp(self.log_std, -5.0, 2.0))

    def forward_with_heads(self, obs):
        pf, _ = self._encode(obs)
        batch, ns, h3 = pf.shape
        flat = pf.reshape(batch * ns, h3)
        action = torch.sigmoid(self.action_net(flat).reshape(batch, ns))
        a1 = self.a1_head(flat).reshape(batch, ns)
        a2 = self.a2_head(flat).reshape(batch, ns)
        peak = self.peak_head(flat).reshape(batch, ns)
        b1 = self.b1_head(flat).reshape(batch, ns)
        return action, a1, a2, peak, b1

    def compute_aux_loss(self, obs, a1_label, a2_label, valid_mask, peak_label, b1_label):
        pf, _ = self._encode(obs)
        batch, ns, h3 = pf.shape
        flat = pf.reshape(batch * ns, h3)
        logits = [head(flat).reshape(batch, ns) for head in (
            self.a1_head, self.a2_head, self.peak_head, self.b1_head)]
        labels = [torch.as_tensor(x, device=pf.device, dtype=torch.float32)
                  for x in (a1_label, a2_label, peak_label, b1_label)]
        valid = torch.as_tensor(valid_mask, device=pf.device, dtype=torch.bool)
        weights = [self._a1_pos_weight, self._a2_pos_weight,
                   self._peak_pos_weight, self._b1_pos_weight]
        losses = []
        for logit, label, pos_weight in zip(logits, labels, weights):
            mask = (label >= 0) & valid
            if mask.any():
                losses.append(nn.functional.binary_cross_entropy_with_logits(
                    logit[mask], label[mask],
                    pos_weight=torch.tensor(pos_weight, device=pf.device)))
            else:
                losses.append(torch.zeros((), device=pf.device))
        l1, l2, lp, lb = losses
        total = self._a1_lambda * l1 + self._a2_lambda * l2 + self._peak_lambda * lp + self._b1_lambda * lb
        return total, l1, l2, lp, lb
