"""LSTM+PPO Joint Policy — one network, end-to-end, outputs position weights directly."""
from __future__ import annotations
from functools import partial
from typing import Any

import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class LstmJointPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 *args: Any, lstm_hidden: int = 128, lstm_layers: int = 1,
                 _n_stocks: int = 0, _window: int = 20, _n_factors: int = 0,
                 **kwargs: Any):
        self._lstm_hidden = lstm_hidden
        self._lstm_layers = lstm_layers
        self._n_stocks = _n_stocks
        self._window = _window
        self._n_factors = _n_factors
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule):
        # Bug AT 修复：_n_factors 默认 0 表示自动推断，避免 298 硬编码与 667 实际面板错位导致 ONNX 推理维度错误
        if self._n_factors == 0:
            try:
                import numpy as np
                obs_dim = int(np.prod(self.observation_space.shape))
                if self._n_stocks > 0 and self._window > 0 and obs_dim % (self._n_stocks * self._window) == 0:
                    self._n_factors = obs_dim // (self._n_stocks * self._window)
            except Exception:
                pass
            if self._n_factors == 0:
                raise ValueError(f"LstmJointPolicy _n_factors=0 无法推断，请显式传入 _n_factors (obs {self.observation_space.shape}, n_stocks={self._n_stocks}, window={self._window})")
        super()._build(lr_schedule)
        h = self._lstm_hidden
        self.lstm = nn.LSTM(self._n_factors, h, self._lstm_layers, batch_first=True)
        self.lstm_fc = nn.Linear(h, h)
        self.action_net = nn.Sequential(nn.Linear(h * 3, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value_net = nn.Sequential(nn.Linear(h * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.action_dist = DiagGaussianDistribution(self._n_stocks)
        # log_std 初始 -2.0：适中探索噪声
        # （-1.0 噪声太大 → 靠噪声+beta 假象；-3.0 太窄 → KL 发散。取折中）
        self.log_std = nn.Parameter(torch.clamp(torch.full((self._n_stocks,), -2.0), -5.0, 2.0))
        if getattr(self, "ortho_init", False):
            # gain=1.0：正常初始化，防止 action_net 权重坍缩到 ~0.001
            # （之前 gain=0.01 导致所有股票输出恒定 0.4637，无法选股）
            self.action_net.apply(partial(self.init_weights, gain=1.0))
            self.value_net.apply(partial(self.init_weights, gain=1.0))
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
