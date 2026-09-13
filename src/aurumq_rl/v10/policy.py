"""v10 — ONNX 5 头输出 Policy (action + a1/a2/peak/b1)

策略: 完全 override forward + get_distribution, 不走 SB3 标准 mlp_extractor+action_net
原因: SB3 在 _build 时自动用 proba_distribution_net 覆盖 self.action_net 为 Linear(64, action_dim)
     而我们的 LSTM 网络结构跟 mlp_extractor 没法简单对接
解决: 把所有 SB3 标准调用路径都 override, 让 LSTM 直接产 actions

架构:
  SB3 mlp_extractor 默认 (64 维中间层) — 不用
  self.lstm + self.lstm_fc — 编码
  self.action_net 重新定义为 Sequential[Linear(h*3, h), ReLU, Linear(h, 1)]
  override get_distribution: obs → _encode → action_net(pf_flat) → mean(B, ns) → DiagGaussian
  override evaluate_actions: 同 forward + log_prob
  override predict_values: obs → _encode → self.value_net(mv)
"""
from __future__ import annotations
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class WaveHunterV10Policy(ActorCriticPolicy):
    """v10 Policy — ONNX 5 头输出 (action + a1/a2/peak/b1 logits)."""

    def __init__(self, observation_space, action_space, lr_schedule,
                 *args: Any,
                 lstm_hidden: int = 128, lstm_layers: int = 1,
                 _n_stocks: int = 0, _window: int = 20, _n_factors: int = 323,
                 aux_lambda: float = 0.1,
                 a1_lambda: float = 1.0, a2_lambda: float = 1.0,
                 peak_lambda: float = 1.5, b1_lambda: float = 0.3,
                 a1_pos_weight: float = 20.0, a2_pos_weight: float = 40.0,
                 peak_pos_weight: float = 80.0, b1_pos_weight: float = 10.0,
                 **kwargs: Any):
        self._lstm_hidden = lstm_hidden
        self._lstm_layers = lstm_layers
        self._n_stocks = _n_stocks
        self._window = _window
        self._n_factors = _n_factors
        self._aux_lambda = aux_lambda
        self._a1_lambda = a1_lambda
        self._a2_lambda = a2_lambda
        self._peak_lambda = peak_lambda
        self._b1_lambda = b1_lambda
        self._a1_pos_weight = a1_pos_weight
        self._a2_pos_weight = a2_pos_weight
        self._peak_pos_weight = peak_pos_weight
        self._b1_pos_weight = b1_pos_weight
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build(self, lr_schedule):
        """构建网络. SB3 _build 后会覆盖 self.action_net 为 Linear(64, action_dim),
        我们必须在 _build 末尾再次覆盖, 才能让 LSTM 网络生效.
        """
        super()._build(lr_schedule)
        h = self._lstm_hidden

        # LSTM 编码
        self.lstm = nn.LSTM(self._n_factors, h, self._lstm_layers, batch_first=True)
        self.lstm_fc = nn.Linear(h, h)

        # 覆盖 action_net 为 LSTM 版本 (输入 h*3=192, 输出 1, 最后 reshape 为 ns)
        self.action_net = nn.Sequential(
            nn.Linear(h * 3, h), nn.ReLU(), nn.Linear(h, 1)
        )
        # value_net 输入 mv(B, h*2=128), 输出 1
        self.value_net = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1))

        # 4 头 BCE 监督
        self.a1_head = nn.Sequential(nn.Linear(h * 3, 64), nn.ReLU(), nn.Linear(64, 1))
        self.a2_head = nn.Sequential(nn.Linear(h * 3, 32), nn.ReLU(), nn.Linear(32, 1))
        self.peak_head = nn.Sequential(nn.Linear(h * 3, 32), nn.ReLU(), nn.Linear(32, 1))
        self.b1_head = nn.Sequential(nn.Linear(h * 3, 32), nn.ReLU(), nn.Linear(32, 1))

        if getattr(self, "ortho_init", False):
            for net in (self.action_net, self.a1_head, self.a2_head, self.peak_head, self.b1_head):
                net.apply(partial(self.init_weights, gain=1.0))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _encode(self, obs):
        """LSTM 编码 → pf(B, ns, h*3) + mv(B, h*2)"""
        ns, w, nf, h = self._n_stocks, self._window, self._n_factors, self._lstm_hidden
        x = obs.reshape(-1, ns, w, nf).reshape(-1, w, nf)
        lstm_out, _ = self.lstm(x)
        ps = torch.relu(self.lstm_fc(lstm_out[:, -1, :])).reshape(-1, ns, h)
        mv = torch.cat([ps.mean(1), ps.std(1) + 1e-6], dim=-1)
        pf = torch.cat([ps, mv.unsqueeze(1).expand(-1, ns, h * 2)], dim=-1)
        return pf, mv

    def _policy_forward(self, obs):
        """统一核心: obs → action_logits + value
        Returns: (mean_actions (B, ns), values (B,))"""
        pf, mv = self._encode(obs)
        B, ns, h3 = pf.shape
        # pf_flat (B*ns, h*3) → action_net → (B*ns, 1) → reshape (B, ns)
        action_logits = self.action_net(pf.reshape(B * ns, h3))  # (B*ns, 1)
        # 这里不用 sigmoid — DiagGaussian 直接接受 mean
        mean_actions = action_logits.reshape(B, ns)
        # value
        values = self.value_net(mv).squeeze(-1)  # (B,)
        return mean_actions, values

    # ── SB3 PPO 训练路径 ──
    def forward(self, obs, deterministic=False):
        """SB3 forward 标准签名: 返回 (actions, values, log_prob)"""
        mean_actions, values = self._policy_forward(obs)
        clamped_log_std = torch.clamp(self.log_std, -5.0, 2.0)
        dist = self.action_dist.proba_distribution(mean_actions, clamped_log_std)
        acts = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(acts)
        return acts, values, log_prob

    def evaluate_actions(self, obs, actions):
        """PPO train 用: 返回 (values, log_prob, entropy)"""
        mean_actions, values = self._policy_forward(obs)
        clamped_log_std = torch.clamp(self.log_std, -5.0, 2.0)
        dist = self.action_dist.proba_distribution(mean_actions, clamped_log_std)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs):
        """value 预测 (SB3 调用)"""
        _, mv = self._encode(obs)
        return self.value_net(mv).squeeze(-1)

    def get_distribution(self, obs):
        """_predict 调用: 返回分布"""
        mean_actions, _ = self._policy_forward(obs)
        clamped_log_std = torch.clamp(self.log_std, -5.0, 2.0)
        return self.action_dist.proba_distribution(mean_actions, clamped_log_std)

    def forward_with_heads(self, obs):
        """ONNX 多输出导出: 返回 5 个张量 (action + a1 + a2 + peak + b1)
        每个 shape = (batch, n_stocks)
        """
        pf, _ = self._encode(obs)
        B, ns, h3 = pf.shape
        pf_flat = pf.reshape(B * ns, h3)
        # action: sigmoid 过后的概率 (PPO 训练中是 mean_actions 本身)
        action_logits = self.action_net(pf_flat)  # (B*ns, 1)
        action = torch.sigmoid(action_logits.reshape(B, ns))
        # 4 头 BCE
        a1_logit = self.a1_head(pf_flat).reshape(B, ns)
        a2_logit = self.a2_head(pf_flat).reshape(B, ns)
        peak_logit = self.peak_head(pf_flat).reshape(B, ns)
        b1_logit = self.b1_head(pf_flat).reshape(B, ns)
        return action, a1_logit, a2_logit, peak_logit, b1_logit

    # ── 训练辅助损失 (4 头 BCE + 峰谷加权) ──
    def compute_aux_loss(self, obs, a1_bins, a2_label, valid_mask,
                         peak_label=None, b1_label=None):
        pf, _ = self._encode(obs)
        B, ns, h3 = pf.shape
        pf_flat = pf.reshape(B * ns, h3)

        a1_logit = self.a1_head(pf_flat).reshape(B, ns)
        a1_t = torch.as_tensor(a1_bins, device=pf.device, dtype=torch.float32)
        a1_valid = (a1_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        a1_loss = torch.tensor(0.0, device=pf.device)
        if a1_valid.any():
            logit_v = a1_logit[a1_valid]
            target_v = a1_t[a1_valid]
            pos_weight = torch.tensor(self._a1_pos_weight, device=pf.device, dtype=torch.float32)
            a1_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)

        a2_logit = self.a2_head(pf_flat).reshape(B, ns)
        a2_t = torch.as_tensor(a2_label, device=pf.device, dtype=torch.float32)
        a2_valid = (a2_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        a2_loss = torch.tensor(0.0, device=pf.device)
        if a2_valid.any():
            logit_v = a2_logit[a2_valid]
            target_v = a2_t[a2_valid]
            pos_weight = torch.tensor(self._a2_pos_weight, device=pf.device, dtype=torch.float32)
            a2_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)

        peak_logit = self.peak_head(pf_flat).reshape(B, ns)
        peak_t = torch.as_tensor(peak_label, device=pf.device, dtype=torch.float32) if peak_label is not None else torch.full((B, ns), -1.0, device=pf.device)
        peak_valid = (peak_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        peak_loss = torch.tensor(0.0, device=pf.device)
        if peak_valid.any():
            logit_v = peak_logit[peak_valid]
            target_v = peak_t[peak_valid]
            pos_weight = torch.tensor(self._peak_pos_weight, device=pf.device, dtype=torch.float32)
            peak_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)

        b1_logit = self.b1_head(pf_flat).reshape(B, ns)
        b1_t = torch.as_tensor(b1_label, device=pf.device, dtype=torch.float32) if b1_label is not None else torch.full((B, ns), -1.0, device=pf.device)
        b1_valid = (b1_t >= 0) & torch.as_tensor(valid_mask, device=pf.device)
        b1_loss = torch.tensor(0.0, device=pf.device)
        if b1_valid.any():
            logit_v = b1_logit[b1_valid]
            target_v = b1_t[b1_valid]
            pos_weight = torch.tensor(self._b1_pos_weight, device=pf.device, dtype=torch.float32)
            b1_loss = nn.functional.binary_cross_entropy_with_logits(logit_v, target_v, pos_weight=pos_weight)

        total_aux = (self._a1_lambda * a1_loss + self._a2_lambda * a2_loss
                     + self._peak_lambda * peak_loss + self._b1_lambda * b1_loss)
        return total_aux, a1_loss, a2_loss, peak_loss, b1_loss
