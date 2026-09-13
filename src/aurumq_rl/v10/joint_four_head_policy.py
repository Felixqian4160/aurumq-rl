"""WaveHunter v10 joint four-head policy.

This is a new v10-only policy subclass.  It keeps the existing four binary
heads and routes their differentiable probabilities into the PPO action mean.
EVT labels are not used here; the current experiment changes only the policy
connection, not the label source.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurumq_rl.v10.policy import WaveHunterV10Policy


class JointWaveHunterV10Policy(WaveHunterV10Policy):
    """WaveHunter policy with differentiable four-head action fusion."""

    def __init__(self, observation_space, action_space, lr_schedule, *args: Any,
                 joint_logit_scale: float = 1.0,
                 joint_use_softmax: bool = True,
                 **kwargs: Any):
        self._joint_logit_scale = float(joint_logit_scale)
        self._joint_use_softmax = bool(joint_use_softmax)
        if self._joint_logit_scale < 0.0:
            raise ValueError("joint_logit_scale must be >= 0")
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        # Explicit trainable fusion projection makes the Joint policy structurally
        # distinct from the baseline instead of only changing its forward formula.
        self.joint_fusion_bias = nn.Parameter(torch.zeros(1))

    def _policy_forward(self, obs):
        """Encode once, then make action mean depend on all four head logits."""
        pf, mv = self._encode(obs)
        batch, n_stocks, hidden3 = pf.shape
        flat = pf.reshape(batch * n_stocks, hidden3)
        base_actions = self.action_net(flat).reshape(batch, n_stocks)
        head_logits = [
            head(flat).reshape(batch, n_stocks)
            for head in (self.a1_head, self.a2_head, self.peak_head, self.b1_head)
        ]
        a1, a2, peak, b1 = head_logits
        if self._joint_use_softmax:
            # Cross-sectional softmax keeps the four signals differentiable
            # while preventing an absolute probability offset from dominating.
            a1_p = F.softmax(a1, dim=-1)
            a2_p = F.softmax(a2, dim=-1)
            peak_p = F.softmax(peak, dim=-1)
            b1_p = F.softmax(b1, dim=-1)
        else:
            a1_p, a2_p, peak_p, b1_p = [torch.sigmoid(x) for x in head_logits]
        fused = 0.40 * a1_p + 0.40 * a2_p + 0.20 * b1_p - 0.30 * peak_p
        mean_actions = base_actions + self._joint_logit_scale * fused + self.joint_fusion_bias
        values = self.value_net(mv).squeeze(-1)
        return mean_actions, values

    def joint_head_norms(self) -> dict[str, float]:
        """Return L2 norms of the four trainable head parameter groups."""
        return {
            name: float(torch.sqrt(sum(torch.sum(p.detach() ** 2) for p in head.parameters())).item())
            for name, head in (
                ("a1_head_l2", self.a1_head),
                ("a2_head_l2", self.a2_head),
                ("peak_head_l2", self.peak_head),
                ("b1_head_l2", self.b1_head),
            )
        }
