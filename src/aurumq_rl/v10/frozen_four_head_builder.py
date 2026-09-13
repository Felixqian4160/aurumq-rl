"""WaveHunter v10 四头融合 action_net (冻结四头版本)。

使用方式：
    builder = FrozenFourHeadActionBuilder(four_head_fusion)
    builder.freeze();                  # 锁定现有 head 权重
    fused_logits = builder(pf_flat);   # 与 frozen 旧 policy 同 shape (B*ns, 1)
    # fused_logits 可直接加到旧 policy 的 action_logits

设计原则：
    1. 只读 ONNX/已保存模型，不会重新训练四头
    2. 不修改 frozen v9 旧代码
    3. 调用图保持简单，便于单测
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from aurumq_rl.v10.four_head_fusion import FourHeadActionFusion, FourHeadFusionConfig


class FrozenFourHeadActionBuilder(nn.Module):
    """Lightweight frozen-head action bias builder."""

    def __init__(self, fusion: FourHeadActionFusion | None = None) -> None:
        super().__init__()
        self.fusion = fusion or FourHeadActionFusion(FourHeadFusionConfig())

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, a1_logit: torch.Tensor, a2_logit: torch.Tensor,
                peak_logit: torch.Tensor, b1_logit: torch.Tensor) -> torch.Tensor:
        """Return a (B*ns, 1) action bias compatible with the frozen action_net output."""
        a1 = a1_logit.detach().cpu().numpy()
        a2 = a2_logit.detach().cpu().numpy()
        peak = peak_logit.detach().cpu().numpy()
        b1 = b1_logit.detach().cpu().numpy()
        score = self.fusion.fuse(a1, a2, peak, b1)
        return torch.as_tensor(score.reshape(-1, 1), dtype=a1_logit.dtype, device=a1_logit.device)
