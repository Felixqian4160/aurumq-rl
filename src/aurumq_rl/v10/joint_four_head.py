"""Joint four-head adapter: route four head logits into the action bias.

This adapter is independent from frozen ``WaveHunterV10Policy``.  It produces
a per-stock action bias from the four head logits using the same fusion
formula as ``FourHeadActionFusion``.  The adapter never owns parameters:
the heads remain in the policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aurumq_rl.v10.four_head_fusion import FourHeadActionFusion, FourHeadFusionConfig


@dataclass(frozen=True)
class JointFourHeadConfig:
    enabled: bool = False
    logit_scale: float = 1.0
    use_softmax: bool = True


class JointFourHeadActionAdapter:
    """Attach head logits to a per-stock action bias."""

    def __init__(self, config: JointFourHeadConfig | None = None,
                 fusion: FourHeadActionFusion | None = None) -> None:
        self.config = config or JointFourHeadConfig()
        self.fusion = fusion or FourHeadActionFusion(FourHeadFusionConfig(use_softmax=self.config.use_softmax))
        self._cache: dict[str, torch.Tensor] = {}

    def disable(self) -> None:
        """No-op kept for symmetry with the env config interface."""
        return

    def fuse_logits(self, a1: torch.Tensor, a2: torch.Tensor,
                    peak: torch.Tensor, b1: torch.Tensor) -> np.ndarray:
        a1_np = a1.detach().cpu().numpy()
        a2_np = a2.detach().cpu().numpy()
        peak_np = peak.detach().cpu().numpy()
        b1_np = b1.detach().cpu().numpy()
        return self.fusion.fuse(a1_np, a2_np, peak_np, b1_np)

    def cache_head_logits(self, a1: torch.Tensor, a2: torch.Tensor,
                          peak: torch.Tensor, b1: torch.Tensor) -> None:
        """Cache the most recent head logits for downstream monitoring."""
        self._cache = {
            'a1': a1.detach(),
            'a2': a2.detach(),
            'peak': peak.detach(),
            'b1': b1.detach(),
        }
        norms = {}
        for name, tensor in self._cache.items():
            flat = tensor.reshape(-1)
            norms[f'head_{name}_norm'] = float(torch.norm(flat).item())
        self._last_norms = norms

    def last_norms(self) -> dict[str, float]:
        return dict(getattr(self, '_last_norms', {}))

    def bias(self, a1: torch.Tensor, a2: torch.Tensor,
             peak: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
        """Return per-stock action bias with shape (B, ns)."""
        if not self.config.enabled:
            return None
        score = self.fuse_logits(a1, a2, peak, b1)
        bias = torch.as_tensor(score * float(self.config.logit_scale),
                               dtype=a1.dtype, device=a1.device)
        return bias
