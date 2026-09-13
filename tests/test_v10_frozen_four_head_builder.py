import numpy as np
import pytest
import torch

from aurumq_rl.v10.four_head_fusion import FourHeadActionFusion, FourHeadFusionConfig
from aurumq_rl.v10.frozen_four_head_builder import FrozenFourHeadActionBuilder


def test_frozen_builder_returns_correct_shape():
    builder = FrozenFourHeadActionBuilder()
    builder.freeze()
    a1 = torch.zeros(2, 5)
    a2 = torch.zeros(2, 5)
    peak = torch.zeros(2, 5)
    b1 = torch.zeros(2, 5)
    out = builder(a1, a2, peak, b1)
    assert out.shape == (10, 1)


def test_frozen_builder_uses_fusion_weights():
    fusion = FourHeadActionFusion(FourHeadFusionConfig(weight_a1=1.0, weight_a2=0.0,
                                                    weight_b1=0.0, weight_peak=0.0))
    builder = FrozenFourHeadActionBuilder(fusion)
    builder.freeze()
    rng = np.random.default_rng(0)
    a1 = rng.normal(size=(3, 4))
    out = builder(torch.as_tensor(a1), torch.zeros(3, 4),
                 torch.zeros(3, 4), torch.zeros(3, 4))
    arr = out.detach().cpu().numpy().reshape(3, 4)
    expected = fusion.fuse(a1, np.zeros_like(a1), np.zeros_like(a1), np.zeros_like(a1))
    assert np.allclose(arr, expected)


def test_frozen_builder_freezes_parameters():
    builder = FrozenFourHeadActionBuilder()
    builder.freeze()
    requires_grad = [p.requires_grad for p in builder.parameters()]
    assert all(not r for r in requires_grad)


def test_frozen_builder_does_not_change_during_zero_grad_call():
    builder = FrozenFourHeadActionBuilder()
    builder.freeze()
    a1 = torch.zeros(2, 3, requires_grad=False)
    a2 = torch.zeros(2, 3, requires_grad=False)
    peak = torch.zeros(2, 3, requires_grad=False)
    b1 = torch.zeros(2, 3, requires_grad=False)
    out = builder(a1, a2, peak, b1)
    assert out.requires_grad is False
