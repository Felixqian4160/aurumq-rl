import numpy as np
import pytest
import torch

from aurumq_rl.v10.four_head_fusion import FourHeadActionFusion, FourHeadFusionConfig
from aurumq_rl.v10.joint_four_head import JointFourHeadActionAdapter, JointFourHeadConfig


def test_joint_adapter_bias_disabled_returns_none():
    adapter = JointFourHeadActionAdapter(JointFourHeadConfig(enabled=False))
    rng = np.random.default_rng(0)
    a1 = torch.as_tensor(rng.normal(size=(3, 5)))
    a2 = torch.as_tensor(rng.normal(size=(3, 5)))
    peak = torch.as_tensor(rng.normal(size=(3, 5)))
    b1 = torch.as_tensor(rng.normal(size=(3, 5)))
    assert adapter.bias(a1, a2, peak, b1) is None


def test_joint_adapter_bias_matches_fusion_when_enabled():
    fusion = FourHeadActionFusion(FourHeadFusionConfig())
    adapter = JointFourHeadActionAdapter(
        JointFourHeadConfig(enabled=True, logit_scale=1.0),
        fusion=fusion,
    )
    rng = np.random.default_rng(1)
    a1_np = rng.normal(size=(3, 5))
    a2_np = rng.normal(size=(3, 5))
    peak_np = rng.normal(size=(3, 5))
    b1_np = rng.normal(size=(3, 5))
    expected = fusion.fuse(a1_np, a2_np, peak_np, b1_np)
    a1 = torch.as_tensor(a1_np)
    a2 = torch.as_tensor(a2_np)
    peak = torch.as_tensor(peak_np)
    b1 = torch.as_tensor(b1_np)
    out = adapter.bias(a1, a2, peak, b1)
    assert out is not None
    arr = out.detach().cpu().numpy()
    assert np.allclose(arr, expected)


def test_joint_adapter_records_head_norms_when_enabled():
    adapter = JointFourHeadActionAdapter(JointFourHeadConfig(enabled=True))
    rng = np.random.default_rng(2)
    a1 = torch.as_tensor(rng.normal(size=(3, 5)))
    a2 = torch.as_tensor(rng.normal(size=(3, 5)))
    peak = torch.as_tensor(rng.normal(size=(3, 5)))
    b1 = torch.as_tensor(rng.normal(size=(3, 5)))
    adapter.cache_head_logits(a1, a2, peak, b1)
    norms = adapter.last_norms()
    assert set(norms) == {'head_a1_norm', 'head_a2_norm', 'head_peak_norm', 'head_b1_norm'}
    for name, value in norms.items():
        assert value > 0
