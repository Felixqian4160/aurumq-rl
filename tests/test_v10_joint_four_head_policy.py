import numpy as np
import pytest
import torch

from aurumq_rl.v10.four_head_fusion import FourHeadActionFusion, FourHeadFusionConfig
from aurumq_rl.v10.joint_four_head import JointFourHeadActionAdapter, JointFourHeadConfig


def test_disabled_adapter_returns_none_bias():
    adapter = JointFourHeadActionAdapter(JointFourHeadConfig(enabled=False))
    rng = np.random.default_rng(0)
    a1 = torch.as_tensor(rng.normal(size=(3, 5)))
    a2 = torch.as_tensor(rng.normal(size=(3, 5)))
    peak = torch.as_tensor(rng.normal(size=(3, 5)))
    b1 = torch.as_tensor(rng.normal(size=(3, 5)))
    assert adapter.bias(a1, a2, peak, b1) is None


def test_enabled_adapter_matches_fusion():
    fusion = FourHeadActionFusion(FourHeadFusionConfig())
    adapter = JointFourHeadActionAdapter(JointFourHeadConfig(enabled=True), fusion=fusion)
    rng = np.random.default_rng(1)
    a1 = rng.normal(size=(3, 5))
    a2 = rng.normal(size=(3, 5))
    peak = rng.normal(size=(3, 5))
    b1 = rng.normal(size=(3, 5))
    expected = fusion.fuse(a1, a2, peak, b1)
    out = adapter.bias(torch.as_tensor(a1), torch.as_tensor(a2),
                       torch.as_tensor(peak), torch.as_tensor(b1))
    assert out is not None
    assert np.allclose(out.detach().cpu().numpy(), expected)


def test_norm_recording_matches_tensor_layout():
    adapter = JointFourHeadActionAdapter(JointFourHeadConfig(enabled=True))
    rng = np.random.default_rng(2)
    a1 = torch.as_tensor(rng.normal(size=(3, 5)))
    a2 = torch.as_tensor(rng.normal(size=(3, 5)))
    peak = torch.as_tensor(rng.normal(size=(3, 5)))
    b1 = torch.as_tensor(rng.normal(size=(3, 5)))
    adapter.cache_head_logits(a1, a2, peak, b1)
    norms = adapter.last_norms()
    assert set(norms) == {'head_a1_norm', 'head_a2_norm', 'head_peak_norm', 'head_b1_norm'}
    for v in norms.values():
        assert v > 0
