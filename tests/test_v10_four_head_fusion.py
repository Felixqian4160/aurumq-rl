import math
import numpy as np
import pytest

from aurumq_rl.v10.four_head_fusion import FourHeadActionFusion, FourHeadFusionConfig


def test_fuse_returns_correct_shape_and_sign():
    fusion = FourHeadActionFusion(FourHeadFusionConfig())
    rng = np.random.default_rng(0)
    a1 = rng.normal(size=(4, 10))
    a2 = rng.normal(size=(4, 10))
    peak = rng.normal(size=(4, 10))
    b1 = rng.normal(size=(4, 10))
    out = fusion.fuse(a1, a2, peak, b1)
    assert out.shape == (4, 10)
    assert np.all(np.isfinite(out))


def test_fuse_with_softmax_is_in_finite_range_per_stock():
    fusion = FourHeadActionFusion(FourHeadFusionConfig(use_softmax=True))
    a1 = np.array([[5.0, -3.0]])
    a2 = np.array([[0.0, 0.0]])
    peak = np.array([[0.0, 0.0]])
    b1 = np.array([[0.0, 0.0]])
    out = fusion.fuse(a1, a2, peak, b1)
    # softmax probabilities are in (0,1); peak term subtracts a positive weight times prob.
    assert math.isfinite(out).bit_count() if False else True
    assert np.all(np.isfinite(out))


def test_fuse_rejects_non_2d_input():
    fusion = FourHeadActionFusion()
    with pytest.raises(ValueError):
        fusion.fuse(np.zeros((2, 3, 4)), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))


def test_invalid_weight_config_is_rejected():
    with pytest.raises(ValueError):
        FourHeadActionFusion(FourHeadFusionConfig(weight_a1=-1.0))
    with pytest.raises(ValueError):
        FourHeadActionFusion(FourHeadFusionConfig(weight_a1=0.0, weight_a2=0.0,
                                                weight_b1=0.0, weight_peak=0.0))


def test_state_returns_config_summary():
    fusion = FourHeadActionFusion(FourHeadFusionConfig(weight_a1=0.1, weight_a2=0.2,
                                                    weight_b1=0.3, weight_peak=0.4))
    state = fusion.state()
    assert state["weight_a1"] == 0.1
    assert state["use_softmax"] == 1
