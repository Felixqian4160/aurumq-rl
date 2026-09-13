import json
from pathlib import Path

import pytest

from scripts.v10_timingfix.preflight_pair import (
    KS_PVALUE_THRESHOLD,
    LEGACY_NEWSTACK,
    check_pair,
)


def _write_model(path: Path, *, onnx_bytes: bytes, policy_class: str,
                 params: int, joint: bool, evt: bool,
                 obs_columns, obs_shape, legacy=None, evt_columns=None,
                 evt_target_loaded=False, evt_target_count=0):
    path.mkdir()
    (path / "policy.onnx").write_bytes(onnx_bytes)
    legacy = legacy or dict(LEGACY_NEWSTACK)
    metadata = {
        "hyperparams": {
            "policy_class": policy_class,
            "observation_names": obs_columns,
            "evt_columns": evt_columns or {},
            "joint_four_head": joint,
            "use_evt_labels": evt,
            **legacy,
        },
        "obs_shape": obs_shape,
    }
    (path / "metadata.json").write_text(json.dumps(metadata))
    (path / "training_summary.json").write_text(json.dumps({
        "parameter_count": params,
        "evt_target_loaded": evt_target_loaded,
        "evt_target_count": evt_target_count,
    }))


def test_rejects_identical_models(tmp_path):
    _write_model(tmp_path / "b", onnx_bytes=b"a" * 32, policy_class="Baseline",
                 params=100, joint=False, evt=False,
                 obs_columns=["alpha"], obs_shape=[60])
    _write_model(tmp_path / "n", onnx_bytes=b"a" * 32, policy_class="NewStack",
                 params=110, joint=True, evt=True,
                 obs_columns=["alpha"], obs_shape=[60],
                 evt_columns={"peak": "v10_evt_peak", "valley": "v10_evt_valley"})
    with pytest.raises(SystemExit) as exc:
        check_pair(tmp_path / "b", tmp_path / "n")
    assert "onnx_hash_different" in str(exc.value)


def test_rejects_same_parameter_count(tmp_path):
    _write_model(tmp_path / "b", onnx_bytes=b"a" * 32, policy_class="Baseline",
                 params=100, joint=False, evt=False,
                 obs_columns=["alpha"], obs_shape=[60])
    _write_model(tmp_path / "n", onnx_bytes=b"b" * 32, policy_class="NewStack",
                 params=100, joint=True, evt=True,
                 obs_columns=["alpha"], obs_shape=[60],
                 evt_columns={"peak": "v10_evt_peak", "valley": "v10_evt_valley"})
    with pytest.raises(SystemExit) as exc:
        check_pair(tmp_path / "b", tmp_path / "n")
    assert "parameter_count_different" in str(exc.value)


def test_rejects_same_policy_class(tmp_path):
    _write_model(tmp_path / "b", onnx_bytes=b"a" * 32, policy_class="Baseline",
                 params=100, joint=False, evt=False,
                 obs_columns=["alpha"], obs_shape=[60])
    _write_model(tmp_path / "n", onnx_bytes=b"b" * 32, policy_class="Baseline",
                 params=110, joint=True, evt=True,
                 obs_columns=["alpha"], obs_shape=[60],
                 evt_columns={"peak": "v10_evt_peak", "valley": "v10_evt_valley"})
    with pytest.raises(SystemExit) as exc:
        check_pair(tmp_path / "b", tmp_path / "n")
    assert "policy_class_different" in str(exc.value)


def test_rejects_label_in_observation(tmp_path):
    _write_model(tmp_path / "b", onnx_bytes=b"a" * 32, policy_class="Baseline",
                 params=100, joint=False, evt=False,
                 obs_columns=["alpha"], obs_shape=[60])
    _write_model(tmp_path / "n", onnx_bytes=b"b" * 32, policy_class="NewStack",
                 params=110, joint=True, evt=True,
                 obs_columns=["alpha", "v10_a1_point"], obs_shape=[60],
                 evt_columns={"peak": "v10_evt_peak", "valley": "v10_evt_valley"})
    with pytest.raises(SystemExit) as exc:
        check_pair(tmp_path / "b", tmp_path / "n")
    assert "labels_excluded_from_observation" in str(exc.value)


def test_rejects_legacy_param_drift(tmp_path):
    bad_legacy = dict(LEGACY_NEWSTACK); bad_legacy["peak_lambda"] = 0.0
    _write_model(tmp_path / "b", onnx_bytes=b"a" * 32, policy_class="Baseline",
                 params=100, joint=False, evt=False,
                 obs_columns=["alpha"], obs_shape=[60])
    _write_model(tmp_path / "n", onnx_bytes=b"b" * 32, policy_class="NewStack",
                 params=110, joint=True, evt=True,
                 obs_columns=["alpha"], obs_shape=[60], legacy=bad_legacy,
                 evt_columns={"peak": "v10_evt_peak", "valley": "v10_evt_valley"})
    with pytest.raises(SystemExit) as exc:
        check_pair(tmp_path / "b", tmp_path / "n")
    assert "new_stack_legacy_hyperparams" in str(exc.value)


def test_accepts_real_new_stack(tmp_path):
    base = tmp_path / "b"
    new = tmp_path / "n"
    base.mkdir(); new.mkdir()
    (base / "policy.onnx").write_bytes(b"a" * 32)
    (new / "policy.onnx").write_bytes(b"b" * 32)
    base_metadata = {
        "hyperparams": {"policy_class": "Baseline", "observation_names": ["alpha"]},
        "obs_shape": [60], "joint_four_head": False, "use_evt_labels": False,
    }
    new_metadata = {
        "hyperparams": {
            "policy_class": "JointNewStack", "observation_names": ["alpha"],
            "joint_four_head": True, "use_evt_labels": True,
            "aux_lambda": 0.1, "a1_lambda": 1.0, "a2_lambda": 1.0,
            "peak_lambda": 1.5, "b1_lambda": 0.3,
        },
        "obs_shape": [60],
        "evt_columns": {"peak": "v10_evt_peak", "valley": "v10_evt_valley"},
    }
    (base / "metadata.json").write_text(json.dumps(base_metadata))
    (new / "metadata.json").write_text(json.dumps(new_metadata))
    (base / "training_summary.json").write_text(json.dumps({"parameter_count": 100}))
    (new / "training_summary.json").write_text(json.dumps({"parameter_count": 128,
                                                            "evt_target_loaded": True,
                                                            "evt_target_count": 50}))
    assert check_pair(tmp_path / "b", tmp_path / "n")["passed"]


def test_ks_check_required_when_enabled(tmp_path):
    _write_model(tmp_path / "b", onnx_bytes=b"a" * 32, policy_class="Baseline",
                 params=100, joint=False, evt=False,
                 obs_columns=["alpha"], obs_shape=[60])
    _write_model(tmp_path / "n", onnx_bytes=b"b" * 32, policy_class="JointNewStack",
                 params=128, joint=True, evt=True,
                 obs_columns=["alpha"], obs_shape=[60],
                 evt_columns={"peak": "v10_evt_peak", "valley": "v10_evt_valley"},
                 evt_target_loaded=True, evt_target_count=50)
    with pytest.raises(SystemExit) as exc:
        check_pair(tmp_path / "b", tmp_path / "n", with_ks=True)
    assert "obs_distribution_different" in str(exc.value)


def test_threshold_exposed():
    assert KS_PVALUE_THRESHOLD == 0.01


def test_head_mapping_consistent(tmp_path):
    """Preflight item #10: training script, signal_engine, and preflight
    must all reference the same HEAD_MAPPING object. If any side imports a
    stale local copy the check fails immediately."""
    from scripts.v10_timingfix import preflight_pair
    ok, info = preflight_pair._head_mapping_consistent()
    assert ok is True, info
    assert info["same_object"] is True
    assert info["train_uses_same"] is True
    assert info["decision_match"] is True
    assert info["buy_head"] == "a1_head"
    assert info["sell_head"] == "peak_head"
