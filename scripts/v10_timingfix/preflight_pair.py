#!/usr/bin/env python3
"""Pre-flight guard for paired v10_timingfix experiments.

Implements the eight-item contract from the v10_timingfix statistical
verification specification. Designed to run before any paired smoke test so
that a "silent regression" (Baseline and New Stack collapsing into the same
model) is caught within seconds instead of after a 100k training run.

Deterministic assertions:
  1.  onnx hash differs
  2.  parameter count differs
  3.  policy class differs
  4.  observation shape contract holds
  5.  label columns (A1/A2/Peak/B1/ZigZag/EVT) stay outside observation
  6.  New Stack joint_four_head enabled
  7.  New Stack use_evt_labels enabled and EVT columns recorded
  8.  legacy New Stack hyperparameters match exactly
  9.  New Stack EVT target array actually loaded into the training buffer
 10.  head ↔ target ↔ decision mapping is the same canonical object across
      the training script, the simulation decision engine, and this check

Statistical assertion (optional, opt-in via --with-ks):
  - KS two-sample test on sampled observation features between the two models.
    Comparison is restricted to legitimate observation features (never to
    label columns), to avoid leaking labels into the check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

LEGACY_NEWSTACK = {
    "aux_lambda": 0.1,
    "a1_lambda": 1.0,
    "a2_lambda": 1.0,
    "peak_lambda": 1.5,
    "b1_lambda": 0.3,
}
FORBIDDEN_OBS = ("a1_label", "a2_label", "peak_label", "b1_label",
                 "v10_a1_point", "v10_a2_interval", "v10_zig_peak", "v10_b1_interval",
                 "v10_evt_peak", "v10_evt_valley")
KS_PVALUE_THRESHOLD = 0.01


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(model: Path) -> tuple[dict, dict]:
    metadata = json.loads((model / "metadata.json").read_text())
    summary_path = model / "training_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return metadata, summary


def _params(metadata: dict, summary: dict) -> int:
    if "parameter_count" in summary:
        return int(summary["parameter_count"])
    return int(metadata.get("parameter_count", -1))


def _policy_class(metadata: dict) -> str:
    hp = metadata.get("hyperparams", {}) or {}
    return hp.get("policy_class") or metadata.get("policy_class") or ""


def _obs_columns(metadata: dict) -> list[str]:
    hp = metadata.get("hyperparams", {}) or {}
    return list(hp.get("observation_names") or metadata.get("observation_names")
                or metadata.get("factor_names") or [])


def _evt_columns(metadata: dict) -> dict[str, str]:
    hp = metadata.get("hyperparams", {}) or {}
    return dict(hp.get("evt_columns") or metadata.get("evt_columns") or {})


def _new_stack_flags(metadata: dict) -> tuple[bool, bool]:
    hp = metadata.get("hyperparams", {}) or {}
    joint = bool(metadata.get("joint_four_head") or hp.get("joint_four_head"))
    evt = bool(metadata.get("use_evt_labels") or hp.get("use_evt_labels"))
    return joint, evt


def _forbidden_columns(metadata: dict) -> list[str]:
    return [c for c in _obs_columns(metadata)
            if any(token.lower() in str(c).lower() for token in FORBIDDEN_OBS)]


def _legacy_match(metadata: dict) -> tuple[bool, dict[str, object]]:
    hp = metadata.get("hyperparams", {}) or {}
    actual = {k: hp.get(k) for k in LEGACY_NEWSTACK}
    expected = dict(LEGACY_NEWSTACK)
    ok = all(actual[k] == expected[k] for k in expected)
    return ok, {"expected": expected, "actual": actual}


def _ks_check(baseline_obs: Iterable[float], new_obs: Iterable[float]) -> tuple[bool, float]:
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return False, float("nan")
    stat, p = ks_2samp(list(baseline_obs), list(new_obs))
    return p < KS_PVALUE_THRESHOLD, float(p)


def _head_mapping_consistent() -> tuple[bool, dict[str, object]]:
    """Verify the head ↔ target ↔ decision mapping is the same canonical
    object across train_loop, signal_engine, and this preflight script.

    The risk we guard against: silently renaming or reassigning the mapping
    on one side, breaking the buy/sell semantic without changing the other
    side. Identity comparison catches this immediately.
    """
    import sys as _sys
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    if str(repo_root / "src") not in _sys.path:
        _sys.path.insert(0, str(repo_root / "src"))
    if str(here) not in _sys.path:
        _sys.path.insert(0, str(here))
    try:
        import aurumq_rl.v10.head_mapping as h_src
        import head_mapping as h_scripts
        import train_loop as tl
        from aurumq_rl.v10.signal_engine import HEAD_MAPPING as h_signal
        from aurumq_rl.v10.signal_engine import BUY_HEAD, SELL_HEAD
        same_object = (h_src.HEAD_MAPPING is h_scripts.HEAD_MAPPING is h_signal)
        train_uses_same = (getattr(tl, "HEAD_MAPPING", None) is h_src.HEAD_MAPPING)
        # Active (targeted) heads must match what signal_engine selects
        active = [name for name, spec in h_src.HEAD_MAPPING.items() if spec.target is not None]
        buy_head = next((n for n, s in h_src.HEAD_MAPPING.items() if s.decision == "buy"), None)
        sell_head = next((n for n, s in h_src.HEAD_MAPPING.items() if s.decision == "sell"), None)
        decision_match = (buy_head == BUY_HEAD) and (sell_head == SELL_HEAD)
        ok = same_object and train_uses_same and decision_match
        return ok, {
            "same_object": same_object,
            "train_uses_same": train_uses_same,
            "decision_match": decision_match,
            "active_heads": active,
            "buy_head": buy_head,
            "sell_head": sell_head,
        }
    except Exception as exc:  # pragma: no cover
        return False, {"error": repr(exc)}


def check_pair(
    baseline: str | Path,
    new_stack: str | Path,
    *,
    with_ks: bool = False,
    baseline_obs_sample: Iterable[float] | None = None,
    new_stack_obs_sample: Iterable[float] | None = None,
) -> dict:
    base, new = Path(baseline), Path(new_stack)
    bm, bs = _load(base); nm, ns = _load(new)
    joint_new, evt_new = _new_stack_flags(nm)
    joint_base, evt_base = _new_stack_flags(bm)

    legacy_ok, legacy_diff = _legacy_match(nm)

    head_ok, head_diff = _head_mapping_consistent()

    checks: dict[str, bool] = {
        "onnx_hash_different": _sha256(base / "policy.onnx") != _sha256(new / "policy.onnx"),
        "parameter_count_different": _params(bm, bs) != _params(nm, ns),
        "policy_class_different": bool(_policy_class(bm)) and bool(_policy_class(nm)) and _policy_class(bm) != _policy_class(nm),
        "observation_shape_valid": bool(bm.get("obs_shape")) and bm.get("obs_shape") == nm.get("obs_shape"),
        "labels_excluded_from_observation": not _forbidden_columns(bm) and not _forbidden_columns(nm),
        "new_stack_joint_enabled": joint_new and not joint_base,
        "new_stack_evt_enabled_and_recorded": evt_new and not evt_base and bool(_evt_columns(nm)),
        "new_stack_legacy_hyperparams": legacy_ok,
        "new_stack_evt_target_loaded": bool(ns.get("evt_target_loaded")) and int(ns.get("evt_target_count", 0)) > 0
        and not bool(bs.get("evt_target_loaded")),
        "head_mapping_consistent": head_ok,
    }

    result: dict = {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline": {
            "path": str(base),
            "onnx_sha256": _sha256(base / "policy.onnx"),
            "parameter_count": _params(bm, bs),
            "policy_class": _policy_class(bm),
            "obs_shape": bm.get("obs_shape"),
            "obs_columns": _obs_columns(bm),
            "joint_four_head": joint_base,
            "use_evt_labels": evt_base,
        },
        "new_stack": {
            "path": str(new),
            "onnx_sha256": _sha256(new / "policy.onnx"),
            "parameter_count": _params(nm, ns),
            "policy_class": _policy_class(nm),
            "obs_shape": nm.get("obs_shape"),
            "obs_columns": _obs_columns(nm),
            "joint_four_head": joint_new,
            "use_evt_labels": evt_new,
            "evt_columns": _evt_columns(nm),
        },
        "legacy_newstack_params": legacy_diff,
    }

    if with_ks:
        if baseline_obs_sample is None or new_stack_obs_sample is None:
            checks["obs_distribution_different"] = False
            result["ks_note"] = "obs sample not provided"
        else:
            ok, p = _ks_check(baseline_obs_sample, new_stack_obs_sample)
            checks["obs_distribution_different"] = ok
            result["ks_pvalue"] = p
        result["passed"] = all(checks.values())

    if not result["passed"]:
        failed = [k for k, v in checks.items() if not v]
        raise SystemExit(json.dumps({**result, "failed_checks": failed}, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--new-stack", required=True)
    parser.add_argument("--with-ks", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_pair(args.baseline, args.new_stack, with_ks=args.with_ks),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
