"""Strict training/simulation contract for v10.1.

The model metadata is the source of truth for parameters that affect the
executable portfolio.  Simulation must pass those values explicitly.  A
legacy LstmWeightEnv model does not charge slippage during training; that
known mismatch may only be used when the config explicitly acknowledges it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

CONTRACT_VERSION = "v10.1_training_sim_contract_v2"

PARAMETER_NAMES = (
    "cost_bps",
    "slippage_bps",
    "max_position_pct",
    "top_k",
    "rebalance_days",
)

EXECUTION_CONTRACT = {
    "signal_timing": "T_close",
    "execution_timing": "T+1_raw_open",
    "mark_timing": "T+1_close",
    "order_priority": "sell_before_buy",
    "terminal_action": "end_of_period_force_close",
}


class ContractViolationError(ValueError):
    """Raised when a simulation config violates the model contract."""

    def __init__(self, violations: list[dict[str, Any]]):
        self.violations = violations
        lines = [f"Training/Simulation contract violation ({len(violations)} item(s)):"]
        for violation in violations:
            lines.append(
                f"  - {violation['param']}: {violation['description']}\n"
                f"      expected: {violation['expected']}\n"
                f"      actual: {violation['actual']}"
            )
        super().__init__("\n".join(lines))


def _number(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"metadata.hyperparams missing {name}")
    return float(value)


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-9
    return left == right


def panel_identity(path: str | Path) -> dict[str, Any]:
    """Return a stable identity for the panel referenced by a model/config."""
    panel = Path(path).resolve()
    if not panel.is_file():
        raise FileNotFoundError(panel)
    stat = panel.stat()
    return {"path": str(panel), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_training_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable contract from one training run's metadata.

    ``simulation`` describes the friction that a ledger is expected to use.
    For the legacy A path, ``training.slippage_bps`` is zero while simulation
    slippage is still the declared execution friction.  The resulting
    difference is retained as a visible, explicit legacy exception; it is not
    silently inferred by the runner.
    """
    hp = metadata.get("hyperparams")
    if not isinstance(hp, dict):
        raise ValueError("metadata.hyperparams must be a dict")

    use_shared_sm = bool(hp.get("use_shared_sm", False))
    cost_bps = _number(hp.get("cost_bps"), "cost_bps")
    sm_cost_bps = hp.get("sm_cost_bps")
    if use_shared_sm and sm_cost_bps is not None:
        training_cost_bps = float(sm_cost_bps)
    else:
        training_cost_bps = cost_bps

    simulation_slippage_bps = _number(
        hp.get("sm_slippage_bps", 10.0), "sm_slippage_bps"
    )
    training_slippage_bps = simulation_slippage_bps if use_shared_sm else 0.0

    values = {
        "cost_bps": {
            "training": training_cost_bps,
            "simulation": cost_bps,
            "must_match": True,
        },
        "slippage_bps": {
            "training": training_slippage_bps,
            "simulation": simulation_slippage_bps,
            "must_match": use_shared_sm,
        },
        "max_position_pct": {
            "training": _number(hp.get("max_position_pct"), "max_position_pct"),
            "simulation": _number(hp.get("max_position_pct"), "max_position_pct"),
            "must_match": True,
        },
        "top_k": {
            "training": int(hp.get("top_k")),
            "simulation": int(hp.get("top_k")),
            "must_match": True,
        },
        "rebalance_days": {
            "training": int(hp.get("rebalance_days")),
            "simulation": int(hp.get("rebalance_days")),
            "must_match": True,
        },
    }

    window = hp.get("window")
    forward_period = hp.get("forward_period")
    if window is None or forward_period is None:
        raise ValueError("metadata.hyperparams must include window and forward_period")

    panel_path = metadata.get("panel") or hp.get("panel")
    if not isinstance(panel_path, (str, Path)):
        raise ValueError("metadata.panel must be a filesystem path")
    return {
        "version": CONTRACT_VERSION,
        "parameters": values,
        "use_shared_sm": use_shared_sm,
        "execution_contract": dict(EXECUTION_CONTRACT),
        "training_only": {
            "window": int(window),
            "forward_period": int(forward_period),
        },
        "panel_identity": panel_identity(panel_path),
        "legacy_exceptions": ["slippage_bps"] if not use_shared_sm else [],
    }


def simulation_contract_audit(
    metadata: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Return a complete contract audit without raising."""
    contract = metadata.get("training_contract")
    if not isinstance(contract, dict):
        return {
            "passed": False,
            "contract_version": None,
            "violations": [{
                "param": "training_contract",
                "description": "metadata is missing training_contract",
                "expected": CONTRACT_VERSION,
                "actual": "missing",
            }],
            "explicit_exceptions": [],
        }

    if contract.get("version") != CONTRACT_VERSION:
        return {
            "passed": False,
            "contract_version": contract.get("version"),
            "violations": [{
                "param": "training_contract.version",
                "description": "contract version mismatch",
                "expected": CONTRACT_VERSION,
                "actual": contract.get("version"),
            }],
            "explicit_exceptions": [],
        }

    params = contract.get("parameters")
    if not isinstance(params, dict):
        return {
            "passed": False,
            "contract_version": CONTRACT_VERSION,
            "violations": [{
                "param": "training_contract.parameters",
                "description": "parameter map is missing",
                "expected": list(PARAMETER_NAMES),
                "actual": params,
            }],
            "explicit_exceptions": [],
        }

    use_shared_sm = bool(contract.get("use_shared_sm", False))
    violations: list[dict[str, Any]] = []
    explicit_exceptions: list[str] = []

    for name in PARAMETER_NAMES:
        entry = params.get(name)
        if not isinstance(entry, dict):
            violations.append({
                "param": name,
                "description": "parameter is missing from contract",
                "expected": "training/simulation values",
                "actual": entry,
            })
            continue

        if name not in cfg:
            violations.append({
                "param": name,
                "description": "simulation config must set this parameter explicitly",
                "expected": entry.get("simulation"),
                "actual": "missing",
            })
            continue

        expected_simulation = entry.get("simulation")
        actual = cfg[name]
        if not _equal(expected_simulation, actual):
            violations.append({
                "param": name,
                "description": "simulation value differs from metadata contract",
                "expected": expected_simulation,
                "actual": actual,
            })

        training_value = entry.get("training")
        must_match = bool(entry.get("must_match", True))
        if _equal(training_value, expected_simulation):
            continue
        if must_match:
            violations.append({
                "param": name,
                "description": "training and simulation values differ",
                "expected": training_value,
                "actual": expected_simulation,
            })
        elif name == "slippage_bps":
            if use_shared_sm:
                violations.append({
                    "param": name,
                    "description": "shared-SM training and simulation slippage must match",
                    "expected": training_value,
                    "actual": expected_simulation,
                })
            elif cfg.get("allow_legacy_friction_mismatch") is True:
                explicit_exceptions.append("legacy_training_slippage_not_modeled")
            else:
                violations.append({
                    "param": name,
                    "description": (
                        "legacy training reward and simulation friction differ; "
                        "set allow_legacy_friction_mismatch=true explicitly"
                    ),
                    "expected": True,
                    "actual": cfg.get("allow_legacy_friction_mismatch", False),
                })

    for training_only_name in ("window", "forward_period"):
        if training_only_name in cfg:
            violations.append({
                "param": training_only_name,
                "description": "training-only parameter must be read from metadata",
                "expected": "not present in simulation config",
                "actual": cfg[training_only_name],
            })

    return {
        "passed": not violations,
        "contract_version": CONTRACT_VERSION,
        "violations": violations,
        "explicit_exceptions": explicit_exceptions,
    }


def verify_simulation_contract(
    metadata: dict[str, Any], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return violations; an empty list means the contract passes."""
    return simulation_contract_audit(metadata, cfg)["violations"]


def enforce_simulation_contract(
    metadata: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """Raise on mismatch and return the successful audit."""
    audit = simulation_contract_audit(metadata, cfg)
    if audit["violations"]:
        raise ContractViolationError(audit["violations"])
    return audit
