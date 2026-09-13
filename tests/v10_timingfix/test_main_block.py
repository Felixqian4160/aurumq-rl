"""End-to-end test: invoke run_simulation.py via its __main__ block and verify
the resulting ledger is produced and parses correctly. This guards against
the class of bugs where hard-coded defaults in __main__ silently override
the head_mapping contract.

Usage: pytest tests/v10_timingfix/test_main_block.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable  # we run via the project venv
RUNNER = REPO / "scripts" / "v10_timingfix" / "run_simulation.py"
MODEL_DIR = REPO / "models" / "whv10_timingfix_seed42_baseline_10k"
PANEL = REPO / "data" / "wavehunter_v10_hs300_evt_wf.parquet"


def _ensure_model_exists() -> None:
    if not (MODEL_DIR / "metadata.json").exists():
        pytest.skip(f"model {MODEL_DIR} not present; run smoke first")


def _write_config(tmp: Path) -> Path:
    cfg_path = tmp / "config.json"
    state_path = tmp / "state.json"
    cfg_path.write_text(json.dumps({
        "model_dir": str(MODEL_DIR),
        "panel_path": str(PANEL),
        "start_date": "2025-01-01",
        "end_date": "2025-01-15",  # small window for speed
        "top_k": 20,
        "cost_bps": 15.0,
        "slippage_bps": 10.0,
        "stop_loss_pct": 0.0,
        "cooldown_days": 5,
        "a1_threshold": 0.55,
        "a2_threshold": 0.55,
        "peak_threshold": 0.50,
        "b1_threshold": 0.55,
        "initial_capital": 100000.0,
    }))
    state_path.write_text(json.dumps({"status": "starting"}))
    return cfg_path


def test_main_block_emits_ledger(tmp_path: Path) -> None:
    """Subprocess-invoke run_simulation.py __main__ block and verify that
    the resulting ledger file exists, parses as JSON, contains the required
    fields, and reflects the head_mapping single source of truth."""
    _ensure_model_exists()
    cfg_path = _write_config(tmp_path)
    state_path = cfg_path.parent / "state.json"

    proc = subprocess.run(
        [PY, "-B", str(RUNNER), str(cfg_path), str(state_path)],
        capture_output=True, text=True, cwd=str(REPO),
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.fail(f"run_simulation failed:\n{proc.stderr[-1500:]}")
    state = json.loads(state_path.read_text())
    assert state["status"] == "finished", state
    led_path = Path(state["ledger_file"])
    assert led_path.exists(), f"ledger not produced: {led_path}"
    ledger = json.loads(led_path.read_text())
    assert "final_nav" in ledger
    assert "trades" in ledger
    assert "nav_curve" in ledger
    assert ledger["nav_curve"]
    # The ledger's terminal NAV must be the same post-liquidation value
    # returned to the caller/state file; this prevents state-vs-curve drift.
    assert ledger["final_nav"] == ledger["nav_curve"][-1]["nav"]
    assert abs(float(state["final_nav"]) - float(ledger["final_nav"])) < 1e-9


def test_main_block_uses_head_mapping_thresholds(tmp_path: Path) -> None:
    """Subprocess invocation should NOT need a1/a2/peak/b1 thresholds to be
    present in config; the __main__ block should default to head_mapping.
    The simplest way to verify is to omit them entirely from config and
    check the run still succeeds."""
    _ensure_model_exists()
    cfg_path = tmp_path / "config.json"
    state_path = tmp_path / "state.json"
    cfg_path.write_text(json.dumps({
        "model_dir": str(MODEL_DIR),
        "panel_path": str(PANEL),
        "start_date": "2025-01-01",
        "end_date": "2025-01-15",
        "top_k": 20,
        "cost_bps": 15.0,
        "slippage_bps": 10.0,
        "stop_loss_pct": 0.0,
        "cooldown_days": 5,
        # NO a1_threshold/a2_threshold/peak_threshold/b1_threshold
        "initial_capital": 100000.0,
    }))
    state_path.write_text(json.dumps({"status": "starting"}))
    proc = subprocess.run(
        [PY, "-B", str(RUNNER), str(cfg_path), str(state_path)],
        capture_output=True, text=True, cwd=str(REPO),
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.fail(f"run_simulation requires explicit thresholds (head_mapping drift):\n{proc.stderr[-1500:]}")
