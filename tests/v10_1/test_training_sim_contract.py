"""Regression tests for the v10.1 execution/state contract."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from aurumq_rl.lstm_weight_env import LstmWeightConfig
from aurumq_rl.v10_1.contract import (
    CONTRACT_VERSION,
    build_training_contract,
    simulation_contract_audit,
)
from aurumq_rl.v10_1.env import WaveHunterV10_1Env


def _metadata(*, use_shared_sm: bool) -> dict:
    return {
        "panel": str(Path(__file__).resolve()),
        "hyperparams": {
            "cost_bps": 15.0,
            "sm_cost_bps": None,
            "sm_slippage_bps": 10.0,
            "max_position_pct": 0.05,
            "top_k": 2,
            "rebalance_days": 2,
            "window": 2,
            "forward_period": 20,
            "use_shared_sm": use_shared_sm,
        }
    }


def _simulation_cfg() -> dict:
    return {
        "cost_bps": 15.0,
        "slippage_bps": 10.0,
        "max_position_pct": 0.05,
        "top_k": 2,
        "rebalance_days": 2,
    }


def test_contract_requires_exact_shared_parameters() -> None:
    metadata = _metadata(use_shared_sm=True)
    metadata["training_contract"] = build_training_contract(metadata)
    assert metadata["training_contract"]["version"] == CONTRACT_VERSION
    assert simulation_contract_audit(metadata, _simulation_cfg())["violations"] == []

    for parameter, value in {
        "cost_bps": 16.0,
        "slippage_bps": 11.0,
        "max_position_pct": 0.02,
        "top_k": 1,
        "rebalance_days": 1,
    }.items():
        bad = _simulation_cfg()
        bad[parameter] = value
        audit = simulation_contract_audit(metadata, bad)
        assert any(v["param"] == parameter for v in audit["violations"])


def test_legacy_slippage_mismatch_requires_explicit_acknowledgement() -> None:
    metadata = _metadata(use_shared_sm=False)
    metadata["training_contract"] = build_training_contract(metadata)
    cfg = _simulation_cfg()

    audit = simulation_contract_audit(metadata, cfg)
    assert not audit["passed"]
    assert any(v["param"] == "slippage_bps" for v in audit["violations"])

    cfg["allow_legacy_friction_mismatch"] = True
    audit = simulation_contract_audit(metadata, cfg)
    assert audit["passed"]
    assert audit["explicit_exceptions"] == ["legacy_training_slippage_not_modeled"]


def test_sm_reward_syncs_marked_positions_to_current_weights() -> None:
    n_dates, n_stocks, n_factors = 6, 4, 3
    factors = np.zeros((n_dates, n_stocks, n_factors), dtype=np.float32)
    returns = np.zeros((n_dates, n_stocks), dtype=np.float32)
    opens = np.full((n_dates, n_stocks), 10.0, dtype=np.float64)
    closes = np.full((n_dates, n_stocks), 10.5, dtype=np.float64)
    mask = np.ones((n_dates, n_stocks), dtype=bool)
    labels = np.zeros((n_dates, n_stocks), dtype=np.int32)
    config = LstmWeightConfig(
        start_date=dt.date(2023, 1, 1),
        end_date=dt.date(2023, 1, 10),
        n_factors=n_factors,
        window=2,
        reward_type="absolute_return_v3",
        cost_bps=15.0,
        max_position_pct=0.05,
        top_k=2,
        rebalance_days=2,
        reward_scale=1.0,
    )
    env = WaveHunterV10_1Env(
        config=config,
        factor_panel=factors,
        return_panel=returns,
        tradeable_mask=mask,
        v10_1_a1_point=labels,
        v10_1_a2_interval=labels,
        v10_1_peak=labels,
        v10_1_b1=labels,
        use_shared_sm=True,
        sm_cost_bps=15.0,
        sm_slippage_bps=10.0,
        open_array=opens,
        close_array=closes,
    )
    env.reset(seed=7)
    action = np.zeros(n_stocks, dtype=np.float32)
    action[:2] = 0.5

    _, _, _, _, info = env.step_with_sm_reward(action)

    assert "sm_error" not in info
    assert info["sm_trade_count"] == 2
    expected = np.zeros(n_stocks, dtype=np.float64)
    code_to_index = {f"S_{i}": i for i in range(n_stocks)}
    for code, position in env._sm.positions.items():
        i = code_to_index[code]
        expected[i] = position.shares * closes[env._current_step, i] / env._sm.nav
    np.testing.assert_allclose(env._current_weights, expected)
    assert info["sm_actual_weights_l1_vs_target"] >= 0.0
