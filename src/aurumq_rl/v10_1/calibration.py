"""v10.1 relative-threshold calibration.

The four heads use weighted BCE, so sigmoid outputs are not probability-calibrated:
with a positive class weight, a 0.5 threshold can classify almost every row as
positive. This module freezes relative P90/P95 thresholds from the model's
training/calibration period before any OOS rows are evaluated.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from aurumq_rl.data_loader import (
    FactorPanelLoader,
    UniverseFilter,
    align_panel_to_training_universe,
)
from aurumq_rl.v10_1.inference import MultiHeadV10_1Inference


DEFAULT_QUANTILE_POLICY: dict[str, float] = {
    "a1": 0.90,
    "a2": 0.90,
    "peak": 0.95,
    "b1": 0.90,
}


def _stock_codes(meta: Mapping, panel_path: Path) -> list[str]:
    codes = list(meta.get("stock_codes") or [])
    if codes:
        return codes[: int(meta.get("n_stocks", len(codes)))]
    import polars as pl

    all_codes = (
        pl.scan_parquet(str(panel_path))
        .select("ts_code")
        .unique()
        .sort("ts_code")
        .collect()["ts_code"]
        .to_list()
    )
    return all_codes[: int(meta.get("n_stocks", len(all_codes)))]


def _observations(panel, target_codes: list[str], window: int) -> np.ndarray:
    idx = [panel.stock_codes.index(code) for code in target_codes if code in panel.stock_codes]
    if len(idx) != len(target_codes):
        raise ValueError("calibration panel 缺少训练股票，无法保持 action_shape")
    rows: list[np.ndarray] = []
    for t in range(max(window - 1, 0), len(panel.dates)):
        x = panel.factor_array[max(0, t - window + 1): t + 1, idx, :].transpose(1, 0, 2)
        if x.shape[1] < window:
            x = np.pad(x, ((0, 0), (window - x.shape[1], 0), (0, 0)))
        rows.append(np.nan_to_num(x.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0))
    if not rows:
        raise ValueError("calibration period 没有完整 observation window")
    return np.asarray(rows, dtype=np.float32)


def calibrate_model_thresholds(
    model_dir: str | Path,
    panel_path: str | Path,
    meta: Mapping,
    start_date: datetime.date,
    end_date: datetime.date,
    output_path: str | Path,
    quantile_policy: Mapping[str, float] | None = None,
) -> dict:
    """在训练/校准期计算并持久化相对阈值；不读取任何未来标签。"""
    policy = dict(DEFAULT_QUANTILE_POLICY)
    if quantile_policy:
        policy.update({k: float(v) for k, v in quantile_policy.items()})
    if start_date >= end_date:
        raise ValueError("calibration start_date 必须早于 end_date")
    if any(not 0.5 <= v < 1.0 for v in policy.values()):
        raise ValueError(f"quantile policy 非法: {policy}")
    panel_path = Path(panel_path)
    target_codes = _stock_codes(meta, panel_path)
    n_factors = int(meta["n_factors"])
    factor_names = list(meta.get("factor_names") or [])
    window = int(meta.get("hyperparams", {}).get("window", 20)) if isinstance(meta.get("hyperparams"), dict) else 20
    panel = FactorPanelLoader(panel_path).load_panel(
        start_date,
        end_date,
        n_factors=n_factors,
        forward_period=1,
        factor_names=factor_names or None,
    )
    panel, alignment = align_panel_to_training_universe(panel, target_codes)
    if panel.factor_array.shape[1:] != (int(meta["n_stocks"]), n_factors):
        raise ValueError(f"calibration shape mismatch: {panel.factor_array.shape}")
    observations = _observations(panel, target_codes, window)
    agent = MultiHeadV10_1Inference(model_dir)
    values: dict[str, list[float]] = {k: [] for k in policy}
    # Batch one is deliberate: the exported LSTM has no hidden-state inputs and
    # was exported with a batch-size-1 contract.
    for observation in observations:
        result = agent.predict(observation)
        for name in values:
            values[name].extend(np.asarray(result[name], dtype=np.float64).ravel().tolist())
    thresholds = {
        name: float(np.quantile(np.asarray(samples), policy[name]))
        for name, samples in values.items()
    }
    distributions = {
        name: {
            "n": len(samples),
            "p50": float(np.quantile(samples, 0.50)),
            "p90": float(np.quantile(samples, 0.90)),
            "p95": float(np.quantile(samples, 0.95)),
            "p99": float(np.quantile(samples, 0.99)),
            "min": float(np.min(samples)),
            "max": float(np.max(samples)),
        }
        for name, samples in values.items()
    }
    artifact = {
        "version": "v10.1_relative_thresholds_v1",
        "model": str(Path(model_dir)),
        "panel": str(panel_path),
        "calibration_start_date": start_date.isoformat(),
        "calibration_end_date": end_date.isoformat(),
        "quantile_policy": policy,
        "thresholds": thresholds,
        "distributions": distributions,
        "observation_count": int(len(observations)),
        "alignment": alignment,
        "labels_used": False,
        "oos_safe": True,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    return artifact


def load_or_calibrate(
    model_dir: str | Path,
    panel_path: str | Path,
    meta: Mapping,
    oos_start: datetime.date,
    calibration_start: datetime.date,
    calibration_end: datetime.date,
    output_path: str | Path,
) -> dict:
    """读取冻结阈值；不存在时仅用 OOS 之前的校准期生成。"""
    if calibration_end >= oos_start:
        raise ValueError("calibration_end_date 必须早于 OOS start_date，禁止 OOS 校准")
    output = Path(output_path)
    if output.is_file():
        data = json.loads(output.read_text())
        if data.get("version") != "v10.1_relative_thresholds_v1":
            raise ValueError("threshold calibration artifact version mismatch")
        if data.get("calibration_end_date", "") >= oos_start.isoformat():
            raise ValueError("已存在的 calibration artifact 使用了 OOS 日期")
        return data
    return calibrate_model_thresholds(
        model_dir,
        panel_path,
        meta,
        calibration_start,
        calibration_end,
        output,
    )
