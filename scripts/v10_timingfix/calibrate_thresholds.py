"""Calibrate per-model relative signal thresholds without using OOS.

The existing 100k artifacts were trained through 2024-12-31, so the default
2024-10-01..2024-12-31 window is temporal/post-training calibration, not a
strict gradient-held-out validation set. The output records this limitation.
"""
from __future__ import annotations
import argparse, json, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter, align_panel_to_training_universe
from aurumq_rl.v10.inference import MultiHeadInference


def collect(model_dir: Path, panel_path: Path, start: date, end: date) -> dict[str, np.ndarray]:
    meta = json.loads((model_dir / "metadata.json").read_text())
    loader = FactorPanelLoader(str(panel_path))
    # Load history before the calibration window so the first calibration day
    # receives its real trailing observation context rather than zero padding.
    context_start = start - timedelta(days=45)
    panel = loader.load_panel(context_start, end, n_factors=int(meta["n_factors"]),
                              forward_period=1,
                              universe_filter=UniverseFilter.MAIN_BOARD_NON_ST,
                              factor_names=meta["factor_names"])
    panel, _ = align_panel_to_training_universe(panel, meta["stock_codes"])
    agent = MultiHeadInference(model_dir)
    window = int(meta.get("hyperparams", {}).get("window", 20))
    values = {"a1": [], "peak": []}
    for t in range(len(panel.dates) - 1):
        current = panel.dates[t]
        current_date = current.date() if hasattr(current, "date") else current
        # Only record scores on the frozen calibration interval. The preceding
        # rows are context, never part of the percentile sample.
        if current_date < start or current_date > end:
            continue
        lo = max(0, t - window + 1)
        x = panel.factor_array[lo:t + 1].transpose(1, 0, 2)
        if x.shape[1] < window:
            x = np.pad(x, ((0, 0), (window - x.shape[1], 0), (0, 0)))
        out = agent.predict(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1))
        for key in values:
            values[key].extend(np.asarray(out[key], dtype=float).tolist())
    return {k: np.asarray(v, dtype=float) for k, v in values.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict-holdout", action="store_true",
                    help="record that the calibration interval was excluded from gradient training")
    args = ap.parse_args()
    vals = collect(Path(args.model_dir), Path(args.panel),
                   date.fromisoformat(args.start), date.fromisoformat(args.end))
    result = {
        "model_dir": args.model_dir,
        "calibration_window": {"start": args.start, "end": args.end},
        "calibration_type": "strict_gradient_holdout_calibration" if args.strict_holdout else "post_training_temporal_calibration",
        "strict_holdout": bool(args.strict_holdout),
        "context_start": (date.fromisoformat(args.start) - timedelta(days=45)).isoformat(),
        "limitation": None if args.strict_holdout else "artifact was trained through the calibration window; not strict holdout",
        "n_scores": {k: int(v.size) for k, v in vals.items()},
        "score_stats": {},
        "thresholds": {},
    }
    for key, arr in vals.items():
        result["score_stats"][key] = {
            "min": float(arr.min()), "max": float(arr.max()),
            "mean": float(arr.mean()), "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p97": float(np.percentile(arr, 97)),
            "p99": float(np.percentile(arr, 99)),
        }
        result["thresholds"][key] = {
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p97": float(np.percentile(arr, 97)),
            "p99": float(np.percentile(arr, 99)),
        }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
