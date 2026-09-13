"""矩阵注册表管理模块。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from matrix_config import (
    _AURUMQ_ROOT, POOLS, _build_runs_for_phase, _VALID_POOLS,
)

MATRIX_DIR = _AURUMQ_ROOT / "data" / "wavehunter_v2_matrix"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = MATRIX_DIR / "registry.json"
RESULTS_FILE = MATRIX_DIR / "results.jsonl"
EVENTS_FILE = MATRIX_DIR / "events.jsonl"
SUMMARY_FILE = MATRIX_DIR / "_summary.json"
LEDGER_DIR = _AURUMQ_ROOT / "data" / "ledgers"


def _load_registry() -> dict:
    """加载注册表。"""
    if not REGISTRY_FILE.exists():
        d = {"schema_version": "wavehunter_v3_matrix", "created_at": time.time(),
             "phase": "smoke", "pool": "hs300", "runs": _build_runs_for_phase("smoke", "hs300", "v3")}
        REGISTRY_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        return d
    d = json.loads(REGISTRY_FILE.read_text())
    d.setdefault("phase", "smoke")
    d.setdefault("pool", "hs300")
    return d


def _save_registry(registry: dict) -> None:
    """保存注册表。"""
    REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2))


def _save_summary_cache(registry: dict) -> None:
    """每个 run 完成时刷新 _summary.json，前端 result 端点秒开。"""
    runs = registry.get("runs", [])
    rows = []
    for run in runs:
        rows.append({
            "run_id": run.get("run_id"),
            "pool": run.get("pool"),
            "span": run.get("span"),
            "steps": run.get("steps"),
            "status": run.get("status"),
            "metrics": run.get("metrics", {}),
            "artifacts": {k: v for k, v in (run.get("artifacts") or {}).items() if k in ("ledger", "model_dir")},
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "error": run.get("error", ""),
        })
    SUMMARY_FILE.write_text(json.dumps({
        "schema_version": "wavehunter_v2_matrix_summary",
        "phase": registry.get("phase", "smoke"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": rows,
    }, ensure_ascii=False, indent=2))


def _append_result(row: dict) -> None:
    """按run_id幂等写结果。"""
    rows = []
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    old = json.loads(line)
                    if old.get("run_id") != row.get("run_id"):
                        rows.append(old)
                except json.JSONDecodeError:
                    continue
    rows.append(row)
    RESULTS_FILE.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def _latest_ledger(before: float) -> Optional[Path]:
    """查找指定时间之后的最新 ledger。"""
    files = sorted(LEDGER_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    newer = [p for p in files if p.stat().st_mtime > before]
    return newer[-1] if newer else None


def _status_counts(runs: list[dict]) -> dict:
    """统计各状态的 run 数量。"""
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0, "blocked": 0,
              "stopped": 0, "stale": 0}
    for run in runs:
        status = run.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _event(kind: str, run_id: str, **payload) -> None:
    """记录事件。"""
    from app import log
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "run_id": run_id, **payload}
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"[矩阵/{run_id}] {kind}: {payload}", source="wavehunter_v2_matrix", task_id=run_id, event=kind)


def _fresh_registry(pool: str = "hs300") -> dict:
    """创建新的注册表。"""
    runs = _build_runs_for_phase("smoke", pool)
    return {"schema_version": f"wavehunter_v2_matrix_{pool}", "created_at": time.time(), "runs": runs}
