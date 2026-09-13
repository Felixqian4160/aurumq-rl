"""WaveHunter 全局规律发现矩阵 API。

矩阵 Tab 只编排已有 WaveHunter 训练和模拟功能：
- 训练调用 wavehunter_api.wavehunter_train_start
- 模拟调用 app.start_sim
不提供旁路 CLI/训练入口；所有动作都由 WebUI Tab 触发。
"""
from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from app import AURUMQ_ROOT, app, log, start_sim, SimRequest
from wavehunter_api import wavehunter_train_start, WaveHunterTrainRequest

MATRIX_DIR = AURUMQ_ROOT / "data" / "wavehunter_matrix_v2"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = MATRIX_DIR / "registry.json"
RESULTS_FILE = MATRIX_DIR / "results.jsonl"
EVENTS_FILE = MATRIX_DIR / "events.jsonl"
LEDGER_DIR = AURUMQ_ROOT / "data" / "ledgers"

_lock = threading.Lock()
_matrix_thread: Optional[threading.Thread] = None
_matrix_stop = threading.Event()
_matrix_state = {
    "running": False, "status": "idle", "current": 0, "total": 72,
    "run_id": None, "progress": "", "error": "", "completed": 0,
}

POOLS = {
    "hs300": "wavehunter_hs300_20040102_20260804.parquet",
    "csi500": "wavehunter_csi500_20040102_20260804.parquet",
    "cs800": "wavehunter_cs800_20040102_20260804.parquet",
}
SPANS = {
    "6m": ("2023-07-01", "2023-12-31", "2024-01-01", "2024-06-30"),
    "12m": ("2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    "18m": ("2022-07-01", "2023-12-31", "2024-01-01", "2025-06-30"),
    "24m": ("2022-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),
}
STEPS = [100_000, 200_000, 300_000, 500_000, 700_000, 1_000_000]

FIXED_TRAIN = dict(
    window=20, top_k=20, universe_filter="main_board_non_st", n_factors=0,
    forward_period=20, max_position_pct=0.02, rebalance_days=20, cost_bps=15.0,
    learning_rate=1e-4, lstm_hidden=128, lstm_layers=1, reward_type="return",
    n_envs=1, seed=42, max_grad_norm=1.0, target_kl=0.02, batch_size=64,
    n_steps=64, aux_lambda=0.1, a1_lambda=1.0, a2_lambda=1.0,
    a2_pos_weight=25.0, label_window=20, label_entry_days=1, label_n_bins=5,
    label_a2_threshold=0.10, label_a2_window=3, label_min_amount_pct=0.05,
)
FIXED_SIM = dict(
    initial_capital=100_000.0, top_k=20, cost_bps=15.0, slippage_bps=10.0,
    rebalance_days=20, stop_loss_pct=8.0, max_position_pct=0.02,
    max_holding_days=120,
)


class MatrixStartRequest(BaseModel):
    confirm: bool = False
    resume: bool = True

    model_config = {"protected_namespaces": ()}


def _run_list() -> list[dict]:
    runs = []
    for pool, panel in POOLS.items():
        for span, (ts, te, vs, ve) in SPANS.items():
            for steps in STEPS:
                rid = f"{pool}_{span}_{steps // 1000}k_seed42"
                runs.append({
                    "run_id": rid, "pool": pool, "panel": panel, "span": span,
                    "train_start": ts, "train_end": te, "oos_start": vs,
                    "oos_end": ve, "steps": steps, "status": "pending",
                    "train": dict(FIXED_TRAIN, total_timesteps=steps),
                    "simulation": dict(FIXED_SIM), "artifacts": {}, "metrics": {},
                    "gates": {}, "error": "",
                })
    return runs


def _load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        d = {"schema_version": "wavehunter_matrix_v2", "created_at": time.time(), "runs": _run_list()}
        REGISTRY_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        return d
    return json.loads(REGISTRY_FILE.read_text())


def _event(kind: str, run_id: str, **payload):
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "run_id": run_id, **payload}
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"[矩阵/{run_id}] {kind}: {payload}", source="wavehunter_matrix")


def _append_result(row: dict):
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _latest_ledger(before: float) -> Optional[Path]:
    files = sorted(LEDGER_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    newer = [p for p in files if p.stat().st_mtime > before]
    return newer[-1] if newer else None


def _wait_train(run_id: str, timeout: float = 60 * 60 * 8) -> bool:
    t0 = time.time()
    from wavehunter_api import wavehunter_train_status
    while time.time() - t0 < timeout:
        if _matrix_stop.is_set(): return False
        st = wavehunter_train_status()
        if not st.get("running", False):
            return True
        time.sleep(5)
    _event("train_timeout", run_id)
    return False


def _wait_sim(run_id: str, timeout: float = 60 * 30) -> bool:
    t0 = time.time()
    from app import sim_status
    while time.time() - t0 < timeout:
        if _matrix_stop.is_set(): return False
        st = sim_status()
        if not st.get("running", False):
            return st.get("status") == "done"
        time.sleep(3)
    _event("sim_timeout", run_id)
    return False


def _worker(resume: bool):
    global _matrix_state
    registry = _load_registry()
    runs = registry["runs"]
    for index, run in enumerate(runs, 1):
        if _matrix_stop.is_set(): break
        if resume and run.get("status") == "done":
            continue
        pool, rid = run["pool"], run["run_id"]
        panel_path = AURUMQ_ROOT / "data" / run["panel"]
        with _lock:
            _matrix_state.update(current=index, run_id=rid, progress=f"{index}/72 {rid}")
        if not panel_path.exists():
            run.update(status="blocked", error=f"v3面板不存在: {panel_path}")
            _event("blocked", rid, reason=run["error"])
            _append_result(run)
            continue
        out_name = f"matrix_{rid}"
        run["status"] = "running"
        run["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _event("run_start", rid, index=index, total=72, config=run)
        try:
            req = WaveHunterTrainRequest(
                panel=run["panel"], start_date=run["train_start"], end_date=run["train_end"],
                total_timesteps=run["steps"], out_dir=out_name, **FIXED_TRAIN,
            )
            # 复用已有 WaveHunter Tab 的训练函数，不调用脚本旁路。
            start_resp = wavehunter_train_start(req)
            run["artifacts"]["train_response"] = start_resp
            if not _wait_train(rid):
                run.update(status="failed", error="训练失败、停止或超时")
                _append_result(run); continue
            # 所有新矩阵模型统一写入 models/。
            model_dir = AURUMQ_ROOT / "models" / out_name
            run["artifacts"]["model_dir"] = str(model_dir)
            if not (model_dir / "policy.onnx").exists() or not (model_dir / "metadata.json").exists():
                raise RuntimeError("训练完成但缺少 policy.onnx 或 metadata.json")
            before = max([p.stat().st_mtime for p in LEDGER_DIR.glob("*.json")], default=0.0)
            sim_req = SimRequest(
                model_dir=f"runs:{out_name}", panel=run["panel"],
                start_date=run["oos_start"], end_date=run["oos_end"], **FIXED_SIM,
            )
            sim_resp = start_sim(sim_req)
            run["artifacts"]["sim_response"] = sim_resp
            if not _wait_sim(rid):
                raise RuntimeError("模拟失败、停止或超时")
            ledger = _latest_ledger(before)
            if ledger is None: raise RuntimeError("模拟完成但未找到新 ledger")
            run["artifacts"]["ledger"] = str(ledger)
            data = json.loads(ledger.read_text())
            run["metrics"] = data.get("metrics", {})
            run["gates"] = {"model_artifacts": "pass", "ledger": "pass", "signal_audit": "pending", "trade_audit": "pending"}
            run["status"] = "done"
            run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _event("run_done", rid, metrics=run["metrics"], ledger=str(ledger))
        except Exception as e:
            run.update(status="failed", error=str(e))
            _event("run_failed", rid, error=str(e))
        _append_result(run)
        registry["runs"] = runs
        REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2))
    with _lock:
        _matrix_state.update(running=False, status="done" if not _matrix_stop.is_set() else "stopped", run_id=None)


@app.get("/api/wavehunter/matrix/config")
def matrix_config():
    d = _load_registry()
    panels = {k: {"file": v, "exists": (AURUMQ_ROOT / "data" / v).exists()} for k, v in POOLS.items()}
    return {"total": len(d["runs"]), "pools": panels, "spans": list(SPANS), "steps": STEPS, "fixed_train": FIXED_TRAIN, "fixed_sim": FIXED_SIM}


@app.get("/api/wavehunter/matrix/runs")
def matrix_runs():
    return _load_registry()


@app.post("/api/wavehunter/matrix/start")
def matrix_start(req: MatrixStartRequest):
    global _matrix_thread
    if not req.confirm:
        raise HTTPException(400, "必须在矩阵 Tab 勾选确认后启动")
    if _matrix_state.get("running"):
        raise HTTPException(400, "矩阵已有任务运行中")
    d = _load_registry()
    missing = [p for p in POOLS.values() if not (AURUMQ_ROOT / "data" / p).exists()]
    if missing:
        raise HTTPException(400, f"缺少 WaveHunter v3 面板，不能启动72-run: {missing}")
    _matrix_stop.clear()
    _matrix_state.update(running=True, status="running", current=0, completed=0, error="")
    _matrix_thread = threading.Thread(target=_worker, args=(req.resume,), daemon=True)
    _matrix_thread.start()
    return {"status": "started", "total": len(d["runs"]), "mode": "reuse_wavehunter_and_sim_tabs"}


@app.post("/api/wavehunter/matrix/stop")
def matrix_stop():
    _matrix_stop.set()
    from wavehunter_api import wavehunter_train_stop
    try: wavehunter_train_stop()
    except Exception: pass
    return {"status": "stopping"}


@app.get("/api/wavehunter/matrix/status")
def matrix_status():
    with _lock: return dict(_matrix_state)


@app.get("/api/wavehunter/matrix/results")
def matrix_results():
    rows=[]
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            if line.strip(): rows.append(json.loads(line))
    return {"results": rows}


@app.get("/api/wavehunter/matrix/events")
def matrix_events():
    rows=[]
    if EVENTS_FILE.exists():
        for line in EVENTS_FILE.read_text().splitlines()[-500:]:
            if line.strip(): rows.append(json.loads(line))
    return {"events": rows}
