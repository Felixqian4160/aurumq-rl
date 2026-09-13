"""WaveHunter v2 全局规律发现矩阵 API — 兼容层。

矩阵 Tab 只编排已有 WaveHunter 训练和模拟功能：
- 训练调用 wavehunter_api.wavehunter_train_start
- 模拟调用 app.start_sim
不提供旁路 CLI/训练入口；所有动作都由 WebUI Tab 触发。
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

from app import AURUMQ_ROOT, app, log
import wavehunter_v2_api as _v2_api
from wavehunter_v2_api import wavehunter_v2_stop

# 导入子模块的公共接口
from matrix_config import (
    POOLS, SPANS, STEPS, SMOKE_STEPS, FULL_STEPS, PHASE_TIPS,
    FIXED_TRAIN, FIXED_SIM, POOL_TRAIN_PROFILES,
    _phase_steps, _build_runs_for_phase, _VALID_POOLS,
)
from matrix_registry import (
    MATRIX_DIR, REGISTRY_FILE, RESULTS_FILE, EVENTS_FILE, SUMMARY_FILE, LEDGER_DIR,
    _load_registry, _save_registry, _save_summary_cache, _append_result,
    _latest_ledger, _status_counts, _event, _fresh_registry,
)
from matrix_worker import (
    _lock, _matrix_stop, _matrix_session_id, _matrix_state,
    _restore_state_from_registry, _effective_train, _worker,
    _find_model_artifacts, _recover_model_artifacts,
    _external_train_identity, _wait_external_train, _wait_train, _wait_sim,
    _has_completed_artifacts, _recover_completed_runs,
)


# ── 请求模型 ──

class MatrixStartRequest(BaseModel):
    confirm: bool = False
    resume: bool = True
    phase: str = "smoke"  # "smoke" | "full"
    pool: str = "hs300"   # "hs300" | "csi500" | "cs800"
    version: str = "v3"   # "v2" | "v3"
    pause_after_run: bool = True

    model_config = {'protected_namespaces': ()}


class MatrixResetRequest(BaseModel):
    confirm: bool = False
    keep_models: bool = False

    model_config = {"protected_namespaces": ()}


# ── 池验证 ──

def _validate_pool(pool: str) -> str:
    if pool not in _VALID_POOLS:
        raise HTTPException(400, f"未知 pool: {pool} (可选: {list(_VALID_POOLS)})")
    panel_path = AURUMQ_ROOT / "data" / POOLS[pool]
    if not panel_path.exists():
        raise HTTPException(400, f"{pool} 面板不存在: {panel_path.name}（先在 '主升浪 v2 面板构建' Tab 构建）")
    return pool


# ── API 路由 ──

@app.get("/api/wavehunter/v2/matrix/config")
def matrix_config():
    d = _load_registry()
    panels = {k: {"file": v, "exists": (AURUMQ_ROOT / "data" / v).exists()} for k, v in POOLS.items()}
    phase = d.get("phase", "smoke")
    pool = d.get("pool", "hs300")
    return {
        "total": len(d["runs"]), "pools": panels, "current_pool": pool,
        "spans": list(SPANS),
        "steps": STEPS, "phase": phase, "smoke_steps": SMOKE_STEPS, "full_steps": FULL_STEPS,
        "phase_tips": PHASE_TIPS,
        "fixed_train": FIXED_TRAIN,
        "pool_train_profiles": POOL_TRAIN_PROFILES,
        "fixed_sim": FIXED_SIM,
    }


@app.get("/api/wavehunter/v2/matrix/runs")
def matrix_runs():
    return _load_registry()


@app.post("/api/wavehunter/v2/matrix/start")
def matrix_start(req: MatrixStartRequest):
    global _matrix_thread, _matrix_session_id
    if not req.confirm:
        raise HTTPException(400, "必须在矩阵 Tab 勾选确认后启动")
    if req.phase not in ("smoke", "full"):
        raise HTTPException(400, f"未知 phase: {req.phase}")
    pool = _validate_pool(req.pool)
    if _matrix_state.get("running"):
        raise HTTPException(400, "矩阵已有任务运行中")
    if req.resume and REGISTRY_FILE.exists():
        d = _load_registry()
        if d.get("phase") != req.phase or d.get("pool") != pool:
            d["runs"] = _build_runs_for_phase(req.phase, pool, req.version)
        else:
            existing_ids = {r.get("run_id") for r in d.get("runs", [])}
            fresh = _build_runs_for_phase(req.phase, pool, req.version)
            for fr in fresh:
                if fr.get("run_id") not in existing_ids:
                    d["runs"].append(fr)
    else:
        d = _fresh_registry(pool)
        if d.get("phase") != req.phase or d.get("pool") != pool:
            d["runs"] = _build_runs_for_phase(req.phase, pool, req.version)
    d["phase"] = req.phase
    d["pool"] = pool
    d["version"] = req.version
    for run in d["runs"]:
        run["panel"] = POOLS[pool]
    _save_registry(d)
    _matrix_session_id = time.strftime(f"{pool}_{req.version}_{req.phase}_%Y%m%d_%H%M%S")
    d["session_id"] = _matrix_session_id
    _save_registry(d)
    runs = d["runs"]
    run = next((r for r in runs if r.get("run_id") == f"{pool}_6m_{SMOKE_STEPS[0] // 1000}k_seed42"), None)
    if run and not req.resume:
        run["status"] = "pending"
        run["error"] = "训练本体已完成，等待后处理恢复"
        _save_registry({**d, "runs": runs})
    _matrix_stop.clear()
    done = sum(1 for r in runs if r.get("status") == "done")
    _matrix_state.update(
        running=True, status="running", current=0, total=len(runs),
        completed=done, error="", phase=req.phase, done=done,
    )
    import threading
    _matrix_thread = threading.Thread(
        target=_worker, args=(req.resume, req.pause_after_run), daemon=True,
    )
    _matrix_thread.start()
    return {
        "status": "started", "total": len(d["runs"]), "phase": req.phase,
        "mode": "reuse_wavehunter_and_sim_tabs", "session_id": _matrix_session_id,
    }


@app.post("/api/wavehunter/v2/matrix/stop")
def matrix_stop():
    _matrix_stop.set()
    try:
        wavehunter_v2_stop()
    except Exception:
        pass
    try:
        from wavehunter_v2_api import wavehunter_v3_stop
        wavehunter_v3_stop()
    except Exception:
        pass
    return {"status": "stopping"}


@app.post("/api/wavehunter/v2/matrix/reset")
def matrix_reset(req: MatrixResetRequest):
    if not req.confirm:
        raise HTTPException(400, "必须在调用 reset 时显式 confirm=true")
    global _matrix_session_id
    _matrix_stop.set()
    try:
        wavehunter_v2_stop()
    except Exception:
        pass
    time.sleep(2)

    removed = {"registry": 0, "summary": 0, "events": 0, "results": 0,
               "progress_relay": 0, "model_dirs": 0, "training_logs": 0,
               "ledgers": 0, "runtime_reset": False}

    for f, key in [(REGISTRY_FILE, "registry"), (SUMMARY_FILE, "summary"),
                   (EVENTS_FILE, "events"), (RESULTS_FILE, "results"),
                   (MATRIX_DIR / "progress_relay.json", "progress_relay")]:
        try:
            if f.exists():
                f.unlink()
                removed[key] = 1
        except Exception as e:
            removed[f"{key}_error"] = str(e)

    if not req.keep_models:
        models_root = AURUMQ_ROOT / "models"
        for d in models_root.glob("matrix_*"):
            try:
                import shutil
                shutil.rmtree(d)
                removed["model_dirs"] += 1
            except Exception as e:
                removed[f"model_dir_{d.name}_error"] = str(e)

    log_dir = AURUMQ_ROOT / "data" / "training_logs"
    for f in log_dir.glob("matrix_*"):
        try:
            f.unlink()
            removed["training_logs"] += 1
        except Exception:
            pass

    ledgers_root = LEDGER_DIR
    cutoff = time.time() - 14 * 86400
    for f in ledgers_root.glob("*matrix*.json"):
        try:
            if f.stat().st_mtime >= cutoff:
                f.unlink()
                removed["ledgers"] += 1
        except Exception:
            pass

    runtime = AURUMQ_ROOT / "data" / "wavehunter_v2_runtime.json"
    try:
        runtime.write_text(json.dumps({
            "out_name": None, "status": "idle",
            "reset_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=2))
        removed["runtime_reset"] = True
    except Exception:
        pass

    with _lock:
        _matrix_state.update(
            running=False, status="idle", current=0,
            total=0, completed=0, run_id=None,
            done=0, pending=0, failed=0, blocked=0,
            eta_seconds=0, current_run=None,
        )
    _matrix_session_id = None
    _matrix_stop.clear()

    log(f"🧹 矩阵 reset 完成: {removed}",
        source="wavehunter_v2_matrix", task_id="reset", event="reset")
    return {"status": "ok", "removed": removed,
            "keep_models": req.keep_models,
            "next_start_will_create_fresh_registry": True}


@app.get("/api/wavehunter/v2/matrix/status")
def matrix_status():
    d = _load_registry()
    runs = d.get("runs", [])
    external = _external_train_identity()
    if external:
        actual = next((r for r in runs if r.get("run_id") == external["run_id"]), None)
        for other in runs:
            if other.get("status") == "running" and other is not actual:
                other["status"] = "pending"
                other["error"] = "等待真实进程完成后续跑"
        if actual:
            with _lock:
                _matrix_state.update(
                    running=True, status="running", run_id=external["run_id"],
                    progress=f"{runs.index(actual)+1}/{len(runs)} {external['run_id']}",
                    current=runs.index(actual)+1, total=len(runs), error="",
                    current_run=external["run_id"],
                )
            actual.setdefault("artifacts", {})["external_process"] = external
            actual["status"] = "running"
            actual["error"] = ""
            dirty = any(other is not actual and other.get("status") == "running" for other in runs)
            if dirty:
                d["runs"] = runs
                _save_registry(d)
    counts = _status_counts(runs)
    if not _matrix_state.get("running"):
        _restore_state_from_registry()
    eta_seconds = 0
    done_runs = [r for r in runs if r.get("status") == "done" and r.get("started_at") and r.get("finished_at")]
    remaining_runs = counts.get("pending", 0)
    current_progress = 0.0
    current_total = 0
    avg_run_seconds = 0.0
    if done_runs:
        total_sec = 0.0
        for r in done_runs:
            try:
                t0 = datetime.datetime.strptime(r["started_at"], "%Y-%m-%d %H:%M:%S").timestamp()
                t1 = datetime.datetime.strptime(r["finished_at"], "%Y-%m-%d %H:%M:%S").timestamp()
                total_sec += max(0.0, t1 - t0)
            except Exception:
                continue
        if done_runs:
            avg_run_seconds = total_sec / len(done_runs)
            for r in runs:
                if r.get("status") == "running":
                    current_total = int(r.get("steps") or 0)
                    break
            try:
                rt_path = AURUMQ_ROOT / "data" / "wavehunter_v2_runtime.json"
                if rt_path.exists():
                    rt = json.loads(rt_path.read_text())
                    cur = int(rt.get("current") or 0)
                    if current_total > 0 and 0 < cur <= current_total:
                        current_progress = cur / current_total
            except Exception:
                pass
            current_remaining = max(0.0, 1.0 - current_progress)
            eta_seconds = int(avg_run_seconds * (remaining_runs - 1 + current_remaining)) if remaining_runs else int(avg_run_seconds * current_remaining)
    with _lock:
        state = dict(_matrix_state)
    state.update({
        "done": counts.get("done", 0), "completed": counts.get("done", 0),
        "running_count": counts.get("running", 0),
        "pending": counts.get("pending", 0), "failed": counts.get("failed", 0),
        "blocked": counts.get("blocked", 0),
        "phase": d.get("phase", "smoke"),
        "eta_seconds": eta_seconds,
        "current_run": state.get("run_id") or state.get("current_run"),
    })
    state["session_id"] = _matrix_session_id or d.get("session_id")
    state["total"] = len(runs)
    if not state.get("run_id"):
        last = None
        for r in runs:
            if r.get("status") in ("done", "stopped", "failed"):
                if not last or (r.get("finished_at") or "") > (last.get("finished_at") or ""):
                    last = r
        if last:
            state["last_run"] = last["run_id"]
            if not state.get("current_run"):
                state["current_run"] = last["run_id"]
    return state


@app.get("/api/wavehunter/v2/matrix/results")
def matrix_results():
    # 读取 registry 最新真实状态，避免 summary 缓存长期保留空 metrics。
    reg = _load_registry()
    registry_by_id = {r.get("run_id"): r for r in reg.get("runs", [])}
    if SUMMARY_FILE.exists():
        try:
            cached = json.loads(SUMMARY_FILE.read_text())
            results = []
            for x in cached.get("runs", []):
                current = registry_by_id.get(x.get("run_id"), {})
                merged = {**x, "status": current.get("status", x.get("status")),
                          "metrics": current.get("metrics") or x.get("metrics", {}),
                          "gates": current.get("gates", x.get("gates", {})),
                          "artifacts": current.get("artifacts") or x.get("artifacts", {}),
                          "error": current.get("error", x.get("error", ""))}
                if merged.get("status") not in (None, "pending", "running"):
                    results.append(merged)
            return {"results": results, "phase": cached.get("phase", reg.get("phase", "smoke")),
                    "updated_at": cached.get("updated_at", "")}
        except Exception:
            pass
    rows = []
    if RESULTS_FILE.exists():
        latest = {}
        for line in RESULTS_FILE.read_text().splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                    latest[row.get("run_id")] = row
                except json.JSONDecodeError:
                    continue
        rows = [x for x in latest.values() if x.get("status") not in (None, "pending", "running")]
    if not rows:
        try:
            rows = [{"run_id": r.get("run_id"), "status": r.get("status"),
                     "metrics": r.get("metrics", {}), "artifacts": r.get("artifacts", {}),
                     "error": r.get("error", "")}
                    for r in _load_registry().get("runs", [])
                    if r.get("status") in ("done", "failed", "stopped", "blocked")]
        except Exception:
            rows = []
    reg = _load_registry()
    return {"results": rows, "phase": reg.get("phase", "smoke"),
            "pool": reg.get("pool", "hs300"), "updated_at": ""}


@app.get("/api/wavehunter/v2/matrix/events")
def matrix_events():
    rows = []
    session = _matrix_session_id or _load_registry().get("session_id")
    if session and EVENTS_FILE.exists():
        try:
            size = EVENTS_FILE.stat().st_size
            if size > 5 * 1024 * 1024:
                lines = EVENTS_FILE.read_text(encoding="utf-8").splitlines()
                EVENTS_FILE.write_text("\n".join(lines[-2000:]) + "\n", encoding="utf-8")
        except Exception:
            pass
        for line in EVENTS_FILE.read_text().splitlines()[-500:]:
            if line.strip():
                row = json.loads(line)
                if row.get("session_id") != session:
                    continue
                rows.append(row)
    return {"events": rows, "session_id": session}
