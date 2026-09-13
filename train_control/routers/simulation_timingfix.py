"""Independent WebUI API for v10_timingfix simulation.

统一进程模型：runtime JSON + start_new_session=True 立即脱离父进程；
启动检查、watchdog、按 task_id 停止、停止请求持久化。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .common import AURUMQ_ROOT, PYTHON_EXE
from core.process import pid_alive

router = APIRouter(tags=["v10-timingfix"])
_PROC: Optional[subprocess.Popen] = None
_RUNTIME = AURUMQ_ROOT / "data" / "task_runtime"
_RUNTIME.mkdir(parents=True, exist_ok=True)
_STARTUP_TIMEOUT_SECONDS = 15


def _stop_request_path(task_id: str) -> Path:
    return _RUNTIME / f"{task_id}.stop"


def _write_runtime(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _read_runtime(task_id: str) -> dict:
    path = _RUNTIME / f"{task_id}.json"
    if not path.exists():
        return {"status": "not_found", "running": False, "task_id": task_id}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "failed", "running": False, "task_id": task_id,
                "progress": "runtime 文件不可读"}
    if state.get("status") == "running" and not pid_alive(state.get("pid")):
        state.update(running=False, status="failed",
                      progress="模拟进程已退出")
        _write_runtime(path, state)
    state["running"] = state.get("status") == "running"
    return state


def _current_running() -> Optional[dict]:
    candidates: list[tuple[float, Path]] = []
    for path in _RUNTIME.glob("timingfix_sim_*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if st.get("status") in {"starting", "running", "stopping"}:
            candidates.append((st.get("started_at", 0), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, path = candidates[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _watch_simulation_start(task_id: str, started_at: float) -> None:
    time.sleep(_STARTUP_TIMEOUT_SECONDS)
    path = _RUNTIME / f"{task_id}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    if state.get("status") not in {"starting", "running"}:
        return
    pid = state.get("pid")
    if isinstance(pid, int) and pid_alive(pid):
        return
    state.update({"status": "failed", "running": False,
                  "progress": f"模拟进程在 {_STARTUP_TIMEOUT_SECONDS}s 内未进入活动状态"})
    _write_runtime(path, state)
    try:
        from core.logging import log
        log(f"❌ TimingFix 模拟启动失败: {task_id}", source="timingfix", task_id=task_id, event="failed")
    except Exception:
        pass


class TimingFixSimulationRequest(BaseModel):
    model_dir: str = ""
    panel: str = ""
    initial_capital: float = Field(100000.0, gt=0)
    start_date: str
    end_date: str
    top_k: int = Field(20, ge=1, le=100)
    cost_bps: float = Field(15.0, ge=0, le=100)
    slippage_bps: float = Field(10.0, ge=0, le=100)
    stop_loss_pct: float = Field(0.0, ge=0, lt=100)
    cooldown_days: int = Field(5, ge=0, le=252)
    a1_threshold: float = Field(0.55, ge=0, le=1)
    a2_threshold: float = Field(0.55, ge=0, le=1)
    peak_threshold: float = Field(0.50, ge=0, le=1)
    b1_threshold: float = Field(0.55, ge=0, le=1)


@router.post("/api/timingfix/simulation/start")
def start_timingfix_simulation(req: TimingFixSimulationRequest):
    global _PROC
    if _PROC and _PROC.poll() is None:
        raise HTTPException(409, "timingfix 模拟已在运行")
    if not req.model_dir or not req.panel:
        raise HTTPException(400, "必须选择 timingfix 模型和面板")
    model = AURUMQ_ROOT / "models" / req.model_dir
    panel = AURUMQ_ROOT / "data" / req.panel
    runner = AURUMQ_ROOT / "scripts" / "v10_timingfix" / "run_simulation.py"
    if not (model.is_dir() and (model / "metadata.json").exists()):
        raise HTTPException(400, f"模型不存在或缺少 metadata.json: {req.model_dir}")
    if not panel.exists():
        raise HTTPException(400, f"面板不存在: {req.panel}")
    if not runner.exists():
        raise HTTPException(500, "timingfix runner 不存在")
    ts = time.time()
    task_id = (f"timingfix_sim_{req.model_dir}_"
               f"{int(ts * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:6]}")
    cfg_path = _RUNTIME / f"{task_id}.config.json"
    log_path = _RUNTIME / f"{task_id}.log"
    state_path = _RUNTIME / f"{task_id}.json"
    payload = req.model_dump()
    payload.update({"model_dir": str(model), "panel_path": str(panel), "version": "v10_timingfix"})
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    state = {"version": "v10_timingfix", "task_id": task_id, "status": "starting", "running": False,
             "model": req.model_dir, "panel": req.panel, "log_path": str(log_path),
             "config_path": str(cfg_path), "started_at": ts}
    _write_runtime(state_path, state)
    try:
        with log_path.open("w") as out:
            _PROC = subprocess.Popen([PYTHON_EXE, str(runner), str(cfg_path), str(state_path)],
                                      cwd=str(AURUMQ_ROOT),
                                      stdout=out, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL,
                                      start_new_session=True,
                                      env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except Exception as exc:
        state.update({"status": "failed", "running": False,
                      "progress": f"启动失败: {exc}"})
        _write_runtime(state_path, state)
        raise HTTPException(500, f"模拟启动失败: {exc}") from exc
    state["pid"] = _PROC.pid
    state["status"] = "running"
    state["running"] = True
    _write_runtime(state_path, state)
    import threading as _th
    _th.Thread(target=_watch_simulation_start, args=(task_id, ts), daemon=True).start()
    return {"status": "started", "version": "v10_timingfix",
            "task_id": task_id, "log_path": str(log_path)}


@router.get("/api/timingfix/simulation/status")
def timingfix_simulation_status(task_id: str = ""):
    if task_id:
        return _read_runtime(task_id)
    latest = _current_running()
    if latest is None:
        return {"status": "idle", "running": False, "version": "v10_timingfix"}
    return _read_runtime(latest["task_id"])


@router.get("/api/timingfix/simulation/log")
def timingfix_simulation_log(task_id: str = ""):
    if task_id:
        path = _RUNTIME / f"{task_id}.log"
        lines = path.read_text(errors="replace").splitlines()[-500:] if path.exists() else []
        return {"version": "v10_timingfix", "task_id": task_id,
                "status": _read_runtime(task_id), "lines": lines}
    latest = _current_running()
    if latest is None:
        return {"version": "v10_timingfix", "status": {"status": "idle"}, "lines": []}
    task_id = latest["task_id"]
    path = _RUNTIME / f"{task_id}.log"
    lines = path.read_text(errors="replace").splitlines()[-500:] if path.exists() else []
    return {"version": "v10_timingfix", "task_id": task_id,
            "status": _read_runtime(task_id), "lines": lines}


@router.post("/api/timingfix/simulation/stop")
def stop_timingfix_simulation(task_id: str = ""):
    if not task_id:
        latest = _current_running()
        if latest is None:
            return {"status": "idle", "version": "v10_timingfix"}
        task_id = latest["task_id"]
    state_path = _RUNTIME / f"{task_id}.json"
    if not state_path.exists():
        raise HTTPException(404, f"任务不存在: {task_id}")
    state = json.loads(state_path.read_text())
    if state.get("status") in {"done", "failed", "stopped", "blocked", "stale"}:
        return {"status": state["status"], "task_id": task_id}
    _stop_request_path(task_id).write_text("stop\n")
    state.update({"status": "stopping", "stop_requested_at": time.time()})
    _write_runtime(state_path, state)
    pid = state.get("pid")
    if isinstance(pid, int) and pid_alive(pid):
        try:
            os.killpg(pid, 15)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return {"status": "stopping", "task_id": task_id, "version": "v10_timingfix"}
