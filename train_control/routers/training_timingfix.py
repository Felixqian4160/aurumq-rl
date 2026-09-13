"""Independent WebUI API for v10_timingfix training.

统一进程模型：runtime JSON + start_new_session=True 立即脱离父进程；
启动检查、watchdog、按 task_id 停止、停止请求持久化。
"""
from __future__ import annotations

import json
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

router = APIRouter(tags=["v10-timingfix-training"])

_PROC: Optional[subprocess.Popen] = None
_RUNTIME = AURUMQ_ROOT / "data" / "task_runtime"
_TRAINING_RUNTIME_DIR = AURUMQ_ROOT / "data" / "training_runtime"
_RUNTIME.mkdir(parents=True, exist_ok=True)
_TRAINING_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
_STARTUP_TIMEOUT_SECONDS = 15


def _stop_request_path(task_id: str) -> Path:
    return _RUNTIME / f"{task_id}.stop"


def _write_runtime(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _training_artifacts(state: dict) -> tuple[list[str], bool]:
    out_dir = Path(str(state.get("out_dir", "")))
    required = ("ppo_final.zip", "policy.onnx", "metadata.json",
                "training_metrics.jsonl", "training_summary.json")
    missing = [name for name in required
               if not (out_dir / name).is_file()
               or (out_dir / name).stat().st_size <= 0]
    log_path = Path(str(state.get("log_path", "")))
    complete = False
    if log_path.is_file():
        complete = "[timingfix] complete:" in log_path.read_text(
            encoding="utf-8", errors="replace")[-8192:]
    return missing, complete


def _finalize_training_state(state: dict, returncode: int) -> dict:
    missing, complete = _training_artifacts(state)
    stop_requested = state.get("status") == "stopping" or _stop_request_path(
        state.get("task_id", "")).exists()
    if stop_requested:
        status = "stopped"
        progress = "训练已停止"
    elif returncode == 0 and not missing and complete:
        status = "done"
        progress = "训练、导出及指标产物已验证"
    elif returncode == 0:
        status = "failed"
        progress = "训练进程退出，但产物校验失败: 缺少 " + ", ".join(missing or ["完成标记"])
    else:
        status = "failed"
        progress = f"训练进程退出 (returncode={returncode})"
    state.update({"status": status, "running": False, "returncode": returncode,
                  "finished_at": time.time(), "progress": progress})
    stop_path = _stop_request_path(state.get("task_id", ""))
    if stop_path.exists():
        stop_path.unlink()
    return state


def _watch_training_process(task_id: str, proc: subprocess.Popen) -> None:
    returncode = proc.wait()
    path = _RUNTIME / f"{task_id}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    _write_runtime(path, _finalize_training_state(state, returncode))


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
                      progress="训练进程已退出")
        _write_runtime(path, state)
    state["running"] = state.get("status") == "running"
    return state


def _current_running() -> Optional[dict]:
    """从所有训练 runtime 中取最新一条 running/failed 任务。"""
    candidates: list[tuple[float, Path]] = []
    for path in _RUNTIME.glob("timingfix_train_*.json"):
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if st.get("status") in {"starting", "running", "stopping"}:
            candidates.append((st.get("started_at", 0), path))
        elif st.get("status") == "failed" and not pid_alive(st.get("pid")):
            candidates.append((st.get("started_at", 0), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, path = candidates[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _watch_training_start(task_id: str, started_at: float) -> None:
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
                  "progress": f"训练进程在 {_STARTUP_TIMEOUT_SECONDS}s 内未进入活动状态"})
    _write_runtime(path, state)
    try:
        from core.logging import log
        log(f"❌ TimingFix 训练启动失败: {task_id}", source="timingfix", task_id=task_id, event="failed")
    except Exception:
        pass


class TimingFixTrainingRequest(BaseModel):
    panel: str = ""
    start_date: str
    end_date: str
    out_dir: str
    total_timesteps: int = Field(10000, ge=1000, le=10_000_000)
    seed: int = 42
    window: int = Field(20, ge=1, le=120)
    top_k: int = Field(20, ge=1, le=100)
    max_position_pct: float = Field(0.05, gt=0, le=1)
    cost_bps: float = Field(15.0, ge=0, le=100)
    slippage_bps: float = Field(10.0, ge=0, le=100)
    stop_loss_pct: float = Field(0.0, ge=0, lt=100)
    cooldown_days: int = Field(5, ge=0, le=252)
    lstm_hidden: int = Field(32, ge=8, le=64)
    lstm_layers: int = Field(1, ge=1, le=4)
    n_steps: int = Field(32, ge=8, le=256)
    batch_size: int = Field(32, ge=8, le=256)
    learning_rate: float = Field(3e-5, gt=0, le=0.01)


@router.post("/api/v10_1/training/start")
def start_v10_1_training(req: TimingFixTrainingRequest):
    """独立 v10.1 训练入口：只接受 v10_1 面板，输出 models/whv10_1_*。"""
    global _PROC
    if _PROC and _PROC.poll() is None:
        raise HTTPException(409, "v10.1 训练已在运行")
    if not req.panel.startswith("wavehunter_v10_1_") or not req.out_dir:
        raise HTTPException(400, "v10.1 训练必须选择 wavehunter_v10_1_ 面板和输出目录")
    panel = AURUMQ_ROOT / "data" / req.panel
    if not panel.is_file():
        raise HTTPException(400, f"v10.1 面板不存在: {req.panel}")
    if req.out_dir.startswith("models/"):
        out_dir = AURUMQ_ROOT / req.out_dir
    else:
        out_dir = AURUMQ_ROOT / "models" / Path(req.out_dir).name
    if not out_dir.name.startswith("whv10_1_"):
        raise HTTPException(400, "v10.1 模型目录必须以 whv10_1_ 开头")
    runner = AURUMQ_ROOT / "scripts" / "v10_1" / "train_wavehunter_v10_1.py"
    if not runner.is_file():
        raise HTTPException(500, "v10.1 training runner 不存在")
    task_id = f"v10_1_train_{out_dir.name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    state_path = _RUNTIME / f"{task_id}.json"
    log_path = _RUNTIME / f"{task_id}.log"
    state = {"version":"v10.1", "task_id":task_id, "status":"starting", "running":False,
             "panel":req.panel, "out_dir":str(out_dir), "log_path":str(log_path),
             "started_at":time.time()}
    _write_runtime(state_path,state)
    args = [PYTHON_EXE,str(runner),"--panel",str(panel),"--start-date",req.start_date,"--end-date",req.end_date,
            "--out-dir",str(out_dir),"--total-timesteps",str(req.total_timesteps),"--seed",str(req.seed),
            "--window",str(req.window),"--top-k",str(req.top_k),"--max-position-pct",str(req.max_position_pct),
            "--cost-bps",str(req.cost_bps),"--lstm-hidden",str(req.lstm_hidden),"--lstm-layers",str(req.lstm_layers),
            "--n-steps",str(min(req.n_steps,32)),"--batch-size",str(min(req.batch_size,32)),"--learning-rate",str(req.learning_rate),
            "--stop-loss-pct",str(req.stop_loss_pct),"--cooldown-days",str(req.cooldown_days)]
    try:
        with log_path.open("w") as out:
            _PROC=subprocess.Popen(args,cwd=str(AURUMQ_ROOT),stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                   start_new_session=True,env={**os.environ,"PYTHONUNBUFFERED":"1"})
    except Exception as exc:
        state.update(status="failed",progress=f"启动失败: {exc}");_write_runtime(state_path,state)
        raise HTTPException(500,f"v10.1 训练启动失败: {exc}") from exc
    state.update(pid=_PROC.pid,status="running",running=True,command=args)
    _write_runtime(state_path,state)
    import threading as _th
    _th.Thread(target=_watch_training_process,args=(task_id,_PROC),daemon=True).start()
    return {"status":"started","version":"v10.1","task_id":task_id,"log_path":str(log_path),"out_dir":str(out_dir)}


@router.get("/api/v10_1/training/status")
def v10_1_training_status(task_id: str = ""):
    if task_id:
        return _read_runtime(task_id)
    latest = sorted(_RUNTIME.glob("v10_1_train_*.json"), key=lambda p:p.stat().st_mtime, reverse=True)
    return _read_runtime(latest[0].stem) if latest else {"version":"v10.1","status":"idle","running":False}


@router.get("/api/v10_1/training/log")
def v10_1_training_log(task_id: str = ""):
    state = v10_1_training_status(task_id)
    path = Path(state.get("log_path", ""))
    return {"version":"v10.1","task_id":state.get("task_id"),"status":state,
            "lines":path.read_text(errors="replace").splitlines()[-500:] if path.is_file() else []}


@router.post("/api/v10_1/training/stop")
def stop_v10_1_training(task_id: str = ""):
    state = v10_1_training_status(task_id)
    if not state.get("task_id"):
        return {"version":"v10.1","status":"idle"}
    path = _RUNTIME / f"{state['task_id']}.json"
    state.update(status="stopping",stop_requested_at=time.time())
    _write_runtime(path,state)
    pid = state.get("pid")
    if isinstance(pid,int) and pid_alive(pid):
        try: os.killpg(pid,15)
        except (ProcessLookupError,PermissionError,OSError): pass
    _stop_request_path(state["task_id"]).write_text("stop\n")
    return {"version":"v10.1","task_id":state["task_id"],"status":"stopping"}



def start_timingfix_training(req: TimingFixTrainingRequest):
    global _PROC
    if _PROC and _PROC.poll() is None:
        raise HTTPException(409, "timingfix 训练已在运行")
    if not req.panel or not req.out_dir:
        raise HTTPException(400, "必须选择面板和输出目录")
    panel = AURUMQ_ROOT / "data" / req.panel
    out_dir = (AURUMQ_ROOT / req.out_dir if req.out_dir.startswith("models/")
                else AURUMQ_ROOT / "models" / Path(req.out_dir).name)
    runner = AURUMQ_ROOT / "scripts" / "v10_timingfix" / "train_timingfix.py"
    if not panel.exists():
        raise HTTPException(400, f"面板不存在: {req.panel}")
    if not runner.exists():
        raise HTTPException(500, "timingfix training runner 不存在")
    ts = time.time()
    task_id = (f"timingfix_train_{out_dir.name}_"
               f"{int(ts * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:6]}")
    cfg_path = _RUNTIME / f"{task_id}.config.json"
    log_path = _RUNTIME / f"{task_id}.log"
    state_path = _RUNTIME / f"{task_id}.json"
    payload = req.model_dump()
    payload.update(panel=str(panel), out_dir=str(out_dir))
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    state = {
        "version": "v10_timingfix",
        "task_id": task_id,
        "status": "starting",
        "running": False,
        "panel": req.panel,
        "out_dir": str(out_dir),
        "log_path": str(log_path),
        "config_path": str(cfg_path),
        "started_at": ts,
    }
    _write_runtime(state_path, state)

    args = [PYTHON_EXE, str(runner),
            "--panel", str(panel),
            "--start-date", req.start_date,
            "--end-date", req.end_date,
            "--out-dir", str(out_dir),
            "--total-timesteps", str(req.total_timesteps),
            "--seed", str(req.seed),
            "--window", str(req.window),
            "--top-k", str(req.top_k),
            "--max-position-pct", str(req.max_position_pct),
            "--cost-bps", str(req.cost_bps),
            "--slippage-bps", str(req.slippage_bps),
            "--stop-loss-pct", str(req.stop_loss_pct),
            "--cooldown-days", str(req.cooldown_days),
            "--lstm-hidden", str(req.lstm_hidden),
            "--lstm-layers", str(req.lstm_layers),
            "--n-steps", str(req.n_steps),
            "--batch-size", str(req.batch_size),
            "--learning-rate", str(req.learning_rate)]
    try:
        with log_path.open("w") as out:
            _PROC = subprocess.Popen(args, cwd=str(AURUMQ_ROOT),
                                      stdout=out, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL,
                                      start_new_session=True,
                                      env={**os.environ, "PYTHONUNBUFFERED": "1"})
    except Exception as exc:
        state.update({"status": "failed", "running": False,
                      "progress": f"启动失败: {exc}"})
        _write_runtime(state_path, state)
        raise HTTPException(500, f"训练启动失败: {exc}") from exc
    state["pid"] = _PROC.pid
    state["status"] = "running"
    state["running"] = True
    _write_runtime(state_path, state)
    import threading as _th
    _th.Thread(target=_watch_training_process, args=(task_id, _PROC), daemon=True).start()
    return {"status": "started", "version": "v10_timingfix",
            "task_id": task_id, "log_path": str(log_path)}


@router.get("/api/timingfix/training/status")
def timingfix_training_status(task_id: str = ""):
    if task_id:
        return _read_runtime(task_id)
    latest = _current_running()
    if latest is None:
        return {"status": "idle", "running": False, "version": "v10_timingfix"}
    state = _read_runtime(latest["task_id"])
    return state


@router.get("/api/timingfix/training/log")
def timingfix_training_log(task_id: str = ""):
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


@router.post("/api/timingfix/training/stop")
def stop_timingfix_training(task_id: str = ""):
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
