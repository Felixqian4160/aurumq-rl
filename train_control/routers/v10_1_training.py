"""v10.1 standalone training API for the WebUI."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .common import AURUMQ_ROOT, PYTHON_EXE
from core.process import pid_alive

router = APIRouter(tags=["v10.1-training"])
_PROC: Optional[subprocess.Popen] = None
_RUNTIME = AURUMQ_ROOT / "data" / "task_runtime"
_RUNTIME.mkdir(parents=True, exist_ok=True)


class V101TrainingRequest(BaseModel):
    panel: str
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    out_dir: str = "models/whv10_1_csi500_smoke_10k"
    total_timesteps: int = Field(10000, ge=100, le=10_000_000)
    seed: int = 42
    window: int = Field(20, ge=1, le=120)
    top_k: int = Field(20, ge=1, le=100)
    max_position_pct: float = Field(0.02, gt=0, le=1)
    cost_bps: float = Field(15.0, ge=0, le=100)
    stop_loss_pct: float = Field(0.0, ge=0, lt=100)
    cooldown_days: int = Field(5, ge=0, le=252)
    lstm_hidden: int = Field(64, ge=8, le=64)
    lstm_layers: int = Field(1, ge=1, le=4)
    n_steps: int = Field(32, ge=8, le=32)
    batch_size: int = Field(32, ge=8, le=32)
    learning_rate: float = Field(3e-5, gt=0, le=0.01)
    aux_lambda: float = Field(0.1, ge=0, le=10)
    a1_pos_weight: float = Field(20.0, gt=0, le=1000)
    a2_pos_weight: float = Field(40.0, gt=0, le=1000)
    peak_pos_weight: float = Field(80.0, gt=0, le=1000)
    b1_pos_weight: float = Field(10.0, gt=0, le=1000)


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _read(task_id: str) -> dict:
    path = _RUNTIME / f"{task_id}.json"
    if not path.exists():
        return {"task_id": task_id, "version": "v10.1", "status": "not_found", "running": False}
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"task_id": task_id, "version": "v10.1", "status": "failed", "running": False}
    if state.get("status") == "running" and not pid_alive(state.get("pid")):
        state.update(status="failed", running=False, progress="训练进程已退出")
        _write(path, state)
    state["running"] = state.get("status") == "running"
    return state


def _latest() -> dict:
    paths = sorted(_RUNTIME.glob("v10_1_train_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return _read(paths[0].stem) if paths else {"version": "v10.1", "status": "idle", "running": False}


def _watch(task_id: str, proc: subprocess.Popen) -> None:
    rc = proc.wait()
    path = _RUNTIME / f"{task_id}.json"
    state = _read(task_id)
    out_dir = Path(state.get("out_dir", ""))
    required = [out_dir / n for n in ("ppo_final.zip", "policy.onnx", "metadata.json", "training_summary.json")]
    missing = [str(p.name) for p in required if not p.is_file() or p.stat().st_size <= 0]
    requested = state.get("status") in {"stopping", "stopped"}
    if requested:
        status, progress = "stopped", "训练已停止"
    elif rc == 0 and not missing:
        status, progress = "done", "训练、ONNX、metadata、summary 已验证"
    elif rc == 0:
        status, progress = "failed", "训练退出但缺少: " + ", ".join(missing)
    else:
        status, progress = "failed", f"训练进程退出 (returncode={rc})"
    state.update(status=status, running=False, returncode=rc, finished_at=time.time(), progress=progress, missing_artifacts=missing)
    _write(path, state)


@router.post("/api/v10_1/training/start")
def start_v101_training(req: V101TrainingRequest):
    global _PROC
    if _PROC and _PROC.poll() is None:
        raise HTTPException(409, "v10.1 训练已在运行")
    if not req.panel.startswith("wavehunter_v10_1_"):
        raise HTTPException(400, "v10.1 训练只接受 wavehunter_v10_1_ 面板")
    panel = AURUMQ_ROOT / "data" / req.panel
    if not panel.is_file():
        raise HTTPException(400, f"面板不存在: {req.panel}")
    out_dir = AURUMQ_ROOT / req.out_dir if req.out_dir.startswith("models/") else AURUMQ_ROOT / "models" / Path(req.out_dir).name
    if not out_dir.name.startswith("whv10_1_"):
        raise HTTPException(400, "v10.1 模型目录必须以 whv10_1_ 开头")
    runner = AURUMQ_ROOT / "scripts" / "v10_1" / "train_wavehunter_v10_1.py"
    if not runner.is_file():
        raise HTTPException(500, "v10.1 training runner 不存在")
    task_id = f"v10_1_train_{out_dir.name}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    state_path = _RUNTIME / f"{task_id}.json"
    log_path = _RUNTIME / f"{task_id}.log"
    state = {"version":"v10.1", "task_id":task_id, "status":"starting", "running":False,
             "panel":req.panel, "out_dir":str(out_dir), "log_path":str(log_path), "started_at":time.time()}
    _write(state_path, state)
    args = [PYTHON_EXE, str(runner), "--panel", str(panel), "--start-date", req.start_date,
            "--end-date", req.end_date, "--out-dir", str(out_dir), "--total-timesteps", str(req.total_timesteps),
            "--seed", str(req.seed), "--window", str(req.window), "--top-k", str(req.top_k),
            "--max-position-pct", str(req.max_position_pct), "--cost-bps", str(req.cost_bps),
            "--lstm-hidden", str(req.lstm_hidden), "--lstm-layers", str(req.lstm_layers),
            "--n-steps", str(req.n_steps), "--batch-size", str(req.batch_size),
            "--learning-rate", str(req.learning_rate), "--aux-lambda", str(req.aux_lambda),
            "--a1-pos-weight", str(req.a1_pos_weight), "--a2-pos-weight", str(req.a2_pos_weight),
            "--peak-pos-weight", str(req.peak_pos_weight), "--b1-pos-weight", str(req.b1_pos_weight)]
    try:
        with log_path.open("w") as out:
            _PROC = subprocess.Popen(args, cwd=str(AURUMQ_ROOT), stdout=out, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True,
                                      env={**os.environ, "PYTHONUNBUFFERED":"1"})
    except Exception as exc:
        state.update(status="failed", progress=f"启动失败: {exc}"); _write(state_path, state)
        raise HTTPException(500, f"v10.1 训练启动失败: {exc}") from exc
    state.update(pid=_PROC.pid, status="running", running=True, command=args)
    _write(state_path, state)
    threading.Thread(target=_watch, args=(task_id, _PROC), daemon=True).start()
    return {"status":"started", "version":"v10.1", "task_id":task_id, "log_path":str(log_path), "out_dir":str(out_dir)}


@router.get("/api/v10_1/training/status")
def v101_training_status(task_id: str = ""):
    return _read(task_id) if task_id else _latest()


@router.get("/api/v10_1/training/log")
def v101_training_log(task_id: str = ""):
    state = v101_training_status(task_id)
    path = Path(state.get("log_path", ""))
    return {"version":"v10.1", "task_id":state.get("task_id"), "status":state,
            "lines":path.read_text(errors="replace").splitlines()[-500:] if path.is_file() else []}


@router.post("/api/v10_1/training/stop")
def stop_v101_training(task_id: str = ""):
    state = v101_training_status(task_id)
    if not state.get("task_id"):
        return {"version":"v10.1", "status":"idle"}
    path = _RUNTIME / f"{state['task_id']}.json"
    state.update(status="stopping", stop_requested_at=time.time()); _write(path, state)
    _RUNTIME.joinpath(f"{state['task_id']}.stop").write_text("stop\n")
    pid = state.get("pid")
    if isinstance(pid, int) and pid_alive(pid):
        try: os.killpg(pid, 15)
        except (ProcessLookupError, PermissionError, OSError): pass
    return {"version":"v10.1", "task_id":state["task_id"], "status":"stopping"}
