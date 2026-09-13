"""模拟交易 v2 路由。独立于 routers/simulation.py 与 /api/sim/start。

对外只提供最终买卖信号配置：内部 A1/A2/Peak/B1 由 v10 runner 学习，
本 Tab 不把旧模拟状态混入新状态。
"""
from __future__ import annotations
import json, math, os, subprocess, threading, time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from .common import AURUMQ_ROOT, PYTHON_EXE

router = APIRouter(tags=["simulation-v2"])
_V2_THREAD: Optional[threading.Thread] = None
_V2_PROC = None
_V2_STATE = AURUMQ_ROOT / "data" / "sim_v2_runtime.json"
_V2_DIR = AURUMQ_ROOT / "data" / "task_runtime"
_V2_DIR.mkdir(parents=True, exist_ok=True)

class SimulationV2Request(BaseModel):
    model_dir: str = ""
    panel: str = ""
    initial_capital: float = 100000.0
    start_date: str = "2024-01-01"
    end_date: str = ""
    top_k: int = 20
    cost_bps: float = 15.0
    slippage_bps: float = 10.0
    rebalance_days: int = -1
    stop_loss_pct: float = -1.0
    max_position_pct: float = -1.0
    max_holding_days: int = -1
    take_profit_pct: float = 0.0
    a1_threshold: float = 0.55
    a2_threshold: float = 0.55
    peak_threshold: float = 0.50
    b1_threshold: float = 0.55
    min_buy_score: float = 0.40

@router.post("/api/sim-v2/start")
def start_simulation_v2(req: SimulationV2Request):
    global _V2_THREAD, _V2_PROC
    if _V2_THREAD and _V2_THREAD.is_alive():
        raise HTTPException(409, "模拟交易 v2 已在运行")
    if not req.model_dir:
        raise HTTPException(400, "请选择 v10 模型")
    for key, value, upper in (("cost_bps", req.cost_bps, 100.0), ("slippage_bps", req.slippage_bps, 100.0)):
        if not math.isfinite(value) or value < 0 or value > upper:
            raise HTTPException(400, f"{key} 必须在 0~{upper} bps 范围内，收到 {value}")
    model = AURUMQ_ROOT / "models" / req.model_dir
    if not model.is_dir() or not (model / "policy.onnx").exists():
        raise HTTPException(400, f"v10 模型不存在或缺少 policy.onnx: {req.model_dir}")
    panel = AURUMQ_ROOT / "data" / req.panel if req.panel else next(iter((AURUMQ_ROOT / "data").glob("wavehunter_v9_*.parquet")), None)
    if panel is None or not panel.exists():
        raise HTTPException(400, "v9 面板不存在")
    task_id = f"sim_v2_{req.model_dir}_{int(time.time())}"
    cfg_path = _V2_DIR / f"{task_id}.config.json"
    log_path = _V2_DIR / f"{task_id}.log"
    state_path = _V2_DIR / f"{task_id}.json"
    payload = req.model_dump()
    payload.update({"version": "simulation-v2", "costfix": True, "model_dir": str(model), "panel_path": str(panel), "end_date": req.end_date or "2026-08-04"})
    cfg_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    state = {"version":"simulation-v2", "task_id":task_id, "status":"starting", "running":False, "model":req.model_dir, "log_path":str(log_path), "runtime_path":str(state_path), "started_at":time.time()}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    runner = AURUMQ_ROOT / "scripts" / "v10" / "run_sim_v10.py"
    if not runner.exists():
        raise HTTPException(500, "v10 模拟入口不存在")
    def run():
        global _V2_PROC
        try:
            state["status"] = "running"; state["running"] = True
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)); _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            with log_path.open("a") as out:
                _V2_PROC = subprocess.Popen([PYTHON_EXE, str(runner), str(cfg_path), str(state_path)], cwd=str(AURUMQ_ROOT), stdout=out, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED":"1"})
                state["pid"] = _V2_PROC.pid; state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
                rc = _V2_PROC.wait()
            child = json.loads(state_path.read_text()) if state_path.exists() else {}
            state.update(child); state["returncode"] = rc; state["running"] = False; state["finished_at"] = time.time()
            if rc == 0 and state.get("status") == "running": state["status"] = "finished"
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)); _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        except Exception as exc:
            state.update({"status":"failed", "running":False, "error":str(exc), "finished_at":time.time()})
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)); _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        finally:
            _V2_PROC = None
    _V2_THREAD = threading.Thread(target=run, daemon=True); _V2_THREAD.start()
    return {"status":"started", "version":"simulation-v2", "task_id":task_id, "model":req.model_dir, "log_path":str(log_path)}

@router.post("/api/sim-v2/stop")
def stop_simulation_v2():
    if _V2_PROC and _V2_PROC.poll() is None:
        _V2_PROC.terminate()
    return {"status":"stopping", "version":"simulation-v2"}

@router.get("/api/sim-v2/status")
def simulation_v2_status():
    if not _V2_STATE.exists(): return {"status":"idle", "running":False, "version":"simulation-v2"}
    try:
        d=json.loads(_V2_STATE.read_text()); d["running"]=bool(_V2_THREAD and _V2_THREAD.is_alive()); return d
    except Exception as exc: return {"status":"failed", "running":False, "version":"simulation-v2", "error":str(exc)}

@router.get("/api/sim-v2/log")
def simulation_v2_log():
    d=simulation_v2_status(); p=d.get("log_path"); lines=Path(p).read_text(errors="replace").splitlines()[-500:] if p and Path(p).exists() else []
    return {"version":"simulation-v2", "status":d, "lines":lines}
