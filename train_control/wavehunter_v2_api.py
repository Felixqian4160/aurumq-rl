"""WaveHunter v2 WebUI API.

独立于 v1 API：只调用 scripts/train_wavehunter_v2.py，使用独立状态和 whv2_* 目录。
当前 72-run 占用 GPU 时拒绝启动，避免并发训练破坏实验矩阵。
"""
from __future__ import annotations

import datetime
import os
import shlex
import signal
import subprocess
import threading
import json
import re
import time
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

from app import AURUMQ_ROOT, app, log
from wavehunter_api import WaveHunterTrainRequest
from wavehunter_v2_schema import schema as v2_schema

_V2_THREAD: threading.Thread | None = None
_V2_PROC: subprocess.Popen | None = None
_V2_CMD: str = ""
_MATRIX_INTERNAL = False


def set_matrix_internal(enabled: bool) -> None:
    global _MATRIX_INTERNAL
    _MATRIX_INTERNAL = enabled
_V2_STOP = threading.Event()
_V2_CONFIGS = AURUMQ_ROOT / "data" / "wavehunter_v2_saved_configs.json"
_V2_STATE = AURUMQ_ROOT / "data" / "wavehunter_v2_runtime.json"
_V2_LOG_DIR = AURUMQ_ROOT / "data" / "training_logs"
_V2_LOG_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/wavehunter/v2/schema")
def wavehunter_v2_schema():
    return v2_schema()


def _load_v2_configs() -> list:
    if _V2_CONFIGS.exists():
        try:
            return json.loads(_V2_CONFIGS.read_text())
        except Exception:
            pass
    return []


def _save_v2_configs(items: list) -> None:
    _V2_CONFIGS.write_text(json.dumps(items, ensure_ascii=False, indent=2))


def _matrix_running() -> bool:
    """阻止 v1/v2 任一矩阵期间手动启动 v2，避免 GPU 并发。"""
    # 矩阵 worker 本身必须复用本模块的单模型入口；仅阻止外部手动并发。
    if _MATRIX_INTERNAL or getattr(threading.current_thread(), "_wavehunter_v2_matrix_worker", False):
        return False
    for module_name in ("wavehunter_matrix_api", "wavehunter_v2_matrix_api"):
        try:
            module = __import__(module_name)
            thread = getattr(module, "_matrix_thread", None)
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                return True
        except Exception:
            continue
    return False


def _validate(req: WaveHunterTrainRequest) -> Path:
    try:
        start = datetime.date.fromisoformat(req.start_date)
        end = datetime.date.fromisoformat(req.end_date)
    except ValueError as exc:
        raise HTTPException(400, f"日期格式非法: {exc}") from exc
    if start >= end:
        raise HTTPException(400, "训练开始日期必须早于结束日期")
    if end >= datetime.date(2024, 1, 1):
        raise HTTPException(400, "训练结束日期必须早于 2024-01-01")
    is_v9_panel = bool(req.panel and req.panel.startswith("wavehunter_v9_"))
    # 审计 P0: 与 train_wavehunter_v2.py 强制降级阈值对齐 (RTX 3060 12GB 实测边界: hidden=64/n_steps 64/batch 64 速度最快; v9 已修复 EMA 归一)
    if is_v9_panel:
        if req.n_steps > 64:
            raise HTTPException(400, f"v9 要求 n_steps<=64, 当前 {req.n_steps}")
        if req.batch_size > 64:
            raise HTTPException(400, f"v9 要求 batch_size<=64, 当前 {req.batch_size}")
    else:
        if req.n_steps > 32:
            raise HTTPException(400, f"v2安全配置要求 n_steps<=32, 当前 {req.n_steps}")
        if req.batch_size > 32:
            raise HTTPException(400, f"v2安全配置要求 batch_size<=32, 当前 {req.batch_size}")
    if req.n_envs not in (1, 2):
        raise HTTPException(400, "v2安全配置要求 n_envs=1或2（n_envs=2仅限10k烟测）")
    if req.lstm_hidden > 64:
        raise HTTPException(400, f"v2安全配置要求 lstm_hidden<=64, 当前 {req.lstm_hidden}")
    panel = AURUMQ_ROOT / "data" / req.panel
    if not panel.exists() and req.panel.startswith("wavehunter_"):
        # v2 面板实际命名为 wavehunter_v2_*；v9 面板命名为 wavehunter_v9_*；兼容旧 schema 默认名。
        for prefix in ("wavehunter_v2_", "wavehunter_v9_"):
            candidate = AURUMQ_ROOT / "data" / req.panel.replace("wavehunter_", prefix, 1)
            if candidate.exists():
                panel = candidate
                break
    if not panel.exists():
        raise HTTPException(400, f"面板不存在: {req.panel}")
    return panel


def _command(req: WaveHunterTrainRequest, out_name: str, panel: Path) -> list[str]:
    # 根据面板版本选择训练脚本
    is_v9 = "wavehunter_v9" in panel.name
    if is_v9:
        script = AURUMQ_ROOT / "scripts" / "train_wavehunter_v9.py"
    else:
        script = AURUMQ_ROOT / "scripts" / "train_wavehunter_v2.py"

    values = [
        ("--panel", str(panel)), ("--start-date", req.start_date), ("--end-date", req.end_date),
        ("--out-dir", str(AURUMQ_ROOT / "models" / Path(out_name).name)),
        ("--total-timesteps", req.total_timesteps), ("--window", req.window), ("--top-k", req.top_k),
        ("--universe-filter", req.universe_filter), ("--n-factors", req.n_factors), ("--forward-period", req.forward_period),
        ("--max-position-pct", req.max_position_pct), ("--rebalance-days", req.rebalance_days),
        ("--cost-bps", req.cost_bps), ("--learning-rate", req.learning_rate),
        ("--lstm-hidden", req.lstm_hidden), ("--lstm-layers", req.lstm_layers),
        ("--reward-type", "absolute_return_v3"),
        ("--n-envs", req.n_envs), ("--seed", req.seed), ("--max-grad-norm", req.max_grad_norm),
        ("--target-kl", req.target_kl), ("--batch-size", req.batch_size), ("--n-steps", req.n_steps),
        ("--aux-lambda", req.aux_lambda), ("--a1-lambda", req.a1_lambda),
        ("--a1-pos-weight", req.a1_pos_weight), ("--a2-lambda", req.a2_lambda),
        ("--a2-pos-weight", req.a2_pos_weight),
        ("--label-window", req.label_window), ("--label-entry-days", req.label_entry_days),
        ("--label-n-bins", req.label_n_bins), ("--label-a2-threshold", req.label_a2_threshold),
        ("--label-a2-window", req.label_a2_window), ("--label-min-amount-pct", req.label_min_amount_pct),
        ("--hit-rate-bonus-weight", getattr(req, 'hit_rate_bonus_weight', 0.0)),
        ("--continuation-bonus-weight", getattr(req, 'continuation_bonus_weight', 0.0)),
        ("--drawdown-penalty", getattr(req, 'drawdown_penalty', 0.0)),
        ("--excess-weight", getattr(req, 'excess_weight', 0.0)),
        ("--reward-scale", getattr(req, 'reward_scale', 100.0)),
    ]
    # v9 特有参数
    if is_v9:
        values += [
            ("--reward-w-abs", getattr(req, 'reward_w_abs', 0.60)),
            ("--reward-w-dd", getattr(req, 'reward_w_dd', 0.25)),
            ("--reward-w-hit", getattr(req, 'reward_w_hit', 0.15)),
            ("--knn-k", getattr(req, 'knn_k', 20)),
            ("--peak-lambda", getattr(req, 'peak_lambda', 1.0)),
            ("--peak-pos-weight", getattr(req, 'peak_pos_weight', 25.0)),
            ("--b1-lambda", getattr(req, 'b1_lambda', 1.0)),
            ("--b1-pos-weight", getattr(req, 'b1_pos_weight', 25.0)),
        ]
    cmd = [os.environ.get("PYTHON", "/usr/bin/python3"), str(script)]
    for key, value in values:
        cmd += [key, str(value)]
    return cmd


@app.post("/api/wavehunter/v2/train/start")
def wavehunter_v2_start(req: WaveHunterTrainRequest):
    global _V2_THREAD, _V2_PROC, _V2_CMD
    if not req.matrix_internal and _matrix_running():
        raise HTTPException(409, "72-run矩阵正在运行，v2训练暂不能并发启动")
    if _V2_THREAD and _V2_THREAD.is_alive():
        raise HTTPException(400, "已有 WaveHunter v2 训练运行中")
    panel = _validate(req)
    # 使用实际解析后的 v2 面板文件名，避免 schema 默认旧名称传入训练命令。
    req.panel = panel.name
    pool = panel.stem.split("_")[2] if panel.stem.startswith("wavehunter_v2_") else (panel.stem.split("_")[1] if "_" in panel.stem else "hs300")
    out_name = Path(req.out_dir).name if req.out_dir else f"whv2_{pool}_{datetime.datetime.now():%m%d}_{req.total_timesteps // 1000}k"
    cmd = _command(req, out_name, panel)
    _V2_CMD = shlex.join(cmd)
    _V2_STOP.clear()

    log_path = _V2_LOG_DIR / f"{out_name}.log"
    state = {"out_name": out_name, "cmd": _V2_CMD, "log_path": str(log_path), "status": "starting", "pid": None, "total": req.total_timesteps, "current": 0, "percent": 0.0, "reward": "absolute_return", "started_at": time.time()}
    _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    def run() -> None:
        global _V2_PROC
        started_at = time.monotonic()
        try:
            log(f"🌊 WaveHunter v2 启动: {out_name} | absolute_return", source="wavehunter_v2", task_id=out_name, event="start")
            state["status"] = "running"
            _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log_path.touch(exist_ok=True)
            child_cmd = shlex.join(cmd) + ' >> ' + shlex.quote(str(log_path)) + ' 2>&1'
            systemd_cmd = ["systemd-run", "--user", "--scope", "--unit", f"aurumq-v2-{out_name}", "--collect", "--quiet", "--", "/bin/bash", "-lc", child_cmd]
            _V2_PROC = subprocess.Popen(
                systemd_cmd, cwd=str(AURUMQ_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True, start_new_session=True,
            )
            _V2_PROC.wait()
            time.sleep(0.5)
            child_pid = None
            for _ in range(5):
                try:
                    child = subprocess.check_output(["pgrep", "-f", f"train_wavehunter_v2.py.*{out_name}"], text=True).splitlines()
                    child_pid = int(child[0]) if child else None
                except Exception:
                    child_pid = None
                if child_pid: break
                time.sleep(0.2)
            state.update({"pid": child_pid or _V2_PROC.pid, "status": "running"})
            _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            while child_pid and Path(f"/proc/{child_pid}").exists():
                time.sleep(2)
            # systemd-run 外层退出后，子 PID 可能已消失，不能把 ps 查询失败误判为 rc=1。
            # 以真实训练产物作为完成真源；主动停止优先记录 stopped。
            out_dir = AURUMQ_ROOT / "models" / Path(out_name).name
            completed = (out_dir / "ppo_final.zip").exists() and (out_dir / "training_summary.json").exists()
            stopped = _V2_STOP.is_set()
            if completed:
                final_status, rc = "finished", 0
            elif stopped:
                final_status, rc = "stopped", -15
            else:
                final_status, rc = "failed", 1
            state.update({"status": final_status, "returncode": rc, "finished_at": time.time()})
            _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log(f"🌊 WaveHunter v2 结束 exit={rc}", source="wavehunter_v2", task_id=out_name, event="finish" if rc == 0 else "error", level="INFO" if rc == 0 else "ERROR")
        except Exception as exc:
            state.update({"status": "failed", "error": str(exc), "finished_at": time.time()})
            _V2_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log(f"🌊 WaveHunter v2 异常: {exc}", source="wavehunter_v2", task_id=out_name, event="error", level="ERROR")
        finally:
            _V2_PROC = None

    _V2_THREAD = threading.Thread(target=run, daemon=True)
    _V2_THREAD.start()
    return {"status": "started", "version": "v2", "out_name": out_name, "cmd": _V2_CMD}


def _command_v3(req: WaveHunterTrainRequest, out_name: str, panel: Path,
                wave_threshold: float = 0.10,
                min_pullback_days: int = 5,
                hit_rate_weight: float = 0.3,
                continuation_weight: float = 0.2,
                drawdown_penalty: float = 0.1) -> list[str]:
    """v3 双头训练命令 (共享 v2 几乎所有参数, 仅新增 5 个 v3 专属)."""
    script = AURUMQ_ROOT / "scripts" / "train_wavehunter_v3.py"
    values = [
        ("--panel", str(panel)), ("--start-date", req.start_date), ("--end-date", req.end_date),
        ("--out-dir", str(AURUMQ_ROOT / "models" / Path(out_name).name)),
        ("--total-timesteps", req.total_timesteps), ("--window", req.window), ("--top-k", req.top_k),
        ("--universe-filter", req.universe_filter), ("--n-factors", req.n_factors),
        ("--forward-period", req.forward_period), ("--max-position-pct", req.max_position_pct),
        ("--rebalance-days", req.rebalance_days), ("--cost-bps", req.cost_bps),
        ("--learning-rate", req.learning_rate), ("--lstm-hidden", req.lstm_hidden),
        ("--lstm-layers", req.lstm_layers), ("--reward-type", "absolute_return_v3"),
        ("--n-envs", req.n_envs), ("--seed", req.seed), ("--max-grad-norm", req.max_grad_norm),
        ("--target-kl", req.target_kl), ("--batch-size", req.batch_size), ("--n-steps", req.n_steps),
        ("--aux-lambda", req.aux_lambda), ("--a1-lambda", req.a1_lambda),
        ("--a1-pos-weight", req.a1_pos_weight),
        ("--a2-lambda", req.a2_lambda), ("--a2-pos-weight", req.a2_pos_weight),
        ("--label-window", req.label_window), ("--label-entry-days", req.label_entry_days),
        ("--label-n-bins", req.label_n_bins), ("--label-a2-threshold", req.label_a2_threshold),
        ("--label-a2-window", req.label_a2_window), ("--label-min-amount-pct", req.label_min_amount_pct),
        # v3 独有参数
        ("--wave-threshold", wave_threshold),
        ("--min-pullback-days", min_pullback_days),
        ("--hit-rate-weight", hit_rate_weight),
        ("--continuation-weight", continuation_weight),
        ("--drawdown-penalty", drawdown_penalty),
        ("--hit-rate-bonus-weight", 0.5),
        ("--continuation-bonus-weight", 0.3),
    ]
    cmd = [os.environ.get("PYTHON", "/usr/bin/python3"), str(script)]
    for key, value in values:
        cmd += [key, str(value)]
    return cmd


class WaveHunterV3TrainRequest(WaveHunterTrainRequest):
    """v3 训练请求 — 继承 v2 所有字段, 仅新增 5 个 v3 专属."""
    a1_pos_weight: float = 3.0   # v3: 降低正样本权重，避免压制 PPO 信号
    a2_pos_weight: float = 3.0   # v3: 降低正样本权重，避免压制 PPO 信号
    # v3 独有
    wave_threshold: float = 0.10
    min_pullback_days: int = 5
    hit_rate_weight: float = 0.3
    continuation_weight: float = 0.2
    drawdown_penalty: float = 0.1

    model_config = {"protected_namespaces": ()}


_V3_THREAD: threading.Thread | None = None
_V3_PROC: subprocess.Popen | None = None
_V3_CMD: str = ""
_V3_STOP = threading.Event()
_V3_STATE = AURUMQ_ROOT / "data" / "wavehunter_v3_state.json"
_V3_LOG_DIR = AURUMQ_ROOT / "data" / "training_logs"


@app.post("/api/wavehunter/v3/train/start")
def wavehunter_v3_start(req: WaveHunterV3TrainRequest):
    global _V3_THREAD, _V3_PROC, _V3_CMD
    if _matrix_running():
        raise HTTPException(409, "72-run矩阵正在运行，v3训练暂不能并发启动")
    if _V3_THREAD and _V3_THREAD.is_alive():
        persisted = {}
        try:
            if _V3_STATE.exists():
                persisted = json.loads(_V3_STATE.read_text())
        except Exception:
            pass
        if persisted.get("status") in ("starting", "running"):
            raise HTTPException(400, "已有 WaveHunter v3 训练运行中")
        # 旧终态线程对象只是在做收尾，回收引用后允许下一个串行 run。
        _V3_THREAD = None
    panel = _validate(req)
    pool = panel.stem.split("_")[2] if panel.stem.startswith("wavehunter_") else "hs300"
    out_name = Path(req.out_dir).name if req.out_dir else f"whv3_{pool}_{datetime.datetime.now():%m%d}_{req.total_timesteps // 1000}k"
    cmd = _command_v3(req, out_name, panel,
                       wave_threshold=req.wave_threshold,
                       min_pullback_days=req.min_pullback_days,
                       hit_rate_weight=req.hit_rate_weight,
                       continuation_weight=req.continuation_weight,
                       drawdown_penalty=req.drawdown_penalty)
    _V3_CMD = shlex.join(cmd)
    _V3_STOP.clear()

    log_path = _V3_LOG_DIR / f"{out_name}.log"
    state = {"out_name": out_name, "cmd": _V3_CMD, "log_path": str(log_path),
             "status": "starting", "pid": None, "total": req.total_timesteps,
             "current": 0, "percent": 0.0, "reward": "absolute_return_v3",
             "started_at": time.time(), "version": "v3"}
    _V3_STATE.parent.mkdir(parents=True, exist_ok=True)
    _V3_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    def run() -> None:
        global _V3_PROC
        try:
            log(f"🌊 WaveHunter v3 启动 (双头): {out_name}", source="wavehunter_v3", task_id=out_name, event="start")
            state["status"] = "running"
            _V3_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=True)
            child_cmd = shlex.join(cmd) + ' >> ' + shlex.quote(str(log_path)) + ' 2>&1'
            systemd_cmd = ["systemd-run", "--user", "--scope", "--unit", f"aurumq-v3-{out_name}",
                           "--collect", "--quiet", "--", "/bin/bash", "-lc", child_cmd]
            _V3_PROC = subprocess.Popen(systemd_cmd, cwd=str(AURUMQ_ROOT),
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        text=True, start_new_session=True)
            _V3_PROC.wait()
            time.sleep(0.5)
            child_pid = None
            for _ in range(5):
                try:
                    child = subprocess.check_output(["pgrep", "-f", f"train_wavehunter_v3.py.*{out_name}"], text=True).splitlines()
                    child_pid = int(child[0]) if child else None
                except Exception:
                    child_pid = None
                if child_pid: break
                time.sleep(0.2)
            state.update({"pid": child_pid or _V3_PROC.pid, "status": "running"})
            _V3_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            while child_pid and Path(f"/proc/{child_pid}").exists():
                time.sleep(2)
            out_dir = AURUMQ_ROOT / "models" / Path(out_name).name
            completed = (out_dir / "ppo_final.zip").exists() and (out_dir / "metadata.json").exists()
            stopped = _V3_STOP.is_set()
            if stopped:
                state.update({"status": "stopped", "running": False, "finished_at": time.time()})
            elif completed:
                state.update({"status": "done", "running": False, "finished_at": time.time(),
                              "current": req.total_timesteps, "percent": 100.0})
                log(f"✅ WaveHunter v3 完成: {out_name}", source="wavehunter_v3", task_id=out_name, event="done")
            else:
                state.update({"status": "failed", "running": False, "finished_at": time.time(),
                              "error": "产物缺失"})
                log(f"❌ WaveHunter v3 失败: {out_name}", source="wavehunter_v3", task_id=out_name, event="failed")
            _V3_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        except Exception as e:
            log(f"❌ WaveHunter v3 异常: {e}", source="wavehunter_v3", event="error")
            state.update({"status": "failed", "error": str(e)})
            _V3_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    _V3_THREAD = threading.Thread(target=run, daemon=True)
    _V3_THREAD.start()
    return {"status": "started", "version": "v3", "out_name": out_name, "cmd": _V3_CMD}


@app.post("/api/wavehunter/v3/train/stop")
def wavehunter_v3_stop():
    _V3_STOP.set()
    if _V3_PROC and _V3_PROC.poll() is None:
        try:
            os.killpg(_V3_PROC.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return {"status": "stopping", "version": "v3"}


@app.get("/api/wavehunter/v3/train/status")
def wavehunter_v3_status():
    if not _V3_STATE.exists():
        return {"running": False, "status": "idle", "version": "v3"}
    try:
        return {**json.loads(_V3_STATE.read_text()), "version": "v3"}
    except Exception:
        return {"running": False, "status": "idle", "version": "v3"}


@app.get("/api/wavehunter/v3/train/log")
def wavehunter_v3_log():
    status = wavehunter_v3_status()
    path = status.get("log_path")
    if not path or not Path(path).exists():
        return {"log_path": path, "lines": [], "status": status}
    lines = Path(path).read_text(errors="replace").splitlines()
    return {"log_path": path, "lines": lines[-500:], "status": status}


@app.post("/api/wavehunter/v2/train/stop")
def wavehunter_v2_stop():
    _V2_STOP.set()
    if _V2_PROC and _V2_PROC.poll() is None:
        try:
            os.killpg(_V2_PROC.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    return {"status": "stopping", "version": "v2"}


@app.get("/api/wavehunter/v2/train/log")
def wavehunter_v2_log():
    status = wavehunter_v2_status()
    path = status.get("log_path")
    if not path or not Path(path).exists():
        return {"log_path": path, "lines": [], "status": status}
    lines = Path(path).read_text(errors="replace").splitlines()
    return {"log_path": path, "lines": lines[-500:], "status": status}


@app.get("/api/wavehunter/v2/train/status")
def wavehunter_v2_status():
    persisted = {}
    if _V2_STATE.exists():
        try: persisted = json.loads(_V2_STATE.read_text())
        except Exception: pass
    if persisted.get("log_path"):
        try:
            tail = Path(persisted["log_path"]).read_text(errors="replace")[-200000:]
            matches = re.findall(r"\[wh\]\s+(\d+)\s+步", tail)
            if matches:
                current = int(matches[-1]); total = int(persisted.get("total") or 0)
                persisted.update({"current": current, "total": total, "percent": round(current / total * 100, 2) if total else None})
        except Exception: pass
    # WebUI 重启后线程内 PID 丢失：从持久化命令查找真实训练子进程回填。
    if persisted.get("status") == "running" and not persisted.get("pid"):
        try:
            pattern = str(persisted.get("out_name", ""))
            ps = subprocess.check_output(["pgrep", "-af", f"train_wavehunter_v2.py.*{pattern}"], text=True)
            for line in ps.splitlines():
                parts = line.split(None, 1)
                if parts and parts[0].isdigit():
                    persisted["pid"] = int(parts[0]); break
            if persisted.get("pid"):
                _V2_STATE.write_text(json.dumps(persisted, ensure_ascii=False, indent=2))
        except Exception:
            pass
    live = bool(_V2_THREAD and _V2_THREAD.is_alive())
    persisted_live = bool(persisted.get("status") == "running" and persisted.get("pid") and Path(f"/proc/{persisted['pid']}").exists())
    if persisted.get("status") == "running" and not live and not persisted_live:
        out = Path(persisted.get("out_name", ""))
        if not out.is_absolute():
            out = AURUMQ_ROOT / "models" / out
        completed = (out / "training_summary.json").exists() or (out / "ppo_final.zip").exists()
        persisted["status"] = "finished" if completed else "stale"
        if completed:
            persisted.setdefault("returncode", 0)
        else:
            persisted["error"] = f"训练进程 PID {persisted.get('pid')} 已不存在"
        try: _V2_STATE.write_text(json.dumps(persisted, ensure_ascii=False, indent=2))
        except Exception: pass
    return {"running": live or persisted_live, "version": "v2", "cmd": _V2_CMD or persisted.get("cmd", ""), "stopping": _V2_STOP.is_set(), "reward": "absolute_return", **persisted}


@app.get("/api/wavehunter/v2/models")
def wavehunter_v2_models():
    models = []
    # v2 训练统一写入 models/，旧 runs/ 不再作为新产物目录。
    model_root = AURUMQ_ROOT / "models"
    for d in sorted(model_root.glob("*"), reverse=True):
        if not d.is_dir() or not (d.name.startswith("whv2_") or d.name.startswith("whv10_") or d.name.startswith("matrix_") or d.name.startswith("whm_")):
            continue
        summary = d / "training_summary.json"
        item = {"name": d.name, "has_ppo": (d / "ppo_final.zip").exists(), "has_summary": summary.exists(), "model_version": "wavehunter_v2"}
        if summary.exists():
            try:
                data = json.loads(summary.read_text())
                item.update({"timesteps": data.get("total_timesteps"), "panel": data.get("panel"), "aux_stats": data.get("aux_stats"), "hyperparams": data.get("hyperparams", data.get("config", {}))})
            except Exception:
                pass
        models.append(item)
    return {"models": models}


@app.delete("/api/wavehunter/v2/models/{name}")
def wavehunter_v2_model_delete(name: str):
    """删除统一 models/ 下的 WaveHunter 模型。"""
    target = AURUMQ_ROOT / "models" / name
    if not target.exists() or not name.startswith(('whv2_', 'matrix_', 'whm_')):
        raise HTTPException(404, f"v2模型不存在: {name}")
    import shutil
    shutil.rmtree(target)
    return {"status": "deleted", "version": "v2"}


@app.get("/api/wavehunter/v2/configs")
def wavehunter_v2_configs():
    return {"configs": _load_v2_configs(), "version": "v2"}


@app.post("/api/wavehunter/v2/configs/save")
def wavehunter_v2_config_save(req: dict):
    name = str(req.get("name", "")).strip()
    if not name:
        raise HTTPException(400, "缺少预设名称")
    items = [x for x in _load_v2_configs() if x.get("name") != name]
    items.append({"name": name, "params": req.get("params", {}), "version": "v2"})
    _save_v2_configs(items)
    return {"status": "saved", "version": "v2"}


@app.delete("/api/wavehunter/v2/configs/{name}")
def wavehunter_v2_config_delete(name: str):
    _save_v2_configs([x for x in _load_v2_configs() if x.get("name") != name])
    return {"status": "deleted", "version": "v2"}
