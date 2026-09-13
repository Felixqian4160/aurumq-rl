"""
train_control/services/panel_service.py — 面板构建服务单一真源

统一 v1/v3/v8 构建的状态读写与启动逻辑，routers/panel_build.py 仅做参数校验与委托。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import date
from pathlib import Path

from fastapi import HTTPException

from core.config import AURUMQ_ROOT, BUILD_LOG_DIR, BUILD_RUNTIME_DIR
from core.process import build_pid_alive
from core.state import read_build_state as _read_state  # reuse core

_BUILD_RUNTIME_DIR = BUILD_RUNTIME_DIR
_BUILD_LOG_DIR = BUILD_LOG_DIR
_PYTHON_EXE = "/usr/bin/python3"
_STARTUP_TIMEOUT_SECONDS = 15


def _stop_request_path(task_id: str) -> Path:
    return _BUILD_RUNTIME_DIR / f"{task_id}.stop"


def _write_runtime(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _persistent_build_state(prefix: str) -> dict:
    paths = sorted(_BUILD_RUNTIME_DIR.glob(f"{prefix}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        return {"running": False, "status": "idle", "log": [], "result": None}
    path = paths[0]
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"running": False, "status": "failed", "progress": "runtime 文件不可读", "log": [], "result": None}
    # 与 driver 自己的状态同步：脚本最后会把 status 写成 done/failed 并落 result。
    status = state.get("status")
    log_path = Path(state.get("log_path", ""))
    last_line = ""
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            state["log"] = lines[-50:]
            for line in reversed(lines):
                if line.strip():
                    last_line = line.strip()[:400]
                    break
        except OSError:
            state["log"] = []
    else:
        state["log"] = []
    # 仅在 PID 真已退出、状态仍是 running 的情况下判 stale。
    if status == "running" and not build_pid_alive(state.get("pid")):
        state.update(running=False, status="failed", progress="失败：构建进程已退出", returncode=None)
        _write_runtime(path, state)
    elif status == "running":
        # driver 写出的最后一行的进度信息
        state["progress"] = state.get("progress") or last_line or "构建中..."
    elif status in ("done", "completed", "finished", "ok", "success"):
        state["progress"] = last_line or state.get("progress") or "完成"
        state["running"] = False
    else:
        state["running"] = False
    return state


def _watch_build_start(task_id: str, started_at: float) -> None:
    time.sleep(_STARTUP_TIMEOUT_SECONDS)
    path = _BUILD_RUNTIME_DIR / f"{task_id}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    if state.get("status") not in {"starting", "running"}:
        return
    pid = state.get("pid")
    if isinstance(pid, int) and build_pid_alive(pid):
        return
    state.update({"status": "failed", "running": False,
                  "progress": f"后台构建进程在 {_STARTUP_TIMEOUT_SECONDS}s 内未进入活动状态",
                  "returncode": None})
    _write_runtime(path, state)
    try:
        from core.logging import log
        log(f"❌ 面板构建启动失败: {task_id}", source="panel_build", task_id=task_id, event="failed")
    except Exception:
        pass


def _start_persistent_build(prefix: str, mode: str, config: dict) -> dict:
    """启动面板构建 — subprocess.run 显式启动，并带启动超时 watchdog。"""
    current = _persistent_build_state(prefix)
    if current.get("running"):
        raise HTTPException(400, "已有构建任务运行中")
    task_id = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    runtime_path = _BUILD_RUNTIME_DIR / f"{task_id}.json"
    log_path = _BUILD_LOG_DIR / f"{task_id}.log"

    pool = config["pool"]
    start_date = config.get("start_date", "20040102").replace("-", "")
    end_date = config.get("end_date", "").replace("-", "")
    include_fund = "1" if config.get("include_fundamental", True) else "0"

    state = {
        "task_id": task_id,
        "status": "starting",
        "running": True,
        "mode": mode,
        "pool": pool,
        "progress": "任务已提交",
        "result": None,
        "pid": None,
        "log_path": str(log_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "started_at": time.time(),
    }
    _write_runtime(runtime_path, state)

    driver = str(AURUMQ_ROOT / "scripts" / "panel_build_driver.py")
    cmd = [_PYTHON_EXE, driver, mode, pool, start_date, end_date, include_fund, task_id]
    try:
        with open(log_path, "a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, timeout=10, check=False)
    except Exception as exc:
        state.update({"status": "failed", "running": False,
                      "progress": f"后台启动异常: {exc}"})
        _write_runtime(runtime_path, state)
        raise HTTPException(500, f"后台构建启动失败: {exc}") from exc
    if result.returncode != 0:
        state.update({"status": "failed", "running": False,
                      "progress": f"启动返回码 {result.returncode}",
                      "returncode": result.returncode})
        _write_runtime(runtime_path, state)
        raise HTTPException(500, f"后台构建启动失败，返回码 {result.returncode}")
    # driver 会在启动子进程后立即退出并写入 stdout；成功启动不等于构建完成。
    # 这里仅把启动响应写入日志，最终状态由持久化 runtime + 进程/产物检查决定。
    try:
        raw_response = result.stdout or b""
        response = raw_response.decode(errors="replace").strip() if isinstance(raw_response, bytes) else str(raw_response).strip()
        if response:
            with open(log_path, "a", encoding="utf-8") as log_handle:
                log_handle.write(response + "\n")
            match = re.search(r'"pid"\s*:\s*(\d+)', response)
            if match:
                state["pid"] = int(match.group(1))
                _write_runtime(runtime_path, state)
    except OSError:
        pass
    started_at = state["started_at"]
    import threading as _th
    _th.Thread(target=_watch_build_start, args=(task_id, started_at), daemon=True).start()
    return {"status": "started", "task_id": task_id, "pool": pool}


def _stop_build(task_id: str = "") -> dict:
    """统一停止：按 task_id 精确停止；不传则扫描所有运行中的构建任务。

    同时通过 stop request 文件和 SIGTERM 兜底，确保 driver 在收到 stop request
    后能优雅退出。
    """
    stopped: list[str] = []
    runtime_paths: list[Path] = []
    if task_id:
        path = _BUILD_RUNTIME_DIR / f"{task_id}.json"
        if path.exists():
            runtime_paths.append(path)
    else:
        runtime_paths = sorted(_BUILD_RUNTIME_DIR.glob("*_build_*.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
    for path in runtime_paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("status") not in {"starting", "running"}:
            continue
        task_id = state.get("task_id", path.stem)
        try:
            _stop_request_path(task_id).write_text("stop\n")
        except Exception:
            pass
        pid = state.get("pid")
        if isinstance(pid, int):
            try:
                import os as _os
                _os.killpg(pid, 15)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        stopped.append(task_id)
    if not stopped and not task_id:
        for prefix in ("v1_build", "v3_panel_build", "v8_panel_build",
                       "v9_panel_build", "v10_panel_build", "v10_1_panel_build", "v2_build"):
            state = _persistent_build_state(prefix)
            pid = state.get("pid")
            if state.get("running") and isinstance(pid, int):
                try:
                    import os as _os
                    _os.killpg(pid, 15)
                    stopped.append(prefix)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    return {"status": "stopping", "stopped": stopped}


def validate_pool(pool: str) -> None:
    if pool not in ("hs300", "csi500", "cs800"):
        raise HTTPException(400, f"不支持的股票池: {pool}")


def validate_date_range(start_date: str, end_date: str) -> tuple[date, date]:
    try:
        s = date.fromisoformat(start_date.replace("/", "-"))
        e = date.fromisoformat(end_date.replace("/", "-"))
    except ValueError as exc:
        raise HTTPException(400, f"日期格式非法: {exc}") from exc
    if s >= e:
        raise HTTPException(400, "面板开始日期必须早于结束日期")
    return s, e


def merge_cs800_v3_panels(start_date: str, end_date: str) -> Path:
    """低内存合并 HS300/CSI500 v3 面板为 CS800。"""
    import polars as pl

    hs_path = AURUMQ_ROOT / "data" / f"wavehunter_hs300_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
    cs_path = AURUMQ_ROOT / "data" / f"wavehunter_csi500_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
    out_path = AURUMQ_ROOT / "data" / f"wavehunter_cs800_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
    if not hs_path.exists() or not cs_path.exists():
        raise FileNotFoundError("CS800 合并需要先完成 HS300 和 CSI500 v3 面板")
    hs_codes = pl.scan_parquet(str(hs_path)).select("ts_code").unique().collect()["ts_code"].to_list()
    hs_lazy = pl.scan_parquet(str(hs_path))
    cs_lazy = pl.scan_parquet(str(cs_path)).filter(~pl.col("ts_code").is_in(hs_codes))
    pl.concat([hs_lazy, cs_lazy], how="vertical_relaxed").sink_parquet(str(out_path), compression="zstd")
    return out_path
