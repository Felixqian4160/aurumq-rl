"""
routers/common.py — 共享工具和全局状态（兼容层）

路径与状态读写已迁移至 core/config.py 与 core/state.py；
日志已迁移至 core/logging.py，此文件仅做 re-export 保持旧 import 路径可用。
"""
import os, sys, time, json, queue, threading, asyncio
from pathlib import Path
from typing import Optional

from core.config import (
    AURUMQ_ROOT,
    PYTHON_EXE,
    STATIC_DIR,
    BUILD_RUNTIME_DIR,
    BUILD_LOG_DIR,
    DATA_DIR,
    PROJECT_DIR,
    TRAIN_RUNTIME_DIR,
)

# ── 全局状态由 core/logging.py 统一管理，此处 re-export 保持兼容 ──
from core.logging import (
    _log_queue,
    _subscribers,
    _emit_event,
    _recent_events,
    init_event_logger,
    log,
    log_callback,
)

# ── 进程管理状态（仍保留此处，供各 router 共享停止信号） ──
_build_process: Optional[threading.Thread] = None
_build_stop_flag = threading.Event()
_train_process: Optional[threading.Thread] = None
_train_stop_flag = threading.Event()
_sim_process: Optional[threading.Thread] = None
_sim_stop_flag = threading.Event()
_p22c_train_process: Optional[threading.Thread] = None
_p22c_train_stop_flag = threading.Event()
_p22c_eval_process: Optional[threading.Thread] = None
_p22c_eval_stop_flag = threading.Event()
_wf_stop_flag = threading.Event()
_mtx_stop_flag = threading.Event()

# ── 状态文件读写（委托 core/state.py，保留兼容） ──
from core.state import (
    read_build_state as _read_build_state_core,
    write_build_state as _write_build_state_core,
    read_train_state as _read_train_state_core,
    list_parquet_files as _list_parquet_files_core,
    list_model_dirs as _list_model_dirs_core,
)


def read_build_state(prefix: str) -> dict:
    return _read_build_state_core(prefix)


def write_build_state(task_id: str, state: dict):
    return _write_build_state_core(task_id, state)


def read_train_state() -> dict:
    return _read_train_state_core()


# ── 进程存活检测（委托 core/process.py，保留兼容） ──
from core.process import pid_alive as _pid_alive_core, build_pid_alive as _build_pid_alive_core


def pid_alive(pid: int) -> bool:
    return _pid_alive_core(pid)


def build_pid_alive(pid: int) -> bool:
    return _build_pid_alive_core(pid)


# ── 文件扫描（委托 core/state.py，保留兼容） ──
def list_parquet_files(directory: Path, pattern: str = '*.parquet') -> list:
    return _list_parquet_files_core(directory, pattern)


def list_model_dirs(directory: Path) -> list:
    return _list_model_dirs_core(directory)
