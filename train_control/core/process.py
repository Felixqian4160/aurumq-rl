"""
train_control/core/process.py — 进程存活检测单一真源
"""
from __future__ import annotations

import os
from pathlib import Path


def pid_alive(pid: int) -> bool:
    """检测进程是否存活；僵尸进程不视为活动进程。"""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(state) > 2 and state[2] != "Z"
    except (OSError, ProcessLookupError, FileNotFoundError, PermissionError, ValueError):
        return False


def build_pid_alive(pid: int) -> bool:
    """检测构建进程是否存活（校验 cmdline）。"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode("utf-8", errors="replace")
        build_markers = (
            "panel_build_driver.py",
            "build_wavehunter_panel",
            "build_wavehunter_v9_panel",
            "build_wavehunter_v10_independent",
            "build_wavehunter_v10_1_independent",
            "build_factor_panel",
        )
        return any(marker in cmdline for marker in build_markers)
    except (FileNotFoundError, PermissionError):
        return False


def v2_build_pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        return len(state) > 2 and state[2] != "Z" and "v2_build_worker.py" in cmdline
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
        return False
