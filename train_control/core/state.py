"""
train_control/core/state.py — 状态文件与文件扫描单一真源

所有 task_runtime/*.json 的读写与扫描逻辑集中于此。
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import BUILD_RUNTIME_DIR, TRAIN_RUNTIME_DIR


def read_build_state(prefix: str) -> dict:
    """读取构建任务状态（取最新一个）。"""
    rt_dir = BUILD_RUNTIME_DIR
    if not rt_dir.exists():
        return {"status": "idle", "running": False}
    files = sorted(rt_dir.glob(f"{prefix}_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return {"status": "idle", "running": False}
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return {"status": "idle", "running": False}


def write_build_state(task_id: str, state: dict) -> None:
    """写入构建任务状态。"""
    state_file = BUILD_RUNTIME_DIR / f"{task_id}.json"
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def read_train_state() -> dict:
    """读取训练状态（取最新一个）。"""
    rt_dir = TRAIN_RUNTIME_DIR
    if not rt_dir.exists():
        return {"running": False}
    files = sorted(rt_dir.glob("train_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return {"running": False}
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return {"running": False}


def list_parquet_files(directory: Path, pattern: str = "*.parquet") -> list[Path]:
    """列出目录下的 parquet 文件（按 mtime 倒序）。"""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)


def list_model_dirs(directory: Path) -> list[Path]:
    """列出模型目录（按 mtime 倒序）。"""
    if not directory.exists():
        return []
    dirs: list[Path] = []
    for d in sorted(directory.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "policy.onnx").exists():
            dirs.append(d)
        elif d.is_file() and d.suffix == ".zip":
            dirs.append(d)
    return dirs
