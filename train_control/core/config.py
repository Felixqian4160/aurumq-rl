"""
train_control/core/config.py — 路径与常量单一真源

所有路径常量集中于此，避免在 routers/* 中重复定义导致不一致。
"""
import os
from pathlib import Path

AURUMQ_ROOT = Path(os.environ.get("AURUMQ_RL_ROOT", Path(__file__).resolve().parents[2]))
PYTHON_EXE = os.environ.get("AURUMQ_PYTHON", "/usr/bin/python3")
STATIC_DIR = Path(__file__).parent.parent / "static"

# 运行时状态目录（文件状态契约：task_runtime/*.json + training_logs/*.log）
BUILD_RUNTIME_DIR = AURUMQ_ROOT / "data" / "task_runtime"
BUILD_LOG_DIR = AURUMQ_ROOT / "data" / "training_logs"
DATA_DIR = AURUMQ_ROOT / "data"
PROJECT_DIR = AURUMQ_ROOT  # alias for backward compatibility
TRAIN_RUNTIME_DIR = BUILD_RUNTIME_DIR  # 训练与构建共用同一目录

BUILD_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
BUILD_LOG_DIR.mkdir(parents=True, exist_ok=True)
