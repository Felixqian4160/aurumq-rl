"""
train_control/services/train_service.py — 训练服务单一真源

统一 train / p22c / joint 的子进程启动、运行时文件与日志透传；
routers/training.py 等仅做参数校验与委托。
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from core.config import AURUMQ_ROOT, PYTHON_EXE

# 由 routers 注入的停止信号与回调，service 不持有全局状态


def clamp_train_params(n_steps: Optional[int], batch_size: Optional[int], lstm_hidden: Optional[int]) -> tuple[int | None, int | None, int | None]:
    """按 RTX 3060 12GB 安全边界 clamp。"""
    if n_steps is not None and n_steps > 128:
        n_steps = 128
    if batch_size is not None and batch_size > 64:
        batch_size = 64
    if lstm_hidden is not None and lstm_hidden > 128:
        lstm_hidden = 128
    return n_steps, batch_size, lstm_hidden


def build_train_cmd(
    *,
    panel_path: str,
    out_dir: str,
    start_date: str,
    end_date: str,
    env_type: str = "portfolio_weight_lstm",
    total_timesteps: int = 100000,
    top_k: int = 20,
    forward_period: int = 10,
    window: int = 10,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    n_envs: int = 1,
    learning_rate: float = 3e-4,
    cost_bps: float = 15,
    max_position_pct: float = 0.05,
    reward_type: str = "return",
    seed: int = 42,
    max_grad_norm: float = 1.0,
    n_steps: int = 128,
    target_kl: Optional[float] = 0.02,
    batch_size: Optional[int] = None,
    n_epochs: Optional[int] = None,
    rebalance_days: int = 20,
    n_factors: Optional[int] = None,
    policy_kwargs_json: Optional[str] = None,
) -> list[str]:
    cmd = [
        PYTHON_EXE,
        str(AURUMQ_ROOT / "scripts" / "train.py"),
        "--data-path",
        panel_path,
        "--out-dir",
        out_dir,
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--env-type",
        env_type,
        "--total-timesteps",
        str(total_timesteps),
        "--top-k",
        str(top_k),
        "--forward-period",
        str(forward_period),
        "--window",
        str(window),
        "--lstm-hidden",
        str(lstm_hidden),
        "--lstm-layers",
        str(lstm_layers),
        "--n-envs",
        str(n_envs),
        "--learning-rate",
        str(learning_rate),
        "--cost-bps",
        str(cost_bps),
        "--max-position-pct",
        str(max_position_pct),
        "--reward-type",
        reward_type,
        "--seed",
        str(seed),
        "--max-grad-norm",
        str(max_grad_norm),
        "--n-steps",
        str(n_steps),
        "--rebalance-days",
        str(rebalance_days),
    ]
    if n_factors:
        cmd += ["--n-factors", str(n_factors)]
    if target_kl:
        cmd += ["--target-kl", str(target_kl)]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if n_epochs:
        cmd += ["--n-epochs", str(n_epochs)]
    if policy_kwargs_json:
        cmd += ["--policy-kwargs-json", policy_kwargs_json]
    return cmd
