"""
train_runner.py — 训练包装器（子进程模式）
==========================================
通过 subprocess 运行 train.py，逐行捕获 stdout 并推送到 log callback。
避免 import 导致的模块副作用和递归问题。
"""
import os
import sys
import time
import json
import subprocess
import threading
from pathlib import Path
from typing import Optional

_log_callbacks = []

def add_log_callback(fn):
    if fn not in _log_callbacks:
        _log_callbacks.append(fn)

def log(msg: str):
    for cb in _log_callbacks:
        try:
            cb(msg)
        except Exception:
            pass


AURUMQ_ROOT = Path(__file__).resolve().parent.parent


def list_available_panels() -> list:
    """列出 data/ 目录下所有因子面板"""
    data_dir = AURUMQ_ROOT / 'data'
    files = []
    for f in sorted(data_dir.glob('factor_panel_*.parquet')):
        stat = f.stat()
        files.append({
            'name': f.name,
            'path': str(f),
            'size_gb': round(stat.st_size / 1024 ** 3, 3),
        })
    return files


def run_training(
    panel_path: str,
    algorithm: str = 'PPO',
    total_timesteps: int = 100000,
    start_date: str = '2023-01-01',
    end_date: str = '2025-06-30',
    universe_filter: str = 'main_board_non_st',
    n_envs: int = 4,
    n_factors: Optional[int] = None,
    top_k: int = 30,
    forward_period: int = 10,
    cost_bps: float = 30.0,
    learning_rate: float = 0.0003,
    target_kl: Optional[float] = 0.05,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    seed: int = 42,
    out_dir: str = 'models/ppo_v1',
    stop_flag: Optional[threading.Event] = None,
    policy_kwargs_json: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> dict:
    """通过子进程运行 train.py，实时捕获 stdout 推送日志"""

    script = AURUMQ_ROOT / 'scripts' / 'train.py'
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    runtime_path = out_path / 'runtime.json'
    persistent_log_path = out_path / 'training.log'
    runtime = {'status': 'starting', 'pid': None, 'current': 0, 'total': total_timesteps,
               'percent': 0.0, 'out_dir': out_dir, 'log_path': str(persistent_log_path),
               'started_at': time.time()}
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2))

    log(f"{'='*50}")
    log(f"🚀 AurumQ-RL 训练启动 (子进程)")
    log(f"{'='*50}")
    log(f"📁 面板: {panel_path}")
    log(f"⚙  算法: {algorithm} | 步数: {total_timesteps:,} | 环境: {n_envs}")
    log(f"📅 {start_date} ~ {end_date} | 股票池: {universe_filter}")
    log(f"🔢 因子: {n_factors if n_factors is not None else '全部(自动)'} | Top-K: {top_k} | 收益窗口: {forward_period}d")
    if policy_kwargs_json:
        log(f"🧠 策略网络: {policy_kwargs_json}")
    log(f"💵 成本: {cost_bps}bps | 学习率: {learning_rate} | 种子: {seed}")
    log(f"📂 输出: {out_dir}")
    log(f"{'='*50}")

    if stop_flag and stop_flag.is_set():
        return {'status': 'cancelled_before_start'}

    # 构建命令行
    cmd = [
        "/usr/bin/python3", str(script),
        '--algorithm', algorithm,
        '--total-timesteps', str(total_timesteps),
        '--data-path', panel_path,
        '--start-date', start_date,
        '--end-date', end_date,
        '--universe-filter', universe_filter,
        '--n-envs', str(n_envs),
        '--top-k', str(top_k),
        '--forward-period', str(forward_period),
        '--cost-bps', str(cost_bps),
        '--learning-rate', str(learning_rate),
        '--seed', str(seed),
        '--out-dir', out_dir,
    ]
    if n_factors is not None and n_factors > 0:
        cmd += ['--n-factors', str(n_factors)]
    if target_kl is not None:
        cmd += ['--target-kl', str(target_kl)]
    if n_steps is not None:
        cmd += ['--n-steps', str(n_steps)]
    if batch_size is not None:
        cmd += ['--batch-size', str(batch_size)]
    if policy_kwargs_json:
        cmd += ['--policy-kwargs-json', policy_kwargs_json]
    if resume_from:
        cmd += ['--resume-from', resume_from]
        log(f'🔄 续训模式: 从 {resume_from} 加载模型')

    log(f"$ {' '.join(cmd)}")

    try:
        # 启动子进程
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(AURUMQ_ROOT),
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
            start_new_session=True,
        )
        runtime.update({'status': 'running', 'pid': proc.pid})
        runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2))

        # stdout同时写入持久化日志；WebUI掉线不影响训练
        with persistent_log_path.open('a', encoding='utf-8', buffering=1) as saved_log:
            start_ts = time.time()
            for line in (proc.stdout or []):
                saved_log.write(line)
                line = line.rstrip('\n\r')
                if line:
                    log(line)
                    import re
                    m = re.search(r'(\d[\d,]*)\s*(?:steps?|步)', line)
                    if m:
                        current = int(m.group(1).replace(',', ''))
                        runtime.update({'current': current, 'percent': round(current / total_timesteps * 100, 2), 'speed_steps_per_sec': round(current / max(time.time() - start_ts, 1), 3)})
                        runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2))

        proc.wait()
        elapsed = time.time() - start_ts
        return_code = proc.returncode

        log(f"\n{'='*50}")
        if return_code == 0 and not (stop_flag and stop_flag.is_set()):
            runtime.update({'status': 'finished', 'current': total_timesteps, 'percent': 100.0, 'returncode': return_code, 'finished_at': time.time()})
            log(f"✅ 训练完成! 耗时 {elapsed:.0f}s")
            status = 'ok'
        elif stop_flag and stop_flag.is_set():
            log(f"⏹ 训练被取消")
            status = 'cancelled'
        else:
            log(f"❌ 训练失败 (exit={return_code})")
            status = 'failed'
        log(f"{'='*50}")

        runtime.update({'status': status, 'returncode': return_code, 'finished_at': time.time()})
        runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2))

    except FileNotFoundError as e:
        log(f"❌ 找不到 train.py: {e}")
        status = 'failed'
        return_code = -1
        elapsed = 0
    except Exception as e:
        log(f"❌ 子进程异常: {e}")
        import traceback
        log(traceback.format_exc()[-300:])
        status = 'failed'
        return_code = -1
        elapsed = 0

    return {
        'status': status,
        'algorithm': algorithm,
        'total_timesteps': total_timesteps,
        'out_dir': out_dir,
        'exit_code': return_code,
        'elapsed_seconds': round(elapsed),
    }
