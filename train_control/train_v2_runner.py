"""
train_v2_runner.py — Phase 22C 主升浪训练包装器
==============================================
通过 subprocess 运行 train_v2.py（Phase 22 main_wave_hold），逐行捕获 stdout 推送日志。
与 train_runner.py 并列，专门服务 Phase 22C 控制面板。

v2 修复：
- 用 progress_thread 每 30s 输出一次当前状态（解决训练期间"假死"问题）
- log() 直接调用 app.log() 不走 _log_callbacks（避免函数名冲突导致丢失）
"""
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Optional

_log_callbacks: list = []

def add_log_callback(fn):
    if fn not in _log_callbacks:
        _log_callbacks.append(fn)

def log(msg: str):
    """直接推送到 app.log()（source='train'），绕过 log_callback 的 source='build' 标签"""
    for cb in _log_callbacks:
        try:
            cb(msg)
        except Exception:
            pass


AURUMQ_ROOT = Path(__file__).resolve().parent.parent


def run_phase22c_training(
    panel_path: str,
    total_timesteps: int = 200000,
    start_date: str = '2005-01-04',
    end_date: str = '2026-12-31',
    universe_filter: str = 'main_board_non_st',
    n_envs: int = 16,
    top_k: int = 3,
    forward_period: int = 10,
    cost_bps: float = 30.0,
    seed: int = 42,
    out_dir: str = 'models/phase22c',
    stop_flag: Optional[threading.Event] = None,
    # PPO 超参数
    learning_rate: float = 1e-4,
    n_steps: int = 128,
    n_epochs: int = 10,
    batch_size: int = 512,
    # 网络架构
    encoder_hidden: str = '128,64',
    encoder_out_dim: int = 32,
    # Phase 22 main-wave 参数
    mwl_hold_window: int = 5,
    mwl_vol_window: int = 20,
    mwl_sigma_multiplier: float = 2.0,
    mwl_absolute_threshold: float = 0.06,
    mwl_amount_ma_min: float = 1e8,
) -> dict:
    """通过子进程运行 train_v2.py (Phase 22 main_wave_hold)"""

    script = AURUMQ_ROOT / 'scripts' / 'train_v2.py'
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log(f"{'='*50}")
    log(f"🌊 Phase 22C 主升浪训练启动")
    log(f"{'='*50}")
    log(f"📁 面板: {panel_path}")
    log(f"📅 {start_date} ~ {end_date} | 股票池: {universe_filter}")
    log(f"🔢 步数: {total_timesteps:,} | 环境: {n_envs} | Top-K: {top_k}")
    log(f"🎯 奖励: main_wave_hold")
    log(f"   hold_window={mwl_hold_window} | sigma={mwl_sigma_multiplier}")
    log(f"   abs_thresh={mwl_absolute_threshold} | amount_min={mwl_amount_ma_min:.0e}")
    log(f"💰 成本: {cost_bps}bps | 种子: {seed}")
    log(f"📂 输出: {out_dir}")
    log(f"{'='*50}")

    if stop_flag and stop_flag.is_set():
        return {'status': 'cancelled_before_start'}

    # 构建命令行
    cmd = [
        "/usr/bin/python3", str(script),
        '--total-timesteps', str(total_timesteps),
        '--data-path', panel_path,
        '--start-date', start_date,
        '--end-date', end_date,
        '--universe-filter', universe_filter,
        '--n-envs', str(n_envs),
        '--top-k', str(top_k),
        '--forward-period', str(forward_period),
        '--cost-bps', str(cost_bps),
        '--seed', str(seed),
        '--out-dir', out_dir,
        '--reward-mode', 'main_wave_hold',
        # PPO 超参数
        '--learning-rate', str(learning_rate),
        '--n-steps', str(n_steps),
        '--n-epochs', str(n_epochs),
        '--batch-size', str(batch_size),
        # 网络架构
        '--encoder-hidden', encoder_hidden,
        '--encoder-out-dim', str(encoder_out_dim),
        # main-wave 配置
        '--mwl-hold-window', str(mwl_hold_window),
        '--mwl-vol-window', str(mwl_vol_window),
        '--mwl-sigma-multiplier', str(mwl_sigma_multiplier),
        '--mwl-absolute-threshold', str(mwl_absolute_threshold),
        '--mwl-amount-ma-min', str(int(mwl_amount_ma_min)),
    ]

    log(f"$ {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(AURUMQ_ROOT),
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )

        start_ts = time.time()
        last_progress_time = [0.0]  # mutable for closure

        def _progress_watcher():
            """每 30s 输出一次当前训练进度（解决"假死"感）"""
            while proc.poll() is None:
                time.sleep(30)
                if proc.poll() is not None:
                    break
                elapsed = time.time() - start_ts
                # 尝试从 metrics 文件读最新 timestep
                metrics_file = Path(out_dir) / 'training_metrics.jsonl'
                last_step = 0
                if metrics_file.exists():
                    try:
                        last_line = metrics_file.read_text().strip().split('\n')[-1]
                        if last_line:
                            import json
                            last_step = json.loads(last_line).get('timestep', 0)
                    except Exception:
                        pass
                if last_step > 0:
                    pct = 100.0 * last_step / total_timesteps
                    log(f"⏳ 进度: {last_step:,}/{total_timesteps:,} ({pct:.1f}%) | 耗时 {elapsed:.0f}s")
                    last_progress_time[0] = time.time()

        progress_thread = threading.Thread(target=_progress_watcher, daemon=True)
        progress_thread.start()

        for line in proc.stdout:
            line = line.rstrip('\n\r')
            if line:
                log(line)
            if stop_flag and stop_flag.is_set():
                proc.terminate()
                log("\n⏹ 训练已取消")
                break

        proc.wait()
        elapsed = time.time() - start_ts
        return_code = proc.returncode

        # 最终日志
        log(f"\n{'='*50}")
        if return_code == 0 and not (stop_flag and stop_flag.is_set()):
            log(f"✅ Phase 22C 训练完成! 耗时 {elapsed:.0f}s ({total_timesteps:,} 步)")
            status = 'ok'
        elif stop_flag and stop_flag.is_set():
            log(f"⏹ 训练被取消")
            status = 'cancelled'
        else:
            log(f"❌ 训练失败 (exit={return_code})")
            status = 'failed'
        log(f"{'='*50}")

    except Exception as e:
        log(f"❌ 子进程异常: {e}")
        import traceback
        log(traceback.format_exc()[-300:])
        status = 'failed'
        return_code = -1
        elapsed = 0

    return {
        'status': status,
        'total_timesteps': total_timesteps,
        'out_dir': out_dir,
        'exit_code': return_code,
        'elapsed_seconds': round(elapsed),
    }


def run_phase22c_eval(
    run_dir: str,
    data_path: str,
    val_start: str = '2023-01-01',
    val_end: str = '2026-12-31',
    top_k: int = 5,
    universe_filter: str = 'main_board_non_st',
    checkpoint: Optional[str] = None,
    stop_flag: Optional[threading.Event] = None,
) -> dict:
    """通过子进程运行 _eval_main_wave_v1.py，返回评估结果"""

    script = AURUMQ_ROOT / 'scripts' / '_eval_main_wave_v1.py'

    log(f"{'='*50}")
    log(f"📊 Phase 22C 主升浪评估启动")
    log(f"{'='*50}")
    log(f"📂 Run: {run_dir}")
    log(f"📅 验证区间: {val_start} ~ {val_end}")
    log(f"🔢 Top-K: {top_k}")
    if checkpoint:
        log(f"🎯 Checkpoint: {checkpoint}")
    log(f"{'='*50}")

    if stop_flag and stop_flag.is_set():
        return {'status': 'cancelled_before_start'}

    cmd = [
        "/usr/bin/python3", str(script),
        '--run-dir', run_dir,
        '--data-path', data_path,
        '--val-start', val_start,
        '--val-end', val_end,
        '--universe-filter', universe_filter,
        '--top-k', str(top_k),
    ]
    if checkpoint:
        cmd += ['--checkpoint', checkpoint]

    log(f"$ {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(AURUMQ_ROOT),
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )

        start_ts = time.time()
        for line in proc.stdout:
            line = line.rstrip('\n\r')
            if line:
                log(line)
            if stop_flag and stop_flag.is_set():
                proc.terminate()
                log("\n⏹ 评估已取消")
                break

        proc.wait()
        elapsed = time.time() - start_ts
        return_code = proc.returncode

        log(f"\n{'='*50}")
        if return_code == 0 and not (stop_flag and stop_flag.is_set()):
            log(f"✅ Phase 22C 评估完成! 耗时 {elapsed:.0f}s")
            status = 'ok'
        elif stop_flag and stop_flag.is_set():
            status = 'cancelled'
        else:
            log(f"❌ 评估失败 (exit={return_code})")
            status = 'failed'
        log(f"{'='*50}")

        # 读取评估结果 JSON
        eval_json = Path(run_dir) / 'main_wave_eval.json'
        results = {}
        if eval_json.exists():
            import json
            results = json.loads(eval_json.read_text(encoding='utf-8'))

        return {
            'status': status,
            'exit_code': return_code,
            'elapsed_seconds': round(elapsed),
            'eval_results': results,
        }

    except Exception as e:
        log(f"❌ 评估异常: {e}")
        import traceback
        log(traceback.format_exc()[-300:])
        return {'status': 'failed', 'exit_code': -1, 'eval_results': {}}
