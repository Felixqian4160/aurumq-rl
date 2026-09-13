"""WaveHunter 主升浪猎手 — WebUI API 模块 (独立, 不动 app.py 现有路由).

在 app.py 末尾 `import wavehunter_api` 即可注册:
  POST /api/wavehunter/train/start     启动训练 (后台线程)
  POST /api/wavehunter/train/stop      停止训练
  GET  /api/wavehunter/train/status    训练状态 + 实时 aux_stats
  GET  /api/wavehunter/panels          面板列表 (复用 resolve_panel)
  GET  /api/wavehunter/models          已训练模型列表 (runs/wh_*)
  DELETE /api/wavehunter/models/{name} 删除模型
  POST /api/wavehunter/configs/save    保存参数预设
  GET  /api/wavehunter/configs         预设列表

模型命名: runs/wh_{pool}_{MMDD}_{steps} (WaveHunter 命名铁律)
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 复用 app.py 的实例与日志 (同进程, 共享 _log_queue)
try:
    from app import AURUMQ_ROOT, log, app
except ImportError:
    # 直接运行时回退
    from core.config import AURUMQ_ROOT
    app = FastAPI(title='WaveHunter API')
    def log(msg: str, source: str = 'wavehunter'):
        print(f'[{source}] {msg}', flush=True)

RUNS_DIR = AURUMQ_ROOT / 'models'
RUNS_DIR.mkdir(exist_ok=True)
CONFIGS_FILE = AURUMQ_ROOT / 'data' / 'wavehunter_saved_configs.json'

# ── 全局状态 ──
_wh_train_process: Optional[threading.Thread] = None
_wh_train_stop_flag = threading.Event()
_wh_train_cmd: str = ''


def _load_configs() -> list:
    if CONFIGS_FILE.exists():
        try:
            return json.loads(CONFIGS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_configs(configs: list):
    CONFIGS_FILE.write_text(json.dumps(configs, indent=2, ensure_ascii=False))


class WaveHunterTrainRequest(BaseModel):
    panel: str = 'wavehunter_hs300_20040102_20260804.parquet'
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    total_timesteps: int = 200_000
    window: int = 20
    top_k: int = 20
    universe_filter: str = 'main_board_non_st'
    n_factors: int = 0
    forward_period: int = 20
    max_position_pct: float = 0.02
    rebalance_days: int = 20
    cost_bps: float = 15.0
    learning_rate: float = 1e-4
    lstm_hidden: int = 128
    lstm_layers: int = 1
    reward_type: str = 'return'
    n_envs: int = 1
    seed: int = 42
    max_grad_norm: float = 1.0
    target_kl: float = 0.02
    batch_size: int = 64
    n_steps: int = 64
    # WaveHunter 专属
    aux_lambda: float = 0.1
    a1_lambda: float = 1.0
    a1_pos_weight: float = 25.0
    a2_lambda: float = 1.0
    a2_pos_weight: float = 25.0
    peak_lambda: float = 1.0
    peak_pos_weight: float = 25.0
    b1_lambda: float = 1.0
    b1_pos_weight: float = 25.0
    # 标签参数
    label_window: int = 20
    label_entry_days: int = 1
    label_n_bins: int = 5
    label_a2_threshold: float = 0.10
    label_a2_window: int = 3
    label_min_amount_pct: float = 0.05
    # v4 reward 组件
    hit_rate_bonus_weight: float = 0.0
    continuation_bonus_weight: float = 0.0
    drawdown_penalty: float = 0.0
    excess_weight: float = 0.0
    reward_scale: float = 100.0
    # ── 市场中性对冲 ──
    hedge_ratio: float = 0.0
    # ── 信号过滤 ──
    signal_threshold: float = 0.0
    # ── 动态止损 ──
    dynamic_stop_loss: bool = False
    # ── 止盈 ──
    take_profit_pct: float = 0.0
    # ── v9 综合 Reward ──
    reward_w_abs: float = 0.60
    reward_w_dd: float = 0.25
    reward_w_hit: float = 0.15
    reward_norm_vol: float = 0.02
    # ── v9 ZigZag + KNN ──
    zigzag_threshold: float = 0.10
    zigzag_pullback: int = 5
    knn_enabled: bool = False
    knn_k: int = 20
    knn_metric: str = "cosine"
    label_version: str = "v9"
    out_dir: str = ''
    matrix_internal: bool = False

    model_config = {'protected_namespaces': ()}


def _build_cmd(req: WaveHunterTrainRequest, out_name: str) -> str:
    """组装 train_wavehunter.py CLI 命令."""
    try:
        start_dt = datetime.date.fromisoformat(req.start_date)
        end_dt = datetime.date.fromisoformat(req.end_date)
    except ValueError as e:
        raise HTTPException(400, f'日期格式非法: {e}') from e
    if start_dt >= end_dt:
        raise HTTPException(400, '训练开始日期必须早于结束日期')
    if end_dt > datetime.date.today():
        raise HTTPException(400, f'训练结束日期不能晚于今天: {datetime.date.today()}')
    if req.n_steps > 64 or req.batch_size > 64:
        raise HTTPException(400, 'n_steps 和 batch_size 必须 <= 64（RTX 3060 显存安全边界）')
    panel_path = AURUMQ_ROOT / 'data' / req.panel
    if not panel_path.exists():
        raise HTTPException(400, f'面板不存在: {req.panel}')
    out_dir = AURUMQ_ROOT / 'models' / out_name
    parts = [
        "/usr/bin/python3", str(AURUMQ_ROOT / 'scripts' / 'train_wavehunter.py'),
        f'--panel', str(panel_path),
        f'--start-date', req.start_date,
        f'--end-date', req.end_date,
        f'--out-dir', str(out_dir),
        f'--total-timesteps', str(req.total_timesteps),
        f'--window', str(req.window),
        f'--top-k', str(req.top_k),
        f'--universe-filter', req.universe_filter,
        f'--n-factors', str(req.n_factors),
        f'--forward-period', str(req.forward_period),
        f'--max-position-pct', str(req.max_position_pct),
        f'--rebalance-days', str(req.rebalance_days),
        f'--cost-bps', str(req.cost_bps),
        f'--learning-rate', str(req.learning_rate),
        f'--lstm-hidden', str(req.lstm_hidden),
        f'--lstm-layers', str(req.lstm_layers),
        f'--reward-type', req.reward_type,
        f'--n-envs', str(req.n_envs),
        f'--seed', str(req.seed),
        f'--max-grad-norm', str(req.max_grad_norm),
        f'--target-kl', str(req.target_kl),
        f'--batch-size', str(req.batch_size),
        f'--n-steps', str(req.n_steps),
        f'--aux-lambda', str(req.aux_lambda),
        f'--a1-lambda', str(req.a1_lambda),
        f'--a1-pos-weight', str(req.a1_pos_weight),
        f'--a2-lambda', str(req.a2_lambda),
        f'--a2-pos-weight', str(req.a2_pos_weight),
        f'--label-window', str(req.label_window),
        f'--label-entry-days', str(req.label_entry_days),
        f'--label-n-bins', str(req.label_n_bins),
        f'--label-a2-threshold', str(req.label_a2_threshold),
        f'--label-a2-window', str(req.label_a2_window),
        f'--label-min-amount-pct', str(req.label_min_amount_pct),
        # ── 市场中性对冲 ──
        f'--hedge-ratio', str(req.hedge_ratio),
        # ── 信号过滤 ──
        f'--signal-threshold', str(req.signal_threshold),
        # ── 动态止损 ──
        f'--dynamic-stop-loss' if req.dynamic_stop_loss else '',
        # ── 止盈 ──
        f'--take-profit-pct', str(req.take_profit_pct),
        # ── v9 综合 Reward ──
        f'--reward-w-abs', str(req.reward_w_abs),
        f'--reward-w-dd', str(req.reward_w_dd),
        f'--reward-w-hit', str(req.reward_w_hit),
        f'--reward-norm-vol', str(req.reward_norm_vol),
        # ── v9 ZigZag + KNN ──
        f'--zigzag-threshold', str(req.zigzag_threshold),
        f'--zigzag-pullback', str(req.zigzag_pullback),
        f'--knn-enabled' if req.knn_enabled else '',
        f'--knn-k', str(req.knn_k),
        f'--knn-metric', req.knn_metric,
        f'--label-version', req.label_version,
    ]
    # 移除空字符串
    parts = [p for p in parts if p]
    return ' '.join(parts)


@app.post('/api/wavehunter/train/start')
def wavehunter_train_start(req: WaveHunterTrainRequest):
    global _wh_train_process, _wh_train_cmd
    if _wh_train_process and _wh_train_process.is_alive():
        raise HTTPException(400, '已有 WaveHunter 训练运行中')

    # 命名: runs/wh_{pool}_{MMDD}_{steps}
    pool = Path(req.panel).stem.split('_')[1] if '_' in Path(req.panel).stem else 'hs300'
    out_name = req.out_dir or f'wh_{pool}_{time.strftime("%m%d")}_{req.total_timesteps//1000}k'
    cmd = _build_cmd(req, out_name)
    _wh_train_cmd = cmd
    _wh_train_stop_flag.clear()

    def run():
        try:
            log(f'🌊 WaveHunter 训练启动: {req.panel} | {req.total_timesteps:,} 步')
            log(f'🌊 CMD: {cmd[:120]}...')
            # 审计 P1-2 修复: shell=False + 参数列表, 使 proc 直接是 python 进程,
            # 这样 terminate()/kill() 真正终止 train_wavehunter.py (shell=True 时
            # proc 是 /bin/sh, terminate 只杀 shell, python 变孤儿继续跑)。
            import shlex
            cmd_args = shlex.split(cmd)
            proc = subprocess.Popen(
                cmd_args, cwd=str(AURUMQ_ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,  # 进程组隔离 (gateway 重启不连坐)
            )
            _wh_train_process._proc = proc
            try:
                stdout = proc.stdout
                if stdout is not None:
                    for line in iter(stdout.readline, ''):
                        line = line.rstrip()
                        if line:
                            log(line, source='wavehunter')
                        if _wh_train_stop_flag.is_set() and proc.poll() is None:
                            # 终止整个进程组 (含 python 及其子进程)
                            import signal
                            try:
                                os.killpg(proc.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                            proc.terminate()
            finally:
                # 停止后确保进程组被清理
                if _wh_train_stop_flag.is_set() and proc.poll() is None:
                    import signal
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.kill()
            proc.wait()
            log(f'🌊 WaveHunter 训练结束 (exit={proc.returncode})')
        except Exception as e:
            log(f'🌊 训练异常: {e}')

    t = threading.Thread(target=run, daemon=True)
    t._panel = req.panel
    t._out_name = out_name
    _wh_train_process = t
    t.start()
    return {'status': 'started', 'out_name': out_name, 'cmd': cmd[:200]}


@app.post('/api/wavehunter/train/stop')
def wavehunter_train_stop():
    _wh_train_stop_flag.set()
    return {'status': 'stopping'}


@app.get('/api/wavehunter/train/status')
def wavehunter_train_status():
    running = _wh_train_process is not None and _wh_train_process.is_alive()
    return {
        'running': running,
        'panel': getattr(_wh_train_process, '_panel', None) if _wh_train_process else None,
        'out_name': getattr(_wh_train_process, '_out_name', None) if _wh_train_process else None,
        'cmd': _wh_train_cmd,
        'stopping': _wh_train_stop_flag.is_set(),
    }


@app.get('/api/wavehunter/panels')
def wavehunter_panels():
    """列出 data/ 下所有因子面板 (*.parquet)."""
    data_dir = AURUMQ_ROOT / 'data'
    panels = []
    for f in sorted(data_dir.glob('*.parquet')):
        panels.append({'name': f.name, 'size_mb': round(f.stat().st_size / 1e6, 1)})
    return {'panels': panels}


@app.get('/api/wavehunter/models')
def wavehunter_models():
    """列出 models/ 下的 WaveHunter v1 产物。"""
    models = []
    for d in sorted(RUNS_DIR.glob('wh_*'), reverse=True):
        ppo = d / 'ppo_final.zip'
        summary = d / 'training_summary.json'
        info = {
            'name': d.name,
            'has_ppo': ppo.exists(),
            'has_summary': summary.exists(),
            'ppo_size_mb': round(ppo.stat().st_size / 1e6, 1) if ppo.exists() else 0,
        }
        if summary.exists():
            try:
                s = json.loads(summary.read_text())
                info['total_timesteps'] = s.get('total_timesteps')
                info['aux_stats'] = s.get('aux_stats', {})
                info['panel'] = s.get('panel')
            except Exception:
                pass
        models.append(info)
    return {'models': models}


@app.delete('/api/wavehunter/models/{name}')
def wavehunter_model_delete(name: str):
    target = RUNS_DIR / name
    if not target.exists():
        raise HTTPException(404, f'模型不存在: {name}')
    shutil.rmtree(target)
    return {'status': 'deleted'}


@app.post('/api/wavehunter/configs/save')
def wavehunter_config_save(req: dict):
    """保存参数预设. body: {name, params}"""
    name = req.get('name', '')
    params = req.get('params', {})
    if not name:
        raise HTTPException(400, '缺少预设名称')
    configs = _load_configs()
    # 同名覆盖
    configs = [c for c in configs if c.get('name') != name]
    configs.append({'name': name, 'params': params,
                    'created': time.strftime('%Y-%m-%d %H:%M')})
    _save_configs(configs)
    return {'status': 'saved', 'count': len(configs)}


@app.get('/api/wavehunter/configs')
def wavehunter_configs():
    return {'configs': _load_configs()}


@app.delete('/api/wavehunter/configs/{name}')
def wavehunter_config_delete(name: str):
    configs = [c for c in _load_configs() if c.get('name') != name]
    _save_configs(configs)
    return {'status': 'deleted'}


# ⚠️ 本模块必须在 app.py 末尾 import (from wavehunter_api import *) 才能注册路由。
if __name__ == '__main__':
    print('WaveHunter API 模块 — 由 app.py 导入注册, 勿直接运行')
