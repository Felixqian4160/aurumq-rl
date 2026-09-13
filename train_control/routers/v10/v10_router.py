"""v10 模拟路由 - 注册到 app 上 (跟随 frozen v9 模式)"""
from __future__ import annotations
import os, json, time, threading, subprocess, shlex
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel

from app import AURUMQ_ROOT, app, log

_V10_THREAD: Optional[threading.Thread] = None
_V10_PROC = None
_V10_STOP = threading.Event()
_V10_STATE = AURUMQ_ROOT / 'data' / 'sim_v10_runtime.json'
_V10_LOG_DIR = AURUMQ_ROOT / 'data' / 'task_runtime'
_V10_LOG_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_EXE = '/usr/bin/python3'


class V10SimRequest(BaseModel):
    model_config = {'protected_namespaces': ()}
    model_dir: str = ''
    panel: str = ''
    initial_capital: float = 100000.0
    start_date: str = '2024-01-01'
    end_date: str = ''
    top_k: int = 20
    cost_bps: float = -1.0
    slippage_bps: float = 10.0
    rebalance_days: int = -1
    stop_loss_pct: float = -1.0
    max_position_pct: float = -1.0
    max_holding_days: int = -1
    take_profit_pct: float = 0.0
    hedge_ratio: float = 0.0
    signal_threshold: float = 0.0
    # v10 四段阈值
    a1_threshold: float = 0.55
    a2_threshold: float = 0.55
    peak_threshold: float = 0.50
    b1_threshold: float = 0.55
    min_buy_score: float = 0.4


@app.post("/api/sim/v10/start")
def v10_sim_start(req: V10SimRequest):
    global _V10_THREAD, _V10_PROC
    if _V10_THREAD and _V10_THREAD.is_alive():
        raise HTTPException(400, '已有 v10 模拟运行中')
    if not req.model_dir:
        raise HTTPException(400, '请选择模型')
    model_dir = AURUMQ_ROOT / 'models' / req.model_dir
    if not model_dir.exists():
        raise HTTPException(400, f'模型不存在: {req.model_dir}')
    if not (model_dir / 'policy.onnx').exists():
        raise HTTPException(400, f'模型无 ONNX: {req.model_dir}')

    if req.panel:
        panel_path = AURUMQ_ROOT / 'data' / req.panel
    else:
        for f in (AURUMQ_ROOT / 'data').glob('wavehunter_v9_*.parquet'):
            panel_path = f; break
    if not panel_path.exists():
        raise HTTPException(400, 'panel 不存在')

    end_date = req.end_date or '2026-08-04'
    task_id = f'sim_v10_{req.model_dir}_{int(time.time())}'
    runtime_path = _V10_LOG_DIR / f'{task_id}.json'
    log_path = _V10_LOG_DIR / f'{task_id}.log'

    cfg_payload = {
        'version': 'v10',
        'model_dir': str(model_dir),
        'panel_path': str(panel_path),
        'initial_capital': req.initial_capital,
        'start_date': req.start_date,
        'end_date': end_date,
        'top_k': req.top_k,
        'cost_bps': req.cost_bps,
        'slippage_bps': req.slippage_bps,
        'rebalance_days': req.rebalance_days,
        'stop_loss_pct': req.stop_loss_pct,
        'max_position_pct': req.max_position_pct,
        'max_holding_days': req.max_holding_days,
        'take_profit_pct': req.take_profit_pct,
        'hedge_ratio': req.hedge_ratio,
        'signal_threshold': req.signal_threshold,
        'v10_thresholds': {
            'a1': req.a1_threshold, 'a2': req.a2_threshold,
            'peak': req.peak_threshold, 'b1': req.b1_threshold,
            'min_buy_score': req.min_buy_score,
        },
    }
    cfg_path = _V10_LOG_DIR / f'{task_id}.config.json'
    cfg_path.write_text(json.dumps(cfg_payload, ensure_ascii=False, indent=2))

    state = {
        'task_id': task_id, 'version': 'v10', 'model': req.model_dir,
        'status': 'starting', 'pid': None, 'total': 1, 'current': 0, 'percent': 0.0,
        'log_path': str(log_path), 'runtime_path': str(runtime_path),
        'started_at': time.time(),
    }
    runtime_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))

    runner = AURUMQ_ROOT / 'scripts' / 'v10' / 'run_sim_v10.py'
    if not runner.exists():
        raise HTTPException(500, 'run_sim_v10.py 未就绪')

    cmd = [PYTHON_EXE, str(runner), str(cfg_path), str(runtime_path)]

    def run():
        global _V10_PROC
        try:
            state['status'] = 'running'
            runtime_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log(f'🚀 v10 模拟启动: {req.model_dir}', source='simulation_v10', task_id=task_id, event='start')
            with open(log_path, 'a') as flog:
                _V10_PROC = subprocess.Popen(cmd, cwd=str(AURUMQ_ROOT),
                                              stdout=flog, stderr=subprocess.STDOUT,
                                              start_new_session=True,
                                              env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            _V10_PROC.wait()
            time.sleep(0.3)
            try:
                state2 = json.loads(runtime_path.read_text())
                state['status'] = state2.get('status', 'finished')
                state['finished_at'] = time.time()
                runtime_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            except Exception:
                pass
            log(f'✅ v10 模拟结束: {req.model_dir}', source='simulation_v10', task_id=task_id, event='done')
        except Exception as e:
            state['status'] = 'failed'
            state['error'] = str(e)
            runtime_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log(f'❌ v10 异常: {e}', source='simulation_v10', task_id=task_id, event='error', level='ERROR')
        finally:
            _V10_PROC = None

    _V10_THREAD = threading.Thread(target=run, daemon=True)
    _V10_THREAD.start()
    return {'status': 'started', 'version': 'v10', 'model': req.model_dir,
            'task_id': task_id, 'cmd': shlex.join(cmd)}


@app.post("/api/sim/v10/stop")
def v10_sim_stop():
    _V10_STOP.set()
    if _V10_PROC and _V10_PROC.poll() is None:
        try:
            os.killpg(_V10_PROC.pid, 15)
        except Exception:
            pass
    return {'status': 'stopping', 'version': 'v10'}


@app.get("/api/sim/v10/status")
def v10_sim_status():
    if not _V10_STATE.exists():
        return {'running': False, 'status': 'idle', 'version': 'v10'}
    try:
        st = json.loads(_V10_STATE.read_text())
        st['running'] = bool(_V10_THREAD and _V10_THREAD.is_alive())
        st['version'] = 'v10'
        return st
    except Exception:
        return {'running': False, 'status': 'idle', 'version': 'v10'}


@app.get("/api/sim/v10/result")
def v10_sim_result():
    """返回最新 ledger (按 mtime)"""
    ledgers_dir = AURUMQ_ROOT / 'data' / 'ledgers'
    if not ledgers_dir.exists():
        return {'version': 'v10', 'status': 'no_result'}
    candidates = sorted(ledgers_dir.glob('*v10*.json'),
                         key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        # 找最新 1h 内的任意 ledger
        for f in sorted(ledgers_dir.glob('*.json'),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if int(time.time() - f.stat().st_mtime) < 3600:
                candidates = [f]; break
    if not candidates:
        return {'version': 'v10', 'status': 'no_result'}
    latest = candidates[0]
    try:
        return {'version': 'v10', 'ledger_file': str(latest),
                'data': json.loads(latest.read_text())}
    except Exception as e:
        return {'version': 'v10', 'status': 'parse_err', 'error': str(e)}
