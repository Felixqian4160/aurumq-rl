"""
routers/simulation.py — 模拟交易 API

统一进程模型：runtime JSON + start_new_session=True 立即脱离父进程；
启动检查、watchdog、按 task_id 停止、停止请求持久化。
"""
import datetime
import os, signal, sys, time, json, uuid, subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .common import AURUMQ_ROOT, PYTHON_EXE, log, list_model_dirs, list_parquet_files
from .ml_joint import resolve_panel
from core.process import pid_alive

router = APIRouter(tags=['simulation'])

# ════════════════════════════════════════════════════════════
# 模拟交易 API
# ════════════════════════════════════════════════════════════

_RUNTIME_DIR = AURUMQ_ROOT / 'data' / 'task_runtime'
_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
_SIM_CONFIGS_FILE = AURUMQ_ROOT / 'data' / 'sim_saved_configs.json'
_STARTUP_TIMEOUT_SECONDS = 15


def _stop_request_path(task_id: str) -> Path:
    return _RUNTIME_DIR / f'{task_id}.stop'


def _write_runtime(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _read_runtime(task_id: str) -> dict:
    path = _RUNTIME_DIR / f'{task_id}.json'
    if not path.exists():
        return {'status': 'not_found', 'running': False, 'task_id': task_id}
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'status': 'failed', 'running': False, 'task_id': task_id,
                'progress': 'runtime 文件不可读'}
    if state.get('status') == 'running' and not pid_alive(state.get('pid')):
        state.update(running=False, status='failed',
                      progress=f'模拟进程 PID {state.get("pid")} 已不存在',
                      error=state.get('progress') or 'PID 已退出')
        _write_runtime(path, state)
    state['running'] = state.get('status') in {'starting', 'running', 'stopping'}
    return state


def _current_running() -> Optional[dict]:
    candidates: list[tuple[float, Path]] = []
    for path in _RUNTIME_DIR.glob('sim_*.json'):
        try:
            st = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        if st.get('status') in {'starting', 'running', 'stopping'}:
            candidates.append((st.get('started_at', 0), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _, path = candidates[0]
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _watch_sim_start(task_id: str, started_at: float) -> None:
    time.sleep(_STARTUP_TIMEOUT_SECONDS)
    path = _RUNTIME_DIR / f'{task_id}.json'
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        state = {}
    if state.get('status') not in {'starting', 'running'}:
        return
    pid = state.get('pid')
    if isinstance(pid, int) and pid_alive(pid):
        return
    state.update({'status': 'failed', 'running': False,
                  'progress': f'模拟进程在 {_STARTUP_TIMEOUT_SECONDS}s 内未进入活动状态'})
    _write_runtime(path, state)
    try:
        from core.logging import log as _log
        _log(f'❌ 模拟启动失败: {task_id}', source='simulation', task_id=task_id, event='failed')
    except Exception:
        pass


def _load_sim_configs() -> list:
    if _SIM_CONFIGS_FILE.exists():
        try:
            return json.loads(_SIM_CONFIGS_FILE.read_text())
        except Exception:
            pass
    return []


def _save_sim_configs(configs: list):
    _SIM_CONFIGS_FILE.write_text(json.dumps(configs, indent=2, ensure_ascii=False))


class SimRequest(BaseModel):
    model_dir: str = ''
    panel: str = ''
    initial_capital: float = 100000.0
    start_date: str = '2024-01-01'
    end_date: str = ''
    top_k: int = -1
    cost_bps: float = -1.0
    slippage_bps: float = -1.0
    rebalance_days: int = -1
    stop_loss_pct: float = -1.0
    max_position_pct: float = -1.0
    max_holding_days: int = -1
    hedge_ratio: float = 0.0
    signal_threshold: float = 0.0
    dynamic_stop_loss: bool = False
    take_profit_pct: float = 0.0
    v9_enabled: bool = False
    v9_a1_threshold: float = 0.5
    v9_a2_threshold: float = 0.5
    v9_peak_threshold: float = 0.5
    v9_b1_threshold: float = 0.5

    model_config = {'protected_namespaces': ()}


@router.post('/api/sim/start')
def start_sim(req: SimRequest):
    current = _current_running()
    if current is not None:
        pid = current.get('pid')
        if isinstance(pid, int) and pid_alive(pid):
            raise HTTPException(400, f'已有模拟任务运行中 (PID {pid})')

    if not req.model_dir:
        raise HTTPException(400, '请选择模型')

    panel_name = resolve_panel(req.panel, 'hs300')
    if not panel_name:
        raise HTTPException(400, '未找到任何因子面板，请先在 P3 构建')
    req.panel = panel_name
    panel_path = str(AURUMQ_ROOT / 'data' / panel_name)
    if not Path(panel_path).exists():
        raise HTTPException(400, f'面板文件不存在: {panel_name}')

    model_path = AURUMQ_ROOT / 'models' / req.model_dir
    if ':' in req.model_dir:
        source, name = req.model_dir.split(':', 1)
        if source != 'models' or not name or '..' in name or '/' in name or '\\' in name:
            raise HTTPException(400, f'非法模型标识: {req.model_dir}')
        model_path = AURUMQ_ROOT / 'models' / name
    if not model_path.exists():
        raise HTTPException(400, f'模型目录不存在: {req.model_dir}')

    end_date = req.end_date or datetime.date.today().isoformat()
    ts = time.time()
    task_id = f'sim_{Path(req.model_dir).name}_{int(ts * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:6]}'
    runtime_path = _RUNTIME_DIR / f'{task_id}.json'
    log_path = _RUNTIME_DIR / f'{task_id}.log'
    cfg_path = _RUNTIME_DIR / f'{task_id}.config.json'

    cfg_payload = {
        'model_dir': str(model_path), 'panel_path': panel_path,
        'initial_capital': req.initial_capital, 'start_date': req.start_date,
        'end_date': end_date, 'top_k': req.top_k, 'cost_bps': req.cost_bps,
        'slippage_bps': req.slippage_bps, 'rebalance_days': req.rebalance_days,
        'stop_loss_pct': req.stop_loss_pct, 'max_position_pct': req.max_position_pct,
        'max_holding_days': req.max_holding_days,
        'hedge_ratio': req.hedge_ratio,
        'signal_threshold': req.signal_threshold,
        'dynamic_stop_loss': req.dynamic_stop_loss,
        'take_profit_pct': req.take_profit_pct,
        'v9_enabled': req.v9_enabled,
        'v9_a1_threshold': req.v9_a1_threshold,
        'v9_a2_threshold': req.v9_a2_threshold,
        'v9_peak_threshold': req.v9_peak_threshold,
        'v9_b1_threshold': req.v9_b1_threshold,
    }
    cfg_path.write_text(json.dumps(cfg_payload, ensure_ascii=False, indent=2))
    initial_state = {
        'task_id': task_id, 'status': 'starting', 'model': req.model_dir,
        'panel': panel_name, 'log_path': str(log_path),
        'runtime_path': str(runtime_path), 'config_path': str(cfg_path),
        'started_at': ts,
    }
    _write_runtime(runtime_path, initial_state)
    cmd = [PYTHON_EXE, str(AURUMQ_ROOT / 'scripts' / 'run_sim_task.py'),
           str(cfg_path), str(runtime_path)]
    try:
        with log_path.open('a') as out:
            proc = subprocess.Popen(
                cmd, cwd=str(AURUMQ_ROOT),
                stdout=out, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    except Exception as exc:
        initial_state.update({'status': 'failed', 'running': False,
                               'progress': f'启动失败: {exc}'})
        _write_runtime(runtime_path, initial_state)
        raise HTTPException(500, f'模拟启动失败: {exc}') from exc
    initial_state['pid'] = proc.pid
    initial_state['status'] = 'running'
    initial_state['running'] = True
    _write_runtime(runtime_path, initial_state)
    import threading as _th
    _th.Thread(target=_watch_sim_start, args=(task_id, ts), daemon=True).start()
    log(f'🚀 模拟交易启动: {req.model_dir} | ¥{req.initial_capital:,.0f}', source='simulation', task_id=task_id, event='start')
    return {'status': 'started', 'model': req.model_dir, 'task_id': task_id,
            'pid': proc.pid, 'log_path': str(log_path),
            'runtime_path': str(runtime_path)}


@router.post('/api/sim/stop')
def stop_sim_api(task_id: str = ''):
    if not task_id:
        current = _current_running()
        if current is None:
            return {'status': 'idle', 'message': '没有运行中的模拟任务'}
        task_id = current['task_id']
    state_path = _RUNTIME_DIR / f'{task_id}.json'
    if not state_path.exists():
        raise HTTPException(404, f'任务不存在: {task_id}')
    state = json.loads(state_path.read_text())
    if state.get('status') in {'done', 'failed', 'stopped', 'blocked', 'stale'}:
        return {'status': state['status'], 'task_id': task_id}
    _stop_request_path(task_id).write_text('stop\n')
    state.update({'status': 'stopping', 'stop_requested_at': time.time()})
    _write_runtime(state_path, state)
    pid = state.get('pid')
    if isinstance(pid, int) and pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return {'status': 'stopping', 'task_id': task_id}


@router.get('/api/sim/status')
def sim_status(task_id: str = ''):
    if task_id:
        state = _read_runtime(task_id)
        return {
            'running': state.get('running', False),
            'status': state.get('status', 'not_found'),
            'progress': state.get('progress', ''),
            'model': state.get('model'),
            'task_id': state.get('task_id'),
            'pid': state.get('pid'),
            'log_path': state.get('log_path'),
            'runtime_path': str(_RUNTIME_DIR / f"{task_id}.json"),
            'error': state.get('error'),
        }
    current = _current_running()
    if current is None:
        return {'running': False, 'status': 'idle'}
    state = _read_runtime(current['task_id'])
    return {
        'running': state.get('running', False),
        'status': state.get('status', 'idle'),
        'progress': state.get('progress', ''),
        'model': state.get('model'),
        'task_id': state.get('task_id'),
        'pid': state.get('pid'),
        'log_path': state.get('log_path'),
        'runtime_path': current.get('runtime_path'),
        'error': state.get('error'),
    }


@router.get('/api/sim/result')
def sim_result(task_id: str = ''):
    if not task_id:
        current = _current_running()
        if current is None:
            candidates = sorted(_RUNTIME_DIR.glob('sim_*.json'),
                                key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                return {'status': 'idle', 'message': '没有模拟记录'}
            task_id = candidates[0].stem
    state = _read_runtime(task_id)
    if state.get('result'):
        return state['result']
    return {
        'status': state.get('status', 'not_found'),
        'progress': state.get('progress', ''),
        'task_id': task_id,
        'model': state.get('model'),
        'error': state.get('error'),
    }


# ── 模拟参数保存/加载 ──
class SimConfigSave(BaseModel):
    name: str
    params: dict

    model_config = {'protected_namespaces': ()}


@router.get('/api/sim/configs')
def list_sim_configs():
    return {'configs': _load_sim_configs()}


@router.post('/api/sim/configs/save')
def save_sim_config(req: SimConfigSave):
    configs = _load_sim_configs()
    configs = [c for c in configs if c.get('name') != req.name]
    configs.append({'name': req.name, 'params': req.params,
                    'saved_at': time.strftime('%Y-%m-%d %H:%M')})
    _save_sim_configs(configs)
    return {'status': 'saved', 'name': req.name}


@router.delete('/api/sim/configs/{name}')
def delete_sim_config(name: str):
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(400, '非法配置名')
    configs = _load_sim_configs()
    configs = [c for c in configs if c.get('name') != name]
    _save_sim_configs(configs)
    return {'status': 'deleted', 'name': name}


# ── 账本管理 ──
@router.get('/api/sim/ledgers')
def list_ledgers():
    from aurumq_rl.simulation import list_ledgers as _list
    return {'ledgers': _list()}


@router.get('/api/sim/ledgers/{ledger_id}')
def get_ledger(ledger_id: str):
    if '..' in ledger_id or '/' in ledger_id or '\\' in ledger_id:
        raise HTTPException(400, '非法账本 ID')
    from aurumq_rl.simulation import load_ledger
    return load_ledger(ledger_id)


@router.delete('/api/sim/ledgers/{ledger_id}')
def del_ledger(ledger_id: str):
    if '..' in ledger_id or '/' in ledger_id or '\\' in ledger_id:
        raise HTTPException(400, '非法账本 ID')
    from aurumq_rl.simulation import delete_ledger
    ok = delete_ledger(ledger_id)
    return {'status': 'deleted' if ok else 'not_found'}


class BatchDeleteLedgersRequest(BaseModel):
    ids: list[str] = []

    model_config = {'protected_namespaces': ()}


@router.post('/api/sim/ledgers/batch-delete')
def batch_del_ledgers(req: BatchDeleteLedgersRequest):
    """批量删除账本 — body: {ids: [id1, id2, ...]}"""
    from aurumq_rl.simulation import delete_ledger
    ids = req.ids or []
    invalid = [i for i in ids if '..' in i or '/' in i or '\\' in i]
    if invalid:
        raise HTTPException(400, f'非法账本 ID: {invalid[:3]}')
    deleted, not_found = 0, 0
    for lid in ids:
        if delete_ledger(lid):
            deleted += 1
        else:
            not_found += 1
    return {'deleted': deleted, 'not_found': not_found, 'total': len(ids)}
