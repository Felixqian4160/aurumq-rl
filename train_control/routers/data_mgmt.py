"""
routers/data_mgmt.py — 数据管理 API（瘦身版）
仅做参数校验 + 委托 services/data_service 与独立 driver 进程。
"""
import os, time, json, threading, subprocess, uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .common import AURUMQ_ROOT, log, BUILD_LOG_DIR, BUILD_RUNTIME_DIR as RUNTIME_DIR
import services.data_service as ds

router = APIRouter(tags=['data_mgmt'])

TOKEN_FILE = AURUMQ_ROOT / '.qbot_token'
DATA_CACHE_DIR = ds.DATA_CACHE_DIR
DOWNLOAD_PROGRESS: dict = {}
_download_start: dict = {}
_STARTUP_TIMEOUT_SECONDS = 15


def _runtime_state_path(task_id: str):
    return RUNTIME_DIR / f'{task_id}.json'


def _stop_request_path(task_id: str):
    return RUNTIME_DIR / f'{task_id}.stop'


def _write_runtime_state(task_id: str, state: dict) -> None:
    path = _runtime_state_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    tmp.replace(path)


def _watch_download_start(task_id: str, unit: str, started_at: float) -> None:
    """确认 systemd unit 在短窗口内真正启动，否则落 failed 状态。"""
    time.sleep(_STARTUP_TIMEOUT_SECONDS)
    path = _runtime_state_path(task_id)
    try:
        state = json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        state = {}
    if state.get('status') != 'starting':
        return
    probe = subprocess.run(
        ['systemctl', '--user', 'is-active', unit],
        capture_output=True, text=True, timeout=5,
    )
    active = probe.returncode == 0 and probe.stdout.strip() == 'active'
    if not active:
        state.update({
            'status': 'failed',
            'error': '后台下载进程未在启动超时内进入 active 状态',
            'unit': unit,
            'elapsed': round(time.time() - started_at),
        })
        _write_runtime_state(task_id, state)
        log(f"❌ 下载任务启动失败: {task_id} | {state['error']}", source='data_mgmt', task_id=task_id, event='failed')

class TokenInput(BaseModel):
    token: str


class DownloadRequest(BaseModel):
    pool: str = 'hs300'
    start_date: str = '20040102'
    end_date: str = ''
    data_type: str = 'ohlcv'


# ── Token ──
@router.get('/api/data/token')
def get_token():
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
        return {'token': token[:8] + '...' + token[-4:], 'set': True}
    env = os.environ.get('TUSHARE_TOKEN', '')
    if env:
        return {'token': env[:8] + '...' + env[-4:], 'set': True, 'source': 'env'}
    return {'token': '', 'set': False}

@router.post('/api/data/token')
def set_token(req: TokenInput):
    TOKEN_FILE.write_text(req.token.strip())
    os.environ['TUSHARE_TOKEN'] = req.token.strip()
    return {'status': 'ok'}

@router.post('/api/data/token/test')
def test_token(req: TokenInput):
    try:
        import tushare as ts
        ts.set_token(req.token.strip())
        pro = ts.pro_api()
        df = pro.trade_cal(exchange='SSE', start_date='20260101', end_date='20260110')
        if df is not None and len(df) > 0:
            return {'status': 'ok', 'message': f'连接成功，返回 {len(df)} 条交易日历'}
        return {'status': 'error', 'message': '返回空数据'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@router.get('/api/data/token/status')
def token_status():
    try:
        pro = ds._get_pro()
        df = pro.trade_cal(exchange='SSE', start_date='20260101', end_date='20260110')
        if df is not None and len(df) > 0:
            return {'status': 'valid', 'message': 'Token 有效'}
        return {'status': 'invalid', 'message': 'Token 无效'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ── 下载（走独立 driver 进程） ──
@router.post('/api/data/download')
def start_download(req: DownloadRequest):
    end_date = req.end_date or datetime.now().strftime('%Y%m%d')
    if req.data_type in ('daily_basic', 'fina_indicator') and req.pool == 'all':
        codes = ds.get_all_cached_codes()
        if not codes:
            codes = ds.get_all_pool_codes()
            if not codes:
                raise HTTPException(400, '无已缓存 OHLCV且获取股票池失败，请先下载日线')
        pool_effective = 'all(已缓存全量)'
    else:
        codes = ds._get_pool_codes(req.pool)
        pool_effective = req.pool
    if not codes:
        raise HTTPException(400, f'获取股票池 {req.pool} 失败')
    if req.data_type not in {'ohlcv', 'daily_basic', 'fina_indicator'}:
        raise HTTPException(400, f'未知数据类型: {req.data_type}')
    task_id = f"dl_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    pool_for_driver = 'all' if (req.data_type in ('daily_basic', 'fina_indicator') and req.pool == 'all') else req.pool
    driver = str(AURUMQ_ROOT / 'scripts' / 'data_download_driver.py')
    log_file = str(BUILD_LOG_DIR / f'{task_id}.log')
    unit = f'dl-{task_id}'
    started_at = time.time()
    _write_runtime_state(task_id, {
        'status': 'starting', 'task_id': task_id, 'unit': unit,
        'pool': pool_for_driver, 'data_type': req.data_type,
        'start_date': req.start_date, 'end_date': end_date,
        'total': len(codes), 'done': 0, 'failed': 0, 'pct': 0,
        'started_at': started_at,
    })
    cmd = ['systemd-run', '--user', '--unit', unit, '--collect',
           '/usr/bin/python3', driver, task_id, pool_for_driver,
           req.start_date, end_date, req.data_type]
    try:
        with open(log_file, 'a', encoding='utf-8') as log_handle:
            result = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                                    timeout=10, check=False)
    except Exception as exc:
        _write_runtime_state(task_id, {'status': 'failed', 'task_id': task_id,
                                        'error': f'后台启动异常: {exc}', 'unit': unit})
        raise HTTPException(500, f'后台下载启动失败: {exc}') from exc
    if result.returncode != 0:
        error = f'systemd-run 返回码 {result.returncode}'
        _write_runtime_state(task_id, {'status': 'failed', 'task_id': task_id,
                                        'error': error, 'unit': unit,
                                        'returncode': result.returncode})
        raise HTTPException(500, f'后台下载启动失败: {error}')
    _download_start[task_id] = started_at
    DOWNLOAD_PROGRESS[task_id] = {'status': 'starting', 'total': len(codes), 'done': 0}
    threading.Thread(target=_watch_download_start, args=(task_id, unit, started_at), daemon=True).start()
    log(f"📥 下载任务已提交: {req.data_type.upper()} | {pool_effective} | {len(codes)} 只 | {req.start_date}~{end_date}", source='data_mgmt', task_id=task_id, event='start')
    return {'task_id': task_id, 'total': len(codes), 'data_type': req.data_type, 'status': 'starting'}

@router.get('/api/data/download/progress')
def download_progress(task_id: str = ''):
    if not task_id:
        tasks = {}
        for f in RUNTIME_DIR.glob('dl_*.json'):
            try: tasks[f.stem] = json.loads(f.read_text())
            except: pass
        return {'tasks': tasks}
    state_file = RUNTIME_DIR / f'{task_id}.json'
    if not state_file.exists():
        raise HTTPException(404, '任务不存在')
    return json.loads(state_file.read_text())


@router.post('/api/data/download/stop')
def stop_download(task_id: str):
    if not task_id or not task_id.startswith('dl_'):
        raise HTTPException(400, '无效下载任务 ID')
    state_path = _runtime_state_path(task_id)
    if not state_path.exists():
        raise HTTPException(404, '下载任务不存在')
    try:
        state = json.loads(state_path.read_text())
    except Exception as exc:
        raise HTTPException(500, f'读取任务状态失败: {exc}') from exc
    if state.get('status') in {'done', 'failed', 'stopped', 'blocked', 'stale'}:
        return {'status': state['status'], 'task_id': task_id}
    _stop_request_path(task_id).write_text('stop\n')
    state.update({'status': 'stopping', 'stop_requested_at': time.time()})
    _write_runtime_state(task_id, state)
    log(f"⏹ 已请求停止下载: {task_id}", source='data_mgmt', task_id=task_id, event='stop_request')
    return {'status': 'stopping', 'task_id': task_id}


# ── 缓存 / 校验 ──
@router.get('/api/data/cached')
def list_cached():
    return ds.cache_status()

@router.get('/api/data/daily-basic-status')
def daily_basic_status():
    return ds.daily_basic_status()

@router.get('/api/data/fina-indicator-status')
def fina_indicator_status():
    return ds.fina_indicator_status()

@router.get('/api/data/adj-status')
def adj_status():
    return ds.adj_status()

@router.get('/api/data/verify')
def verify_cache():
    return ds.verify_cache()


# ── 一键更新 ──
@router.post('/api/data/update-all')
def update_all():
    end_date = datetime.now().strftime('%Y%m%d')
    codes = ds.get_all_cached_codes()
    if not codes:
        codes = ds.get_all_pool_codes()
    if not codes:
        raise HTTPException(500, '获取股票池失败')
    task_id = f"update_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    driver = str(AURUMQ_ROOT / 'scripts' / 'data_download_driver.py')
    unit = f'dl-{task_id}'
    log_file = str(BUILD_LOG_DIR / f'{task_id}.log')
    started_at = time.time()
    _write_runtime_state(task_id, {
        'status': 'starting', 'task_id': task_id, 'unit': unit,
        'pool': 'all', 'data_type': 'ohlcv', 'start_date': '20040102',
        'end_date': end_date, 'total': len(codes), 'done': 0, 'failed': 0,
        'pct': 0, 'started_at': started_at,
    })
    cmd = ['systemd-run', '--user', '--unit', unit, '--collect',
           '/usr/bin/python3', driver, task_id, 'all', '20040102', end_date, 'ohlcv']
    try:
        with open(log_file, 'a', encoding='utf-8') as log_handle:
            result = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                                    timeout=10, check=False)
    except Exception as exc:
        _write_runtime_state(task_id, {'status': 'failed', 'task_id': task_id,
                                        'error': f'后台更新启动异常: {exc}', 'unit': unit})
        raise HTTPException(500, f'后台更新启动失败: {exc}') from exc
    if result.returncode != 0:
        error = f'systemd-run 返回码 {result.returncode}'
        _write_runtime_state(task_id, {'status': 'failed', 'task_id': task_id,
                                        'error': error, 'unit': unit,
                                        'returncode': result.returncode})
        raise HTTPException(500, f'后台更新启动失败: {error}')
    _download_start[task_id] = started_at
    DOWNLOAD_PROGRESS[task_id] = {'status': 'starting', 'total': len(codes), 'done': 0}
    threading.Thread(target=_watch_download_start, args=(task_id, unit, started_at), daemon=True).start()
    log(f"🔄 更新任务已提交: {len(codes)} 只 | 20040102~{end_date}", source='data_mgmt', task_id=task_id, event='start')
    return {'task_id': task_id, 'total': len(codes), 'status': 'starting'}


@router.post('/api/data/update-all/stop')
def stop_update(task_id: str = ''):
    if not task_id:
        candidates = []
        for path in RUNTIME_DIR.glob('update_*.json'):
            try:
                state = json.loads(path.read_text())
                if state.get('status') in {'starting', 'running', 'stopping'}:
                    candidates.append((state.get('started_at', 0), path.stem))
            except Exception:
                continue
        if candidates:
            task_id = max(candidates)[1]
    if not task_id or not task_id.startswith('update_'):
        return {'status': 'idle', 'message': '没有运行中的更新任务'}
    state_path = _runtime_state_path(task_id)
    if not state_path.exists():
        raise HTTPException(404, '更新任务不存在')
    state = json.loads(state_path.read_text())
    if state.get('status') in {'done', 'failed', 'stopped', 'blocked', 'stale'}:
        return {'status': state['status'], 'task_id': task_id}
    _stop_request_path(task_id).write_text('stop\n')
    state.update({'status': 'stopping', 'stop_requested_at': time.time()})
    _write_runtime_state(task_id, state)
    log(f"⏹ 已请求停止更新: {task_id}", source='data_mgmt', task_id=task_id, event='stop_request')
    return {'status': 'stopping', 'task_id': task_id}
