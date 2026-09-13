"""
routers/panel_build.py — 因子面板构建 API (v1/v2/v3/v8)
瘦身为参数校验 + 委托 services/panel_service.py。
"""
import os, sys, time, json, datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.config import AURUMQ_ROOT, PYTHON_EXE, BUILD_RUNTIME_DIR, BUILD_LOG_DIR
from core.process import build_pid_alive
from services.panel_service import (
    _persistent_build_state,
    _start_persistent_build,
    _stop_build,
    validate_pool,
    validate_date_range,
    merge_cs800_v3_panels,
)

router = APIRouter(tags=['panel_build'])

_build_runtime_dir = BUILD_RUNTIME_DIR
_build_log_dir = BUILD_LOG_DIR
class BuildRequest(BaseModel):
    pool: str = 'hs300'
    start_date: str = '20040102'
    end_date: str = ''
    include_fundamental: bool = True

    model_config = {'protected_namespaces': ()}


class WaveHunterPanelBuildRequest(BaseModel):
    pool: str = 'hs300'
    start_date: str = '20040102'
    end_date: str = '20260804'

    model_config = {'protected_namespaces': ()}


# _build_runtime/log 已由 core/config 初始化


@router.post('/api/wavehunter/panel/build/start')
def start_wavehunter_panel_build(req: WaveHunterPanelBuildRequest):
    """通过 WebUI 构建 WaveHunter v3 面板；不允许并行构建。"""
    if req.pool not in ('hs300', 'csi500', 'cs800'):
        raise HTTPException(400, f'不支持的股票池: {req.pool}')
    try:
        start_dt = datetime.date.fromisoformat(req.start_date.replace('/', '-'))
        end_dt = datetime.date.fromisoformat(req.end_date.replace('/', '-'))
    except ValueError as e:
        raise HTTPException(400, f'日期格式非法: {e}') from e
    if start_dt >= end_dt:
        raise HTTPException(400, '面板开始日期必须早于结束日期')
    return _start_persistent_build('v3_panel_build', 'v3', {
        'pool': req.pool, 'start_date': req.start_date.replace('-', ''),
        'end_date': req.end_date.replace('-', '')})


@router.get('/api/wavehunter/panel/build/status')
def wavehunter_panel_build_status():
    return _persistent_build_state('v3_panel_build')


@router.post('/api/wavehunter/v8/panel/build/start')
def start_wavehunter_v8_panel_build(req: WaveHunterPanelBuildRequest):
    """通过独立 v8 worker 构建 WaveHunter v8 因子面板。"""
    if req.pool not in ('hs300', 'csi500', 'cs800'):
        raise HTTPException(400, f'不支持的股票池: {req.pool}')
    try:
        start_dt = datetime.date.fromisoformat(req.start_date.replace('/', '-'))
        end_dt = datetime.date.fromisoformat(req.end_date.replace('/', '-'))
    except ValueError as e:
        raise HTTPException(400, f'日期格式非法: {e}') from e
    if start_dt >= end_dt:
        raise HTTPException(400, '面板开始日期必须早于结束日期')
    return _start_persistent_build('v8_panel_build', 'v8', {
        'pool': req.pool, 'start_date': req.start_date.replace('-', ''),
        'end_date': req.end_date.replace('-', ''), 'include_fundamental': True})


@router.get('/api/wavehunter/v8/panel/build/status')
def wavehunter_v8_panel_build_status():
    return _persistent_build_state('v8_panel_build')

@router.post('/api/wavehunter/v9/panel/build/start')
def start_wavehunter_v9_panel_build(req: WaveHunterPanelBuildRequest):
    """构建 v9 高低点面板：在 v8 基础上追加 ZigZag 真实高低点标签。"""
    if req.pool not in ('hs300', 'csi500', 'cs800'):
        raise HTTPException(400, f'不支持的股票池: {req.pool}')
    try:
        start_dt = datetime.date.fromisoformat(req.start_date.replace('/', '-'))
        end_dt = datetime.date.fromisoformat(req.end_date.replace('/', '-'))
    except ValueError as e:
        raise HTTPException(400, f'日期格式非法: {e}') from e
    if start_dt >= end_dt:
        raise HTTPException(400, '面板开始日期必须早于结束日期')
    return _start_persistent_build('v9_panel_build', 'v9', {
        'pool': req.pool, 'start_date': req.start_date.replace('-', ''),
        'end_date': req.end_date.replace('-', ''), 'include_fundamental': True})

@router.get('/api/wavehunter/v9/panel/build/status')
def wavehunter_v9_panel_build_status():
    return _persistent_build_state('v9_panel_build')


@router.post('/api/wavehunter/v10/panel/build/start')
def start_wavehunter_v10_panel_build(req: WaveHunterPanelBuildRequest):
    """从原始缓存独立构建 WaveHunter v10 面板，不读取 v8/v9 面板。"""
    if req.pool not in ('hs300', 'csi500', 'cs800'):
        raise HTTPException(400, f'不支持的股票池: {req.pool}')
    try:
        start_dt = datetime.date.fromisoformat(req.start_date.replace('/', '-'))
        end_dt = datetime.date.fromisoformat(req.end_date.replace('/', '-'))
    except ValueError as e:
        raise HTTPException(400, f'日期格式非法: {e}') from e
    if start_dt >= end_dt:
        raise HTTPException(400, '面板开始日期必须早于结束日期')
    return _start_persistent_build('v10_panel_build', 'v10', {
        'pool': req.pool, 'start_date': req.start_date.replace('-', ''),
        'end_date': req.end_date.replace('-', ''), 'include_fundamental': True})


@router.get('/api/wavehunter/v10/panel/build/status')
def wavehunter_v10_panel_build_status():
    return _persistent_build_state('v10_panel_build')


@router.post('/api/wavehunter/v10.1/panel/build/start')
def start_wavehunter_v10_1_panel_build(req: WaveHunterPanelBuildRequest):
    """从原始缓存构建 v10.1：峰谷中心±1日，前一日方向性3%过滤。"""
    if req.pool not in ('hs300', 'csi500', 'cs800'):
        raise HTTPException(400, f'不支持的股票池: {req.pool}')
    try:
        start_dt = datetime.date.fromisoformat(req.start_date.replace('/', '-'))
        end_dt = datetime.date.fromisoformat(req.end_date.replace('/', '-'))
    except ValueError as e:
        raise HTTPException(400, f'日期格式非法: {e}') from e
    if start_dt >= end_dt:
        raise HTTPException(400, '面板开始日期必须早于结束日期')
    return _start_persistent_build('v10_1_panel_build', 'v10_1', {
        'pool': req.pool, 'start_date': req.start_date.replace('-', ''),
        'end_date': req.end_date.replace('-', ''), 'include_fundamental': True})


@router.get('/api/wavehunter/v10.1/panel/build/status')
def wavehunter_v10_1_panel_build_status():
    return _persistent_build_state('v10_1_panel_build')


# ── API: 启动构建 ──
@router.post('/api/build/start')
def start_build(req: BuildRequest):
    if req.pool not in ('hs300', 'csi500', 'cs800'):
        raise HTTPException(400, f'不支持的股票池: {req.pool}')
    end = req.end_date or time.strftime('%Y%m%d')
    try:
        if datetime.date.fromisoformat(req.start_date.replace('/', '-')) >= datetime.date.fromisoformat(end.replace('/', '-')):
            raise HTTPException(400, '面板开始日期必须早于结束日期')
    except ValueError as e:
        raise HTTPException(400, f'日期格式非法: {e}') from e
    return _start_persistent_build('v1_build', 'v1', {
        'pool': req.pool, 'start_date': req.start_date, 'end_date': end,
        'include_fundamental': req.include_fundamental})


# ── API: 停止构建 ──
@router.post('/api/build/stop')
def stop_build():
    """停止 v1 构建（兼容旧入口）。"""
    state = _persistent_build_state('v1_build')
    pid = state.get('pid')
    if state.get('running') and isinstance(pid, int):
        try:
            os.killpg(pid, 15)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    return {'status': 'stopping'}


@router.post('/api/panel/build/stop')
def stop_panel_build(task_id: str = ''):
    """统一停止面板构建：按 task_id 精确停止；不传则扫描所有运行中的构建任务。"""
    result = _stop_build(task_id)
    if task_id and not result.get('stopped'):
        from fastapi import HTTPException
        raise HTTPException(404, f'未找到 task_id={task_id} 的运行任务')
    return result


# ── API: v2 市场面板构建 ──
_v2_runtime_dir = AURUMQ_ROOT / 'data' / 'task_runtime'
_v2_log_dir = AURUMQ_ROOT / 'data' / 'training_logs'
_v2_runtime_dir.mkdir(parents=True, exist_ok=True)
_v2_log_dir.mkdir(parents=True, exist_ok=True)


def _v2_read_runtime(task_id: str = ''):
    from services.panel_service import _BUILD_RUNTIME_DIR, _BUILD_LOG_DIR  # noqa: F401
    paths = sorted(_v2_runtime_dir.glob('v2_build_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        return {'running': False, 'status': 'idle', 'log': [], 'result': None}
    if task_id:
        match = next((p for p in paths if p.stem == task_id), None)
        path = match or paths[0]
    else:
        path = paths[0]
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'running': False, 'status': 'failed', 'progress': 'runtime 文件不可读', 'log': [], 'result': None}
    if state.get('status') == 'running' and not build_pid_alive(state.get('pid')):
        state.update(running=False, status='failed', progress='失败：构建 PID 已失效（stale）', returncode=None)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    log_path = Path(state.get('log_path', ''))
    try:
        lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-50:]
    except OSError:
        lines = []
    state['log'] = lines
    state['running'] = state.get('status') == 'running'
    state['task_id'] = state.get('task_id', path.stem)
    return state


@router.post('/api/build/v2')
async def start_v2_build(req: Request):
    """构建 v2 市场面板：统一通过 panel_service 启动并复用停止入口。"""
    current = _v2_read_runtime()
    if current.get('running'):
        return {'error': 'v2 构建正在运行中'}
    body = await req.json()
    pool = str(body.get('pool', 'hs300'))
    validate_pool(pool)
    start_date = str(body.get('start_date', '20040102')).replace('-', '')
    end_date = str(body.get('end_date', '')).replace('-', '') or time.strftime('%Y%m%d')
    validate_date_range(start_date, end_date)
    include_fundamental = bool(body.get('include_fundamental', False))
    config = {
        'pool': pool, 'start_date': start_date, 'end_date': end_date,
        'include_fundamental': include_fundamental,
    }
    result = _start_persistent_build('v2_panel_build', 'v2', config)
    return {'status': 'started', 'task_id': result['task_id']}


@router.get('/api/build/v2/status')
def v2_build_status(task_id: str = ''):
    return _v2_read_runtime(task_id=task_id)


# ── API: 可用股票池 ──
@router.get('/api/pools')
def get_pools():
    return {
        'pools': [
            {'id': 'hs300', 'name': '沪深300', 'desc': '约 354 只'},
            {'id': 'csi500', 'name': '中证500', 'desc': '约 500 只'},
            {'id': 'cs800', 'name': '沪深300+中证500', 'desc': '约 854 只'},
        ]
    }


# ── API: 已有构建文件 ──
@router.get('/api/panels')
def list_panels():
    data_dir = AURUMQ_ROOT / 'data'
    # 已生成面板包含普通 factor_panel、WaveHunter v3 和 v2 主升浪面板。
    # 不能只扫描 factor_panel_*.parquet，否则 v2 成功生成后列表仍为空。
    panel_files = list(data_dir.glob('factor_panel_*.parquet'))
    panel_files += list(data_dir.glob('wavehunter_*.parquet'))
    files = []
    unique_files = {p.resolve(): p for p in panel_files}
    for f in sorted(unique_files.values(), key=lambda p: p.stat().st_mtime, reverse=True):
        stats = f.stat()
        files.append({
            'name': f.name,
            'size_gb': round(stats.st_size / 1024**3, 3),
            'size_mb': round(stats.st_size / 1024**2, 2),
            'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(stats.st_mtime)),
        })
    return {'panels': files}


@router.delete('/api/panels/{name}')
def delete_panel(name: str):
    """删除因子面板 parquet 文件（仅限 factor_panel_*.parquet/wavehunter_*.parquet，路径校验防越界）。

    同时清理 build_stats_*.json 残留，防止 stats 文件悬空。
    """
    if '..' in name or '/' in name or not (name.startswith('factor_panel_') or name.startswith('wavehunter_')) or not name.endswith('.parquet'):
        raise HTTPException(status_code=400, detail='非法文件名')
    target = (AURUMQ_ROOT / 'data' / name).resolve()
    data_dir = (AURUMQ_ROOT / 'data').resolve()
    if not str(target).startswith(str(data_dir)):
        raise HTTPException(status_code=400, detail='路径越界')
    if not target.exists():
        raise HTTPException(status_code=404, detail='文件不存在')
    size_mb = round(target.stat().st_size / 1024**2, 2)
    target.unlink()
    stem = name[:-len('.parquet')]
    removed_stats: list[str] = []
    patterns = [
        f"build_stats_{stem}.json",
        f"build_stats_{stem.replace('factor_panel_', '')}.json",
        f"build_stats_{stem.replace('wavehunter_', '')}.json",
    ]
    for pattern in patterns:
        stats_file = data_dir / pattern
        if stats_file.exists():
            stats_file.unlink()
            removed_stats.append(pattern)
    return {'status': 'deleted', 'name': name, 'size_mb': size_mb,
            'removed_stats': removed_stats}
@router.get('/api/panels/{name}/stats')
def panel_stats(name: str):
    """获取面板统计信息（A1/A2 密度等）。

    读取 parquet schema 仅做列名/类型扫描，不读取数据；统计采样前 10000 行或全局轻量聚合，
    避免在 1~4GB 面板上做多次全量读取导致 WebUI 首屏延迟。
    """
    import polars as pl

    if '..' in name or '/' in name or not (name.startswith('factor_panel_') or name.startswith('wavehunter_')) or not name.endswith('.parquet'):
        raise HTTPException(status_code=400, detail='非法文件名')
    panel_path = (AURUMQ_ROOT / 'data' / name).resolve()
    data_dir = (AURUMQ_ROOT / 'data').resolve()
    if not str(panel_path).startswith(str(data_dir)):
        raise HTTPException(status_code=400, detail='路径越界')
    if not panel_path.exists():
        raise HTTPException(404, f"面板不存在: {name}")
    try:
        schema = pl.scan_parquet(str(panel_path)).collect_schema()
        total_rows = pl.scan_parquet(str(panel_path)).select(pl.len()).collect().item()
        stats = {
            'name': name,
            'total_rows': total_rows,
            'columns': len(schema.names()),
            'v9_columns': [c for c in schema.names() if c.startswith('v9_')],
            'v10_columns': [c for c in schema.names() if c.startswith('v10_') and not c.startswith('v10_1_')],
            'v10_1_columns': [c for c in schema.names() if c.startswith('v10_1_')],
        }
        sample_n = min(int(total_rows), 10000) if total_rows else 0
        if sample_n:
            sample = pl.scan_parquet(str(panel_path)).head(sample_n).collect()
            stats['sample_rows'] = sample.height
        # v10.1 / v10 / v9 标签密度（优先识别更长前缀，避免 v10_1 被误归入 v10）。
        v101_prefix = any(c.startswith('v10_1_') for c in schema.names())
        v10_prefix = (not v101_prefix) and any(c.startswith('v10_') for c in schema.names())
        v9_prefix = (not v101_prefix) and (not v10_prefix) and any(c.startswith('v9_') for c in schema.names())
        if v101_prefix:
            label_cols = [c for c in schema.names()
                          if c.startswith(('v10_1_zig_', 'v10_1_peak_zone', 'v10_1_valley_zone', 'v10_1_a', 'v10_1_b1', 'v10_1_down'))]
            prefix = 'v10_1_'
        elif v10_prefix:
            label_cols = [c for c in schema.names()
                          if c.startswith(('v10_zig_', 'v10_a', 'v10_b1', 'v10_down'))]
            prefix = 'v10_'
        elif v9_prefix:
            label_cols = [c for c in schema.names()
                          if c.startswith(('v9_zig_', 'v9_a', 'v9_b1', 'v9_down'))]
            prefix = 'v9_'
        else:
            label_cols = []
            prefix = ''
        if label_cols:
            agg = (pl.scan_parquet(str(panel_path))
                   .select([pl.len().alias('n')] + label_cols)
                   .select([pl.col('n')] + [(pl.col(c) == 1).sum().alias(c) for c in label_cols])
                   .collect())
            n = int(agg['n'][0])
            labels = {}
            for c in label_cols:
                s = float(agg[c][0]) if c in agg.columns else 0.0
                labels[c.replace(prefix, '')] = {
                    'count': int(s),
                    'density_pct': round(s / n * 100, 2) if n else 0,
                }
            stats['v10_1_labels' if v101_prefix else 'v10_labels'] = labels
        return stats
    except Exception as e:
        raise HTTPException(500, f"读取面板失败: {e}")
