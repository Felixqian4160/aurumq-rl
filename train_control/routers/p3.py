"""
routers/p3.py — P3 数据包构建 API
"""
import os, sys, time, json, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from .common import AURUMQ_ROOT, PYTHON_EXE, log
from .ml_joint import resolve_panel

router = APIRouter(tags=['p3'])


def _normalize_cli_date(value: str, default: str = '') -> str:
    """将页面允许的 YYYYMMDD 输入转换为构建脚本使用的 ISO 日期。"""
    value = (value or '').strip()
    if not value:
        return default
    if len(value) == 8 and value.isdigit():
        return f'{value[:4]}-{value[4:6]}-{value[6:]}'
    return value[:10]

# ── P3 数据包构建 ──
from threading import Thread
import subprocess

_p3_build_state = {'running': False, 'output': '', 'result': None}

@router.post('/api/p3/build/start')
def start_p3_build(req: Request):
    """启动 P3 数据包构建"""
    import asyncio
    task_id = f"p3_build_{time.strftime('%Y%m%d_%H%M%S')}"
    async def _run():
        _p3_build_state.update(running=True, output='', result=None)
        log('🚀 P3 bundle构建启动', source='p3', task_id=task_id, event='start')
        try:
            proc = await asyncio.create_subprocess_exec(
                PYTHON_EXE, 'scripts/p3/build_p3_bundle.py',
                '--factor-panel', f'data/{resolve_panel(req.query_params.get("factor_panel", ""), "hs300")}',
                '--out', f'data/{req.query_params.get("out_dir", "p3_hs300")}',
                '--forward-period', req.query_params.get('forward_period', '10'),
                '--start-date', _normalize_cli_date(req.query_params.get('start_date', ''), '2004-01-01'),
                '--end-date', _normalize_cli_date(req.query_params.get('end_date', ''), '2026-07-24'),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                cwd=str(AURUMQ_ROOT), start_new_session=True)
            # 逐行读取并推送日志
            lines = []
            if proc.stdout:
                async for line in proc.stdout:
                    text = line.decode(errors='replace').rstrip('\n\r')
                    if text:
                        lines.append(text)
                        log(text, source='p3', task_id=task_id, event='progress')
            await proc.wait()
            _p3_build_state['output'] = '\n'.join(lines)
            if proc.returncode == 0:
                _p3_build_state['result'] = {'status': 'ok'}
                log('✅ P3 bundle构建完成', source='p3', task_id=task_id, event='finish')
            else:
                _p3_build_state['result'] = {'status': 'error', 'code': proc.returncode}
                log(f'❌ P3 bundle构建失败 exit={proc.returncode}', source='p3', task_id=task_id, event='error', level='ERROR')
        except Exception as e:
            _p3_build_state['result'] = {'status': 'error', 'msg': str(e)}
            log(f'❌ P3 bundle异常: {e}', source='p3', task_id=task_id, event='error', level='ERROR')
        finally:
            _p3_build_state['running'] = False
    Thread(target=lambda: asyncio.run(_run()), daemon=True).start()
    return {'status': 'started'}

@router.get('/api/p3/build/status')
def p3_build_status():
    return _p3_build_state

@router.get('/api/p3/bundles')
def list_p3_bundles():
    """列出已构建的 P3 数据包"""
    data_dir = AURUMQ_ROOT / 'data'
    bundles = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.startswith('p3_'):
            meta_path = d / 'meta.json'
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding='utf-8'))
                    # 计算目录大小
                    total_size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
                    size_gb = round(total_size / (1024**3), 3)
                    # 获取修改时间
                    mtime = datetime.datetime.fromtimestamp(d.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                    bundles.append({
                        'name': d.name,
                        'n_features': meta.get('n_features', 0),
                        'n_stocks': meta.get('n_stocks', 0),
                        'n_dates': meta.get('n_dates', 0),
                        'date_range': meta.get('date_range', []),
                        'splits': meta.get('splits', {}),
                        'size_gb': size_gb,
                        'modified': mtime,
                    })
                except:
                    pass
    return {'bundles': bundles}

@router.delete('/api/p3/bundles/{name}')
def delete_p3_bundle(name: str):
    """删除 P3 数据包"""
    import shutil
    if '..' in name or '/' in name:
        raise HTTPException(400, '非法名称')
    target = AURUMQ_ROOT / 'data' / name
    if not target.exists() or not target.is_dir():
        raise HTTPException(404, '数据包不存在')
    shutil.rmtree(target)
    return {'status': 'deleted', 'name': name}
