"""
AurumQ-RL 训练控制台 (模块化版)
================================
FastAPI 主入口，挂载所有 router。
原 app.py 3022行拆分为 9 个 router 模块。
"""
import os, sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── 路径设置 ──
# wavehunter 模块用 `from app import app` 导入，必须设置 sys.modules 别名
if __name__ == '__main__':
    sys.modules.setdefault('app', sys.modules[__name__])

from pathlib import Path

from core.config import AURUMQ_ROOT
STATIC_DIR = Path(__file__).parent / 'static'
STATIC_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(AURUMQ_ROOT / 'scripts'))
sys.path.insert(0, str(AURUMQ_ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

# ── 创建 FastAPI 应用 ──
app = FastAPI(title='AurumQ-RL 训练控制台')
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

# ── 挂载所有 Router ──
from routers.panel_build import router as panel_build_router
from routers.training import router as training_router
from routers.simulation import router as simulation_router
from routers.p22c import router as p22c_router
from routers.p3 import router as p3_router
from routers.ml_joint import router as ml_joint_router
from routers.walkforward import router as walkforward_router
from routers.data_mgmt import router as data_mgmt_router
from routers.matrix_ledger_api import router as matrix_ledger_router

app.include_router(panel_build_router)
app.include_router(training_router)
app.include_router(simulation_router)
app.include_router(p22c_router)
app.include_router(p3_router)
app.include_router(ml_joint_router)
app.include_router(walkforward_router)
app.include_router(data_mgmt_router)
app.include_router(matrix_ledger_router)
from routers.pivots import router as pivots_router
app.include_router(pivots_router)

# ── 通用端点 ──
import core.logging as _clog
from routers.simulation import start_sim, SimRequest, sim_status  # re-export for wavehunter
import json, time, asyncio

_clog.init_event_logger()
log = _clog.log  # 供 wavehunter 等延迟导入复用


@app.get('/api/{module}/schema')
def module_schema(module: str):
    """返回模块参数schema"""
    try:
        from module_schemas import get_schema
        return get_schema(module)
    except Exception:
        return {}


@app.get('/api/logs')
async def stream_logs():
    """SSE 日志流"""
    async def event_generator():
        q = asyncio.Queue(maxsize=5000)
        _clog._subscribers.append(q)
        try:
            while True:
                entry = await q.get()
                yield f'data: {json.dumps(entry, ensure_ascii=False)}\n\n'
        except asyncio.CancelledError:
            pass
        finally:
            if q in _clog._subscribers:
                _clog._subscribers.remove(q)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/api/logs/recent')
def recent_logs(limit: int = 200, source: str = '', task_id: str = ''):
    if _clog._recent_events is None:
        return {'events': []}
    return {'events': _clog._recent_events(max(1, min(limit, 2000)), source, task_id)}


@app.get('/api/status')
def get_status():
    from routers.common import read_build_state
    return read_build_state('v1_build')


@app.get('/', response_class=HTMLResponse)
def index_page():
    """训练控制台首页"""
    index = STATIC_DIR / 'index.html'
    if not index.exists():
        raise HTTPException(500, 'WebUI index.html不存在')
    from fastapi.responses import HTMLResponse
    resp = HTMLResponse(index.read_text(encoding='utf-8'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


# ── WaveHunter 模块（延迟导入，避免循环依赖） ──
# 这些模块做 `from app import app, AURUMQ_ROOT, log, start_sim, SimRequest`
# 必须在 app.py 完全初始化后才导入
from wavehunter_api import *  # noqa: E402,F401
from wavehunter_matrix_api import *  # noqa: E402,F401
from wavehunter_v2_api import *  # noqa: E402,F401
from wavehunter_v2_matrix_api import *  # noqa: E402,F401

from routers.v10_1_training import router as v10_1_training_router
app.include_router(v10_1_training_router)
from routers.v10_1_simulation import router as v10_1_simulation_router
app.include_router(v10_1_simulation_router)

# v10 四段概率策略路由 (独立于 frozen 链路)
try:
    from routers.v10.v10_router import _V10_STATE  # noqa
    from routers.v10.v10_router import v10_sim_start  # noqa
    from routers.v10.v10_router import v10_sim_stop  # noqa
    from routers.v10.v10_router import v10_sim_status  # noqa
    from routers.v10.v10_router import v10_sim_result  # noqa
    from routers.simulation_v2 import router as simulation_v2_router  # noqa
    from routers.simulation_timingfix import router as timingfix_simulation_router  # noqa
    from routers.training_timingfix import router as timingfix_training_router  # noqa
    app.include_router(simulation_v2_router)
    app.include_router(timingfix_simulation_router)
    app.include_router(timingfix_training_router)
    print("[app] v10 路由已挂载 (4 个端点)")
except Exception as e:
    print(f"[app] v10 路由加载失败: {e}")


if __name__ == '__main__':
    print('AurumQ-RL 训练控制台 → http://0.0.0.0:8082')
    uvicorn.run(app, host='0.0.0.0', port=8082)
