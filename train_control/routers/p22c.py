"""
routers/p22c.py — Phase 22C 主升浪训练 + 评估 API
"""
import os, sys, time, json, asyncio, subprocess, threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .common import AURUMQ_ROOT, PYTHON_EXE, log, log_callback, list_model_dirs, list_parquet_files
from .ml_joint import resolve_panel

router = APIRouter(tags=['p22c'])

# ── 状态 ──
_p22c_train_process: Optional[subprocess.Popen] = None
_p22c_train_stop = False
_p22c_eval_process: Optional[subprocess.Popen] = None
_p22c_eval_stop = False
_p22c_configs_file = AURUMQ_ROOT / 'data' / 'p22c_saved_configs.json'

# ════════════════════════════════════════════════════════════
# Phase 22C 主升浪 API
# ════════════════════════════════════════════════════════════

_p22c_train_process: Optional[threading.Thread] = None
_p22c_train_stop_flag = threading.Event()
_p22c_eval_process: Optional[threading.Thread] = None
_p22c_eval_stop_flag = threading.Event()
_p22c_last_eval: dict = {}  # 最新评估结果缓存


_p22c_configs_file = AURUMQ_ROOT / 'data' / 'p22c_saved_configs.json'

def _load_p22c_configs() -> list:
    if _p22c_configs_file.exists():
        try:
            return json.loads(_p22c_configs_file.read_text())
        except Exception:
            pass
    return []

def _save_p22c_configs(configs: list):
    _p22c_configs_file.write_text(json.dumps(configs, indent=2, ensure_ascii=False))


class P22cTrainRequest(BaseModel):
    panel: str = ''
    total_timesteps: int = 200000
    start_date: str = '2005-01-04'
    end_date: str = '2026-12-31'
    universe_filter: str = 'main_board_non_st'
    n_envs: int = 16
    top_k: int = 3
    forward_period: int = 10
    cost_bps: float = 30.0
    seed: int = 0  # 0 = 自动随机
    out_dir: str = ''
    # PPO 超参数
    learning_rate: float = 1e-4
    n_steps: int = 128
    n_epochs: int = 10
    batch_size: int = 512
    # 网络架构
    encoder_hidden: str = '128,64'  # 逗号分隔
    encoder_out_dim: int = 32
    # main-wave 配置
    mwl_hold_window: int = 5
    mwl_vol_window: int = 20
    mwl_sigma_multiplier: float = 2.0
    mwl_absolute_threshold: float = 0.06
    mwl_amount_ma_min: float = 1e8

    model_config = {'protected_namespaces': ()}


@router.post('/api/p22c/train/start')
def start_p22c_train(req: P22cTrainRequest):
    global _p22c_train_process

    if _p22c_train_process and _p22c_train_process.is_alive():
        raise HTTPException(400, '已有 Phase 22C 训练运行中')

    panel_name = resolve_panel(req.panel, 'sse50')
    if not panel_name:
        raise HTTPException(400, '未找到任何因子面板，请先在 P3 构建')
    req.panel = panel_name
    panel_path = str(AURUMQ_ROOT / 'data' / panel_name)
    if not Path(panel_path).exists():
        raise HTTPException(400, f'面板文件不存在: {panel_name}')

    if req.start_date >= req.end_date:
        raise HTTPException(400, f'开始日期 ({req.start_date}) 必须早于结束日期 ({req.end_date})')

    # seed=0 → 自动随机
    seed = req.seed if req.seed > 0 else int.from_bytes(os.urandom(4), 'big') % 100000

    out_dir = req.out_dir or f'models/p22c_{time.strftime("%Y%m%d_%H%M")}'
    if not str(out_dir).startswith('models/'):
        out_dir = f"models/{Path(out_dir).name}"

    _p22c_train_stop_flag.clear()

    def run():
        try:
            from train_v2_runner import add_log_callback, run_phase22c_training
            add_log_callback(log_callback)

            log(f'🌊 Phase 22C 训练启动: {req.panel} | {req.total_timesteps:,} 步')

            stats = run_phase22c_training(
                panel_path=panel_path,
                total_timesteps=req.total_timesteps,
                start_date=req.start_date,
                end_date=req.end_date,
                universe_filter=req.universe_filter,
                n_envs=req.n_envs,
                top_k=req.top_k,
                forward_period=req.forward_period,
                cost_bps=req.cost_bps,
                seed=seed,
                out_dir=out_dir,
                stop_flag=_p22c_train_stop_flag,
                # PPO 超参数
                learning_rate=req.learning_rate,
                n_steps=req.n_steps,
                n_epochs=req.n_epochs,
                batch_size=req.batch_size,
                # 网络架构
                encoder_hidden=req.encoder_hidden,
                encoder_out_dim=req.encoder_out_dim,
                # main-wave 配置
                mwl_hold_window=req.mwl_hold_window,
                mwl_vol_window=req.mwl_vol_window,
                mwl_sigma_multiplier=req.mwl_sigma_multiplier,
                mwl_absolute_threshold=req.mwl_absolute_threshold,
                mwl_amount_ma_min=req.mwl_amount_ma_min,
            )
            if stats.get('status') == 'ok':
                log(f'✅ Phase 22C 训练完成: {out_dir}')
            elif stats.get('status') == 'cancelled':
                log(f'⏹ Phase 22C 训练已取消')
            else:
                log(f'⚠ Phase 22C 训练异常: {stats.get("status")}')
        except Exception as e:
            log(f'❌ Phase 22C 训练失败: {str(e)}')
            import traceback
            log(traceback.format_exc()[-500:])
        finally:
            log('--- 任务结束 ---')

    t = threading.Thread(target=run, daemon=True)
    t._panel = req.panel
    t._out_dir = out_dir
    t._total_steps = req.total_timesteps
    t._seed = seed
    _p22c_train_process = t
    t.start()

    return {'status': 'started', 'out_dir': out_dir}


@router.post('/api/p22c/train/stop')
def stop_p22c_train():
    _p22c_train_stop_flag.set()
    return {'status': 'stopping'}


@router.get('/api/p22c/train/status')
def p22c_train_status():
    return {
        'running': _p22c_train_process is not None and _p22c_train_process.is_alive(),
        'panel': getattr(_p22c_train_process, '_panel', None),
        'out_dir': getattr(_p22c_train_process, '_out_dir', None),
        'total_timesteps': getattr(_p22c_train_process, '_total_steps', None),
    }


# ── P22C 参数保存/加载 ──
class P22cConfigSave(BaseModel):
    name: str
    params: dict

    model_config = {'protected_namespaces': ()}

@router.get('/api/p22c/configs')
def list_p22c_configs():
    return {'configs': _load_p22c_configs()}

@router.post('/api/p22c/configs/save')
def save_p22c_config(req: P22cConfigSave):
    configs = _load_p22c_configs()
    configs = [c for c in configs if c.get('name') != req.name]
    configs.append({'name': req.name, 'params': req.params,
                    'saved_at': time.strftime('%Y-%m-%d %H:%M')})
    _save_p22c_configs(configs)
    return {'status': 'saved', 'name': req.name}

@router.delete('/api/p22c/configs/{name}')
def delete_p22c_config(name: str):
    configs = _load_p22c_configs()
    configs = [c for c in configs if c.get('name') != name]
    _save_p22c_configs(configs)
    return {'status': 'deleted', 'name': name}


class P22cEvalRequest(BaseModel):
    run_dir: str = 'runs/phase22c'
    data_path: str = ''
    val_start: str = '2023-01-01'
    val_end: str = '2026-12-31'
    top_k: int = 5
    universe_filter: str = 'main_board_non_st'
    checkpoint: str = ''

    model_config = {'protected_namespaces': ()}


@router.post('/api/p22c/eval/start')
def start_p22c_eval(req: P22cEvalRequest):
    global _p22c_eval_process, _p22c_last_eval

    if _p22c_eval_process and _p22c_eval_process.is_alive():
        raise HTTPException(400, '已有 Phase 22C 评估运行中')

    run_dir = str(AURUMQ_ROOT / req.run_dir) if not req.run_dir.startswith('/') else req.run_dir
    if not req.data_path:
        panel_name = resolve_panel('', 'sse50')
        req.data_path = f'data/{panel_name}' if panel_name else ''
    data_path = str(AURUMQ_ROOT / req.data_path) if not req.data_path.startswith('/') else req.data_path

    if not Path(run_dir).exists():
        raise HTTPException(400, f'Run 目录不存在: {run_dir}')
    if not Path(data_path).exists():
        raise HTTPException(400, f'数据文件不存在: {req.data_path}')

    _p22c_eval_stop_flag.clear()
    _p22c_last_eval = {}

    def run():
        global _p22c_last_eval
        try:
            from train_v2_runner import add_log_callback, run_phase22c_eval
            add_log_callback(log_callback)

            log(f'📊 Phase 22C 评估启动: {req.run_dir}')

            stats = run_phase22c_eval(
                run_dir=run_dir,
                data_path=data_path,
                val_start=req.val_start,
                val_end=req.val_end,
                top_k=req.top_k,
                universe_filter=req.universe_filter,
                checkpoint=req.checkpoint or None,
                stop_flag=_p22c_eval_stop_flag,
            )
            _p22c_last_eval = stats.get('eval_results', {})
            if stats.get('status') == 'ok':
                log(f'✅ Phase 22C 评估完成')
            else:
                log(f'⚠ Phase 22C 评估: {stats.get("status")}')
        except Exception as e:
            log(f'❌ Phase 22C 评估失败: {str(e)}')
            import traceback
            log(traceback.format_exc()[-500:])
        finally:
            log('--- 评估结束 ---')

    t = threading.Thread(target=run, daemon=True)
    _p22c_eval_process = t
    t.start()

    return {'status': 'started'}


@router.post('/api/p22c/eval/stop')
def stop_p22c_eval():
    """停止 Phase 22C 评估（协作式 stop_flag）"""
    global _p22c_eval_process
    _p22c_eval_stop_flag.set()
    log('⏹ Phase 22C 评估停止信号已发送')
    # 进程会在下一轮检查 stop_flag 后退出；线程不强制 kill
    return {'status': 'stopping'}


@router.get('/api/p22c/eval/status')
def p22c_eval_status():
    return {
        'running': _p22c_eval_process is not None and _p22c_eval_process.is_alive(),
    }


@router.get('/api/p22c/eval/result')
def p22c_eval_result():
    """返回最新评估结果（缓存在内存中）"""
    return _p22c_last_eval


@router.get('/api/p22c/runs')
def list_p22c_runs():
    """列出 models/ 目录下所有 Phase 22C 相关的模型（统一模型目录）"""
    runs_dir = AURUMQ_ROOT / 'models'
    if not runs_dir.exists():
        return {'runs': []}
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        onnx = list(d.glob('*.onnx'))
        metrics_files = list(d.glob('*metrics*.jsonl'))
        summary = d / 'training_summary.json'
        meta = d / 'metadata.json'
        final_zip = d / 'ppo_final.zip'
        ckpt_dir = d / 'checkpoints'
        ckpt_size = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file()) if ckpt_dir.exists() else 0
        has_ckpt = ckpt_dir.exists() and ckpt_size > 0
        total_size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())

        info = {
            'name': d.name,
            'path': str(d),
            'has_onnx': len(onnx) > 0,
            'has_metrics': len(metrics_files) > 0,
            'has_final_zip': final_zip.exists(),
            'has_checkpoints': has_ckpt,
            'checkpoints_size_mb': round(ckpt_size / 1024**2, 1),
            'size_mb': round(total_size / 1024**2, 1),
            'size_no_ckpt_mb': round((total_size - ckpt_size) / 1024**2, 1),
            'n_stocks': None,
            'total_timesteps': None,
            'reward_mode': '',
            'top_k': 0,
        }
        if meta.exists():
            try:
                m = json.loads(meta.read_text(encoding='utf-8'))
                # P22C format: stock_codes (list) + training_timesteps
                # Path1-5 format: n_stocks (int) + total_timesteps
                info['n_stocks'] = len(m.get('stock_codes', [])) or m.get('n_stocks')
                info['total_timesteps'] = m.get('training_timesteps') or m.get('total_timesteps')
                info['reward_mode'] = m.get('reward_mode', '')
                info['top_k'] = m.get('top_k', 0)
                # 训练参数（模拟交易对齐用）
                info['rebalance_days'] = m.get('rebalance_days')
                info['max_position_pct'] = m.get('max_position_pct')
                info['cost_bps'] = m.get('cost_bps')
                info['max_holding_days'] = m.get('max_holding_days')
            except Exception:
                pass
        if summary.exists():
            try:
                s = json.loads(summary.read_text(encoding='utf-8'))
                info['total_timesteps'] = s.get('total_timesteps', info['total_timesteps'])
                if not info['n_stocks']:
                    info['n_stocks'] = s.get('n_stocks')
            except Exception:
                pass

        eval_json = d / 'main_wave_eval.json'
        info['has_eval'] = eval_json.exists()
        if eval_json.exists():
            try:
                ev = json.loads(eval_json.read_text(encoding='utf-8'))
                rows = ev.get('rows', [])
                if rows:
                    best = max(rows, key=lambda r: r.get('eval_score', 0))
                    info['best_eval'] = {
                        'top_k': best.get('top_k'),
                        'hit_rate': best.get('main_wave_hit_rate', 0),
                        'win_rate': best.get('basic_win_rate', 0),
                        'avg_hold': best.get('avg_hold_return', 0),
                        'eval_score': best.get('eval_score', 0),
                    }
            except Exception:
                pass

        mtime = d.stat().st_mtime
        info['modified'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
        runs.append(info)
    # 最新在前
    runs.sort(key=lambda x: x.get('modified', ''), reverse=True)
    return {'runs': runs}


@router.delete('/api/p22c/runs/{name}')
def delete_p22c_run(name: str):
    """删除 Phase 22C Run 目录（路径校验防越界）"""
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail='非法名称')
    target = (AURUMQ_ROOT / 'models' / name).resolve()
    runs_dir = (AURUMQ_ROOT / 'models').resolve()
    if not str(target).startswith(str(runs_dir)):
        raise HTTPException(status_code=400, detail='路径越界')
    if not target.exists():
        raise HTTPException(status_code=404, detail='Run 不存在')
    total_size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
    size_mb = round(total_size / 1024**2, 2)
    import shutil
    shutil.rmtree(target)
    return {'status': 'deleted', 'name': name, 'size_mb': size_mb}


@router.post('/api/p22c/runs/{name}/clean-checkpoints')
def clean_p22c_checkpoints(name: str):
    """清理 Phase 22C Run 下的 checkpoints，保留 ppo_final.zip / 指标"""
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail='非法名称')
    target = (AURUMQ_ROOT / 'models' / name).resolve()
    runs_dir = (AURUMQ_ROOT / 'models').resolve()
    if not str(target).startswith(str(runs_dir)):
        raise HTTPException(status_code=400, detail='路径越界')
    if not target.exists():
        raise HTTPException(status_code=404, detail='Run 不存在')
    ckpt_dir = target / 'checkpoints'
    if not ckpt_dir.exists():
        return {'status': 'already_clean', 'name': name}
    size_before = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())
    import shutil
    shutil.rmtree(ckpt_dir)
    return {
        'status': 'cleaned',
        'name': name,
        'cleaned_mb': round(size_before / 1024**2, 1),
    }


@router.get('/api/p22c/eval/json/{run_name}')
def get_p22c_eval_json(run_name: str):
    """读取指定 run 的 main_wave_eval.json"""
    if '..' in run_name or '/' in run_name:
        raise HTTPException(400, '非法名称')
    target = (AURUMQ_ROOT / 'models' / run_name / 'main_wave_eval.json')
    if not target.exists():
        raise HTTPException(404, '评估结果不存在')
    return json.loads(target.read_text(encoding='utf-8'))
