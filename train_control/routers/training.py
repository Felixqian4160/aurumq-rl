"""
routers/training.py — PPO/LSTM 模型训练 API
"""
import os, sys, time, json, threading, uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .ml_joint import resolve_panel

from .common import AURUMQ_ROOT, PYTHON_EXE, log, log_callback, list_model_dirs, list_parquet_files

router = APIRouter(tags=['training'])

_train_process = None
_train_stop_flag = __import__('threading').Event()  # 线程安全的停止信号
_train_task_id = ''
_train_runtime_path: Optional[Path] = None

# ════════════════════════════════════════════════════════════
# 训练 API
# ════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    panel: str = ''
    algorithm: str = 'PPO'
    total_timesteps: int = Field(100000, ge=10000, le=10000000)
    start_date: str = '2023-01-01'
    end_date: str = '2025-06-30'
    universe_filter: str = 'main_board_non_st'
    n_envs: int = 4
    n_factors: int = 0  # 0 = auto
    top_k: int = 30
    forward_period: int = 10
    cost_bps: float = 30.0
    learning_rate: float = 0.0003
    target_kl: float = 0.05
    n_steps: int = 0  # 0 = 默认
    batch_size: int = 0  # 0 = 默认
    seed: int = 42
    out_dir: str = ''
    policy_kwargs_json: str = ''
    resume_from: str = ''

    model_config = {'protected_namespaces': ()}

@router.post('/api/train/start')
def start_train(req: TrainRequest):
    global _train_process, _train_stop_flag, _train_task_id, _train_runtime_path

    if _train_process and _train_process.is_alive():
        raise HTTPException(400, '已有训练任务运行中')

    # 参数修正
    panel_name = resolve_panel(req.panel, 'hs300')
    if not panel_name:
        raise HTTPException(400, '未找到任何因子面板，请先在 P3 构建')
    req.panel = panel_name
    panel_path = str(AURUMQ_ROOT / 'data' / panel_name)
    if req.out_dir:
        out_dir = req.out_dir if str(req.out_dir).startswith('models/') else f"models/{Path(req.out_dir).name}"
    else:
        # 生成描述性命名: ppo_{pool}_{steps}_{network}_{date}
        pool = req.panel.replace('factor_panel_', '').replace('.parquet', '').split('_')[0]
        steps_label = f'{req.total_timesteps // 1_000_000}M' if req.total_timesteps >= 1_000_000 else f'{req.total_timesteps // 1000}K'
        net = ''
        if req.policy_kwargs_json:
            try:
                arch = json.loads(req.policy_kwargs_json).get('net_arch', [])
                net = '_'.join(str(x) for x in arch) if arch else ''
            except Exception:
                pass
        date_str = time.strftime('%m%d')
        name_parts = ['ppo', pool, steps_label]
        if net:
            name_parts.append(net)
        name_parts.append(date_str)
        out_dir = 'models/' + '_'.join(name_parts)

    task_id = f"train_{Path(out_dir).name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    runtime_dir = AURUMQ_ROOT / 'data' / 'task_runtime'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _train_task_id = task_id
    _train_runtime_path = runtime_dir / f"{task_id}.json"
    _train_runtime_path.write_text(json.dumps({
        'version': 'training', 'task_id': task_id, 'status': 'starting',
        'running': False, 'panel': req.panel, 'out_dir': out_dir,
        'total_timesteps': req.total_timesteps, 'algorithm': req.algorithm,
        'started_at': time.time(),
    }, ensure_ascii=False, indent=2))
    _train_stop_flag.clear()

    def run():
        try:
            from train_runner import add_log_callback, run_training
            add_log_callback(log_callback)

            log(f'🚀 训练启动: {req.algorithm} | {req.total_timesteps:,} 步 | {req.panel}', source='train', task_id=task_id, event='start')
            if _train_runtime_path:
                state = json.loads(_train_runtime_path.read_text())
                state.update({'status': 'running', 'running': True, 'started_at': state.get('started_at', time.time())})
                _train_runtime_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))

            kwargs = {
                'panel_path': panel_path,
                'algorithm': req.algorithm,
                'total_timesteps': req.total_timesteps,
                'start_date': req.start_date,
                'end_date': req.end_date,
                'universe_filter': req.universe_filter,
                'n_envs': req.n_envs,
                'n_factors': req.n_factors if req.n_factors > 0 else None,
                'top_k': req.top_k,
                'forward_period': req.forward_period,
                'cost_bps': req.cost_bps,
                'learning_rate': req.learning_rate,
                'target_kl': req.target_kl if req.target_kl > 0 else None,
                'n_steps': req.n_steps if req.n_steps > 0 else None,
                'batch_size': req.batch_size if req.batch_size > 0 else None,
                'seed': req.seed,
                'out_dir': out_dir,
                'stop_flag': _train_stop_flag,
                'policy_kwargs_json': req.policy_kwargs_json or None,
                'resume_from': req.resume_from or None,
            }
            stats = run_training(**kwargs)
            if _train_runtime_path:
                state = json.loads(_train_runtime_path.read_text())
                state.update({
                    'status': 'done' if stats.get('status') == 'ok' else ('stopped' if stats.get('status') == 'cancelled' else 'failed'),
                    'running': False, 'returncode': stats.get('exit_code'),
                    'finished_at': time.time(), 'progress': stats.get('status'),
                    'artifact_paths': [str(Path(out_dir) / name) for name in ('ppo_final.zip', 'policy.onnx', 'metadata.json', 'training_summary.json') if (Path(out_dir) / name).is_file()],
                })
                _train_runtime_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            if stats.get('status') == 'ok':
                log(f'✅ 训练完成: {out_dir}', source='train', task_id=task_id, event='finish')
            elif stats.get('status') == 'cancelled':
                log(f'⏹ 训练已取消', source='train', task_id=task_id, event='cancel')
            else:
                log(f'⚠ 训练异常: {stats.get("status")}', source='train', task_id=task_id, event='error', level='WARN')
        except Exception as e:
            if _train_runtime_path:
                state = json.loads(_train_runtime_path.read_text()) if _train_runtime_path.exists() else {}
                state.update({'status': 'failed', 'running': False, 'returncode': -1, 'finished_at': time.time(), 'error': str(e)})
                _train_runtime_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            log(f'❌ 训练失败: {str(e)}', source='train', task_id=task_id, event='error', level='ERROR')
            import traceback
            log(traceback.format_exc()[-500:])
        finally:
            log('--- 任务结束 ---')

    t = threading.Thread(target=run, daemon=True)
    t._panel = req.panel          # P2: 存储训练参数给 status API
    t._algo = req.algorithm
    t._out_dir = out_dir
    t._total_steps = req.total_timesteps
    _train_process = t
    t.start()

    return {'status': 'started', 'version': 'training', 'task_id': task_id, 'algorithm': req.algorithm, 'out_dir': out_dir}

@router.post('/api/train/stop')
def stop_train():
    _train_stop_flag.set()
    return {'status': 'stopping'}

@router.get('/api/train/status')
def train_status():
    state = {
        'running': _train_process is not None and _train_process.is_alive(),
        'task_id': _train_task_id or None,
        'version': 'training',
        'algorithm': getattr(_train_process, '_algo', None),
        'out_dir': getattr(_train_process, '_out_dir', None),
        'panel': getattr(_train_process, '_panel', None),
        'total_timesteps': getattr(_train_process, '_total_steps', None),
    }
    if _train_runtime_path and _train_runtime_path.exists():
        try:
            persisted = json.loads(_train_runtime_path.read_text())
            state.update(persisted)
            state['running'] = bool(_train_process is not None and _train_process.is_alive())
        except Exception:
            pass
    return state

@router.get('/api/train/panels')
def list_train_panels():
    """列出可用于训练的因子面板（含 v8 wavehunter 面板）"""
    data_dir = AURUMQ_ROOT / 'data'
    files = []
    for f in sorted(data_dir.glob('factor_panel_*.parquet')):
        stat = f.stat()
        files.append({
            'name': f.name,
            'path': str(f),
            'size_gb': round(stat.st_size / 1024**3, 3),
        })
    # 也包含 wavehunter v8 面板
    for f in sorted(data_dir.glob('wavehunter_v8_*.parquet')):
        stat = f.stat()
        files.append({
            'name': f.name,
            'path': str(f),
            'size_gb': round(stat.st_size / 1024**3, 3),
        })
    return {'panels': files}

@router.get('/api/train/models')
def list_train_models():
    """列出已训练模型（含日期、大小、股票数、ONNX/指标）"""
    models_dir = AURUMQ_ROOT / 'models'
    if not models_dir.exists():
        return {'models': []}
    models = []
    for d in sorted(models_dir.iterdir()):
        if not d.is_dir():
            continue
        onnx = list(d.glob('*.onnx'))
        metrics_files = list(d.glob('*metrics*.jsonl'))
        summary = d / 'training_summary.json'
        final_zip = d / 'ppo_final.zip'
        # Checkpoints 单独算
        ckpt_dir = d / 'checkpoints'
        ckpt_size = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file()) if ckpt_dir.exists() else 0
        has_ckpt = ckpt_dir.exists() and ckpt_size > 0
        # 完整大小（含 checkpoint）
        total_size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
        # 股票数 / 步数 / 时间
        n_stocks = None
        total_steps = None
        train_date = None
        if summary.exists():
            try:
                s = json.loads(summary.read_text())
                n_stocks = s.get('n_stocks')
                total_steps = s.get('total_timesteps')
            except Exception:
                pass
        # 训练时间（目录 mtime）
        mtime = d.stat().st_mtime
        models.append({
            'name': d.name,
            'path': str(d),
            'has_onnx': len(onnx) > 0,
            'has_metrics': len(metrics_files) > 0,
            'has_training_summary': summary.exists(),
            'has_tensorboard': any(d.glob('tb_logs/**/events.out.tfevents.*')),
            'has_ledger': any((Path(str(AURUMQ_ROOT)) / 'data' / 'ledgers').glob(f'*{d.name}*.json')),
            'has_final_zip': final_zip.exists(),
            'has_checkpoints': has_ckpt,
            'checkpoints_size_mb': round(ckpt_size / 1024**2, 1),
            'size_mb': round(total_size / 1024**2, 1),
            'size_no_ckpt_mb': round((total_size - ckpt_size) / 1024**2, 1),
            'n_stocks': n_stocks,
            'total_timesteps': total_steps,
            'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)),
        })
    # 最新在前
    models.sort(key=lambda x: x['modified'], reverse=True)
    return {'models': models}


@router.delete('/api/train/models/{name}')
def delete_train_model(name: str):
    """删除已训练模型目录（路径校验防越界）"""
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail='非法名称')
    target = (AURUMQ_ROOT / 'models' / name).resolve()
    models_dir = (AURUMQ_ROOT / 'models').resolve()
    if not str(target).startswith(str(models_dir)):
        raise HTTPException(status_code=400, detail='路径越界')
    if not target.exists():
        raise HTTPException(status_code=404, detail='模型不存在')
    # 算大小
    total_size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
    size_mb = round(total_size / 1024**2, 2)
    import shutil
    shutil.rmtree(target)
    return {'status': 'deleted', 'name': name, 'size_mb': size_mb}


# ── 清理 Checkpoints ──
@router.post('/api/train/models/{name}/clean-checkpoints')
def clean_checkpoints(name: str):
    """删除模型目录下的 checkpoints，保留 ppo_final.zip / onnx / 指标"""
    # P1 fix: regex 统一为单反斜杠检查（与 delete model 一致）
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail='非法名称')
    target = (AURUMQ_ROOT / 'models' / name).resolve()
    models_dir = (AURUMQ_ROOT / 'models').resolve()
    if not str(target).startswith(str(models_dir)):
        raise HTTPException(status_code=400, detail='路径越界')
    if not target.exists():
        raise HTTPException(status_code=404, detail='模型不存在')
    ckpt_dir = target / 'checkpoints'
    if not ckpt_dir.exists():
        return {'status': 'already_clean', 'name': name}
    # 算大小
    size_before = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())
    import shutil
    shutil.rmtree(ckpt_dir)
    return {
        'status': 'cleaned',
        'name': name,
        'cleaned_mb': round(size_before / 1024**2, 1),
    }


# ── GPU 监控 ──
@router.get('/api/gpu/stats')
def get_gpu_stats():
    """实时 GPU + 系统（CPU/内存）状态"""
    import subprocess
    import psutil
    result = {'available': True, 'cpu': {}, 'memory': {}}
    # 系统
    try:
        result['cpu']['pct'] = psutil.cpu_percent(interval=None)
        freq = psutil.cpu_freq()
        if freq:
            result['cpu']['freq_current'] = round(freq.current)
            result['cpu']['freq_max'] = round(freq.max)
        mem = psutil.virtual_memory()
        result['memory']['total_gb'] = round(mem.total / 1024**3, 1)
        result['memory']['used_gb'] = round(mem.used / 1024**3, 1)
        result['memory']['pct'] = mem.percent
    except Exception as e:
        result['memory'] = {'error': str(e)}
    # GPU
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory',
             '--format=csv,noheader,nounits'],
            timeout=3, text=True
        )
        parts = out.strip().split(', ')
        result['gpu'] = {
            'index': int(parts[0]),
            'name': parts[1],
            'memory_total_mb': int(float(parts[2])),
            'memory_used_mb': int(float(parts[3])),
            'gpu_util_pct': float(parts[4]),
            'memory_util_pct': float(parts[5]),
            'temp_c': int(parts[6]),
            'power_w': float(parts[7]),
            'clock_graphics_mhz': int(parts[8]),
            'clock_memory_mhz': int(parts[9]),
            'memory_free_mb': int(float(parts[2])) - int(float(parts[3])),
        }
    except Exception as e:
        result['gpu'] = {'error': str(e)}
    return result


# ── 训练图表 ──
@router.get('/api/train/metrics/{name}')
def get_train_metrics(name: str):
    """返回指定模型 training_metrics.jsonl 数据（兼容 models/ 旧路径）"""
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail='非法名称')
    target = (AURUMQ_ROOT / 'models' / name / 'training_metrics.jsonl')
    if not target.exists():
        raise HTTPException(status_code=404, detail='指标文件不存在')
    lines = []
    for line in target.read_text().splitlines():
        if line.strip():
            lines.append(json.loads(line))
    return {'metrics': lines, 'model': name, 'count': len(lines)}


def _read_jsonl_metrics(target: Path) -> list[dict]:
    rows = []
    if not target.exists():
        return rows
    for line in target.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@router.get('/api/train/metrics-detail/{source}/{name}')
def get_train_metrics_detail(source: str, name: str):
    """训练图表统一数据：指标、超参数、metadata、GPU日志和进度。

    source=models|runs；仅读训练产物，不触发训练或模拟。
    """
    if source not in ('models', 'runs') or '..' in name or '/' in name or '\\' in name:
        raise HTTPException(400, '非法训练产物名称')
    root = AURUMQ_ROOT / 'models' / name
    if not root.is_dir():
        # 某些历史训练误写为 models/models/name，保留读取兼容。
        nested = AURUMQ_ROOT / 'models' / 'models' / name
        if nested.is_dir():
            root = nested
        else:
            raise HTTPException(404, '训练产物不存在')
    metrics = _read_jsonl_metrics(root / 'training_metrics.jsonl')
    gpu = _read_jsonl_metrics(root / 'gpu.jsonl')
    metadata = {}
    for fn in ('metadata.json', 'training_summary.json'):
        p = root / fn
        if p.exists():
            try:
                metadata[fn[:-5]] = json.loads(p.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                metadata[fn[:-5]] = {}
    # WaveHunter v2 使用 Stable-Baselines3 TensorBoard event 文件，不一定生成
    # training_metrics.jsonl；使用内置轻量 reader，避免强依赖 TensorBoard。
    if not metrics:
        try:
            event_files = list((root / 'tb_logs').rglob('events.out.tfevents.*'))
            if event_files:
                try:
                    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                    acc = EventAccumulator(str(event_files[-1])).Reload()
                    by_step = {}
                    for tag in acc.Tags().get('scalars', []):
                        for ev in acc.Scalars(tag):
                            by_step.setdefault(ev.step, {}).setdefault('extra', {})[tag] = ev.value
                            by_step[ev.step]['timestep'] = ev.step
                    metrics = [by_step[k] for k in sorted(by_step)]
                except ImportError:
                    from tensorboard_scalars import read as read_tensorboard_scalars
                    metrics = read_tensorboard_scalars(event_files[-1])
        except (ImportError, OSError, ValueError, struct.error):
            pass
    return {'source': source, 'name': name, 'metrics': metrics, 'gpu': gpu,
            'metadata': metadata, 'count': len(metrics)}


# ── 合并模型列表（models/ + runs/）──
def _list_models_in(dir_path: Path, source: str) -> list:
    """列出单个目录下的所有训练模型（统一格式，含 source 字段）

    过滤规则（runs/ 目录）：
      - pipeline/  保留（Path5 最终结果，用户可见）
      - p22c_*/    保留（Phase 22C 主模型）
      - intermediate/ 过滤掉（Path1-4 中间产物）
      - sl_path*/  过滤掉（Path1/2/4 ensemble 产物）
      - path*-*_*/ 过滤掉（Path1/2/3 训练产物）
    """
    items = []
    if not dir_path.exists():
        return items
    for d in sorted(dir_path.iterdir()):
        if not d.is_dir():
            continue
        # 过滤 runs/ 下的中间产物
        if source == 'runs':
            name = d.name
            if name.startswith('intermediate'):
                continue
            if name.startswith('sl_path'):
                continue
            if name.startswith('path1-') or name.startswith('path2-') or name.startswith('path3-') or name.startswith('path4-'):
                continue
            if name.startswith('path1_') or name.startswith('path2_') or name.startswith('path3_') or name.startswith('path4_'):
                continue
        onnx = list(d.glob('*.onnx'))
        # hybrid 目录: stage1_knn_lstm / stage2_ppo 子目录里有 onnx
        if not onnx and d.name.startswith('hybrid'):
            stage2 = d / 'stage2_ppo'
            if (stage2 / 'policy.onnx').exists():
                onnx = [stage2 / 'policy.onnx']
        metrics_files = list(d.glob('*metrics*.jsonl'))
        summary = d / 'training_summary.json'
        meta = d / 'metadata.json'
        final_zip = d / 'ppo_final.zip'
        ckpt_dir = d / 'checkpoints'
        ckpt_size = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file()) if ckpt_dir.exists() else 0
        has_ckpt = ckpt_dir.exists() and ckpt_size > 0
        total_size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())

        info = {
            'source': source,           # 'models' | 'runs'
            'path': str(d),             # 完整绝对路径（sim 直接用）
            'name': d.name,
            'display_name': ('🧬 ' + d.name) if d.name.startswith('hybrid') else d.name,
            'has_onnx': len(onnx) > 0,
            'has_metrics': len(metrics_files) > 0,
            'has_training_summary': summary.exists(),
            'has_metadata': meta.exists(),
            'has_tensorboard': any(d.glob('tb_logs/**/events.out.tfevents.*')),
            'has_ledger': any((Path(str(AURUMQ_ROOT)) / 'data' / 'ledgers').glob(f'*{d.name}*.json')),
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

        # runs 专有字段
        if source == 'runs':
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
        items.append(info)
    return items


@router.get('/api/all/models')
def list_all_models():
    """合并列出 models/ 和 runs/ 下所有可推理的模型（带 source 标记）

    用于图表 Tab 和模拟交易 Tab 的下拉选择器。
    Value 格式：<source>:<name>，例如 'models:ppo_hs300_...' 或 'runs:p22c_...'
    """
    # 所有新训练模型统一写入 models/；runs/ 仅保留历史兼容读取。
    # 所有新模型统一在 models/；兼容旧 API 名称但不再读取 runs/。
    models = _list_models_in(AURUMQ_ROOT / 'models', 'models')
    runs = []
    # runs 加 🌊 前缀方便区分
    for r in runs:
        r['display_name'] = f"[🌊] {r['name']}"
    all_items = models + runs
    all_items.sort(key=lambda x: x.get('modified', ''), reverse=True)
    return {'models': all_items}


@router.get('/api/train/metrics-runs/{name}')
def get_train_metrics_runs(name: str):
    """返回 runs/ 目录下指定 run 的 training_metrics.jsonl（Phase 22C）"""
    if '..' in name or '/' in name or '\\' in name:
        raise HTTPException(status_code=400, detail='非法名称')
    target = (AURUMQ_ROOT / 'models' / name / 'training_metrics.jsonl')
    if not target.exists():
        raise HTTPException(status_code=404, detail='指标文件不存在')
    lines = []
    for line in target.read_text().splitlines():
        if line.strip():
            lines.append(json.loads(line))
    return {'metrics': lines, 'model': name, 'count': len(lines), 'source': 'runs'}
