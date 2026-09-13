"""
routers/ml_joint.py — ML Path1-5 / 联合训练 / KNN LSTM API
"""
import os, sys, time, json, asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .common import AURUMQ_ROOT, PYTHON_EXE, log, list_model_dirs

router = APIRouter(tags=['ml_joint'])

# ═══════════════════════════════════════════════════════════════
# ML Path1-5 API
# ═══════════════════════════════════════════════════════════════

import asyncio, subprocess

_ml_train_state = {'running': False, 'proc': None, 'last_rc': None, 'last_path': None}

# Reuse AURUMQ_ROOT as project root
PROJECT_DIR = AURUMQ_ROOT
SCRIPTS_DIR = AURUMQ_ROOT / 'scripts'
DATA_DIR = AURUMQ_ROOT / 'data'

class MLTrainRequest(BaseModel):
    path: str  # path1/path2/path3/path4/path5
    bundle: str  # P3 bundle name
    train_start: str = '2018-01-02'
    train_end: str = '2024-12-31'
    valid_start: str = '2025-01-02'
    valid_end: str = '2025-06-30'
    test_start: str = '2025-07-01'
    test_end: str = '2026-07-24'
    top_k: int = 5
    seed: int = 42
    # Path1 (LightGBM)
    num_leaves: int = 63
    learning_rate: float = 0.03
    min_data_in_leaf: int = 31
    num_boost_round: int = 2000
    feat_fraction: float = 0.7
    grid_search: bool = True
    gpu: bool = False  # Use GPU for LightGBM
    # Path2 (CatBoost)
    iterations: int = 2000
    depth: int = 8
    l2_leaf_reg: float = 3.0
    # Path3 (TabNet)
    n_d: int = 64
    n_steps: int = 5
    lambda_sparse: float = 0.001
    epochs: int = 200
    # Path4 (Rank-Z)
    forward_period: int = 10
    method: str = 'rank_z'
    # Path5 (Regime)
    n_regimes: int = 3
    regime_method: str = 'hmm'
    path1_dir: str = ''


def _build_ml_cmd(req: MLTrainRequest, bundle_path: str) -> list[str]:
    """根据 path 类型构建训练命令"""
    base = ["/usr/bin/python3"]
    # 自动生成 out 目录：path1-1, path1-2, path2-1 ...
    # 所有新训练产物统一落在 models/；旧 runs/ 仅作为历史兼容读取目录。
    runs_dir = PROJECT_DIR / 'models'
    prefix = f'{req.path}-'
    existing = [d.name for d in runs_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix) and req.bundle in d.name]
    run_num = len(existing) + 1
    ts = time.strftime('%Y%m%d_%H%M')
    out_base = f'models/{prefix}{run_num}_{req.bundle}_{ts}'

    if req.path == 'path1':
        if req.grid_search:
            # grid_search=True → path1_grid.py（12配置×3seed=36跑）
            # path1_train.py 不支持 --grid-search，只有 path1_grid.py 是真正的网格搜索入口
            script = SCRIPTS_DIR / 'p3' / 'path1_grid.py'
            base += [str(script),
                     '--bundle', bundle_path,
                     '--out-root', out_base,
                     '--num-iterations', str(req.num_boost_round)]
            if req.train_start:
                base += ['--train-start', req.train_start]
            if req.train_end:
                base += ['--train-end', req.train_end]
        else:
            script = SCRIPTS_DIR / 'p3' / 'path1_train.py'
            base += [str(script),
                     '--bundle', bundle_path,
                     '--out', out_base,
                     '--seed', str(req.seed),
                     '--num-leaves', str(req.num_leaves),
                     '--learning-rate', str(req.learning_rate),
                     '--min-data-in-leaf', str(req.min_data_in_leaf),
                     '--num-iterations', str(req.num_boost_round),
                     '--feature-fraction', str(req.feat_fraction)]
            if req.gpu:
                base.append('--gpu')
            if req.train_start:
                base += ['--train-start', req.train_start]
            if req.train_end:
                base += ['--train-end', req.train_end]
    elif req.path == 'path2':
        script = SCRIPTS_DIR / 'p3' / 'path2_train_catboost.py'
        base += [str(script),
                 '--bundle', bundle_path,
                 '--out', out_base,
                 '--seed', str(req.seed),
                 '--depth', str(req.depth),
                 '--learning-rate', str(req.learning_rate),
                 '--num-iterations', str(req.iterations),
                 '--l2-leaf-reg', str(req.l2_leaf_reg)]
        if req.gpu:
            base.append('--gpu')
        if req.train_start:
            base += ['--train-start', req.train_start]
        if req.train_end:
            base += ['--train-end', req.train_end]
    elif req.path == 'path3':
        script = SCRIPTS_DIR / 'p3' / 'path3_train_tabnet.py'
        base += [str(script),
                 '--bundle', bundle_path,
                 '--out', out_base,
                 '--seed', str(req.seed),
                 '--n-d', str(req.n_d),
                 '--n-steps', str(req.n_steps),
                 '--gamma', str(req.lambda_sparse),
                 '--max-epochs', str(req.epochs)]
    elif req.path == 'path4':
        script = SCRIPTS_DIR / 'p3' / 'path4_features.py'
        base += [str(script),
                 '--bundle', bundle_path,
                 '--out', out_base]
    elif req.path == 'path5':
        # path5_regime_stacking.py 只支持 --bundle/--paths/--top-k-configs/--out/--runs-root，
        # 不支持 --n-regimes/--method/--path1-dir（那些是 path5_regime_features.py 的入参，bundle 已预生成）
        script = SCRIPTS_DIR / 'p3' / 'path5_regime_stacking.py'
        base += [str(script),
                 '--bundle', bundle_path,
                 '--out', out_base,
                 '--runs-root', str(PROJECT_DIR / 'models')]
        if req.path1_dir:
            base += ['--paths'] + [p.strip() for p in req.path1_dir.split(',') if p.strip()]
    else:
        raise ValueError(f'Unknown path: {req.path}')

    return base


@router.post('/api/ml/train')
async def ml_train_start(req: MLTrainRequest):
    if _ml_train_state.get('running'):
        return {'error': '已有训练在运行中，请先停止'}

    bundle_path = str(DATA_DIR / req.bundle)
    if not Path(bundle_path).exists():
        return {'error': f'数据包 {req.bundle} 不存在'}

    cmd = _build_ml_cmd(req, bundle_path)
    task_id = f"ml_{req.path}_{time.strftime('%Y%m%d_%H%M%S')}"
    log(f'🤖 启动 {req.path}', source='ml', task_id=task_id, event='start')

    async def _run():
        _ml_train_state['running'] = True
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_DIR), start_new_session=True)
            _ml_train_state['proc'] = proc
            if proc.stdout:
                async for line in proc.stdout:
                    text = line.decode(errors='replace').rstrip('\n\r')
                    if text.strip():
                        log(text, source='ml', task_id=task_id, event='progress')
            await proc.wait()
            rc = proc.returncode if proc.returncode is not None else -1
            if rc == 0:
                log(f'✅ {req.path} 训练完成 (exit=0)', source='ml', task_id=task_id, event='finish')
            elif rc < 0:
                log(f'⏹ {req.path} 训练被终止 (signal={-rc})', source='ml', task_id=task_id, event='cancel')
            else:
                log(f'❌ {req.path} 训练失败 (exit={rc})', source='ml', task_id=task_id, event='error', level='ERROR')
            _ml_train_state['last_rc'] = rc
            _ml_train_state['last_path'] = req.path
        except Exception as e:
            log(f'❌ {req.path} 训练异常: {e}')
            _ml_train_state['last_rc'] = -1
            _ml_train_state['last_path'] = req.path
        finally:
            _ml_train_state.update(running=False, proc=None)

    asyncio.get_event_loop().create_task(_run())
    return {'status': 'started', 'path': req.path, 'bundle': req.bundle}


@router.post('/api/ml/train/stop')
async def ml_train_stop():
    proc = _ml_train_state.get('proc')
    if proc and proc.returncode is None:
        proc.kill()
        log('⏹ ML 训练已停止')
    _ml_train_state.update(running=False, proc=None)
    return {'message': '已停止'}


@router.get('/api/ml/train/status')
def ml_train_status():
    """返回 ML 训练当前状态"""
    running = _ml_train_state.get('running', False)
    return {'running': running,
            'last_rc': _ml_train_state.get('last_rc'),
            'last_path': _ml_train_state.get('last_path')}


@router.get('/api/ml/models')
def ml_models_list():
    """列出 models/ 目录下的 ML 训练产物"""
    runs_dir = PROJECT_DIR / 'models'
    models = []
    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            # 只匹配 Path1-5 模型目录（path1_N 或 path1-N 格式）
            if not any(d.name.startswith(f'path{i}-') or d.name.startswith(f'path{i}_') for i in range(1, 6)):
                continue
            has_model = any(d.glob('*.pkl')) or any(d.glob('*.cbm')) or any(d.glob('*.pth')) or any(d.glob('*.json')) or any(d.glob('*.txt')) or any(d.glob('*.npz'))
            if not has_model:
                continue
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            models.append({
                'name': d.name,
                'path': str(d.relative_to(PROJECT_DIR)),
                'size_mb': size / (1024*1024),
            })
    return {'models': models}


@router.delete('/api/ml/models/{name}')
def ml_model_delete(name: str):
    import shutil
    target = PROJECT_DIR / 'models' / name
    if not target.exists() or not target.is_dir():
        raise HTTPException(404, '模型不存在')
    shutil.rmtree(target)
    return {'status': 'deleted', 'name': name}


# ════════════════════════════════════════════════════════════
# KNN + LSTM 融合模型 API
# ════════════════════════════════════════════════════════════

def resolve_panel(panel: str, prefer_pool: str = 'hs300') -> str:
    """解析因子面板路径。

    若请求的 panel 文件不存在，自动回退到指定股票池的最新面板
    （与训练 Tab 的动态面板加载一致，避免硬编码日期过期）。
    返回相对于 data/ 的文件名。
    支持 factor_panel_* 和 wavehunter_v8_* 两种面板。
    """
    if panel:
        p = Path(DATA_DIR / panel)
        if p.exists():
            return panel
    # 面板不存在或未指定 → 找 prefer_pool 最新面板
    candidates = sorted(DATA_DIR.glob(f'factor_panel_{prefer_pool}_*.parquet'))
    if not candidates:
        # 也找 wavehunter v8 面板
        candidates = sorted(DATA_DIR.glob(f'wavehunter_v8_{prefer_pool}_*.parquet'))
    if not candidates:
        candidates = sorted(DATA_DIR.glob(f'factor_panel_*.parquet'))
    if not candidates:
        candidates = sorted(DATA_DIR.glob(f'wavehunter_v8_*.parquet'))
    if not candidates:
        return ''
    # 名字含日期，倒序取最新（factor_panel_{pool}_{start}_{end}.parquet）
    return candidates[-1].name


class KnnLstmRequest(BaseModel):
    panel: str = ''
    out_dir: str = ''
    window: int = 20
    forward_period: int = 1
    k_neighbors: int = 20
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_lr: float = 0.001
    epochs: int = 50
    batch_size: int = 2048
    num_workers: int = 3
    max_samples_per_stock: int = 2000
    lstm_weight: float = 0.6
    knn_weight: float = 0.4
    confidence: float = 0.55
    train_end: str = '2024-12-31'
    test_start: str = '2025-01-01'
    device: str = 'cuda'
    seed: int = 42
    model_config = {'protected_namespaces': ()}


class IncrementalKnnLstmRequest(BaseModel):
    """滚动分段训练请求"""
    panel: str = ''
    out_dir: str = ''
    segments: str = '2004-2006,2007-2008,2009-2011,2012-2014,2015-2017,2018-2020,2021-2023'
    resume_at: int = 0
    window: int = 20
    forward_period: int = 1
    k_neighbors: int = 20
    lstm_hidden: int = 384
    lstm_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_lr: float = 0.001
    epochs: int = 15
    batch_size: int = 2048
    num_workers: int = 3
    max_samples_per_stock: int = 2000
    lstm_weight: float = 0.6
    knn_weight: float = 0.4
    confidence: float = 0.55
    test_start: str = '2024-01-01'
    device: str = 'cuda'
    seed: int = 42
    model_config = {'protected_namespaces': ()}


_knn_lstm_state: dict = {}


class JointTrainRequest(BaseModel):
    """端到端联合训练请求 — LSTM+PPO 一个网络直接输出仓位权重（选股+择时一体）"""
    panel: str = ''
    out_dir: str = ''
    start_date: str = '2024-01-01'
    end_date: str = '2024-06-30'
    total_timesteps: int = 2_000_000
    top_k: int = 20
    forward_period: int = 10
    window: int = 10
    lstm_hidden: int = 128
    lstm_layers: int = 1
    n_envs: int = 1
    learning_rate: float = 3e-4
    # 补充参数（默认值对齐 train.py）
    n_factors: int | None = None          # None = 自动检测全部列
    cost_bps: float = 30.0
    max_position_pct: float = 0.05
    max_industry_pct: float = 0.30
    reward_type: str = 'return'
    seed: int = 42
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    batch_size: int | None = None         # None = SB3 默认 64
    n_steps: int = 128                     # ⚠️ 关键: 默认128防OOM (SB3默认2048必爆显存)
    n_epochs: int | None = None            # None = SB3 默认 10
    feature_group_weights_json: str = ''   # 特征分组加权 JSON，如 {"roe":10,"roa":10}（精确列名/前缀均可）
    use_knn: bool = False                  # KNN+LSTM+PPO 融合：KNN 特征通道拼入 LSTM 输入
    knn_k: int = 20                        # KNN 邻居数
    knn_ref_lookback: int = 240            # KNN 参考集回看天数
    knn_max_ref_days: int = 12             # KNN 采样天数上限（实测 12 天最优）
    rebalance_days: int = 20               # 调仓周期（天），必须与 SimConfig 一致
    model_config = {'protected_namespaces': ()}


_joint_state: dict = {}


@router.post('/api/joint/train')
async def joint_train_start(req: JointTrainRequest):
    if _joint_state.get('running'):
        return {'error': '已有联合训练在运行中'}

    panel_name = resolve_panel(req.panel, 'hs300')
    if not panel_name:
        return {'error': '未找到任何因子面板，请先构建'}
    req.panel = panel_name
    panel_path = str(DATA_DIR / panel_name)
    if not Path(panel_path).exists():
        return {'error': f'面板文件不存在: {req.panel}'}

    out_dir = req.out_dir or f'models/joint_{time.strftime("%Y%m%d_%H%M")}'
    if not str(out_dir).startswith('models/'):
        out_dir = f"models/{Path(out_dir).name}"
    task_id = str(out_dir)

    # ── 安全 clamp（防 OOM，RTX 3060 12GB 实测边界）──
    if req.n_steps > 128:
        log(f'⚠️ n_steps={req.n_steps} 超出安全边界(≤128)，强制降为 128（防 GPU OOM）')
        req.n_steps = 128
    if req.batch_size and req.batch_size > 64:
        log(f'⚠️ batch_size={req.batch_size} 超出安全边界(≤64)，强制降为 64（防 GPU OOM）')
        req.batch_size = 64
    if req.lstm_hidden > 128:
        log(f'⚠️ lstm_hidden={req.lstm_hidden} 超出安全边界(≤128)，强制降为 128（防 GPU OOM）')
        req.lstm_hidden = 128
    if req.n_envs != 1:
        log(f'⚠️ n_envs={req.n_envs} 强制为 1（DummyVecEnv 防 cgroup OOM）')
        req.n_envs = 1

    cmd = [
        "/usr/bin/python3",
        str(SCRIPTS_DIR / 'train.py'),
        '--data-path', panel_path,
        '--out-dir', out_dir,
        '--start-date', req.start_date,
        '--end-date', req.end_date,
        '--env-type', 'portfolio_weight_lstm',
        '--total-timesteps', str(req.total_timesteps),
        '--top-k', str(req.top_k),
        '--forward-period', str(req.forward_period),
        '--window', str(req.window),
        '--lstm-hidden', str(req.lstm_hidden),
        '--lstm-layers', str(req.lstm_layers),
        '--n-envs', str(req.n_envs),
        '--learning-rate', str(req.learning_rate),
        '--cost-bps', str(req.cost_bps),
        '--max-position-pct', str(req.max_position_pct),
        '--max-industry-pct', str(req.max_industry_pct),
        '--reward-type', req.reward_type,
        '--seed', str(req.seed),
        '--max-grad-norm', str(req.max_grad_norm),
        # ⚠️ n_steps 必须显式传：SB3 默认 2048 → batch=2048×282 → GPU OOM
        '--n-steps', str(req.n_steps),
    ]
    if req.n_factors:
        cmd += ['--n-factors', str(req.n_factors)]
    if req.target_kl:
        cmd += ['--target-kl', str(req.target_kl)]
    if req.batch_size:
        cmd += ['--batch-size', str(req.batch_size)]
    if req.n_epochs:
        cmd += ['--n-epochs', str(req.n_epochs)]
    if req.feature_group_weights_json.strip():
        cmd += ['--feature-group-weights-json', req.feature_group_weights_json.strip()]
    if req.use_knn:
        cmd += ['--use-knn', '--knn-k', str(req.knn_k), '--knn-ref-lookback', str(req.knn_ref_lookback),
                '--knn-max-ref-days', str(req.knn_max_ref_days)]
        log(f'🧬 KNN+LSTM+PPO 融合模式: k={req.knn_k}, ref_lookback={req.knn_ref_lookback}, max_ref_days={req.knn_max_ref_days}')
    cmd += ['--rebalance-days', str(req.rebalance_days)]

    # 打印完整 cmd 到日志（诊断用）
    log(f'📋 CMD: {" ".join(cmd)}')

    log(f'🚀 联合训练启动: {req.panel} (LSTM+PPO, window={req.window}, {req.total_timesteps//1000000}M步)')

    async def _run():
        _joint_state['running'] = True
        _joint_state['out_dir'] = out_dir
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_DIR), start_new_session=True)
            _joint_state['proc'] = proc
            if proc.stdout:
                async for line in proc.stdout:
                    text = line.decode(errors='replace').rstrip('\n\r')
                    if text.strip():
                        log(text, source='joint', task_id=task_id, event='progress')
            await proc.wait()
            rc = proc.returncode
            log(f'🚀 联合训练完成 (exit={rc})', source='joint', task_id=task_id, event='finish' if rc == 0 else 'error', level='INFO' if rc == 0 else 'ERROR')
        except Exception as e:
            log(f'❌ 联合训练异常: {e}', source='joint', task_id=task_id, event='error', level='ERROR')
        finally:
            _joint_state.update(running=False, proc=None)

    asyncio.get_event_loop().create_task(_run())
    return {'status': 'started', 'out_dir': out_dir}


@router.post('/api/joint/train/stop')
async def joint_train_stop():
    proc = _joint_state.get('proc')
    if proc:
        proc.kill()
        _joint_state.update(running=False, proc=None)
        log('⏹ 联合训练已停止')
        return {'status': 'stopped'}
    return {'status': 'idle'}


@router.get('/api/joint/train/status')
def joint_train_status():
    return {'running': _joint_state.get('running', False)}


@router.get('/api/joint/models')
def joint_models_list():
    """统一列出所有 KNN+LSTM+PPO 联合模型。

    联合训练可能写入 models/ 或 runs/，不同 Tab 必须共享这一个扫描结果。
    识别条件：metadata.env_type=portfolio_weight_lstm，或联合模型目录前缀。
    """
    models = []
    seen = set()
    running_out_dir = _joint_state.get('out_dir', '')
    prefixes = ('joint_', 'knn_lstm', 'real_test_joint', 'real_test_knn_lstm', 'whv2_', 'whv3_', 'matrix_')
    for root in (PROJECT_DIR / 'models',):
        if not root.exists():
            continue
        for d in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir() or d.resolve() in seen:
                continue
            meta_file = d / 'metadata.json'
            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding='utf-8'))
                except Exception:
                    meta = {}
            is_joint = meta.get('env_type') in ('portfolio_weight_lstm', 'wavehunter_lstm') or d.name.startswith(prefixes)
            has_trace = any((d / x).exists() for x in ('training_metrics.jsonl', 'gpu.jsonl'))
            if not is_joint or (not meta and not has_trace):
                continue
            seen.add(d.resolve())
            onnx_files = list(d.glob('*.onnx'))
            zip_files = list(d.glob('*_final.zip')) + list(d.glob('*.zip'))
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            mtime = d.stat().st_mtime
            status = 'running' if (_joint_state.get('running') and d.name == Path(running_out_dir).name) else ('done' if zip_files else 'interrupted')
            metrics = meta.get('metrics', {})
            models.append({
                'name': d.name,
                'path': str(d.relative_to(PROJECT_DIR)),
                'source': 'models' if d.parent.name == 'models' else 'runs',
                'size_mb': round(size / (1024 * 1024), 1),
                'has_onnx': bool(onnx_files),
                'has_model': bool(onnx_files or zip_files or (d / 'lstm_best.pt').exists()),
                'has_zip': bool(zip_files),
                'has_final_zip': bool(zip_files),
                'running': status == 'running',
                'status': status,
                'meta': meta,
                'config': meta,
                'metrics': metrics,
                'lstm_only': meta.get('lstm_only', {}),
                'knn_only': meta.get('knn_only', {}),
                'modified': time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)),
            })
    models.sort(key=lambda x: x['modified'], reverse=True)
    return {'models': models}


@router.delete('/api/joint/models/{name}')
def joint_model_delete(name: str):
    import shutil
    target = PROJECT_DIR / 'models' / name
    if not target.exists() or not target.is_dir():
        raise HTTPException(404, '模型不存在')
    shutil.rmtree(target)
    return {'status': 'deleted', 'name': name}


# ════════════════════════════════════════════════════════════════
# KNN+LSTM+PPO 融合 (kj) — 复用 joint 后端逻辑 + KNN 参数
# ════════════════════════════════════════════════════════════════
@router.post('/api/kj/train')
async def kj_train_start(req: JointTrainRequest):
    # 强制开启 KNN 融合（kj Tab 语义）
    req.use_knn = True
    log('🚀 KNN+LSTM+PPO联合训练启动', source='kj', task_id=req.out_dir or 'kj_train', event='start')
    return await joint_train_start(req)


@router.post('/api/kj/train/stop')
async def kj_train_stop():
    return await joint_train_stop()


@router.get('/api/kj/train/status')
def kj_train_status():
    return joint_train_status()


@router.get('/api/kj/models')
def kj_models_list():
    return joint_models_list()


@router.delete('/api/kj/models/{name}')
def kj_model_delete(name: str):
    return joint_model_delete(name)


@router.post('/api/knn_lstm/incremental/train')
async def knn_lstm_incremental_train_start(req: IncrementalKnnLstmRequest):
    """滚动分段训练（20 年数据逐段训练 + 续训）"""
    if _knn_lstm_state.get('running'):
        return {'error': '已有训练在运行中'}

    panel_name = resolve_panel(req.panel, 'hs300')
    if not panel_name:
        return {'error': '未找到任何因子面板，请先在 P3 构建'}
    req.panel = panel_name
    panel_path = str(DATA_DIR / panel_name)
    if not Path(panel_path).exists():
        return {'error': f'面板文件不存在: {req.panel}'}

    out_dir = req.out_dir or f'models/knn_lstm_roll_{time.strftime("%Y%m%d_%H%M")}'
    if not str(out_dir).startswith('models/'):
        out_dir = f"models/{Path(out_dir).name}"

    cmd = [
        "/usr/bin/python3",
        str(SCRIPTS_DIR / 'incremental_train.py'),
        '--panel', panel_path,
        '--out', out_dir,
        '--segments', req.segments,
        '--resume-at', str(req.resume_at),
        '--test-start', req.test_start,
        '--window', str(req.window),
        '--forward-period', str(req.forward_period),
        '--k-neighbors', str(req.k_neighbors),
        '--lstm-hidden', str(req.lstm_hidden),
        '--lstm-layers', str(req.lstm_layers),
        '--lstm-dropout', str(req.lstm_dropout),
        '--lstm-lr', str(req.lstm_lr),
        '--epochs', str(req.epochs),
        '--batch-size', str(req.batch_size),
        '--num-workers', str(req.num_workers),
        '--max-samples-per-stock', str(req.max_samples_per_stock),
        '--lstm-weight', str(req.lstm_weight),
        '--knn-weight', str(req.knn_weight),
        '--confidence', str(req.confidence),
        '--device', req.device,
        '--seed', str(req.seed),
    ]

    log(f'🔬 KNN+LSTM 滚动分段训练启动: {req.segments}')

    async def _run():
        _knn_lstm_state['running'] = True
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_DIR), start_new_session=True)
            _knn_lstm_state['proc'] = proc
            if proc.stdout:
                async for line in proc.stdout:
                    text = line.decode(errors='replace').rstrip('\n\r')
                    if text.strip():
                        log(text)
            await proc.wait()
            rc = proc.returncode
            log(f'🔬 KNN+LSTM 滚动训练完成 (exit={rc})')
        except Exception as e:
            log(f'❌ KNN+LSTM 滚动训练异常: {e}')
        finally:
            _knn_lstm_state.update(running=False, proc=None)

    asyncio.get_event_loop().create_task(_run())
    return {'status': 'started', 'out_dir': out_dir}


@router.post('/api/knn_lstm/train')
async def knn_lstm_train_start(req: KnnLstmRequest):
    if _knn_lstm_state.get('running'):
        return {'error': '已有训练在运行中'}

    panel_name = resolve_panel(req.panel, 'hs300')
    if not panel_name:
        return {'error': '未找到任何因子面板，请先在 P3 构建'}
    req.panel = panel_name
    panel_path = str(DATA_DIR / panel_name)
    if not Path(panel_path).exists():
        return {'error': f'面板文件不存在: {req.panel}'}

    out_dir = req.out_dir or f'models/knn_lstm_{time.strftime("%Y%m%d_%H%M")}'
    if not str(out_dir).startswith('models/'):
        out_dir = f"models/{Path(out_dir).name}"

    cmd = [
        "/usr/bin/python3",
        str(SCRIPTS_DIR / 'knn_lstm_train.py'),
        '--panel', panel_path,
        '--out', out_dir,
        '--window', str(req.window),
        '--forward-period', str(req.forward_period),
        '--k-neighbors', str(req.k_neighbors),
        '--lstm-hidden', str(req.lstm_hidden),
        '--lstm-layers', str(req.lstm_layers),
        '--lstm-dropout', str(req.lstm_dropout),
        '--lstm-lr', str(req.lstm_lr),
        '--epochs', str(req.epochs),
        '--batch-size', str(req.batch_size),
        '--num-workers', str(req.num_workers),
        '--max-samples-per-stock', str(req.max_samples_per_stock),
        '--lstm-weight', str(req.lstm_weight),
        '--knn-weight', str(req.knn_weight),
        '--confidence', str(req.confidence),
        '--train-end', req.train_end,
        '--test-start', req.test_start,
        '--device', req.device,
        '--seed', str(req.seed),
    ]

    log(f'🔬 KNN+LSTM 训练启动: {req.panel}')

    async def _run():
        _knn_lstm_state['running'] = True
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                cwd=str(PROJECT_DIR), start_new_session=True)
            _knn_lstm_state['proc'] = proc
            if proc.stdout:
                async for line in proc.stdout:
                    text = line.decode(errors='replace').rstrip('\n\r')
                    if text.strip():
                        log(text)
            await proc.wait()
            rc = proc.returncode
            log(f'🔬 KNN+LSTM 训练完成 (exit={rc})')
        except Exception as e:
            log(f'❌ KNN+LSTM 训练异常: {e}')
        finally:
            _knn_lstm_state.update(running=False, proc=None)

    asyncio.get_event_loop().create_task(_run())
    return {'status': 'started', 'out_dir': out_dir}


@router.post('/api/knn_lstm/train/stop')
async def knn_lstm_train_stop():
    proc = _knn_lstm_state.get('proc')
    if proc and proc.returncode is None:
        proc.kill()
        log('⏹ KNN+LSTM 训练已停止')
    _knn_lstm_state.update(running=False, proc=None)
    return {'message': '已停止'}


@router.get('/api/knn_lstm/train/status')
def knn_lstm_train_status():
    return {'running': _knn_lstm_state.get('running', False)}


@router.get('/api/knn_lstm/models')
def knn_lstm_models_list():
    """列出 KNN+LSTM 训练产物"""
    models = []
    # 训练入口和列表统一使用 models/；历史 runs/ 不再作为模型目录。
    model_dirs = [PROJECT_DIR / 'models']
    seen = set()
    for models_dir in model_dirs:
      if models_dir.exists():
        for d in sorted(models_dir.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            if not (d.name.startswith('knn_lstm') or d.name.startswith('real_test_knn_lstm')):
                continue
            if d.resolve() in seen:
                continue
            seen.add(d.resolve())
            meta_file = d / 'metadata.json'
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                meta = {}
            has_model = (d / 'lstm_best.pt').exists() or (d / 'lstm_model.onnx').exists()
            size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            models.append({
                'name': d.name,
                'path': str(d.relative_to(PROJECT_DIR)),
                'size_mb': round(size / (1024 * 1024), 1),
                'has_model': has_model,
                'metrics': meta.get('metrics', {}),
                'lstm_only': meta.get('lstm_only', {}),
                'knn_only': meta.get('knn_only', {}),
                'config': {k: v for k, v in meta.items() if k not in ('metrics', 'lstm_only', 'knn_only', 'factor_names')},
            })
    return {'models': models}


@router.delete('/api/knn_lstm/models/{name}')
def knn_lstm_model_delete(name: str):
    import shutil
    candidates = [PROJECT_DIR / 'models' / name]
    target = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if target is None:
        raise HTTPException(404, '模型不存在')
    shutil.rmtree(target)
    return {'status': 'deleted', 'name': name}


# ── 前端 ──
