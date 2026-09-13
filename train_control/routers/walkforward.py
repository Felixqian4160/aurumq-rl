"""
routers/walkforward.py — Walk-forward 验证 + 参数矩阵 API
"""
import os, sys, time, json, subprocess, threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .common import AURUMQ_ROOT, PYTHON_EXE, log, list_parquet_files, DATA_DIR, PROJECT_DIR

router = APIRouter(tags=['walkforward'])


# ═══════════════════════════════════════════════════════════════
# 市场状态识别 API
# ═══════════════════════════════════════════════════════════════

@router.get('/api/market/regime')
def get_market_regime():
    """返回市场状态数据 v2（6维度评分）"""
    import pandas as pd
    import numpy as np
    csv_path = PROJECT_DIR / 'data' / 'market_regime_v2.csv'
    if not csv_path.exists():
        return {'error': 'market_regime_v2.csv 不存在，请先运行 tmp_regime_v2.py'}
    df = pd.read_csv(csv_path)
    df['date'] = df['date'].astype(str).str[:10]

    def clean(vals):
        return [None if pd.isna(v) else round(v, 4) if isinstance(v, float) else v for v in vals]

    dates = df['date'].tolist()
    closes = clean(df['close'])
    scores = clean(df['score']) if 'score' in df.columns else clean(df.get('total_score', pd.Series()))
    regimes = df['regime'].tolist()

    # 各维度得分
    dim_names = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6']
    dim_labels = {'d1': '均线排列', 'd2': 'ADX+RSI', 'd3': '量价配合', 'd4': '波动率', 'd5': '市场广度', 'd6': '动量'}
    dim_data = {}
    for name in dim_names:
        if name in df.columns:
            dim_data[name] = clean(df[name])

    # 连续区间（>=30天）
    df2 = df.copy()
    df2['chg'] = df2['regime'] != df2['regime'].shift(1)
    df2['grp'] = df2['chg'].cumsum()
    periods = []
    for _, gdf in df2.groupby('grp'):
        days = len(gdf)
        if days < 30:
            continue
        sd = gdf['date'].iloc[0]
        ed = gdf['date'].iloc[-1]
        r = gdf['regime'].iloc[0]
        chg_pct = round((gdf['close'].iloc[-1] / gdf['close'].iloc[0] - 1) * 100, 2)
        avg_sc = round(gdf['score'].mean(), 4) if 'score' in gdf.columns else 0
        periods.append({'start': sd, 'end': ed, 'days': int(days),
                        'regime': r, 'return': chg_pct, 'avg_score': avg_sc})

    # 训练方案建议
    train_plan = []
    for regime, emoji in [('bull', '🟢 牛市'), ('bear', '🔴 熊市'), ('oscillation', '🟡 震荡')]:
        bp = sorted([p for p in periods if p['regime'] == regime], key=lambda x: x['days'], reverse=True)[:2]
        if bp:
            train_plan.append({
                'regime': regime,
                'label': emoji,
                'train': [p['start'] + '~' + p['end'] for p in bp],
                'total_days': sum(p['days'] for p in bp),
            })

    return {
        'dates': dates, 'closes': closes, 'scores': scores, 'regimes': regimes,
        'dimensions': dim_data, 'dim_labels': dim_labels,
        'periods': periods, 'train_plan': train_plan,
    }


# ══════════════════════════════════════════════════
# Walk-forward 滚动验证
# ══════════════════════════════════════════════════
_wf_state: dict = {}
_wf_stop_flag = threading.Event()


class WalkForwardRequest(BaseModel):
    panel: str = ''
    steps: int = 100000
    train_years: int = 2
    val_years: int = 1
    # 模拟参数（-1 = 使用模型 metadata）
    top_k: int = -1
    cost_bps: float = -1.0
    slippage_bps: float = -1.0
    rebalance_days: int = -1
    stop_loss_pct: float = -1.0
    max_position_pct: float = -1.0
    max_holding_days: int = -1
    model_config = {'protected_namespaces': ()}


def _read_sim_metrics_from_ledger(ledger_path: Path) -> dict:
    """从 ledger 文件读取模拟指标（ground truth，避开 /api/sim/result 竞态条件）。"""
    try:
        d = json.loads(ledger_path.read_text(encoding='utf-8'))
        m = d.get('metrics', {})
        return {
            'sharpe': float(m.get('sharpe_ratio', 0) or 0),
            'excess': float(m.get('excess_return', 0) or 0),
            'max_dd': float(m.get('max_drawdown', 0) or 0),
            'total_return': float(m.get('total_return', 0) or 0),
        }
    except Exception:
        return {'sharpe': 0, 'excess': 0, 'max_dd': 0, 'total_return': 0}


def _run_sim_and_get_metrics(model_dir: str, panel_path: str, start: str, end: str,
                             sim_params: dict | None = None) -> dict:
    """同步跑模拟并读取 ledger 指标。sim_params: 前端传入的模拟参数，-1/None 表示用 metadata fallback。"""
    from aurumq_rl.simulation import SimConfig, run_simulation
    sp = sim_params or {}
    cfg = SimConfig(
        model_dir=model_dir,
        panel_path=panel_path,
        initial_capital=100000.0,
        start_date=start,
        end_date=end,
        top_k=sp.get('top_k', -1),
        cost_bps=sp.get('cost_bps', -1.0),
        slippage_bps=sp.get('slippage_bps', -1.0),
        rebalance_days=sp.get('rebalance_days', -1),
        stop_loss_pct=sp.get('stop_loss_pct', -1.0),
        max_position_pct=sp.get('max_position_pct', -1.0),
        max_holding_days=sp.get('max_holding_days', -1),
    )
    run_simulation(cfg, lambda msg: log(f'  [WF] {msg}', 'train'))
    # 读最新 ledger
    ledgers = sorted((DATA_DIR / 'ledgers').glob('*.json'), key=lambda p: p.stat().st_mtime)
    if not ledgers:
        return {'sharpe': 0, 'excess': 0, 'max_dd': 0, 'total_return': 0}
    return _read_sim_metrics_from_ledger(ledgers[-1])


def _wf_worker(req: WalkForwardRequest, panel_path: str):
    try:
        # 计算窗口：从面板日期推导
        import pandas as pd
        pdf = pd.read_parquet(panel_path, columns=['trade_date'])
        dates = sorted(pdf['trade_date'].astype(str).unique())
        if len(dates) < 250:
            _wf_state['error'] = '面板数据不足（<1年）'
            _wf_state['running'] = False
            return
        start_date = dates[0][:10]
        end_date = dates[-1][:10]
        start_y = int(start_date[:4])
        end_y = int(end_date[:4])

        train_days = req.train_years * 252
        val_days = req.val_years * 252
        windows = []
        cursor = start_y
        while cursor + req.train_years + req.val_years <= end_y + 1:
            tr_start = f'{cursor}-01-01'
            tr_end = f'{cursor + req.train_years}-12-31'
            va_start = f'{cursor + req.train_years}-01-01'
            va_end = f'{cursor + req.train_years + req.val_years - 1}-12-31'
            # 验证期不能超过面板末尾
            if va_start[:4] > end_date[:4]:
                break
            windows.append((tr_start, tr_end, va_start, va_end))
            cursor += req.val_years

        if not windows:
            _wf_state['error'] = '窗口计算失败'
            _wf_state['running'] = False
            return

        _wf_state['total_windows'] = len(windows)
        results = []
        for i, (tr_s, tr_e, va_s, va_e) in enumerate(windows):
            if _wf_stop_flag.is_set():
                log(f'⏹ Walk-forward 用户停止 (窗口 {i+1}/{len(windows)})')
                break
            _wf_state['current'] = i + 1
            log(f'📊 [WF {i+1}/{len(windows)}] 训练 {tr_s}~{tr_e} → 验证 {va_s}~{va_e}')

            out_dir = f'models/wf_{time.strftime("%Y%m%d_%H%M%S")}'
            cmd = [
                "/usr/bin/python3", str(SCRIPTS_DIR / 'train.py'),
                '--data-path', panel_path,
                '--out-dir', out_dir,
                '--start-date', tr_s,
                '--end-date', tr_e,
                '--env-type', 'portfolio_weight_lstm',
                '--total-timesteps', str(req.steps),
                '--top-k', '20',
                '--forward-period', '10',
                '--window', '10',
                '--lstm-hidden', '128',
                '--lstm-layers', '1',
                '--n-envs', '1',
                '--learning-rate', '0.0001',
                '--cost-bps', '15',
                '--max-position-pct', '0.02',
                '--max-industry-pct', '0.30',
                '--reward-type', 'return',
                '--seed', '42',
                '--max-grad-norm', '1.0',
                '--n-steps', '128',
                '--target-kl', '0.02',
                '--rebalance-days', '20',
                '--use-knn', '--knn-k', '20', '--knn-ref-lookback', '750', '--knn-max-ref-days', '12',
            ]
            log(f'  CMD: {" ".join(cmd[-12:])} ...')
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=7200)
            if r.returncode != 0:
                log(f'  ❌ 训练失败 (exit={r.returncode}): {r.stderr[-300:]}')
                results.append({'train_start': tr_s, 'train_end': tr_e, 'val_start': va_s, 'val_end': va_e,
                                'sharpe': 0, 'excess': 0, 'max_dd': 0, 'error': 'train_failed'})
                continue

            # 模拟
            model_full = str(PROJECT_DIR / out_dir)
            _sim_p = {
                'top_k': req.top_k, 'cost_bps': req.cost_bps,
                'slippage_bps': req.slippage_bps, 'rebalance_days': req.rebalance_days,
                'stop_loss_pct': req.stop_loss_pct, 'max_position_pct': req.max_position_pct,
                'max_holding_days': req.max_holding_days,
            }
            try:
                m = _run_sim_and_get_metrics(model_full, panel_path, va_s, va_e, sim_params=_sim_p)
                log(f'  ✅ 窗口完成: Sharpe={m["sharpe"]:.3f}, 超额={m["excess"]:.1f}%, 回撤={m["max_dd"]:.1f}%')
            except Exception as e:
                log(f'  ❌ 模拟失败: {e}')
                m = {'sharpe': 0, 'excess': 0, 'max_dd': 0, 'total_return': 0}
            results.append({
                'train_start': tr_s, 'train_end': tr_e, 'val_start': va_s, 'val_end': va_e,
                'sharpe': round(m['sharpe'], 4), 'excess': round(m['excess'], 2),
                'max_dd': round(m['max_dd'], 2), 'total_return': round(m['total_return'], 2),
            })

        _wf_state['results'] = results
        _wf_state['completed'] = not _wf_stop_flag.is_set()
        _wf_state['status'] = 'stopped' if _wf_stop_flag.is_set() else 'done'
        (DATA_DIR / 'walkforward_status.json').write_text(
            json.dumps({'status': _wf_state['status'], 'completed': _wf_state['completed'],
                        'windows': len(results)}, ensure_ascii=False), encoding='utf-8')
        # 落盘
        (DATA_DIR / 'walkforward_results.json').write_text(
            json.dumps({'windows': results}, ensure_ascii=False, indent=2), encoding='utf-8')
        log(f'📊 Walk-forward 全部完成: {len(results)} 窗口')
    except Exception as e:
        import traceback
        log(f'❌ Walk-forward 异常: {e}\n{traceback.format_exc()[-500:]}')
        _wf_state['error'] = str(e)
    finally:
        _wf_state['running'] = False
        _wf_stop_flag.clear()


@router.post('/api/walkforward/start')
def walkforward_start(req: WalkForwardRequest):
    if _wf_state.get('running'):
        return {'error': '已有 Walk-forward 在运行中'}
    panel_name = resolve_panel(req.panel, 'hs300')
    if not panel_name:
        return {'error': '未找到因子面板'}
    panel_path = str(DATA_DIR / panel_name)
    if not Path(panel_path).exists():
        return {'error': f'面板不存在: {panel_name}'}

    _wf_stop_flag.clear()
    _wf_state.update(running=True, completed=False, results=[], error='', total_windows=0, current=0)
    t = threading.Thread(target=_wf_worker, args=(req, panel_path), daemon=True)
    _wf_state['thread'] = t
    t.start()
    return {'status': 'started', 'total_windows': '计算中...'}


@router.get('/api/walkforward/status')
def walkforward_status():
    if not _wf_state.get('running'):
        status_file = DATA_DIR / 'walkforward_status.json'
        result_file = DATA_DIR / 'walkforward_results.json'
        if status_file.exists():
            try:
                persisted = json.loads(status_file.read_text())
                _wf_state.update(status=persisted.get('status', 'idle'),
                                 completed=persisted.get('completed', False))
            except Exception: pass
        if not _wf_state.get('results') and result_file.exists():
            try: _wf_state['results'] = json.loads(result_file.read_text()).get('windows', [])
            except Exception: pass
    running = _wf_state.get('running', False)
    total = _wf_state.get('total_windows', 0)
    current = _wf_state.get('current', 0)
    progress = f'窗口 {current}/{total}' if total and current else ''
    return {
        'running': running,
        'completed': _wf_state.get('completed', False),
        'status': _wf_state.get('status', 'idle'),
        'progress': progress,
        'error': _wf_state.get('error', ''),
    }


@router.get('/api/walkforward/result')
def walkforward_result():
    rows = _wf_state.get('results', [])
    if not rows:
        result_file = DATA_DIR / 'walkforward_results.json'
        if result_file.exists():
            try: rows = json.loads(result_file.read_text()).get('windows', [])
            except Exception: rows = []
    return {'windows': rows}


@router.post('/api/walkforward/stop')
def walkforward_stop():
    _wf_stop_flag.set()
    return {'status': 'stopping'}


# ══════════════════════════════════════════════════
# 参数矩阵扫描
# ══════════════════════════════════════════════════
_mtx_state: dict = {}
_mtx_stop_flag = threading.Event()


class MatrixRequest(BaseModel):
    panel: str = ''
    steps: int = 100000
    max_position_pcts: list = [0.01, 0.02, 0.05, 0.08]
    top_ks: list = [10, 20, 30, 50]
    rebalance_days_list: list = [10, 20, 40]
    lstm_hiddens: list = [64, 128]
    model_config = {'protected_namespaces': ()}


def _mtx_worker(req: MatrixRequest, panel_path: str):
    task_id = f"matrix_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        combos = []
        for mp in req.max_position_pcts:
            for tk in req.top_ks:
                for rb in req.rebalance_days_list:
                    for hd in req.lstm_hiddens:
                        combos.append({'max_position_pct': mp, 'top_k': tk, 'rebalance_days': rb, 'lstm_hidden': hd})
        _mtx_state['total'] = len(combos)
        task_id = f"matrix_{time.strftime('%Y%m%d_%H%M%S')}"
        log(f'🔬 参数矩阵启动: {len(combos)} 配置', source='matrix', task_id=task_id, event='start')
        results = []
        for i, c in enumerate(combos):
            if _mtx_stop_flag.is_set():
                log(f'⏹ 参数矩阵用户停止 (配置 {i+1}/{len(combos)})', source='matrix', task_id=task_id, event='cancel')
                break
            _mtx_state['current'] = i + 1
            log(f'🔬 [MTX {i+1}/{len(combos)}] mp={c["max_position_pct"]} topk={c["top_k"]} rb={c["rebalance_days"]} hidden={c["lstm_hidden"]}', source='matrix', task_id=task_id, event='progress', progress={'current': i+1, 'total': len(combos), 'percent': round((i+1)/len(combos)*100, 2)})

            out_dir = f'models/mtx_{time.strftime("%Y%m%d_%H%M%S")}'
            cmd = [
                "/usr/bin/python3", str(SCRIPTS_DIR / 'train.py'),
                '--data-path', panel_path,
                '--out-dir', out_dir,
                '--start-date', '2024-01-01',
                '--end-date', '2024-06-30',
                '--env-type', 'portfolio_weight_lstm',
                '--total-timesteps', str(req.steps),
                '--top-k', str(c['top_k']),
                '--forward-period', '10',
                '--window', '10',
                '--lstm-hidden', str(c['lstm_hidden']),
                '--lstm-layers', '1',
                '--n-envs', '1',
                '--learning-rate', '0.0001',
                '--cost-bps', '15',
                '--max-position-pct', str(c['max_position_pct']),
                '--max-industry-pct', '0.30',
                '--reward-type', 'return',
                '--seed', '42',
                '--max-grad-norm', '1.0',
                '--n-steps', '128',
                '--target-kl', '0.02',
                '--rebalance-days', str(c['rebalance_days']),
                '--use-knn', '--knn-k', '20', '--knn-ref-lookback', '750', '--knn-max-ref-days', '12',
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=7200)
            if r.returncode != 0:
                log(f'  ❌ 训练失败: {r.stderr[-200:]}')
                results.append({**c, 'sharpe': 0, 'excess': 0, 'max_dd': 0, 'error': 'train_failed'})
                continue

            model_full = str(PROJECT_DIR / out_dir)
            try:
                m = _run_sim_and_get_metrics(model_full, panel_path, '2024-07-01', '2026-08-04')
                log(f'  ✅ Sharpe={m["sharpe"]:.3f} 超额={m["excess"]:.1f}%')
            except Exception as e:
                log(f'  ❌ 模拟失败: {e}')
                m = {'sharpe': 0, 'excess': 0, 'max_dd': 0, 'total_return': 0}
            results.append({**c, 'sharpe': round(m['sharpe'], 4), 'excess': round(m['excess'], 2),
                            'max_dd': round(m['max_dd'], 2)})

        _mtx_state['results'] = results
        _mtx_state['completed'] = True
        (DATA_DIR / 'matrix_results.json').write_text(
            json.dumps({'results': results}, ensure_ascii=False, indent=2), encoding='utf-8')
        log(f'🔬 参数矩阵完成: {len(results)} 配置', source='matrix', task_id=task_id, event='finish')
    except Exception as e:
        import traceback
        log(f'❌ 参数矩阵异常: {e}\n{traceback.format_exc()[-500:]}', source='matrix', task_id=task_id, event='error', level='ERROR')
        _mtx_state['error'] = str(e)
    finally:
        _mtx_state['running'] = False
        _mtx_stop_flag.clear()


@router.post('/api/matrix/start')
def matrix_start(req: MatrixRequest):
    if _mtx_state.get('running'):
        return {'error': '已有参数矩阵在运行中'}
    panel_name = resolve_panel(req.panel, 'hs300')
    if not panel_name:
        return {'error': '未找到因子面板'}
    panel_path = str(DATA_DIR / panel_name)
    if not Path(panel_path).exists():
        return {'error': f'面板不存在: {panel_name}'}

    _mtx_stop_flag.clear()
    _mtx_state.update(running=True, completed=False, results=[], error='', total=0, current=0)
    t = threading.Thread(target=_mtx_worker, args=(req, panel_path), daemon=True)
    _mtx_state['thread'] = t
    t.start()
    return {'status': 'started', 'total': '计算中...'}


@router.get('/api/matrix/status')
def matrix_status():
    running = _mtx_state.get('running', False)
    total = _mtx_state.get('total', 0)
    current = _mtx_state.get('current', 0)
    progress = f'配置 {current}/{total}' if total and current else ''
    return {
        'running': running,
        'completed': _mtx_state.get('completed', False),
        'progress': progress,
        'error': _mtx_state.get('error', ''),
    }


@router.get('/api/matrix/result')
def matrix_result():
    return {'results': _mtx_state.get('results', [])}


@router.post('/api/matrix/stop')
def matrix_stop():
    _mtx_stop_flag.set()
    return {'status': 'stopping'}


# 保存主应用实例。子模块兼容直接运行模式，会导出名为 app 的符号；
