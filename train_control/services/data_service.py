"""
train_control/services/data_service.py — 数据服务单一真源
统一 Tushare 限速、下载与缓存扫描，routers/data_mgmt.py 仅做参数校验与委托。
"""
from __future__ import annotations

import time
import threading
from pathlib import Path
from typing import Optional

from core.config import AURUMQ_ROOT

DATA_CACHE_DIR = AURUMQ_ROOT / "data_cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_FILE = AURUMQ_ROOT / ".qbot_token"

_TUSHARE_CALL_LOCK = threading.Lock()
_TUSHARE_LAST_CALL = 0.0
_TUSHARE_MIN_INTERVAL = 0.6
_TUSHARE_RATE_RETRIES = 3
_download_start: dict = {}
_update_stop_flag = False


def _get_pro():
    import os
    import tushare as ts
    for token_path in [TOKEN_FILE, Path.home() / ".tushare_token"]:
        if token_path.exists():
            ts.set_token(token_path.read_text().strip())
            break
    env_token = os.environ.get("TUSHARE_TOKEN", "")
    if env_token:
        ts.set_token(env_token)
    return ts.pro_api()


def tushare_call(method_name: str, **kwargs):
    """统一串行限速的 Tushare 调用（含 频率超限 重试）。"""
    global _TUSHARE_LAST_CALL
    pro = _get_pro()
    last_exc: Optional[Exception] = None
    for attempt in range(_TUSHARE_RATE_RETRIES + 1):
        with _TUSHARE_CALL_LOCK:
            wait = _TUSHARE_MIN_INTERVAL - (time.monotonic() - _TUSHARE_LAST_CALL)
            if wait > 0:
                time.sleep(wait)
            try:
                result = getattr(pro, method_name)(**kwargs)
                _TUSHARE_LAST_CALL = time.monotonic()
                return result
            except Exception as exc:
                last_exc = exc
                _TUSHARE_LAST_CALL = time.monotonic()
                msg = str(exc)
                if "频率超限" not in msg and "frequency" not in msg.lower():
                    raise
        if attempt >= _TUSHARE_RATE_RETRIES:
            assert last_exc is not None
            raise last_exc
        time.sleep(65)


def _cache_path(ts_code: str) -> Path:
    return DATA_CACHE_DIR / f"{ts_code.replace('.', '_')}_full.parquet"


def _daily_basic_path(ts_code: str) -> Path:
    return DATA_CACHE_DIR / f"{ts_code.replace('.', '_')}_daily_basic.parquet"


def _fina_indicator_path(ts_code: str) -> Path:
    return DATA_CACHE_DIR / f"{ts_code.replace('.', '_')}_fina_indicator.parquet"


def _get_pool_codes(pool: str) -> list:
    try:
        if pool == 'csi800':
            return sorted(set(_get_pool_codes('hs300')) | set(_get_pool_codes('csi500')))
        index_map = {'hs300': '000300.SH', 'csi500': '000905.SH', 'csi1000': '000852.SH', 'sse50': '000016.SH'}
        code = index_map.get(pool)
        if not code:
            return []
        df = tushare_call('index_weight', index_code=code)
        if df is None or df.empty:
            return []
        return sorted(df['con_code'].unique().tolist())
    except Exception as e:
        from core.logging import log
        log(f"获取股票池失败: {e}", source='data_mgmt')
        return []


def get_all_pool_codes() -> list:
    codes = set()
    for pool in ['hs300', 'csi500', 'csi1000']:
        codes.update(_get_pool_codes(pool))
    return sorted(codes)


def get_all_cached_codes() -> list:
    """已缓存 OHLCV 的全部股票（`*_full.parquet` 并集），动态全量。"""
    return sorted({p.stem.replace('_full', '').replace('_', '.') for p in DATA_CACHE_DIR.glob('*_full.parquet')})


def _download_single_ohlcv(ts_code: str, start: str, end: str) -> bool:
    try:
        df = tushare_call('daily', ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return False
        try:
            adj = tushare_call('adj_factor', ts_code=ts_code, start_date=start, end_date=end)
            if adj is not None and not adj.empty:
                df = df.merge(adj[['trade_date', 'adj_factor']], on='trade_date', how='left')
        except Exception:
            pass
        path = _cache_path(ts_code)
        if path.exists():
            import pandas as pd
            old = pd.read_parquet(path)
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['trade_date'], keep='last').sort_values('trade_date').reset_index(drop=True)
            combined.to_parquet(path, index=False)
        else:
            df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _download_single_daily_basic(ts_code: str, start: str, end: str) -> bool:
    try:
        df = tushare_call('daily_basic', ts_code=ts_code, start_date=start, end_date=end,
                          fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv')
        if df is None or df.empty:
            return False
        path = _daily_basic_path(ts_code)
        if path.exists():
            import pandas as pd
            old = pd.read_parquet(path)
            combined = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=['trade_date'], keep='last').sort_values('trade_date').reset_index(drop=True)
            combined.to_parquet(path, index=False)
        else:
            df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _download_single_fina_indicator(ts_code: str, start: str, end: str) -> bool:
    try:
        df = tushare_call('fina_indicator', ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return False
        path = _fina_indicator_path(ts_code)
        if path.exists():
            import pandas as pd
            old = pd.read_parquet(path)
            combined = pd.concat([old, df], ignore_index=True).drop_duplicates(subset=['end_date'], keep='last').sort_values('end_date').reset_index(drop=True)
            combined.to_parquet(path, index=False)
        else:
            df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def _has_adj_factor(path: Path) -> bool:
    """统一判断 OHLCV parquet 是否包含复权因子列。"""
    try:
        import pyarrow.parquet as pq
        return 'adj_factor' in pq.read_schema(path).names
    except Exception:
        return False


def cache_status() -> dict:
    files = sorted(DATA_CACHE_DIR.glob("*_full.parquet"))
    total = 0
    cached: list[dict] = []
    with_adj = 0
    for f in files:
        size = f.stat().st_size
        total += size
        has_adj = _has_adj_factor(f)
        with_adj += int(has_adj)
        code = f.stem.replace("_full", "").replace("_", ".")
        cached.append({"code": code, "size_mb": round(size / 1024 / 1024, 2), "has_adj": has_adj})
    return {
        "stocks": len(files), "with_adj": with_adj,
        "adj_pct": round(with_adj / len(files) * 100, 1) if files else 0,
        "total_size_mb": round(total / 1024 / 1024, 1), "cached": cached[:50]
    }


def daily_basic_status() -> dict:
    return {'stocks': len(list(DATA_CACHE_DIR.glob('*_daily_basic.parquet')))}


def fina_indicator_status() -> dict:
    return {'stocks': len(list(DATA_CACHE_DIR.glob('*_fina_indicator.parquet')))}


def adj_status() -> dict:
    files = list(DATA_CACHE_DIR.glob('*_full.parquet'))
    total = len(files)
    ok = sum(_has_adj_factor(f) for f in files)
    return {'total': total, 'with_adj': ok, 'pct': round(ok / total * 100, 1) if total else 0}


def verify_cache() -> dict:
    """四项新鲜度校验。"""
    from datetime import datetime
    import pandas as pd
    import pyarrow.parquet as pq
    today = datetime.now()
    today_str = today.strftime('%Y%m%d')
    tolerance_days = 5

    def _is_fresh(max_date_str: str, expected_str: str, tol: int) -> bool:
        try:
            d1 = datetime.strptime(max_date_str, '%Y%m%d')
            d2 = datetime.strptime(expected_str, '%Y%m%d')
            return (d2 - d1).days <= tol
        except Exception:
            return False

    def _scan_parquet_dates(pattern: str, date_col: str, tol: int):
        files = list(DATA_CACHE_DIR.glob(pattern))
        total = len(files); fresh = 0; stale = 0; stale_samples: list[str] = []
        for f in files:
            try:
                pf = pq.ParquetFile(f)
                if date_col not in pf.metadata.schema.names:
                    stale += 1; continue
                df = pd.read_parquet(f, columns=[date_col])
                if df.empty or date_col not in df.columns:
                    stale += 1; continue
                mx = df[date_col].astype(str).str.replace('-', '').str[:8].max()
                if _is_fresh(mx, today_str, tol):
                    fresh += 1
                else:
                    stale += 1
                    if len(stale_samples) < 5:
                        stale_samples.append(f'{f.stem}:{mx}')
            except Exception as e:
                stale += 1
                if len(stale_samples) < 5:
                    stale_samples.append(f'{f.name}: {e}')
        return {'total': total, 'fresh': fresh, 'stale': stale, 'fresh_pct': round(fresh/total*100,1) if total else 0, 'expected': today_str, 'tolerance_days': tol, 'stale_samples': stale_samples}

    ohlcv = _scan_parquet_dates('*_full.parquet', 'trade_date', tolerance_days)
    daily_basic = _scan_parquet_dates('*_daily_basic.parquet', 'trade_date', tolerance_days)
    fina = _scan_parquet_dates('*_fina_indicator.parquet', 'end_date', 120)
    files_full = list(DATA_CACHE_DIR.glob('*_full.parquet'))
    total_full = len(files_full); with_adj = 0; adj_fresh = 0
    for f in files_full:
        try:
            if _has_adj_factor(f):
                with_adj += 1
                try:
                    df = pd.read_parquet(f, columns=['trade_date'])
                    mx = df['trade_date'].astype(str).str.replace('-', '').str[:8].max()
                    if _is_fresh(mx, today_str, tolerance_days):
                        adj_fresh += 1
                except Exception:
                    pass
        except Exception:
            pass
    adj = {'total': total_full, 'with_adj': with_adj, 'pct': round(with_adj/total_full*100,1) if total_full else 0, 'fresh': adj_fresh, 'fresh_pct': round(adj_fresh/total_full*100,1) if total_full else 0, 'expected': today_str}
    issues: list[str] = []
    if ohlcv['total']==0: issues.append('无 OHLCV 缓存')
    elif ohlcv['fresh_pct']<90: issues.append(f"OHLCV 仅 {ohlcv['fresh']}/{ohlcv['total']} 新鲜到 {today_str}±{tolerance_days}天")
    if daily_basic['total']==0: issues.append('未下载 daily_basic')
    elif daily_basic['fresh_pct']<90: issues.append(f"daily_basic 仅 {daily_basic['fresh']}/{daily_basic['total']} 新鲜")
    if fina['total']==0: issues.append('未下载 fina_indicator')
    elif fina['fresh_pct']<80: issues.append(f"fina_indicator 仅 {fina['fresh']}/{fina['total']} 新鲜（季频120天）")
    if adj['pct']<90: issues.append(f"复权因子仅 {adj['with_adj']}/{adj['total']} ({adj['pct']}%)")
    status = 'ok' if not issues else 'warning'
    return {'total': ohlcv['total'], 'healthy': ohlcv['fresh'], 'issues': issues[:10], 'status': status, 'detail': {'ohlcv': ohlcv, 'daily_basic': daily_basic, 'fina_indicator': fina, 'adj': adj}}


def download_worker(task_id: str, codes: list, start: str, end: str, download_fn, data_type: str = 'ohlcv'):
    global _update_stop_flag
    done = 0; failed = 0; total = len(codes)
    from core.logging import log
    from core.config import BUILD_RUNTIME_DIR
    log(f"📥 开始下载 {data_type.upper()} | {total} 只 | {start}~{end}", source='data_mgmt', task_id=task_id, event='start')
    t0 = _download_start.get(task_id, time.time())
    for i, code in enumerate(codes):
        if _update_stop_flag:
            log(f"⏹ 下载已停止 ({done}/{total})", source='data_mgmt', task_id=task_id, event='stop'); break
        try:
            ok = download_fn(code, start, end)
            if ok: done += 1
            else: failed += 1
        except Exception:
            failed += 1
        pct = (i+1)/total; bar_len=30; filled=int(bar_len*pct); bar='█'*filled+'░'*(bar_len-filled)
        elapsed = time.time()-t0; speed=(i+1)/elapsed if elapsed else 0; eta=(total-i-1)/speed if speed else 0
        log(f"[{bar}] {pct*100:5.1f}% | {i+1}/{total} | ✅{done} ❌{failed} | {code} | {speed:.1f}/s | ETA {eta:.0f}s", source='data_mgmt', task_id=task_id, event='progress', progress={'pct': round(pct*100,1), 'done': done, 'failed': failed, 'total': total, 'current': code, 'speed': round(speed,1), 'eta': round(eta)})
        # 兼容部分轮询走 RUNTIME_DIR/*.json
        try:
            (BUILD_RUNTIME_DIR / f'{task_id}.json').write_text(__import__('json').dumps({'status': 'running' if i < total-1 else 'done', 'total': total, 'done': done, 'failed': failed, 'current': code, 'pct': round(pct*100,1)}))
        except Exception:
            pass
        time.sleep(0.1)
    _update_stop_flag = False
    log(f"✅ 下载完成 | {data_type.upper()} | {done}/{total} | {failed} 失败 | 耗时 {time.time()-t0:.0f}s", source='data_mgmt', task_id=task_id, event='done')
