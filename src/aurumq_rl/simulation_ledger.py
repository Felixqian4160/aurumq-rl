"""模拟交易账本管理模块。"""
from __future__ import annotations

import json
import datetime
from pathlib import Path

import numpy as np


_AURUMQ_ROOT = Path(__file__).resolve().parent.parent.parent
_LEDGER_DIR = _AURUMQ_ROOT / 'data' / 'ledgers'


def save_ledger(result: dict) -> str:
    """保存账本到 data/ledgers/，返回 ledger_id。"""
    _LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = Path(result.get('config', {}).get('model_dir', '')).name
    ledger_id = f"{ts}_{model_name}"
    path = _LEDGER_DIR / f"{ledger_id}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    return ledger_id


def list_ledgers() -> list:
    """列出所有已保存的账本。"""
    _LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledgers = []
    for f in sorted(_LEDGER_DIR.glob('*.json'), reverse=True):
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            m = data.get('metrics', {})
            cfg = data.get('config', {})
            ledgers.append({
                'id': f.stem,
                'file': f.name,
                'model': Path(cfg.get('model_dir', '')).name,
                'capital': cfg.get('initial_capital', 0),
                'start_date': cfg.get('start_date', ''),
                'end_date': cfg.get('end_date', ''),
                'stop_loss': cfg.get('stop_loss_pct', 0),
                'total_return': m.get('total_return', 0),
                'benchmark_return': m.get('benchmark_return', 0),
                'sharpe': m.get('sharpe_ratio', 0),
                'excess_return': m.get('excess_return', 0),
                'excess': m.get('excess_return', 0),
                'profit_factor': m.get('profit_factor', 0),
                'max_dd': m.get('max_drawdown', 0),
                'final_nav': m.get('final_nav', 0),
                'trades': m.get('total_trades', 0),
                'win_rate': m.get('win_rate', 0),
                'saved_at': f.stat().st_mtime,
            })
        except Exception:
            pass
    return ledgers


def load_ledger(ledger_id: str) -> dict:
    """加载指定账本。"""
    path = _LEDGER_DIR / f"{ledger_id}.json"
    if not path.exists():
        return {'status': 'not_found'}
    return json.loads(path.read_text(encoding='utf-8'))


def delete_ledger(ledger_id: str) -> bool:
    """删除指定账本。"""
    path = _LEDGER_DIR / f"{ledger_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
