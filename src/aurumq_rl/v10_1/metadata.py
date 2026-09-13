"""Metadata/panel universe resolver for v10.1 simulation."""
from __future__ import annotations
import json
from pathlib import Path
import polars as pl


def load_metadata(model_dir: Path, panel_path: Path) -> dict:
    data=json.loads((model_dir/'metadata.json').read_text())
    if not data.get('stock_codes'):
        codes=pl.scan_parquet(str(panel_path)).select('ts_code').unique().sort('ts_code').collect()['ts_code'].to_list()
        data['stock_codes']=codes[:int(data.get('n_stocks',len(codes)))]
    if len(data['stock_codes']) != int(data.get('n_stocks',len(data['stock_codes']))):
        raise ValueError('metadata stock_codes/n_stocks mismatch')
    return data
