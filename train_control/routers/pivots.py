"""routers/pivots.py — 高低点标注校验 API"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import polars as pl
router = APIRouter(tags=["pivots"])

from core.config import AURUMQ_ROOT, DATA_DIR
_V9_NAME = "wavehunter_v9_hs300_20040102_20260804.parquet"
_V8_NAME = "wavehunter_v8_hs300_20040102_20260804.parquet"

_PANEL_NAME_RE = re.compile(r'^wavehunter[A-Za-z0-9._\-]+\.parquet$')
_SCAN_COLS = (
    "v9_zig_peak", "v9_zig_valley", "v9_a1_point", "v9_a2_interval", "v9_a2_start",
)
_V10_SCAN_COLS = (
    "v10_zig_peak", "v10_zig_valley", "v10_a1_point", "v10_a2_interval", "v10_a2_start",
)
_V101_SCAN_COLS = (
    "v10_1_zig_peak", "v10_1_zig_valley", "v10_1_a1_point", "v10_1_a2_interval", "v10_1_a2_start",
)
# v8 主升浪阶段标签（仅 v8 面板有；v9 面板可能也保留兼容字段）
_OLD_PIVOT_COLS = ("a1_reversal_start", "a2_start")
_OHLC_COLS = ("trade_date", "ts_code", "open", "high", "low", "close", "amount", "pct_chg")


def _resolve_panel_path(panel: str | None) -> Path:
    """面板路径解析：仅接受 data/ 目录下的 wavehunter_*.parquet，避免任意文件读取。"""
    if panel:
        if not _PANEL_NAME_RE.match(panel) or ".." in panel or "/" in panel or "\\" in panel:
            raise HTTPException(400, f"非法面板名: {panel}")
        candidate = DATA_DIR / panel
        if not candidate.exists():
            raise HTTPException(404, f"面板不存在: {panel}")
        return candidate
    for name in (_V9_NAME, _V8_NAME):
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    raise HTTPException(500, "未找到 v9/v8 面板")


def _read_schema(p: Path) -> list[str]:
    """仅读 parquet schema，避免读取数据。"""
    return pl.read_parquet(str(p), n_rows=0).columns


def _select_pivot_cols(cols: list[str]) -> list[str]:
    """按面板版本选取实际存在的高低点/阶段标签列。"""
    if any(c.startswith("v10_1_") for c in cols):
        return [c for c in _V101_SCAN_COLS if c in cols]
    if any(c.startswith("v10_") for c in cols):
        return [c for c in _V10_SCAN_COLS if c in cols]
    return [c for c in _SCAN_COLS if c in cols]


def _pivot_prefix(cols: list[str]) -> str:
    if any(c.startswith("v10_1_") for c in cols):
        return "v10_1_"
    return "v10_" if any(c.startswith("v10_") for c in cols) else "v9_"


def _select_old_pivot_cols(cols: list[str]) -> list[str]:
    """从 schema 选取实际存在的 v8 pivot 标签列。"""
    return [c for c in _OLD_PIVOT_COLS if c in cols]


@router.get("/api/pivots/panels")
def pivot_panels():
    """返回可用面板列表（wavehunter_*）"""
    panels = []
    for p in sorted(DATA_DIR.glob("wavehunter_*.parquet")):
        try:
            cols = _read_schema(p)
            has_v9 = "v9_a1_point" in cols
            has_v8 = "a1_reversal_start" in cols
            panels.append({"name": p.name, "size_mb": round(p.stat().st_size/1024**2, 1),
                           "has_v9": has_v9, "has_v8": has_v8, "cols": len(cols)})
        except Exception:
            panels.append({"name": p.name, "size_mb": 0, "has_v9": False, "has_v8": False, "cols": 0})
    default = ""
    for name in (_V9_NAME, _V8_NAME):
        candidate = DATA_DIR / name
        if candidate.exists():
            default = name
            break
    if not default and panels:
        default = panels[0]["name"]
    return {"panels": panels, "default": default}


@router.get("/api/pivots/stocks")
def pivot_stocks(panel: str | None = Query(None, description="面板文件名"),
                limit: int = Query(50, ge=1, le=2000),
                q: str = Query("", description="ts_code 模糊搜索")):
    """返回股票列表（按有标注数量排序优先）"""
    p = _resolve_panel_path(panel)
    schema = _read_schema(p)
    codes = sorted(pl.scan_parquet(str(p)).select(["ts_code"]).unique().collect()["ts_code"].to_list())
    if q:
        ql = q.lower()
        codes = [c for c in codes if ql in c.lower()]
    pivot_cols = [c for c in ("v10_1_a1_point", "v10_1_zig_peak", "v10_1_zig_valley") if c in schema]
    if not pivot_cols:
        pivot_cols = [c for c in ("v10_a1_point", "v10_zig_peak", "v10_zig_valley") if c in schema]
    if not pivot_cols:
        pivot_cols = [c for c in ("v9_a1_point", "v9_zig_peak", "v9_zig_valley") if c in schema]
    if pivot_cols and codes:
        try:
            label_counts = (pl.scan_parquet(str(p))
                            .select(pivot_cols + ["ts_code"])
                            .with_columns([(pl.col(c) == 1).cast(pl.Int8) for c in pivot_cols])
                            .group_by("ts_code")
                            .agg([pl.col(c).sum().alias(c) for c in pivot_cols])
                            .collect())
            ranked = label_counts.sort(pivot_cols, descending=[True] * len(pivot_cols))["ts_code"].to_list()
            codes = [c for c in ranked if c in codes] + [c for c in codes if c not in ranked]
        except Exception:
            pass
    return {"stocks": codes[:limit], "total": len(codes), "panel": p.name}


@router.get("/api/pivots/data")
def pivot_data(
    stock: str = Query(..., description="ts_code e.g. 600000.SH"),
    panel: str | None = Query(None, description="面板文件名"),
    limit: int = Query(500, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    p = _resolve_panel_path(panel)
    schema = _read_schema(p)
    select_cols = [c for c in _OHLC_COLS if c in schema]
    pivot_cols = _select_pivot_cols(schema)
    old_pivot_cols = _select_old_pivot_cols(schema)
    version = _pivot_prefix(schema)
    df = (pl.scan_parquet(str(p))
            .filter(pl.col("ts_code") == stock)
            .select(select_cols + pivot_cols + old_pivot_cols)
            .rename({c: f"__pivot_{c}" for c in pivot_cols + old_pivot_cols})
            .sort("trade_date")
            .collect())
    if len(df) == 0:
        raise HTTPException(404, f"未找到股票 {stock}")
    total = len(df)
    if offset == 0:
        df = df.tail(limit)
    else:
        start = max(0, total - offset - limit)
        end = total - offset
        df = df.slice(start, end - start)
    dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
             for d in df["trade_date"].to_list()]
    ohlc = []
    for o, h, l, c in zip(df["open"].to_list(), df["high"].to_list(),
                          df["low"].to_list(), df["close"].to_list()):
        ohlc.append([round(float(o or 0), 2), round(float(c or 0), 2),
                     round(float(l or 0), 2), round(float(h or 0), 2)])

    def _idx(col):
        col = f"__pivot_{col}"
        if col not in df.columns:
            return []
        idxs = [i for i, v in enumerate(df[col].to_list()) if v == 1]
        return [{"idx": i, "date": dates[i], "price": ohlc[i][1]} for i in idxs]

    def _interval(col):
        col = f"__pivot_{col}"
        if col not in df.columns:
            return []
        return [1 if v == 1 else 0 for v in df[col].to_list()]

    out = {
        "stock": stock,
        "panel": p.name,
        "total": total,
        "returned": len(df),
        "offset": offset,
        "dates": dates,
        "ohlc": ohlc,
        "amount": [round(float(v or 0), 2) for v in df["amount"].to_list()] if "amount" in df.columns else [],
        "pct_chg": [round(float(v or 0), 2) for v in df["pct_chg"].to_list()] if "pct_chg" in df.columns else [],
        "peaks": _idx(f"{version}zig_peak"),
        "valleys": _idx(f"{version}zig_valley"),
        "a1": _idx(f"{version}a1_point"),
        "a2_start": _idx(f"{version}a2_start"),
        "a2_interval": _interval(f"{version}a2_interval"),
        "old_a1": _idx("a1_reversal_start"),
        "old_a2": _idx("a2_start"),
    }
    return out


@router.get("/api/pivots/stats")
def pivot_stats(panel: str | None = Query(None, description="面板文件名")):
    p = _resolve_panel_path(panel)
    schema = _read_schema(p)
    select_cols = _select_pivot_cols(schema)
    if not select_cols:
        return {"panel": p.name, "total_rows": 0, "note": "panel 不包含 pivot 标签列"}
    df = pl.scan_parquet(str(p)).select(select_cols).collect()
    total = len(df)
    out = {"panel": p.name, "total_rows": total}
    for c in df.columns:
        out[c] = int((df[c] == 1).sum())
        out[c + "_pct"] = round(int((df[c] == 1).sum()) / total * 100, 3) if total else 0
    return out
