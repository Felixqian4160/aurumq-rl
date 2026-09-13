"""全局任务事件日志基础设施。

所有 Tab 通过 train_control.app.log() 进入这里：
- 内存队列：实时 SSE
- JSONL：跨进程/重启可追溯
- 统一字段：time、timestamp、source、task_id、event、level、msg、progress
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_CST = timezone(timedelta(hours=8))
_LOCK = threading.Lock()
_LOG_DIR: Path | None = None
_MAX_BYTES = 50 * 1024 * 1024


def configure(root: Path) -> None:
    global _LOG_DIR
    _LOG_DIR = root / "data" / "logs"
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _path() -> Path:
    if _LOG_DIR is None:
        raise RuntimeError("event_logger.configure() must be called first")
    return _LOG_DIR / "app_events.jsonl"


def emit(
    msg: str,
    *,
    source: str = "app",
    task_id: str = "",
    event: str = "log",
    level: str = "INFO",
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(_CST)
    entry: dict[str, Any] = {
        "time": now.strftime("%H:%M:%S"),
        "timestamp": now.isoformat(),
        "source": source,
        "task_id": task_id,
        "event": event,
        "level": level,
        "msg": str(msg),
    }
    if progress is not None:
        entry["progress"] = progress
    path = _path()
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _LOCK:
        if path.exists() and path.stat().st_size > _MAX_BYTES:
            backup = path.with_suffix(".jsonl.1")
            backup.unlink(missing_ok=True)
            path.replace(backup)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    return entry


def recent(limit: int = 200, source: str = "", task_id: str = "") -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with _LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()[-max(limit * 5, limit):]
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if source and row.get("source") != source:
            continue
        if task_id and row.get("task_id") != task_id:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    rows.reverse()
    return rows
