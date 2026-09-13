"""
train_control/core/logging.py — 日志单一真源

统一事件日志器初始化 + SSE 队列推送 + 回调。
routers/common.py 通过 re-export 保持兼容。
"""
from __future__ import annotations

import queue
import time
from pathlib import Path

# 由 app.py 注入的全局状态（与 routers/common.py 共享）
_log_queue: queue.Queue = queue.Queue()
_subscribers: list = []
_emit_event = None
_recent_events = None


def init_event_logger() -> None:
    """初始化事件日志器（委托 event_logger）。"""
    global _emit_event, _recent_events
    try:
        from event_logger import configure, emit, recent  # type: ignore

        # AURUMQ_ROOT 来自 core.config
        from .config import AURUMQ_ROOT

        configure(AURUMQ_ROOT)
        _emit_event = emit
        _recent_events = recent
    except Exception:
        pass


def log(
    msg: str,
    source: str = "train",
    task_id: str = "",
    event: str = "log",
    level: str = "INFO",
    progress: dict | None = None,
) -> None:
    """日志 → JSONL + 队列 + SSE 订阅者。"""
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "msg": msg,
        "source": source,
        "task_id": task_id,
        "event": event,
        "level": level,
    }
    if progress is not None:
        entry["progress"] = progress
    if _emit_event is not None:
        try:
            entry = _emit_event(
                msg, source=source, task_id=task_id, event=event, level=level, progress=progress
            )
        except Exception:
            pass
    _log_queue.put(entry)
    dead = []
    for q in _subscribers:
        try:
            q.put_nowait(entry)
        except Exception:
            dead.append(q)
    for q in dead:
        _subscribers.remove(q)


def log_callback(msg: str) -> None:
    """供 build_factor_panel 脚本调用的日志回调。"""
    log(msg, source="build", event="progress")
