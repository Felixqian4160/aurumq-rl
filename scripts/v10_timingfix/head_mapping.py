"""Re-export the v10 head mapping from src so the training scripts see the
same HEAD_MAPPING object that signal_engine.py imports. Preflight performs an
identity check to guarantee no drift between the two surfaces.
"""
from __future__ import annotations

from aurumq_rl.v10.head_mapping import (
    HEAD_MAPPING,
    HeadSpec,
    active_heads,
    masked_heads,
    threshold_for,
)

__all__ = ["HEAD_MAPPING", "HeadSpec", "active_heads", "masked_heads", "threshold_for"]