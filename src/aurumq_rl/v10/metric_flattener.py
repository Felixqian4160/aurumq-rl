"""Flatten nested component state dicts into scalar reward-metrics keys.

The v10 reward monitoring pipeline persists one JSON line per checkpoint.
Component state such as ``HHIConcentrationPenalty.state({...})`` returns a
nested dict (``{'hhi': ..., 'effective_count': ..., ...}``).  Mixing these
into a top-level row naively stores a JSON object and silently skips the
scalar fields, which has caused three repeated monitoring gaps in this
project.  This module provides a deterministic, prefixed flattening that
always yields flat scalar entries.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


MAX_DEPTH = 4


def flatten_state(prefix: str, state: Mapping[str, Any] | None, allowed_keys: Iterable[str]) -> dict[str, float | int | str]:
    """Project a nested ``state`` mapping into a flat scalar dict.

    ``prefix`` is the namespace, e.g. ``"hhi"``.  Only keys listed in
    ``allowed_keys`` are kept.  Non-scalar leaves are skipped to guarantee
    the row stays JSON-serializable in a single line.
    """
    out: dict[str, float | int | str] = {}
    if state is None:
        return out
    for key in allowed_keys:
        value = _lookup(state, key, MAX_DEPTH)
        if isinstance(value, bool):
            out[f"{prefix}_{key}"] = int(value)
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (int, float, str)):
            out[f"{prefix}_{key}"] = value
    return out


def _lookup(obj: Any, key: str, depth: int) -> Any:
    if depth <= 0 or not isinstance(obj, Mapping):
        return None
    if key in obj:
        return obj[key]
    for value in obj.values():
        if isinstance(value, Mapping):
            found = _lookup(value, key, depth - 1)
            if found is not None:
                return found
    return None
