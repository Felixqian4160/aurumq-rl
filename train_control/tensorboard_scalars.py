"""Minimal TensorBoard event reader for SB3 scalar charts.

Supports the TFRecord event format used by stable-baselines3 without importing
TensorBoard. It intentionally reads scalar summaries only.
"""
from __future__ import annotations

import struct
from pathlib import Path


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while pos < len(buf):
        b = buf[pos]; pos += 1
        value |= (b & 0x7f) << shift
        if not b & 0x80:
            return value, pos
        shift += 7
    raise ValueError("truncated varint")


def _skip_field(data: bytes, pos: int, wire: int) -> int:
    if wire == 0:
        _, return_pos = _varint(data, pos); return return_pos
    if wire == 1: return pos + 8
    if wire == 2:
        n, pos = _varint(data, pos); return pos + n
    if wire == 5: return pos + 4
    raise ValueError("unsupported protobuf wire type")


def _message(data: bytes):
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos); field, wire = key >> 3, key & 7
        if wire == 2:
            n, pos = _varint(data, pos); yield field, data[pos:pos+n]; pos += n
        elif wire == 0:
            value, pos = _varint(data, pos); yield field, value
        elif wire == 5:
            value = struct.unpack('<f', data[pos:pos+4])[0]; pos += 4; yield field, value
        elif wire == 1:
            pos += 8
        else:
            pos = _skip_field(data, pos, wire)


def _scalar(data: bytes) -> tuple[str, float] | None:
    tag = value = None
    for field, item in _message(data):
        if field == 1 and isinstance(item, bytes): tag = item.decode('utf-8', errors='replace')
        elif field == 2 and isinstance(item, (int, float)): value = float(item)
    return (tag, value) if tag is not None and value is not None else None


def read(path: Path) -> list[dict]:
    data = path.read_bytes(); pos = 0; rows = {}
    while pos + 12 <= len(data):
        length = struct.unpack('<Q', data[pos:pos+8])[0]; pos += 12
        if length > len(data) - pos:
            break
        payload = data[pos:pos+length]; pos += length
        # Event: step field 2, summary field 5; summary contains value field 2.
        step = 0; summary = []
        for field, item in _message(payload):
            if field == 2 and isinstance(item, int): step = item
            elif field == 5 and isinstance(item, bytes): summary.append(item)
        for value_msg in summary:
            for field, item in _message(value_msg):
                if field == 2 and isinstance(item, bytes):
                    scalar = _scalar(item)
                    if scalar:
                        tag, value = scalar
                        rows.setdefault(step, {}).setdefault('extra', {})[tag] = value
                        rows[step]['timestep'] = step
    return [rows[k] for k in sorted(rows)]
