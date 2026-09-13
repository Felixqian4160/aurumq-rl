"""Single source of truth for head ↔ training target ↔ decision semantic.

Lives inside aurumq_rl.v10 so the simulation decision logic and the training
loop can both import the same canonical object. Preflight verifies identity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadSpec:
    name: str
    target: str | None
    decision: str | None
    threshold: float | None
    description: str


HEAD_MAPPING: dict[str, HeadSpec] = {
    "a1_head": HeadSpec(
        name="a1_head",
        target="v10_evt_valley",
        decision="buy",
        threshold=0.55,
        description="买入主信号：v10_evt_valley==1 概率高 → 买入",
    ),
    "a2_head": HeadSpec(
        name="a2_head",
        target=None,
        decision=None,
        threshold=None,
        description="保留 head（masked）：未分配训练目标",
    ),
    "peak_head": HeadSpec(
        name="peak_head",
        target="v10_evt_peak",
        decision="sell",
        threshold=0.50,
        description="卖出主信号：v10_evt_peak==1 概率高 → 卖出",
    ),
    "b1_head": HeadSpec(
        name="b1_head",
        target=None,
        decision=None,
        threshold=None,
        description="保留 head（masked）：未分配训练目标",
    ),
}


def masked_heads() -> list[str]:
    return [name for name, spec in HEAD_MAPPING.items() if spec.target is None]


def active_heads() -> list[str]:
    return [name for name, spec in HEAD_MAPPING.items() if spec.target is not None]


def threshold_for(head_name: str) -> float:
    spec = HEAD_MAPPING[head_name]
    if spec.threshold is None:
        raise ValueError(f"head {head_name} has no decision threshold")
    return spec.threshold