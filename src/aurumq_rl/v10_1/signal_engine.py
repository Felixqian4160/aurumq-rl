"""v10.1 simulation signal contract."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class V101Signal:
    action: str
    reason: str
    buy_score: float = 0.0


def decision_to_signal(probs: dict, a1_threshold=0.55, a2_threshold=0.55, peak_threshold=0.50, b1_threshold=0.55) -> V101Signal:
    a1=float(probs.get('a1',0.));a2=float(probs.get('a2',0.));peak=float(probs.get('peak',0.));b1=float(probs.get('b1',0.))
    if peak>=peak_threshold:return V101Signal('sell',f'v10_1_peak={peak:.3f}')
    if a1>=a1_threshold:return V101Signal('buy',f'v10_1_a1={a1:.3f},a2={a2:.3f}',.6*a1+.4*a2)
    if a2>=a2_threshold:return V101Signal('hold',f'v10_1_a2={a2:.3f}',a2)
    if b1>=b1_threshold:return V101Signal('wait',f'v10_1_b1={b1:.3f}')
    return V101Signal('wait','weak')
