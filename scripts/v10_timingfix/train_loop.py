"""TimingFix-only auxiliary label bridge.

Reuses the tested v10 PPO auxiliary-update implementation while mapping
TimingFix EVT targets into policy heads according to the single source of
truth in head_mapping.py. EVT targets remain in info/aux loss and never
enter observations.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np

_V10_LOOP_PATH = Path(__file__).resolve().parents[2] / "scripts" / "v10" / "train_loop_v10.py"
_spec = _ilu.spec_from_file_location("v10_train_loop", _V10_LOOP_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"failed to load v10 train_loop from {_V10_LOOP_PATH}")
_v10_loop = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_v10_loop)
V10LabelBuffer = _v10_loop.V10LabelBuffer
V10LabelWrapper = _v10_loop.V10LabelWrapper
V10PPO = _v10_loop.V10PPO

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from aurumq_rl.v10.head_mapping import HEAD_MAPPING  # noqa: E402

# Buffer slot name -> info dict key. Each slot's target/decision are looked up
# via HEAD_MAPPING. Slot names must match V10LabelBuffer.a1/a2/peak/b1.
_INFO_KEY_BY_SLOT = {
    "a1": "evt_valley",
    "a2": None,   # masked
    "peak": "evt_peak",
    "b1": None,   # masked
}


def _slot_target(slot: str):
    spec = HEAD_MAPPING[f"{slot}_head"]
    return spec.target


class TimingFixLabelBuffer(V10LabelBuffer):
    def add(self, info):
        if not isinstance(info, dict):
            return
        peak = np.asarray(info.get("evt_peak", -1), dtype=np.int64)
        valley = np.asarray(info.get("evt_valley", -1), dtype=np.int64)
        shape = peak.shape
        # Mapping is driven by HEAD_MAPPING. Slots whose target is None get -1
        # (masked) so V10PPO computes a zero loss for them. Other slots get the
        # EVT label matching the head's documented semantic.
        self.a1.append(np.where(_slot_target("a1") is not None, valley, np.full(shape, -1, dtype=np.int64)))
        self.a2.append(np.where(_slot_target("a2") is not None, valley, np.full(shape, -1, dtype=np.int64)))
        self.valid.append(np.zeros(shape, dtype=bool))
        self.peak.append(np.where(_slot_target("peak") is not None, peak, np.full(shape, -1, dtype=np.int64)))
        self.b1.append(np.where(_slot_target("b1") is not None, valley, np.full(shape, -1, dtype=np.int64)))


class TimingFixLabelWrapper(V10LabelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.buffer = TimingFixLabelBuffer()


class TimingFixPPO(V10PPO):
    def train(self):  # type: ignore[override]
        super().train()
        env = self.env
        if not isinstance(env, TimingFixLabelWrapper):
            return
        a1, a2, valid, peak, b1 = env.buffer.arrays()
        if a1 is None:
            return
        import torch as _t
        device = self.device
        obs_n = np.asarray(self.rollout_buffer.observations).reshape(
            -1, self.rollout_buffer.observations.shape[-1]).astype(np.float32)
        a1, a2, valid, peak, b1 = map(np.asarray, (a1, a2, valid, peak, b1))
        if a1.ndim == 1:
            a1 = a1.reshape(1, -1); a2 = a2.reshape(1, -1)
            valid = valid.reshape(1, -1); peak = peak.reshape(1, -1); b1 = b1.reshape(1, -1)
        n = min(len(obs_n), len(a1), len(a2), len(valid), len(peak), len(b1))
        if n <= 0:
            env.buffer.reset(); return
        obs_n = obs_n[:n]; a1 = a1[:n]; a2 = a2[:n]
        valid = valid[:n]; peak = peak[:n]; b1 = b1[:n]
        if n > 32:
            idx = np.random.default_rng().choice(n, 32, replace=False)
            obs_n, a1, a2, valid, peak, b1 = (x[idx] for x in (obs_n, a1, a2, valid, peak, b1))
        # Re-encode observation through the live policy so the aux loss retains
        # a graph connection to the trainable parameters (V10PPO's train() does
        # this for the legacy heads; TimingFix reuses the same compute_aux_loss
        # but with EVT-mapped peak/b1 targets).
        pf, _ = self.policy._encode(_t.as_tensor(obs_n, dtype=_t.float32, device=device))
        B, ns, h3 = pf.shape
        pf_flat = pf.reshape(B * ns, h3)
        a1_t = _t.as_tensor(a1, device=device, dtype=_t.float32)
        a2_t = _t.as_tensor(a2, device=device, dtype=_t.float32)
        valid_t = _t.as_tensor(valid, device=device, dtype=_t.bool)
        peak_t = _t.as_tensor(peak, device=device, dtype=_t.float32)
        b1_t = _t.as_tensor(b1, device=device, dtype=_t.float32)
        a1_logit = self.policy.a1_head(pf_flat).reshape(B, ns)
        a2_logit = self.policy.a2_head(pf_flat).reshape(B, ns)
        peak_logit = self.policy.peak_head(pf_flat).reshape(B, ns)
        b1_logit = self.policy.b1_head(pf_flat).reshape(B, ns)
        valid_a1 = (a1_t >= 0) & valid_t
        valid_a2 = (a2_t >= 0) & valid_t
        valid_pk = (peak_t >= 0)
        valid_b1 = (b1_t >= 0)
        bce = _t.nn.functional.binary_cross_entropy_with_logits
        a1_loss = bce(a1_logit[valid_a1], a1_t[valid_a1]) if valid_a1.any() else _t.tensor(0., device=device)
        a2_loss = bce(a2_logit[valid_a2], a2_t[valid_a2]) if valid_a2.any() else _t.tensor(0., device=device)
        peak_loss = bce(peak_logit[valid_pk], peak_t[valid_pk]) if valid_pk.any() else _t.tensor(0., device=device)
        b1_loss = bce(b1_logit[valid_b1], b1_t[valid_b1]) if valid_b1.any() else _t.tensor(0., device=device)
        total = (self._a1_lambda * a1_loss + self._a2_lambda * a2_loss
                 + self._peak_lambda * peak_loss + self._b1_lambda * b1_loss)
        aux = self.aux_lambda * total
        for p in self.policy.parameters():
            p.grad = None
        aux.backward()
        for name, head in zip(('a1', 'a2', 'peak', 'b1'),
                              (self.policy.a1_head, self.policy.a2_head,
                               self.policy.peak_head, self.policy.b1_head)):
            self._head_l2[name] = float(_t.sqrt(sum(_t.sum(p.detach() ** 2)
                                                   for p in head.parameters())).item())
            self._head_grad_l2[name] = float(_t.sqrt(sum(_t.sum(p.grad ** 2)
                                                        for p in head.parameters() if p.grad is not None)).item())
        _t.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        with _t.no_grad():
            self._last_logits = [l.detach().cpu().numpy() for l in (a1_logit, a2_logit, peak_logit, b1_logit)]
            for name, logit, target, mask in zip(('a1','a2','peak','b1'),
                                                 (a1_logit, a2_logit, peak_logit, b1_logit),
                                                 (a1_t, a2_t, peak_t, b1_t),
                                                 (valid_a1, valid_a2, valid_pk, valid_b1)):
                if not mask.any():
                    self.aux_stats.update({f'{name}_acc': 0.0, f'{name}_precision': 0.0, f'{name}_recall': 0.0})
                    continue
                p = (_t.sigmoid(logit[mask]) > 0.5)
                t = (target[mask] > 0.5)
                tp = (p & t).sum().float()
                self.aux_stats.update({
                    f'{name}_acc': float((p == t).float().mean()),
                    f'{name}_precision': float(tp / p.sum().clamp_min(1.0)),
                    f'{name}_recall': float(tp / t.sum().clamp_min(1.0)),
                })
        self.aux_stats.update({
            'a1_loss': float(a1_loss.detach()), 'a2_loss': float(a2_loss.detach()),
            'peak_loss': float(peak_loss.detach()), 'b1_loss': float(b1_loss.detach()),
            'aux_total': float(aux.detach()),
        })
        env.buffer.reset()
