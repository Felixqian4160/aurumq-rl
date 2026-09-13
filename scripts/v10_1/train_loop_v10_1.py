"""v10.1 isolated four-head PPO training loop.

The loop is intentionally copied into the v10.1 namespace so v10/v9 files are
not modified. Labels are collected from v10.1-specific info keys.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class V10_1LabelBuffer:
    def __init__(self):
        self.a1, self.a2, self.valid, self.peak, self.b1 = [], [], [], [], []

    def reset(self):
        self.a1.clear(); self.a2.clear(); self.valid.clear(); self.peak.clear(); self.b1.clear()

    def add(self, info):
        if not isinstance(info, dict): return
        a1 = info.get('a1_bins'); a2 = info.get('a2_labels')
        if a1 is None or a2 is None: return
        a1 = np.asarray(a1, dtype=np.int64); a2 = np.asarray(a2, dtype=np.int64)
        self.a1.append(a1); self.a2.append(a2)
        self.valid.append(np.asarray(info.get('valid', np.ones_like(a1)), dtype=bool))
        self.peak.append(np.asarray(info.get('v10_1_peak_label', np.full_like(a1, -1)), dtype=np.int64))
        self.b1.append(np.asarray(info.get('v10_1_b1_label', np.full_like(a1, -1)), dtype=np.int64))

    def arrays(self):
        if not self.a1: return None, None, None, None, None
        return tuple(np.stack(x) for x in (self.a1, self.a2, self.valid, self.peak, self.b1))


class V10_1LabelWrapper:
    def __init__(self, env): self._env = env; self.buffer = V10_1LabelBuffer()
    def __getattr__(self, name): return getattr(self._env, name)
    def reset(self, *args, **kwargs): return self._env.reset(*args, **kwargs)
    def step(self, actions):
        result = self._env.step(actions)
        infos = result[-1] if len(result) in (4, 5) else []
        if isinstance(infos, dict): infos = [infos]
        for info in infos: self.buffer.add(info)
        return result


class V10_1PPO(PPO):
    def __init__(self, *args, aux_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_lambda = aux_lambda
        self.aux_stats = {k: 0.0 for k in (
            'a1_loss','a2_loss','peak_loss','b1_loss','a1_acc','a2_acc',
            'peak_acc','b1_acc','a1_precision','a2_precision','peak_precision','b1_precision',
            'a1_recall','a2_recall','peak_recall','b1_recall')}
        self._head_l2 = {}; self._head_grad_l2 = {}

    def learn(self, *args, **kwargs):
        original = self.env
        self.env = V10_1LabelWrapper(original)
        try: return super().learn(*args, **kwargs)
        finally: self.env = original

    @staticmethod
    def _metric(logit, target, valid):
        if not valid.any(): return (torch.tensor(0., device=logit.device),) * 3
        pred = torch.sigmoid(logit[valid]) > 0.5; true = target[valid] > 0.5
        tp = (pred & true).sum().float()
        return ((pred == true).float().mean(), tp / pred.sum().clamp_min(1.0), tp / true.sum().clamp_min(1.0))

    def train(self):
        super().train()
        env = self.env
        if not isinstance(env, V10_1LabelWrapper): return
        a1, a2, valid, peak, b1 = env.buffer.arrays()
        if a1 is None: return
        obs = np.asarray(self.rollout_buffer.observations).reshape(-1, self.rollout_buffer.observations.shape[-1]).astype(np.float32)
        a1, a2, valid, peak, b1 = map(np.asarray, (a1, a2, valid, peak, b1))
        if a1.ndim == 1: a1, a2, valid, peak, b1 = [x.reshape(1, -1) for x in (a1, a2, valid, peak, b1)]
        n = min(len(obs), len(a1), len(a2), len(valid), len(peak), len(b1))
        if n <= 0: env.buffer.reset(); return
        obs, a1, a2, valid, peak, b1 = obs[:n], a1[:n], a2[:n], valid[:n], peak[:n], b1[:n]
        if n > 32:
            idx = np.random.default_rng().choice(n, 32, replace=False)
            obs, a1, a2, valid, peak, b1 = [x[idx] for x in (obs, a1, a2, valid, peak, b1)]
        device = self.device
        result = self.policy.compute_aux_loss(torch.as_tensor(obs, dtype=torch.float32, device=device), a1, a2, valid, peak, b1)
        total, l1, l2, lp, lb = result
        aux = self.aux_lambda * total
        self.policy.optimizer.zero_grad(); aux.backward()
        for name, head in zip(('a1','a2','peak','b1'), (self.policy.a1_head, self.policy.a2_head, self.policy.peak_head, self.policy.b1_head)):
            self._head_l2[name] = float(torch.sqrt(sum(torch.sum(p.detach() ** 2) for p in head.parameters())).item())
            self._head_grad_l2[name] = float(torch.sqrt(sum(torch.sum(p.grad ** 2) for p in head.parameters() if p.grad is not None)).item())
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        with torch.no_grad():
            pf = self.policy._encode(torch.as_tensor(obs, dtype=torch.float32, device=device))[0]
            flat = pf.reshape(pf.shape[0] * pf.shape[1], -1)
            logits = [head(flat).reshape(pf.shape[0], pf.shape[1]) for head in (self.policy.a1_head, self.policy.a2_head, self.policy.peak_head, self.policy.b1_head)]
            targets = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (a1, a2, peak, b1)]
            mask = torch.as_tensor(valid, device=device)
            for name, logit, target in zip(('a1','a2','peak','b1'), logits, targets):
                acc, precision, recall = self._metric(logit, target, (target >= 0) & mask)
                self.aux_stats.update({f'{name}_acc': float(acc), f'{name}_precision': float(precision), f'{name}_recall': float(recall)})
        self.aux_stats.update({'a1_loss': float(l1.detach()), 'a2_loss': float(l2.detach()), 'peak_loss': float(lp.detach()), 'b1_loss': float(lb.detach())})
        env.buffer.reset()


class V10_1Monitor(BaseCallback):
    def __init__(self, model, metrics_path=None, verbose=0):
        super().__init__(verbose); self.model_ref = model; self.metrics_path = Path(metrics_path) if metrics_path else None; self.last = 0; self.rewards=[]
        if self.metrics_path: self.metrics_path.parent.mkdir(parents=True, exist_ok=True); self.metrics_path.write_text('', encoding='utf-8')

    def _on_step(self):
        rewards = np.asarray(self.locals.get('rewards', []), dtype=float).reshape(-1)
        self.rewards.extend(float(x) for x in rewards if np.isfinite(x))
        if self.num_timesteps - self.last >= 100:
            self.last = self.num_timesteps
            arr = np.asarray(self.rewards, dtype=float); self.rewards.clear(); s = self.model_ref.aux_stats
            row = {'timesteps': int(self.num_timesteps), 'reward_count': int(arr.size), 'reward_mean': float(arr.mean()) if arr.size else 0.0, 'reward_std': float(arr.std()) if arr.size else 0.0, 'reward_min': float(arr.min()) if arr.size else 0.0, 'reward_max': float(arr.max()) if arr.size else 0.0}
            row.update({k: float(v) for k,v in s.items()})
            row.update({f'head_{k}_l2':v for k,v in self.model_ref._head_l2.items()}); row.update({f'head_{k}_grad_l2':v for k,v in self.model_ref._head_grad_l2.items()})
            if self.metrics_path:
                with self.metrics_path.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False)+'\n')
            print(f"[v10.1] {self.num_timesteps}步 A1_P={s['a1_precision']*100:.1f}% A2_P={s['a2_precision']*100:.1f}% Peak_P={s['peak_precision']*100:.1f}% B1_P={s['b1_precision']*100:.1f}%", flush=True)
        return True
