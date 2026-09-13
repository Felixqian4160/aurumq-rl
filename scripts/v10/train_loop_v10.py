"""v10 独立 PPO 训练循环。

不导入或修改 frozen scripts/train_loop.py。
负责 v10 的 rollout 标签缓存、PPO 更新后的四头辅助损失和训练监控。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from aurumq_rl.v10.metric_flattener import flatten_state


_BAYES_KEYS = ('alpha', 'beta', 'variance_mean', 'variance_std', 'volatility')
_CVAR_KEYS = ('count', 'var', 'cvar', 'last_drawdown')
_HHI_KEYS = ('hhi', 'effective_count', 'active_count', 'total_weight', 'target_count')


def _flatten_state_window(prefix: str, states: list[dict]) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {}
    keys = _BAYES_KEYS if prefix == 'bayes' else _CVAR_KEYS if prefix == 'cvar' else _HHI_KEYS if prefix == 'hhi' else ()
    if not keys or not states:
        return out
    latest = states[-1] if states else None
    if latest is None:
        return out
    flat = flatten_state(prefix, latest, keys)
    out.update(flat)
    values = {}
    for k in keys:
        sample = [s.get(k) for s in states if s and isinstance(s.get(k), (int, float)) and np.isfinite(s.get(k))]
        if sample:
            arr=np.asarray(sample, dtype=np.float64)
            values[f"{prefix}_{k}_mean"] = float(arr.mean())
            values[f"{prefix}_{k}_std"] = float(arr.std())
    out.update(values)
    return out


class V10LabelBuffer:
    def __init__(self):
        self.a1, self.a2, self.valid, self.peak, self.b1 = [], [], [], [], []

    def reset(self):
        self.a1.clear(); self.a2.clear(); self.valid.clear(); self.peak.clear(); self.b1.clear()

    def add(self, info):
        if not isinstance(info, dict): return
        a1, a2 = info.get('a1_bins'), info.get('a2_labels')
        if a1 is None or a2 is None: return
        a1 = np.asarray(a1, dtype=np.int64); a2 = np.asarray(a2, dtype=np.int64)
        self.a1.append(a1); self.a2.append(a2)
        self.valid.append(np.asarray(info.get('valid', np.ones_like(a1)), dtype=bool))
        self.peak.append(np.asarray(info.get('v9_peak', np.full_like(a1, -1)), dtype=np.int64))
        self.b1.append(np.asarray(info.get('v9_b1', np.full_like(a1, -1)), dtype=np.int64))

    def arrays(self):
        if not self.a1: return None, None, None, None, None
        return tuple(np.stack(x) for x in (self.a1, self.a2, self.valid, self.peak, self.b1))


class V10LabelWrapper:
    def __init__(self, env):
        self._env = env; self.buffer = V10LabelBuffer()
    def __getattr__(self, name): return getattr(self._env, name)
    def reset(self, *args, **kwargs): return self._env.reset(*args, **kwargs)
    def step(self, actions):
        result = self._env.step(actions)
        infos = result[-1] if len(result) in (4, 5) else []
        if isinstance(infos, dict): infos = [infos]
        for info in infos: self.buffer.add(info)
        return result


class V10PPO(PPO):
    def __init__(self, *args, aux_lambda=0.1, policy_factory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_lambda = aux_lambda
        self._policy_factory = policy_factory
        if self._policy_factory is not None:
            # SB3 has already constructed the frozen policy; wrap it.
            wrapped = self._policy_factory(self.policy)
            wrapped._frozen = self.policy
            wrapped.optimizer = self.policy.optimizer
            wrapped.scheduler = self.policy.scheduler
            self.policy = wrapped
        self.aux_stats = {k: 0.0 for k in (
            'a1_loss','a2_loss','peak_loss','b1_loss','a1_acc','a2_acc',
            'peak_acc','b1_acc','a1_precision','a2_precision','peak_precision','b1_precision',
            'a1_recall','a2_recall','peak_recall','b1_recall')}
        self._head_l2 = {}
        self._head_grad_l2 = {}
    def learn(self, *args, **kwargs):
        original = self.env
        self.env = V10LabelWrapper(original)
        try: return super().learn(*args, **kwargs)
        finally: self.env = original

    @staticmethod
    def _metric(logit, target, valid):
        if not valid.any(): return (torch.tensor(0., device=logit.device),) * 3
        pred = torch.sigmoid(logit[valid]) > 0.5; true = target[valid] > 0.5
        tp = (pred & true).sum().float()
        return ( (pred == true).float().mean(),
                 tp / pred.sum().clamp_min(1.0),
                 tp / true.sum().clamp_min(1.0) )

    def train(self):
        super().train()
        env = self.env
        if not isinstance(env, V10LabelWrapper): return
        a1, a2, valid, peak, b1 = env.buffer.arrays()
        if a1 is None: return
        obs = np.asarray(self.rollout_buffer.observations).reshape(-1, self.rollout_buffer.observations.shape[-1]).astype(np.float32)
        a1, a2, valid, peak, b1 = map(np.asarray, (a1, a2, valid, peak, b1))
        if a1.ndim == 1:
            a1, a2, valid, peak, b1 = [x.reshape(1, -1) for x in (a1, a2, valid, peak, b1)]
        n = min(len(obs), len(a1), len(a2), len(valid), len(peak), len(b1))
        if n <= 0: env.buffer.reset(); return
        obs, a1, a2, valid, peak, b1 = obs[:n], a1[:n], a2[:n], valid[:n], peak[:n], b1[:n]
        if n > 32:
            idx = np.random.default_rng().choice(n, 32, replace=False)
            obs, a1, a2, valid, peak, b1 = [x[idx] for x in (obs, a1, a2, valid, peak, b1)]
        device = self.device
        result = self.policy.compute_aux_loss(
            torch.as_tensor(obs, dtype=torch.float32, device=device), a1, a2, valid, peak, b1)
        total, l1, l2, lp, lb = result
        aux = self.aux_lambda * total
        self.policy.optimizer.zero_grad(); aux.backward()
        for name, head in zip(('a1', 'a2', 'peak', 'b1'), (self.policy.a1_head, self.policy.a2_head, self.policy.peak_head, self.policy.b1_head)):
            self._head_l2[name] = float(torch.sqrt(sum(torch.sum(p.detach() ** 2) for p in head.parameters())).item())
            self._head_grad_l2[name] = float(torch.sqrt(sum(torch.sum(p.grad ** 2) for p in head.parameters() if p.grad is not None)).item())
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        with torch.no_grad():
            pf = self.policy._encode(torch.as_tensor(obs, dtype=torch.float32, device=device))[0]
            flat = pf.reshape(pf.shape[0] * pf.shape[1], -1)
            logits = [head(flat).reshape(pf.shape[0], pf.shape[1]) for head in (
                self.policy.a1_head, self.policy.a2_head, self.policy.peak_head, self.policy.b1_head)]
            self._last_logits = [logit.detach().cpu().numpy() for logit in logits]
            targets = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (a1, a2, peak, b1)]
            valids = [(x >= 0) & torch.as_tensor(valid, device=device) for x in targets]
            names = ('a1','a2','peak','b1')
            for name, logit, target, mask in zip(names, logits, targets, valids):
                acc, precision, recall = self._metric(logit, target, mask)
                self.aux_stats.update({f'{name}_acc': float(acc), f'{name}_precision': float(precision), f'{name}_recall': float(recall)})
        self.aux_stats.update({'a1_loss':float(l1.detach()),'a2_loss':float(l2.detach()),'peak_loss':float(lp.detach()),'b1_loss':float(lb.detach())})
        env.buffer.reset()


class V10Monitor(BaseCallback):
    def __init__(self, model, verbose=0, metrics_path=None):
        super().__init__(verbose)
        self.model_ref = model
        self.last = 0
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self._reward_window = []
        self._component_window = {k: [] for k in ('reward_raw', 'r_abs', 'r_dd', 'r_hit', 'r_cycle', 'ema_vol', 'dd', 'r_div', 'effective_count', 'active_count', 'total_weight')}
        self._head_logits_window = {k: [] for k in ('a1_logit_mean', 'a2_logit_mean', 'peak_logit_mean', 'b1_logit_mean',
                                                'a1_logit_std', 'a2_logit_std', 'peak_logit_std', 'b1_logit_std')}
        self._head_weight_window: list[dict] = []
        self._fused_score_window: list[float] = []
        self._bayes_state_window: list[dict] = []
        self._cvar_state_window: list[dict] = []
        self._hhi_state_window: list[dict] = []
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text('', encoding='utf-8')

    def _on_step(self):
        rewards = np.asarray(self.locals.get('rewards', []), dtype=float).reshape(-1)
        self._reward_window.extend(float(x) for x in rewards if np.isfinite(x))
        infos = self.locals.get('infos', [])
        if isinstance(infos, dict): infos = [infos]
        for info in infos:
            if not isinstance(info, dict): continue
            for key, values in self._component_window.items():
                value = info.get(key, np.nan)
                if np.isfinite(value): values.append(float(value))
            bayes_state = info.get('bayes_vol_state')
            cvar_state = info.get('cvar_state')
            hhi_state = info.get('hhi_state')
            if isinstance(bayes_state, dict): self._bayes_state_window.append(bayes_state)
            if isinstance(cvar_state, dict): self._cvar_state_window.append(cvar_state)
            if isinstance(hhi_state, dict): self._hhi_state_window.append(hhi_state)
            fused_score = info.get('fused_score')
            if isinstance(fused_score, (int, float)) and np.isfinite(fused_score):
                self._fused_score_window.append(float(fused_score))
            head_weights = info.get('head_weights')
            if isinstance(head_weights, dict): self._head_weight_window.append(head_weights)
        if self.num_timesteps - self.last >= 1000:
            self.last = self.num_timesteps
            s = self.model_ref.aux_stats
            reward_values = np.asarray(self._reward_window, dtype=float)
            row = {'timesteps': int(self.num_timesteps), 'reward_count': int(reward_values.size),
                   'reward_mean': float(np.mean(reward_values)) if reward_values.size else 0.0,
                   'reward_std': float(np.std(reward_values)) if reward_values.size else 0.0,
                   'reward_min': float(np.min(reward_values)) if reward_values.size else 0.0,
                   'reward_max': float(np.max(reward_values)) if reward_values.size else 0.0}
            self._reward_window.clear()
            for key, values in self._component_window.items():
                if values:
                    arr = np.asarray(values, dtype=float)
                    row[key] = float(np.mean(arr)); row[f'{key}_std'] = float(np.std(arr))
                values.clear()
            last_logits = getattr(self.model_ref, '_last_logits', None) or []
            if last_logits and all(isinstance(x, np.ndarray) for x in last_logits):
                for name, arr in zip(('a1', 'a2', 'peak', 'b1'), last_logits):
                    arr = np.asarray(arr, dtype=np.float64)
                    if arr.size:
                        row[f'{name}_logit_mean'] = float(arr.mean())
                        row[f'{name}_logit_std'] = float(arr.std())
            if self._fused_score_window:
                arr = np.asarray(self._fused_score_window, dtype=np.float64)
                row['fused_score_mean'] = float(arr.mean())
                row['fused_score_std'] = float(arr.std())
                self._fused_score_window.clear()
            if self._head_weight_window:
                latest = self._head_weight_window[-1]
                flat = flatten_state('head', latest, ('a1_w', 'a2_w', 'peak_w', 'b1_w'))
                row.update(flat)
                self._head_weight_window.clear()
            for k, v in s.items():
                self.model_ref.logger.record(f'train/{k}', float(v)); row[k] = float(v)
            for name, value in self.model_ref._head_l2.items(): row[f'head_{name}_l2'] = value
            for name, value in self.model_ref._head_grad_l2.items(): row[f'head_{name}_grad_l2'] = value
            self.model_ref.logger.record('train/reward_mean', row['reward_mean'])
            self.model_ref.logger.record('train/reward_std', row['reward_std'])
            self.model_ref.logger.dump(self.num_timesteps)
            row.update(_flatten_state_window('bayes', self._bayes_state_window))
            row.update(_flatten_state_window('cvar', self._cvar_state_window))
            row.update(_flatten_state_window('hhi', self._hhi_state_window))
            if self.metrics_path:
                with self.metrics_path.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f"[v10] {self.num_timesteps}步 A1_P={s['a1_precision']*100:.1f}% A2_P={s['a2_precision']*100:.1f}% Peak_P={s['peak_precision']*100:.1f}% B1_P={s['b1_precision']*100:.1f}%", flush=True)
        return True
