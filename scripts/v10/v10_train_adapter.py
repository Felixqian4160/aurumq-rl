"""v10 train_loop adapter — 兼容 v10 Policy 的 4 头训练接口

注意：本文件不再 monkey-patch model.policy。v10 Policy 本身已经实现
SB3 forward/evaluate_actions，adapter 仅负责把 v10 的 5 元组 loss
转成 frozen AuxMonitor 需要的 15 元组，并由 v10 训练脚本显式使用。
"""
from __future__ import annotations
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from aurumq_rl.wavehunter_env_v9 import WaveHunterV9Env
from aurumq_rl.lstm_weight_env import LstmWeightConfig
from aurumq_rl.v10.policy import WaveHunterV10Policy


def _make_v10_train_loop(policy):
    """把 v10 compute_aux_loss 5 元素包装成 frozen 期望的 15 元素.
    复用 WaveHunterPPO + LabelCollectWrapper + AuxMonitor, 仅扩展 compute_aux_loss.
    """
    original_compute_aux_loss = policy.compute_aux_loss
    def wrapped_compute_aux_loss(obs, a1_bins, a2_label, valid_mask,
                                peak_label=None, b1_label=None):
        # v10 返回 (total_aux, a1_loss, a2_loss, peak_loss, b1_loss)
        total_aux, a1_loss, a2_loss, peak_loss, b1_loss = original_compute_aux_loss(
            obs, a1_bins, a2_label, valid_mask, peak_label, b1_label
        )
        # 计算准确率/precision/recall 替代 (frozen 期望 15 元素)
        B, ns = a1_bins.shape
        device = obs.device
        a1_p = torch.as_tensor(a1_bins, device=device, dtype=torch.float32)
        a2_p = torch.as_tensor(a2_label, device=device, dtype=torch.float32)
        with torch.no_grad():
            pf = policy._encode(obs)[0]
            pf_flat = pf.reshape(B * ns, -1)
            a1_logit = policy.a1_head(pf_flat).reshape(B, ns)
            a2_logit = policy.a2_head(pf_flat).reshape(B, ns)
            peak_logit = policy.peak_head(pf_flat).reshape(B, ns)
            b1_logit = policy.b1_head(pf_flat).reshape(B, ns)
            def metrics(logit, target, valid):
                if not valid.any(): return (torch.tensor(0.),) * 3
                p = (torch.sigmoid(logit[valid]) > 0.5)
                t = (target[valid] > 0.5)
                tp = (p & t).sum().float()
                acc = (p == t).float().mean()
                prec = tp / p.sum().clamp_min(1.0)
                rec = tp / t.sum().clamp_min(1.0)
                return acc, prec, rec
            v_a1 = (a1_p >= 0)
            v_a2 = (a2_p >= 0)
            v_pk = (peak_label is not None) & (torch.as_tensor(peak_label, device=device) >= 0)
            v_b1 = (b1_label is not None) & (torch.as_tensor(b1_label, device=device) >= 0)
            a1_acc, a1_prec, a1_rec = metrics(a1_logit, a1_p, v_a1)
            a2_acc, a2_prec, a2_rec = metrics(a2_logit, a2_p, v_a2)
            pk_acc, pk_prec, pk_rec = metrics(peak_logit, torch.as_tensor(peak_label, device=device, dtype=torch.float32), v_pk)
            b1_acc, b1_prec, b1_rec = metrics(b1_logit, torch.as_tensor(b1_label, device=device, dtype=torch.float32), v_b1)
        # 15 元素: total_aux, a1_loss, a2_loss, a1_acc, a2_acc, a1_prec, a1_rec, a2_prec, a2_rec, peak_acc, b1_acc, peak_prec, b1_prec, peak_rec, b1_rec
        return (total_aux, a1_loss, a2_loss,
                a1_acc, a2_acc, a1_prec, a1_rec, a2_prec, a2_rec,
                pk_acc, b1_acc, pk_prec, b1_prec, pk_rec, b1_rec)
    policy.compute_aux_loss = wrapped_compute_aux_loss
    return policy
