"""KNN scorer — 训练期 A2区间前20日因子快照库 → 查询窗打分。

v9 首版可关闭（knn_enabled=False），P4.2 再启用。
"""
from __future__ import annotations
import numpy as np
from typing import Optional

def build_knn_library(factor_panel: np.ndarray, a2_interval: np.ndarray,
                      window: int = 20) -> np.ndarray:
    """提取所有 A2区间内样本的前 window 日因子均值作为库。返回 (N, F)"""
    T, S, F = factor_panel.shape
    lib = []
    for t in range(window, T):
        for s in range(S):
            if a2_interval[t, s] == 1:
                # 前 window 日因子
                snap = factor_panel[t-window:t, s, :].mean(axis=0)
                if np.isfinite(snap).all():
                    lib.append(snap)
    if not lib:
        return np.zeros((0, F), dtype=np.float32)
    return np.stack(lib).astype(np.float32)

def knn_score_batch(factor_panel: np.ndarray, knn_lib: np.ndarray,
                    window: int = 20, k: int = 20, metric: str = "cosine") -> np.ndarray:
    """
    对每个 (t,s) 的前 window 日因子均值，在库中找 k 近邻，返回命中率。
    输出 (T,S) float32，0-1。
    未启用或库为空时返回全0。
    """
    T, S, F = factor_panel.shape
    out = np.zeros((T, S), dtype=np.float32)
    if knn_lib.shape[0] == 0 or k <= 0:
        return out
    # 简单实现：逐 t,s 计算，T~5500*354 量大，首版仅在训练期子集上算
    # 这里提供全量但可被调用方裁剪
    for t in range(window, T):
        queries = factor_panel[t-window:t].mean(axis=0)  # (S,F)
        # 过滤无效
        valid = np.isfinite(queries).all(axis=1)
        if not valid.any():
            continue
        # 归一化用于 cosine
        if metric == "cosine":
            q_norm = np.linalg.norm(queries[valid], axis=1, keepdims=True).clip(min=1e-6)
            l_norm = np.linalg.norm(knn_lib, axis=1, keepdims=True).clip(min=1e-6)
            q_n = queries[valid] / q_norm
            l_n = knn_lib / l_norm
            # cosine 距离 = 1 - dot，近邻 dot 大
            sims = q_n @ l_n.T  # (Sv, N)
            # k 近邻索引
            k_eff = min(k, sims.shape[1])
            # 取 top-k sims 均值作分数
            topk = np.partition(sims, -k_eff, axis=1)[:, -k_eff:]
            scores = topk.mean(axis=1)
        else:
            # 欧氏
            diff = queries[valid, None, :] - knn_lib[None, :, :]  # (Sv,N,F) 内存大，改逐行
            # 逐 s 计算避免 OOM
            scores = []
            for qi in queries[valid]:
                dists = np.linalg.norm(knn_lib - qi, axis=1)
                k_eff = min(k, len(dists))
                topk = np.partition(dists, k_eff-1)[:k_eff]
                # 转为相似度
                scores.append(float(np.exp(-topk.mean())))
            scores = np.array(scores)
        out[t, valid] = scores.astype(np.float32)
    return out
