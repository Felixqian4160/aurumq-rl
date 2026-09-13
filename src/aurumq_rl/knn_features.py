"""内存友好版 KNN 特征 — 流式切片，不预计算全量窗口"""
from __future__ import annotations

import numpy as np

KNN_FEATURE_NAMES = ["knn_fwd_ret", "knn_dist"]


def compute_knn_features(
    factor_panel: np.ndarray,
    return_panel: np.ndarray,
    window: int = 20,
    k: int = 20,
    ref_lookback: int = 750,
    max_ref_days: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """计算 KNN 特征面板（内存友好版）。

    每个 t 只构建一次参考集：用 [t-ref_lookback, t-window) 的窗口，跨所有股票。
    每 t 采样 max_ref_days 天 × n_stocks 个窗口，流式构建不驻留全量。

    参数:
        ref_lookback: 回看天数（默认750=3年，实测对结果无影响）
        max_ref_days: 采样天数上限（默认12，实测12天最优）
        k: KNN 邻居数
    """
    from sklearn.neighbors import NearestNeighbors

    n_dates, n_stocks, n_factors = factor_panel.shape
    out = np.full((n_dates, n_stocks, 2), np.nan, dtype=np.float32)
    flat_dim = window * n_factors

    for t in range(window, n_dates):
        ref_start = max(0, t - ref_lookback)
        ref_end = t - window + 1
        if ref_end <= ref_start:
            continue
        # 采样参考天数 — 由 max_ref_days 参数控制
        ref_days = np.arange(ref_start, ref_end)
        ref_days = ref_days[ref_days >= window - 1]  # 窗口不足的天跳过
        if len(ref_days) > max_ref_days:
            ref_days = ref_days[::max(1, len(ref_days) // max_ref_days)]
            ref_days = ref_days[:max_ref_days]

        # 构建参考集（流式）— 限制窗口总数，避免高维 KDTree 退化
        max_ref_windows = 8000
        ref_windows = []
        ref_rets = []
        for rt in ref_days:
            if len(ref_windows) >= max_ref_windows:
                break
            seg = factor_panel[rt - window + 1: rt + 1]  # (window, n_stocks, nf)
            if np.isnan(seg).any():
                continue
            # 所有股票展平
            flat = seg.transpose(1, 0, 2).reshape(n_stocks, flat_dim)
            for s in range(n_stocks):
                ref_windows.append(flat[s])
                ref_rets.append(return_panel[rt, s])
        if len(ref_windows) < k:
            continue
        ref_windows = np.asarray(ref_windows, dtype=np.float32)
        ref_rets = np.asarray(ref_rets, dtype=np.float32)

        # 当前 t 的查询窗口
        seg = factor_panel[t - window + 1: t + 1]
        if np.isnan(seg).any():
            continue
        query = seg.transpose(1, 0, 2).reshape(n_stocks, flat_dim).astype(np.float32)

        # 高维(>100)用 brute 算法：KDTree/ball_tree 在高维退化到 O(n·d)，brute 有 BLAS 加速
        algo = "brute" if flat_dim > 100 else "auto"
        nn = NearestNeighbors(n_neighbors=min(k, len(ref_windows)),
                              algorithm=algo, metric="euclidean", n_jobs=-1)
        nn.fit(ref_windows)
        dists, idx = nn.kneighbors(query)

        for s in range(n_stocks):
            nbr_ret = ref_rets[idx[s]]
            out[t, s, 0] = float(np.nanmean(nbr_ret))
            out[t, s, 1] = float(np.mean(dists[s]))

    out = np.nan_to_num(out, nan=0.0)
    return out
