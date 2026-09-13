"""WaveHunter v10 独立标签统计引擎。

本模块只生成候选峰谷的 EVT/OU 辅助标签，不修改旧版 ZigZag 实现。
所有标签均为事后监督目标，训练时必须排除出 observation。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.stats import genpareto, ttest_ind
except ImportError:  # pragma: no cover - 运行环境通常提供 scipy
    genpareto = None
    ttest_ind = None


@dataclass(frozen=True)
class EVTConfig:
    """EVT 候选点过滤参数。"""

    tail_frac: float = 0.05
    tail_alpha: float = 0.01
    min_excess_samples: int = 20
    trend_window: int = 10
    trend_alpha: float = 0.05


@dataclass(frozen=True)
class OUConfig:
    """OU/AR(1) B1 识别参数。"""

    window: int = 30
    theta_min: float = 0.05
    band_sigma: float = 1.0
    min_samples: int = 10


def _finite_1d(values: np.ndarray | list[float]) -> np.ndarray:
    """返回有限值，供分布估计使用。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def evt_threshold(
    swing_history: np.ndarray | list[float],
    tail_frac: float = 0.05,
    alpha: float = 0.01,
    min_excess_samples: int = 20,
    stats: dict[str, int] | None = None,
) -> float:
    """估计累计摆动幅度的 EVT 尾部阈值。

    ``swing_history`` 必须是候选枢轴之间的累计价格变动幅度，而不是
    候选日的单日收益。样本不足或 scipy 不可用时退化为经验分位数。
    """
    values = np.abs(_finite_1d(swing_history))
    if values.size == 0:
        return float("nan")
    tail_frac = float(np.clip(tail_frac, 1e-4, 0.5))
    alpha = float(np.clip(alpha, 1e-8, 0.5))
    threshold = float(np.quantile(values, 1.0 - tail_frac))
    excess = values[values > threshold] - threshold
    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
        stats["excess_samples"] = stats.get("excess_samples", 0) + int(excess.size)
        if excess.size < int(min_excess_samples):
            stats["fallback_count"] = stats.get("fallback_count", 0) + 1
    # alpha 不小于 POT 基准尾部时，直接使用经验分位数。
    if alpha >= tail_frac:
        return float(np.quantile(values, 1.0 - alpha))
    if excess.size < int(min_excess_samples) or genpareto is None:
        return threshold
    try:
        shape, _, scale = genpareto.fit(excess, floc=0)
        if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
            return threshold
        if abs(shape) < 1e-8:
            extra = -scale * np.log(alpha / tail_frac)
        else:
            extra = scale / shape * ((alpha / tail_frac) ** (-shape) - 1.0)
        result = threshold + extra
        return float(result) if np.isfinite(result) and result > 0 else threshold
    except Exception:
        return threshold


def trend_change_significant(
    before: np.ndarray | list[float],
    after: np.ndarray | list[float],
    alpha: float = 0.05,
) -> bool:
    """判断候选点前后收益率均值/趋势斜率是否显著变化。

    优先比较一阶收益率，避免直接对非平稳价格水平做 t 检验；
    数据不足时再比较价格差分。
    """
    x, y = _finite_1d(before), _finite_1d(after)
    if x.size < 4 or y.size < 4 or ttest_ind is None:
        return False
    x_ret, y_ret = np.diff(x) / np.maximum(np.abs(x[:-1]), 1e-12), np.diff(y) / np.maximum(np.abs(y[:-1]), 1e-12)
    x_ret, y_ret = _finite_1d(x_ret), _finite_1d(y_ret)
    if x_ret.size < 3 or y_ret.size < 3:
        return False
    try:
        _, p_value = ttest_ind(x_ret, y_ret, equal_var=False, nan_policy="omit")
        return bool(np.isfinite(p_value) and p_value < float(alpha))
    except Exception:
        return False


def cumulative_swing(prices: np.ndarray, previous_pivot: int, candidate: int) -> float:
    """计算上一个枢轴到候选点的累计摆动幅度。

    对连续缓慢上涨/下跌的转折，使用端点累计变动而不是单日收益，
    避免漏掉“单日波动小、整个摆动很大”的候选点。
    """
    p = np.asarray(prices, dtype=np.float64).reshape(-1)
    if not (0 <= previous_pivot < p.size and 0 <= candidate < p.size):
        return float("nan")
    start, end = sorted((int(previous_pivot), int(candidate)))
    if not (np.isfinite(p[start]) and np.isfinite(p[end]) and p[start] != 0):
        return float("nan")
    return float(abs(p[end] / p[start] - 1.0))


def fit_ou_params(
    price_window: np.ndarray | list[float],
    min_samples: int = 10,
) -> Optional[dict[str, float]]:
    """用 AR(1) OLS 拟合离散 OU 参数。

    退化条件返回 ``None``：样本不足、非有限值、价格完全走平、
    ``b <= 0``、``b >= 1`` 或参数非有限。这样纯趋势延续和停牌走平段
    不会被误判为均值回复 B1。
    """
    p = _finite_1d(price_window)
    if p.size < max(3, int(min_samples)) or np.any(p <= 0):
        return None
    x, y = p[:-1], p[1:]
    if np.ptp(p) <= np.finfo(np.float64).eps * max(1.0, float(np.mean(p))):
        return None
    try:
        b, a = np.polyfit(x, y, 1)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0.0 or b >= 1.0:
        return None
    theta = float(-np.log(b))
    mu = float(a / (1.0 - b))
    residual = y - (a + b * x)
    sigma = float(np.std(residual, ddof=1)) if residual.size > 1 else 0.0
    if not all(np.isfinite(v) for v in (theta, mu, sigma)) or mu <= 0 or sigma < 0:
        return None
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"a": float(a), "b": float(b), "theta": theta, "mu": mu, "sigma": sigma, "r2": r2}


def ou_b1_candidate(
    price_window: np.ndarray | list[float],
    config: OUConfig | None = None,
) -> bool:
    """判断窗口末端是否满足 OU 筑底候选条件。"""
    cfg = config or OUConfig()
    params = fit_ou_params(price_window, min_samples=cfg.min_samples)
    if params is None or params["theta"] < cfg.theta_min:
        return False
    last = float(_finite_1d(price_window)[-1])
    band = max(params["sigma"] * cfg.band_sigma, np.finfo(float).eps)
    return bool(abs(last - params["mu"]) <= band)


def stop_falling_features(
    prices: np.ndarray | list[float],
    end: int,
    window: int = 5,
    volatility_short: int = 3,
    volatility_long: int = 10,
) -> dict[str, float]:
    """计算 OU 候选之后的连续止跌确认特征。

    这些特征可使用标签构造阶段的后验窗口，但不得进入 observation。
    返回值均为可排序的连续分数，越大表示止跌确认越强。
    """
    p = np.asarray(prices, dtype=np.float64).reshape(-1)
    end = int(end)
    lo = max(0, end - int(window) + 1)
    segment = p[lo:end + 1]
    finite = segment[np.isfinite(segment)]
    if finite.size < 2 or end >= p.size:
        return {"no_new_low": 0.0, "slope_turn": 0.0, "vol_contraction": 0.0}
    recent = p[max(0, end - int(volatility_long) + 1):end + 1]
    returns = np.diff(recent) / np.maximum(np.abs(recent[:-1]), 1e-12)
    returns = _finite_1d(returns)
    no_new_low = float(np.clip((finite[-1] - np.min(finite)) / max(np.ptp(finite), 1e-12), 0.0, 1.0))
    x = np.arange(finite.size, dtype=np.float64)
    slope = float(np.polyfit(x, finite, 1)[0]) if finite.size >= 3 else 0.0
    slope_turn = float(1.0 / (1.0 + np.exp(-slope / max(np.std(finite), 1e-12) * 10.0)))
    if returns.size >= max(2, int(volatility_short)):
        short = float(np.std(returns[-int(volatility_short):]))
        long = float(np.std(returns))
        vol_contraction = float(np.clip(1.0 - short / max(long, 1e-12), 0.0, 1.0))
    else:
        vol_contraction = 0.0
    return {"no_new_low": no_new_low, "slope_turn": slope_turn, "vol_contraction": vol_contraction}


def stop_falling_confirmation_score(features: dict[str, float], weights: tuple[float, float, float] = (0.4, 0.4, 0.2)) -> float:
    """将三个止跌特征合成为连续确认分数。"""
    vals = np.asarray([features.get("no_new_low", 0.0), features.get("slope_turn", 0.0), features.get("vol_contraction", 0.0)], dtype=np.float64)
    vals = np.nan_to_num(np.clip(vals, 0.0, 1.0))
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    return float(np.dot(vals, w))


def post_ou_confirmation_features(
    prices: np.ndarray | list[float],
    ou_end: int,
    window: int = 5,
    pre_window: int = 5,
) -> dict[str, float]:
    """计算 OU 结束后同一摆动实例的止跌确认特征。

    这是标签质量验证/事后标签构造函数，允许使用 ou_end 之后的价格，
    但返回字段严禁进入 observation。三个分数分别衡量：没有再创新低、
    后续斜率转正、波动率相对 OU 前窗口收缩。
    """
    p = np.asarray(prices, dtype=np.float64).reshape(-1)
    start, end = int(ou_end) + 1, int(ou_end) + 1 + int(window)
    future = p[start:end]
    before = p[max(0, int(ou_end) - int(pre_window) + 1):int(ou_end) + 1]
    future = future[np.isfinite(future)]
    before = before[np.isfinite(before)]
    if future.size < 2 or before.size < 2:
        return {"no_new_low": 0.0, "slope_turn": 0.0, "vol_contraction": 0.0}
    pre_low = float(np.min(before))
    no_new_low = float(np.mean(future >= pre_low))
    x = np.arange(future.size, dtype=np.float64)
    slope = float(np.polyfit(x, future, 1)[0]) if future.size >= 3 else 0.0
    slope_turn = float(1.0 / (1.0 + np.exp(-slope / max(np.std(future), 1e-12) * 10.0)))
    pre_ret = np.diff(before) / np.maximum(np.abs(before[:-1]), 1e-12)
    fut_ret = np.diff(future) / np.maximum(np.abs(future[:-1]), 1e-12)
    pre_vol, fut_vol = float(np.std(pre_ret)), float(np.std(fut_ret))
    vol_contraction = float(np.clip(1.0 - fut_vol / max(pre_vol, 1e-12), 0.0, 1.0))
    return {"no_new_low": no_new_low, "slope_turn": slope_turn, "vol_contraction": vol_contraction}


def evt_accept_candidate(
    prices: np.ndarray,
    previous_pivot: int,
    candidate: int,
    swing_history: np.ndarray,
    config: EVTConfig | None = None,
) -> tuple[bool, float]:
    """用累计摆动 EVT + 前后趋势检验接受/拒绝候选点。"""
    cfg = config or EVTConfig()
    swing = cumulative_swing(prices, previous_pivot, candidate)
    threshold = evt_threshold(swing_history, cfg.tail_frac, cfg.tail_alpha, cfg.min_excess_samples)
    if not np.isfinite(swing) or not np.isfinite(threshold) or swing < threshold:
        return False, swing
    w = int(cfg.trend_window)
    before = prices[max(0, candidate - w):candidate]
    after = prices[candidate + 1:candidate + 1 + w]
    return trend_change_significant(before, after, cfg.trend_alpha), swing


def causal_percentile(value: float, history: list[float] | np.ndarray, min_history: int = 30) -> tuple[float, str]:
    """计算只使用严格历史样本的因果百分位。"""
    hist = _finite_1d(history)
    if hist.size < int(min_history) or not np.isfinite(value):
        return float("nan"), "cold_start"
    return float(np.mean(hist <= float(value))), "ready"


def causal_combined_score(
    swing: float,
    trend: float,
    swing_history: list[float] | np.ndarray,
    trend_history: list[float] | np.ndarray,
    min_history: int = 30,
) -> tuple[float, str]:
    """用严格历史分布计算 EVT/趋势几何平均分。"""
    evt_score, s1 = causal_percentile(swing, swing_history, min_history)
    trend_score, s2 = causal_percentile(trend, trend_history, min_history)
    if s1 != "ready" or s2 != "ready":
        return float("nan"), "cold_start"
    return float(np.sqrt(max(evt_score, 0.0) * max(trend_score, 0.0))), "ready"


def _percentile_scores(values: np.ndarray) -> np.ndarray:
    """对同一只股票的候选点做[0,1]内部百分位排名。"""
    x = np.asarray(values, dtype=np.float64)
    out = np.zeros(x.size, dtype=np.float64)
    finite = np.isfinite(x)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return out
    order = idx[np.argsort(x[idx], kind="mergesort")]
    if order.size == 1:
        out[order] = 1.0
    else:
        out[order] = np.linspace(1.0 / order.size, 1.0, order.size)
    return out


def _trend_reversal_stat(prices: np.ndarray, candidate: int, window: int) -> float:
    """计算候选点前后收益率的方向性变化统计量。"""
    p = np.asarray(prices, dtype=np.float64)
    before = p[max(0, candidate - window):candidate]
    after = p[candidate + 1:candidate + 1 + window]
    if before.size < 4 or after.size < 4:
        return float("nan")
    br = np.diff(before) / np.maximum(np.abs(before[:-1]), 1e-12)
    ar = np.diff(after) / np.maximum(np.abs(after[:-1]), 1e-12)
    br, ar = _finite_1d(br), _finite_1d(ar)
    if br.size < 3 or ar.size < 3:
        return float("nan")
    pooled = np.sqrt((np.var(br, ddof=1) / br.size) + (np.var(ar, ddof=1) / ar.size))
    return float(abs(np.mean(ar) - np.mean(br)) / max(pooled, 1e-12))


def score_pivot_candidates(
    prices: np.ndarray,
    pivots: np.ndarray | list[int],
    peak_mask: np.ndarray,
    valley_mask: np.ndarray,
    trend_window: int = 10,
    target_density: float = 0.015,
) -> dict[str, np.ndarray]:
    """按单股内部排名计算 EVT/趋势几何平均分，并控制最终序列密度。

    ``target_density`` 表示最终时间序列密度，不是候选枢轴池内的保留比例。
    每只股票先在自己的候选池内计算分数，再保留约
    ``ceil(T * target_density)`` 个候选点；因此 ZigZag 候选已经较合理时，
    统计层只负责剔除候选池中的低分尾部，避免高波动股票垄断全局 top 分位。

    ``evt_score`` 的基础是候选点到上一个枢轴的累计摆动幅度；
    ``trend_score`` 是前后收益率变化统计量的百分位；最终分数为
    ``sqrt(evt_score * trend_score)``。
    """
    p = np.asarray(prices, dtype=np.float64).reshape(-1)
    piv = np.asarray(pivots, dtype=np.int64).reshape(-1)
    peak = np.asarray(peak_mask, dtype=bool)
    valley = np.asarray(valley_mask, dtype=bool)
    candidates = piv[(piv >= 0) & (piv < p.size)]
    swings, trends, valid_pivots = [], [], []
    for i in range(1, candidates.size):
        c, prev = int(candidates[i]), int(candidates[i - 1])
        swing = cumulative_swing(p, prev, c)
        trend = _trend_reversal_stat(p, c, int(trend_window))
        if np.isfinite(swing) and np.isfinite(trend):
            valid_pivots.append(c)
            swings.append(swing)
            trends.append(trend)
    valid_pivots = np.asarray(valid_pivots, dtype=np.int64)
    evt = _percentile_scores(np.asarray(swings))
    trend = _percentile_scores(np.asarray(trends))
    combined = np.sqrt(evt * trend)
    selected = np.zeros(p.size, dtype=bool)
    if combined.size:
        keep = min(valid_pivots.size, max(1, int(np.ceil(p.size * float(np.clip(target_density, 0.001, 1.0))))))
        chosen = np.argsort(combined)[-keep:]
        selected[valid_pivots[chosen]] = True
    return {
        "pivot": valid_pivots,
        "evt_score": evt,
        "trend_score": trend,
        "combined_score": combined,
        "selected": selected & (peak | valley),
        "selected_peak": selected & peak,
        "selected_valley": selected & valley,
    }


def forward_return_quality(
    prices: np.ndarray,
    selected: np.ndarray | list[int] | np.ndarray,
    horizons: tuple[int, ...] = (5, 10),
    direction: str = "peak",
) -> dict[str, dict[str, float]]:
    """统计候选点后的前瞻收益，仅作标签质量验证。

    Peak 期望未来收益为负，Valley 期望为正；函数不参与训练标签生成。
    """
    p = np.asarray(prices, dtype=np.float64).reshape(-1)
    idx = np.flatnonzero(selected) if np.asarray(selected).dtype == bool else np.asarray(selected, dtype=int)
    result: dict[str, dict[str, float]] = {}
    sign = -1.0 if direction.lower() == "peak" else 1.0
    for horizon in horizons:
        vals = []
        for i in idx:
            j = int(i) + int(horizon)
            if 0 <= i < p.size and j < p.size and p[i] > 0 and np.isfinite(p[i]) and np.isfinite(p[j]):
                vals.append(float(p[j] / p[i] - 1.0))
        arr = np.asarray(vals, dtype=np.float64)
        result[str(horizon)] = {
            "count": float(arr.size),
            "mean_return": float(np.mean(arr)) if arr.size else float("nan"),
            "median_return": float(np.median(arr)) if arr.size else float("nan"),
            "expected_direction_hit_rate": float(np.mean(sign * arr > 0)) if arr.size else float("nan"),
        }
    return result
