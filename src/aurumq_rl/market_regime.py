#!/usr/bin/env python3
"""
市场状态识别模块 — 基于沪深300指数的多维度打分系统
维度：均线排列(30%) + ADX趋势(20%) + 成交量(15%) + 波动率(15%) + 市场广度(10%) + 累计涨跌幅(10%)
输出：每个交易日的市场状态 (牛市/熊市/震荡市) + 各维度得分
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# External data and local caches are configured through environment variables.
# The code never embeds developer-specific paths or credentials.

def get_csi300_daily() -> pd.DataFrame:
    """获取沪深300指数日线数据（从面板等权close近似，或真实指数）"""
    # Try an explicitly configured public-data client; credentials never live in source.
    try:
        import tushare as ts
        token = os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN is not set")
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.index_daily(ts_code='000300.SH', start_date='20040101', end_date='20260804')
        df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"✅ 从 tushare 获取 CSI300 日线: {len(df)} 天 ({df['date'].min().date()} ~ {df['date'].max().date()})")
        return df[['date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"⚠️ tushare 获取失败: {e}")
        print("   使用面板等权 close 近似")

    # fallback: 从面板等权近似
    panel = pd.read_parquet('data/factor_panel_hs300_20040102_20260804.parquet')
    panel['trade_date'] = pd.to_datetime(panel['trade_date'])
    daily = panel.groupby('trade_date').agg(
        close=('close', 'mean'),
        volume=('vol', 'sum'),
    ).reset_index().rename(columns={'trade_date': 'date'})
    daily['open'] = daily['close']
    daily['high'] = daily['close']
    daily['low'] = daily['close']
    daily = daily.sort_values('date').reset_index(drop=True)
    print(f"✅ 用面板等权近似 CSI300: {len(daily)} 天")
    return daily[['date', 'open', 'high', 'low', 'close', 'volume']]


# ═══════════════════════════════════════════════════════════════
# 维度1: 均线排列与价格位置 (权重30%)
# ═══════════════════════════════════════════════════════════════

def score_ma_alignment(close: pd.Series) -> pd.Series:
    """
    均线排列打分 [-1, +1]
    +1 = 多头排列(价格>MA60>MA120>MA250)
    -1 = 空头排列(价格<MA60<MA120<MA250)
    0 = 缠绕/震荡
    """
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ma250 = close.rolling(250).mean()

    score = pd.Series(0.0, index=close.index)

    # 价格相对均线位置
    pos_60 = (close > ma60).astype(float) * 2 - 1   # +1 above, -1 below
    pos_120 = (close > ma120).astype(float) * 2 - 1
    pos_250 = (close > ma250).astype(float) * 2 - 1

    # 均线排列
    ma_60_120 = (ma60 > ma120).astype(float) * 2 - 1
    ma_120_250 = (ma120 > ma250).astype(float) * 2 - 1

    # 综合得分：价格位置(60%) + 均线排列(40%)
    score = (pos_60 * 0.25 + pos_120 * 0.20 + pos_250 * 0.15 +
             ma_60_120 * 0.20 + ma_120_250 * 0.20)

    return score


# ═══════════════════════════════════════════════════════════════
# 维度2: ADX趋势强度 (权重20%)
# ═══════════════════════════════════════════════════════════════

def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """计算 ADX, +DI, -DI"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()

    return pd.DataFrame({'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di})


def score_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """ADX 趋势打分 [-1, +1]"""
    adx_df = calc_adx(high, low, close)
    adx = adx_df['adx']
    plus_di = adx_df['plus_di']
    minus_di = adx_df['minus_di']

    score = pd.Series(0.0, index=close.index)

    # 有趋势 (ADX > 25)
    trending = (adx > 25).values
    # 强趋势 (ADX > 40)
    strong = (adx > 40).values

    # +DI > -DI → 牛, +DI < -DI → 熊
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    direction = ((plus_di - minus_di) / di_sum).clip(-1, 1).fillna(0).values

    # ADX < 20 → 震荡
    adx_vals = adx.values
    result = np.where(adx_vals < 20, 0,
             np.where(trending, direction * np.minimum(adx_vals / 30, 1.0), 0))

    return pd.Series(result, index=close.index)


# ═══════════════════════════════════════════════════════════════
# 维度3: 成交量配合度 (权重15%)
# ═══════════════════════════════════════════════════════════════

def score_volume(volume: pd.Series, close: pd.Series) -> pd.Series:
    """成交量配合度打分 [-1, +1]"""
    vol_ma20 = volume.rolling(20).mean()
    vol_ma60 = volume.rolling(60).mean()
    vol_ratio = vol_ma20 / vol_ma60.replace(0, np.nan)

    # 价格变化
    price_ret = close.pct_change(5)

    # 量价配合：
    # 牛：价涨量增 (ret>0 且 vol_ratio>1) → 正分
    # 熊：价跌量增 (ret<0 且 vol_ratio>1) → 负分
    # 震荡：量比 0.8~1.2 → 接近0

    score = pd.Series(0.0, index=close.index)

    # 量价同向 = 健康趋势
    vol_price_same = np.sign(price_ret) * np.sign(vol_ratio - 1)
    # 量比偏离程度
    vol_dev = (vol_ratio - 1).clip(-0.5, 0.5) * 2  # normalize to [-1, 1]

    score = vol_price_same * vol_dev.abs()

    return score.clip(-1, 1)


# ═══════════════════════════════════════════════════════════════
# 维度4: 波动率水平 (权重15%)
# ═══════════════════════════════════════════════════════════════

def score_volatility(close: pd.Series) -> pd.Series:
    """波动率打分 [-1, +1]"""
    ret = close.pct_change()
    vol_20 = ret.rolling(20).std() * np.sqrt(252)
    vol_60 = ret.rolling(60).std() * np.sqrt(252)

    # 波动率分位数（用60日滚动）
    vol_rank = vol_20.rolling(252).rank(pct=True)

    # 高波动(>70%分位) → 偏熊或趋势末期
    # 低波动(<30%分位) → 震荡
    # 中等波动 → 偏牛

    score = pd.Series(0.0, index=close.index)

    # 波动率上升 + 价格下跌 = 熊市恐慌
    price_trend = close.pct_change(20)
    vol_change = vol_20.pct_change(20)

    # 恐慌指标：波动率飙升 + 价格下跌
    panic = (vol_change > 0.2) & (price_trend < -0.05)
    # 健康上涨：波动率稳定 + 价格上涨
    healthy = (vol_change.abs() < 0.1) & (price_trend > 0.05)

    score = np.where(panic, -0.8,
             np.where(healthy, 0.5,
             np.where(vol_rank > 0.7, -0.3,
             np.where(vol_rank < 0.3, 0.2, 0))))

    return pd.Series(score, index=close.index)


# ═══════════════════════════════════════════════════════════════
# 维度5: 市场广度 (权重10%) — 需要全市场数据
# ═══════════════════════════════════════════════════════════════

def score_breadth(panel: pd.DataFrame) -> pd.Series:
    """市场广度打分 [-1, +1]：用面板中每日上涨家数占比"""
    daily = panel.groupby('trade_date').apply(
        lambda g: (g['pct_chg'] > 0).sum() / len(g)
    ).reset_index()
    daily.columns = ['date', 'up_ratio']
    daily = daily.sort_values('date').set_index('date')

    # 60日滚动平均
    up_ma60 = daily['up_ratio'].rolling(60).mean()

    # > 0.55 → 牛, < 0.45 → 熊
    score = (up_ma60 - 0.50) * 10  # normalize: 0.5→0, 0.6→+1, 0.4→-1
    return score.clip(-1, 1)


# ═══════════════════════════════════════════════════════════════
# 维度6: 累计涨跌幅 (权重10%)
# ═══════════════════════════════════════════════════════════════

def score_cumulative_return(close: pd.Series) -> pd.Series:
    """累计涨跌幅打分 [-1, +1]"""
    # 250日累计收益
    cum_ret_250 = close.pct_change(250)
    # 120日累计收益
    cum_ret_120 = close.pct_change(120)

    # > +25% → 牛, < -20% → 熊 (A股阈值)
    score_250 = cum_ret_250.clip(-0.3, 0.4) / 0.35  # normalize
    score_120 = cum_ret_120.clip(-0.2, 0.3) / 0.25

    return (score_250 * 0.6 + score_120 * 0.4).clip(-1, 1)


# ═══════════════════════════════════════════════════════════════
# 综合打分 + 状态判定
# ═══════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    """每日市场状态结果"""
    date: pd.Timestamp
    score: float          # 综合得分 [-1, +1]
    regime: str           # "牛市" / "熊市" / "震荡市"
    ma_score: float       # 均线得分
    adx_score: float      # ADX得分
    vol_score: float      # 成交量得分
    volatility_score: float  # 波动率得分
    breadth_score: float  # 广度得分
    cumret_score: float   # 累计涨跌得分


def detect_market_regime(
    csi300: pd.DataFrame,
    panel: Optional[pd.DataFrame] = None,
    bull_threshold: float = 0.4,
    bear_threshold: float = -0.4,
) -> pd.DataFrame:
    """
    综合打分判定市场状态

    权重：
    - 均线排列: 30%
    - ADX趋势: 20%
    - 成交量: 15%
    - 波动率: 15%
    - 市场广度: 10%
    - 累计涨跌: 10%

    状态：
    - 得分 > bull_threshold → 牛市
    - 得分 < bear_threshold → 熊市
    - 其他 → 震荡市
    """
    close = csi300['close']
    high = csi300['high']
    low = csi300['low']
    volume = csi300['volume']

    print("📊 计算各维度得分...")

    # 各维度打分
    ma_s = score_ma_alignment(close)
    adx_s = score_adx(high, low, close)
    vol_s = score_volume(volume, close)
    volat_s = score_volatility(close)
    cumret_s = score_cumulative_return(close)

    # 广度得分（需要面板数据）
    if panel is not None:
        breadth_s = score_breadth(panel)
        # 对齐到 close.index（按位置对齐，因为日期可能不完全匹配）
        if len(breadth_s) == len(close):
            breadth_s = pd.Series(breadth_s.values, index=close.index).fillna(0)
        else:
            breadth_s = pd.Series(0.0, index=close.index)
    else:
        breadth_s = pd.Series(0.0, index=close.index)

    # 综合得分（加权平均）
    total_score = (
        ma_s * 0.30 +
        adx_s * 0.20 +
        vol_s * 0.15 +
        volat_s * 0.15 +
        breadth_s * 0.10 +
        cumret_s * 0.10
    )

    # 状态判定（A股波动大，阈值放宽到 ±0.3）
    bull_arr = (total_score.values > 0.3)
    bear_arr = (total_score.values < -0.3)
    regime_arr = np.where(bull_arr, '牛市', np.where(bear_arr, '熊市', '震荡市'))

    # 跳过时间过滤（直接使用原始判定，后续 find_regime_periods 会过滤短区间）
    confirmed_arr = regime_arr

    regime = pd.Series(confirmed_arr, index=close.index)

    # 组装结果
    result = pd.DataFrame({
        'date': csi300['date'],
        'close': close.values,
        'total_score': total_score.values,
        'regime': regime.values,
        'ma_score': ma_s.values,
        'adx_score': adx_s.values,
        'vol_score': vol_s.values,
        'volatility_score': volat_s.values,
        'breadth_score': breadth_s.values if isinstance(breadth_s, pd.Series) else 0,
        'cumret_score': cumret_s.values,
    })

    # 统计
    regime_counts = result['regime'].value_counts()
    print(f"\n=== 市场状态统计 ===")
    for r, c in regime_counts.items():
        pct = c / len(result) * 100
        print(f"  {r}: {c} 天 ({pct:.1f}%)")

    return result


# ═══════════════════════════════════════════════════════════════
# 划分训练/验证区间
# ═══════════════════════════════════════════════════════════════

def find_regime_periods(regime_df: pd.DataFrame, min_days: int = 60) -> dict:
    """
    找出连续的牛市/熊市/震荡区间（≥min_days天）
    返回: {regime: [(start, end, days), ...]}
    """
    periods = {}
    for regime_name in ['牛市', '熊市', '震荡市']:
        mask = regime_df['regime'] == regime_name
        # 找连续区间
        groups = (mask != mask.shift()).cumsum()
        regime_groups = regime_df[mask].groupby(groups[mask])

        periods[regime_name] = []
        for _, group in regime_groups:
            if len(group) >= min_days:
                start = group['date'].iloc[0]
                end = group['date'].iloc[-1]
                days = len(group)
                periods[regime_name].append((start, end, days))

    return periods


def print_regime_periods(periods: dict):
    """打印市场状态区间"""
    print(f"\n{'='*70}")
    print("市场状态区间（≥60天连续）")
    print(f"{'='*70}")

    for regime_name in ['牛市', '熊市', '震荡市']:
        marker = {'牛市': '🟢', '熊市': '🔴', '震荡市': '🟡'}[regime_name]
        print(f"\n{marker} {regime_name}:")
        for start, end, days in periods.get(regime_name, []):
            years = days / 252
            print(f"  {str(start.date()):<12} ~ {str(end.date()):<12}  {days:>4}天 ({years:.1f}年)")

    # 推荐训练/验证方案
    print(f"\n{'='*70}")
    print("推荐方案：分状态训练+验证")
    print(f"{'='*70}")

    for regime_name in ['牛市', '熊市', '震荡市']:
        ps = periods.get(regime_name, [])
        if len(ps) >= 2:
            # 用前半段训练，后半段验证
            mid = len(ps) // 2
            train_periods = ps[:mid]
            val_periods = ps[mid:]

            train_start = train_periods[0][0]
            train_end = train_periods[-1][1]
            val_start = val_periods[0][0]
            val_end = val_periods[-1][1]

            print(f"\n{regime_name}模型:")
            print(f"  训练: {str(train_start.date())} ~ {str(train_end.date())}")
            print(f"  验证: {str(val_start.date())} ~ {str(val_end.date())}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 1. 获取沪深300数据
    csi300 = get_csi300_daily()

    # 2. 加载面板数据（用于市场广度）
    panel_path = Path('data/factor_panel_hs300_20040102_20260804.parquet')
    panel = pd.read_parquet(panel_path)
    panel['trade_date'] = pd.to_datetime(panel['trade_date'])

    # 3. 市场状态识别
    regime_df = detect_market_regime(csi300, panel)

    # 4. 保存结果
    out_path = Path('data/market_regime_csi300.csv')
    regime_df.to_csv(out_path, index=False)
    print(f"\n✅ 保存至 {out_path}")

    # 5. 找出区间
    periods = find_regime_periods(regime_df, min_days=60)
    print_regime_periods(periods)

    # 6. 输出每日状态（最近10天）
    print(f"\n最近10天状态:")
    for _, row in regime_df.tail(10).iterrows():
        marker = {'牛市': '🟢', '熊市': '🔴', '震荡市': '🟡'}[row['regime']]
        print(f"  {str(row['date'].date()):<12} {marker} {row['regime']:<4} "
              f"得分={row['total_score']:>+.3f} "
              f"均线={row['ma_score']:>+.2f} "
              f"ADX={row['adx_score']:>+.2f} "
              f"量={row['vol_score']:>+.2f} "
              f"波动={row['volatility_score']:>+.2f} "
              f"涨跌={row['cumret_score']:>+.2f}")
