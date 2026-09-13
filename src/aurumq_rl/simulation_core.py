from . import simulation_config as _state

import json
import datetime
import os
from pathlib import Path
from dataclasses import asdict
from typing import Callable, Optional

import numpy as np

from .simulation_config import SimConfig, SimResult, get_sim_result, _sanitize
from .simulation_ledger import save_ledger
from .simulation_benchmark import load_csi300_benchmark


def run_simulation(cfg: SimConfig, log_callback: Optional[Callable[[str], None]] = None) -> dict:
    """执行模拟交易。

    Args:
        cfg: 模拟配置
        log_callback: 日志回调函数

    Returns:
        模拟结果字典
    """
    _state._sim_stop = False

    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    _state._sim_result = SimResult(status='running', config=asdict(cfg))
    _log(f"🚀 模拟交易启动: {Path(cfg.model_dir).name}")
    _log(f"💰 ¥{cfg.initial_capital:,.0f} | 调仓 {cfg.rebalance_days}天 | 止损 {cfg.stop_loss_pct}%")

    try:
        import onnxruntime as ort
        import pandas as pd
        from aurumq_rl.data_loader import (
            FactorPanelLoader, UniverseFilter,
            align_panel_to_training_universe, build_tradeable_mask,
        )

        start_dt = datetime.date.fromisoformat(cfg.start_date)
        end_dt = datetime.date.fromisoformat(cfg.end_date) if cfg.end_date else datetime.date.today()

        # 1) 加载模型（支持 RL 模型 policy.onnx 与 KNN+LSTM 模型 lstm_model.onnx）
        model_dir = Path(cfg.model_dir)
        if (model_dir / "policy.onnx").exists():
            from aurumq_rl.inference import RlAgentInference
            agent = RlAgentInference(model_dir)
            is_knn_lstm = False
        elif (model_dir / "lstm_model.onnx").exists():
            from aurumq_rl.knn_lstm_inference import KnnLstmInference
            agent = KnnLstmInference(model_dir)
            is_knn_lstm = True
            _log("🧬 KNN+LSTM 模型检测到，使用窗口推理模式")
        else:
            raise FileNotFoundError(f"模型目录缺少 policy.onnx 或 lstm_model.onnx: {model_dir}")

        meta = agent.metadata
        train_stock_codes = list(meta.stock_codes or [])
        train_factor_names = list(meta.factor_names or [])
        _log(f"📦 模型: {meta.algorithm} | {meta.training_timesteps:,} 步 | {len(train_stock_codes)} 只")

        # ── metadata 参数 fallback（B6 全局修复）──
        _apply_metadata_fallback(cfg, meta)
        _log(f"📋 对齐: max_pos={cfg.max_position_pct} | rebalance={cfg.rebalance_days}天 | cost={cfg.cost_bps}bps | stop={cfg.stop_loss_pct}% | hold={cfg.max_holding_days}天 | hedge={cfg.hedge_ratio} | signal={cfg.signal_threshold} | tp={cfg.take_profit_pct}%")

        # 2) 加载面板（含历史用于 z-score）
        loader = FactorPanelLoader(parquet_path=cfg.panel_path)
        lookback_start = start_dt - datetime.timedelta(days=60)
        _fwd_period = getattr(meta, "forward_period", 1) or 1
        panel = loader.load_panel(
            start_date=lookback_start, end_date=end_dt,
            n_factors=len(train_factor_names) if train_factor_names else None,
            forward_period=_fwd_period,
            universe_filter=UniverseFilter('main_board_non_st'),
            factor_names=train_factor_names if train_factor_names else None,
        )
        if train_stock_codes:
            panel, drift = align_panel_to_training_universe(panel, train_stock_codes)
            _log(f"📊 面板: {drift['kept']}/{len(train_stock_codes)} 保留")

        n_dates, n_stocks, n_factors = panel.factor_array.shape
        tradeable = build_tradeable_mask(panel)

        dates = []
        for d in panel.dates:
            if isinstance(d, datetime.datetime):
                dates.append(d.date())
            elif isinstance(d, str):
                dates.append(datetime.date.fromisoformat(d[:10]))
            else:
                dates.append(d)

        start_idx = None
        for i, d in enumerate(dates):
            if d >= start_dt:
                start_idx = i
                break
        if start_idx is None:
            _log("❌ 起始日期不在面板范围内")
            _state._sim_result.status = 'failed'
            return get_sim_result()

        _log(f"📅 {dates[start_idx]} ~ {dates[-1]} ({len(dates) - start_idx} 天)")

        # 3) 基准
        bench = load_csi300_benchmark(
            [d.isoformat() for d in dates], cfg.start_date,
            cfg.end_date or datetime.date.today().isoformat(),
        )
        has_bench = len(bench) > 0
        _log(f"📈 沪深300: {len(bench)} 天" if has_bench else "⚠ 无基准数据")

        # 4) 推理
        _log("🧠 模型推理...")
        all_scores = _run_inference(agent, panel, n_dates, n_stocks, n_factors, is_knn_lstm, meta, _log)

        # 5) 加载真实股票价格
        _log("📊 加载真实股票价格...")
        price_dict = _load_prices(panel.stock_codes, cfg.start_date, end_dt, n_stocks, _log)

        # 6) 交易循环
        result = _run_trading_loop(
            cfg, panel, dates, start_idx, all_scores, price_dict, bench, has_bench, n_stocks, _log, tradeable
        )

        # 7) 计算指标
        _log("📊 计算指标...")
        metrics = _compute_metrics(result, cfg)

        # 8) 每周市值汇总
        weekly_nav = _compute_weekly_nav(result['nav_curve'])

        # 9) 保存账本
        final_result = {
            'status': 'done',
            'config': result['config'],
            'nav_curve': result['nav_curve'],
            'weekly_nav': weekly_nav,
            'trades': result['trades'],
            'holdings': result['holdings_snapshots'],
            'metrics': metrics,
            'progress': '100%',
        }

        ledger_id = save_ledger(final_result)
        final_result['ledger_id'] = ledger_id
        _log(f"📒 账本已保存: {ledger_id}")

        _state._sim_result = SimResult(
            status='done', ledger_id=ledger_id,
            config=final_result['config'], nav_curve=result['nav_curve'],
            weekly_nav=weekly_nav,
            trades=result['trades'], holdings=result['holdings_snapshots'],
            metrics=metrics, progress='100%',
        )

    except Exception as e:
        import traceback
        _log(f"❌ 失败: {e}")
        _log(traceback.format_exc()[-500:])
        _state._sim_result = SimResult(status='failed', progress=str(e))

    return get_sim_result()


def _apply_metadata_fallback(cfg: SimConfig, meta) -> None:
    """从 metadata 应用参数 fallback。"""
    if cfg.max_position_pct < 0:
        if getattr(meta, "max_position_pct", None) is not None:
            cfg.max_position_pct = meta.max_position_pct
        else:
            cfg.max_position_pct = 0.05
    if cfg.rebalance_days < 0:
        if getattr(meta, "rebalance_days", None) is not None:
            cfg.rebalance_days = meta.rebalance_days
        else:
            cfg.rebalance_days = 20
    if cfg.cost_bps < 0:
        if getattr(meta, "cost_bps", None) is not None:
            cfg.cost_bps = meta.cost_bps
        else:
            cfg.cost_bps = 15.0
    if cfg.slippage_bps < 0:
        cfg.slippage_bps = 0.0
    if cfg.stop_loss_pct < 0:
        cfg.stop_loss_pct = 8.0
    if cfg.max_holding_days < 0:
        cfg.max_holding_days = 0
    if cfg.hedge_ratio < 0:
        cfg.hedge_ratio = 0.0
    if cfg.signal_threshold < 0:
        cfg.signal_threshold = 0.0
    if cfg.take_profit_pct < 0:
        cfg.take_profit_pct = 0.0


def _run_inference(agent, panel, n_dates, n_stocks, n_factors, is_knn_lstm, meta, _log) -> list:
    """执行模型推理。"""
    all_scores = []
    if is_knn_lstm:
        for t in range(n_dates):
            obs = panel.factor_array[t]
            scores = agent.predict(obs)
            all_scores.append(np.asarray(scores, dtype=np.float64).reshape(-1))
        _log(f"   窗口推理完成: {n_dates} 天 (w={agent.window})")
    else:
        factor_flat = panel.factor_array.reshape(n_dates, n_stocks * n_factors).astype(np.float32)
        expected = tuple(meta.obs_shape)
        env_type = getattr(meta, 'env_type', '')
        is_lstm_joint = env_type in ('portfolio_weight_lstm', 'wavehunter_lstm')

        if is_lstm_joint:
            window = int(np.prod(expected) // (n_stocks * n_factors)) if n_stocks * n_factors > 0 else 0
            total_dim = int(np.prod(expected))
            knn_dim = total_dim - n_stocks * window * n_factors
            knn_features = None

            if knn_dim > 0 and knn_dim % (n_stocks * window) == 0:
                n_knn = knn_dim // (n_stocks * window)
                _log(f"   检测到 KNN 通道: +{n_knn} 特征")
                from aurumq_rl.knn_features import compute_knn_features
                _knn_k = getattr(meta, "knn_k", 20) or 20
                _knn_ref = getattr(meta, "knn_ref_lookback", 750) or 750
                _knn_max = getattr(meta, "knn_max_ref_days", 12) or 12
                knn_features = compute_knn_features(
                    factor_panel=panel.factor_array,
                    return_panel=panel.return_array,
                    window=window, k=_knn_k, ref_lookback=_knn_ref, max_ref_days=_knn_max,
                )
            else:
                _log(f"   LSTM 联合模型: window={window} obs={expected[0]:,}")

            for t in range(n_dates):
                if t + 1 >= window:
                    obs3d = panel.factor_array[t - window + 1: t + 1].transpose(1, 0, 2).astype(np.float32)
                    if knn_features is not None:
                        knn3d = knn_features[t - window + 1: t + 1].transpose(1, 0, 2).astype(np.float32)
                        obs3d = np.concatenate([obs3d, knn3d], axis=2)
                else:
                    pad = window - (t + 1)
                    past = panel.factor_array[: t + 1].transpose(1, 0, 2)
                    zeros = np.zeros((n_stocks, pad, n_factors), dtype=np.float32)
                    obs3d = np.concatenate([zeros, past], axis=1).astype(np.float32)
                    if knn_features is not None:
                        knn_past = knn_features[: t + 1].transpose(1, 0, 2).astype(np.float32)
                        knn_zeros = np.zeros((n_stocks, pad, knn_features.shape[2]), dtype=np.float32)
                        knn3d = np.concatenate([knn_zeros, knn_past], axis=1).astype(np.float32)
                        obs3d = np.concatenate([obs3d, knn3d], axis=2)
                obs_flat = obs3d.reshape(-1)
                if obs_flat.size == int(np.prod(expected)):
                    scores = agent.predict(obs_flat.reshape(expected).astype(np.float32))
                    all_scores.append(np.asarray(scores, dtype=np.float64).reshape(-1))
                else:
                    all_scores.append(np.zeros(n_stocks))
            _log(f"   窗口推理完成: {n_dates} 天 (w={window})")
        else:
            for t in range(n_dates):
                obs = factor_flat[t]
                if obs.size == int(np.prod(expected)):
                    scores = agent.predict(obs.reshape(expected).astype(np.float32))
                    all_scores.append(np.asarray(scores, dtype=np.float64).reshape(-1))
                else:
                    all_scores.append(np.zeros(n_stocks))
            _log(f"   推理完成: {n_dates} 天")

    return all_scores


def _load_prices(stock_codes, start_date, end_dt, n_stocks, _log) -> dict:
    """加载真实股票价格。"""
    import pandas as pd
    _QBOT_CACHE = Path(os.environ.get("AURUMQ_PRICE_CACHE", "data/cache"))
    price_dict = {}
    loaded_count = 0
    for code in stock_codes:
        fname = code.replace('.', '_') + '_full.parquet'
        fpath = _QBOT_CACHE / fname
        if not fpath.exists():
            continue
        try:
            df = pd.read_parquet(fpath)
            date_col = 'date' if 'date' in df.columns else 'trade_date'
            df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
            df = df[(df[date_col] >= start_date) & (df[date_col] <= end_dt.isoformat())]
            price_dict[code] = dict(zip(df[date_col], df['close']))
            loaded_count += 1
        except Exception:
            pass
    _log(f"   加载 {loaded_count}/{n_stocks} 只股票价格")
    return price_dict


def _run_trading_loop(cfg, panel, dates, start_idx, all_scores, price_dict, bench, has_bench, n_stocks, _log, tradeable):
    """执行交易循环。"""
    # 这是核心交易逻辑，暂时保留完整实现
    # 后续可以进一步拆分为独立的交易执行模块
    cash = cfg.initial_capital
    holdings = {}
    nav_curve = []
    meta_top_k = 0
    meta_env_type = ''

    trades = []
    holdings_snapshots = []
    prev_nav = cfg.initial_capital
    bench_nav = 1.0
    total_weight_sum = 0.0
    total_weight_count = 0
    rebalance_positions = []

    for t in range(start_idx, len(dates)):
        if _state._sim_stop:
            _log("⏹ 已停止")
            break

        date_str = dates[t].isoformat()
        scores = all_scores[t]
        mask = tradeable[t] if t < tradeable.shape[0] else np.ones(n_stocks, dtype=bool)

        # ── portfolio_weight_lstm / wavehunter_lstm: 模型输出权重，需 top_k 过滤 ──
        if meta_env_type in ('portfolio_weight_lstm', 'wavehunter_lstm') and meta_top_k > 0:
            valid_scores = scores.copy()
            valid_scores[~mask] = -np.inf
            valid_idx = np.where(valid_scores > -np.inf)[0]
            if len(valid_idx) > meta_top_k:
                sorted_idx = valid_idx[np.argsort(valid_scores[valid_idx])[::-1]]
                keep = set(sorted_idx[:meta_top_k])
                for i in valid_idx:
                    if i not in keep:
                        valid_scores[i] = -np.inf
            valid_scores[valid_scores < 0] = -np.inf
            max_pos_raw = cfg.max_position_pct
            max_pos = max_pos_raw
            total_w = valid_scores[valid_scores > -np.inf].sum()
            if total_w > 0:
                normed = valid_scores[valid_scores > -np.inf] / total_w
                normed = np.minimum(normed, max_pos)
                if normed.sum() > 1.0:
                    normed = normed / normed.sum()
                valid_scores[valid_scores > -np.inf] = normed
            scores = valid_scores

        # 获取当日各股票真实收盘价
        day_prices = {}
        for ci in range(n_stocks):
            code = panel.stock_codes[ci]
            if code in price_dict:
                p = price_dict[code].get(date_str)
                if p is not None and p > 0:
                    day_prices[ci] = p

        # 更新持仓: 用真实价格
        for ci in list(holdings.keys()):
            h = holdings[ci]
            cur_price = day_prices.get(ci, h.get('cur_price', h['buy_price']))
            h['cur_price'] = cur_price
            h['market_value'] = h['shares'] * cur_price
            if cur_price > h.get('high_since_entry', h['buy_price']):
                h['high_since_entry'] = cur_price

        # 止损检查（追踪止损：从最高价下跌 stop_loss_pct）
        if cfg.stop_loss_pct > 0:
            for ci in list(holdings.keys()):
                h = holdings[ci]
                high = h.get('high_since_entry', h['buy_price'])
                cur = h.get('cur_price', h['buy_price'])
                # ── 动态止损：根据近期波动率调整止损阈值 ──
                effective_stop = cfg.stop_loss_pct
                if cfg.dynamic_stop_loss and t >= 20:
                    recent_rets = []
                    for rt in range(max(start_idx, t-20), t):
                        rt_date = dates[rt].isoformat() if rt < len(dates) else ''
                        rt_ci_ret = 0
                        if ci in price_dict and rt_date in price_dict[ci]:
                            prev_date = dates[rt-1].isoformat() if rt > start_idx else ''
                            if prev_date and prev_date in price_dict[ci]:
                                p_prev = price_dict[ci].get(prev_date, 0)
                                p_cur = price_dict[ci].get(rt_date, 0)
                                if p_prev > 0:
                                    rt_ci_ret = (p_cur / p_prev - 1)
                        recent_rets.append(rt_ci_ret)
                    if len(recent_rets) >= 10:
                        vol_20d = float(np.std(recent_rets)) * 100
                        effective_stop = max(5.0, min(30.0, cfg.stop_loss_pct * (1 + vol_20d / 20)))
                if high > 0 and (high - cur) / high * 100 >= effective_stop:
                    sell_price = cur
                    sell_amount = h['shares'] * sell_price
                    cost = sell_amount * cfg.cost_bps / 10000.0
                    pnl = (sell_price - h['buy_price']) * h['shares'] - cost
                    cash += sell_amount - cost
                    code = panel.stock_codes[ci]
                    drawdown_pct = round((high - sell_price) / high * 100, 2)
                    entry_str = h.get('entry_date', date_str)[:10]
                    try:
                        held_days_val = int((np.datetime64(date_str[:10]) - np.datetime64(entry_str)) / np.timedelta64(1, 'D'))
                    except Exception:
                        held_days_val = 0
                    trades.append({
                        'date': date_str, 'stock_code': code, 'side': 'stop_loss',
                        'buy_price': round(h['buy_price'], 2),
                        'high_price': round(high, 2),
                        'sell_price': round(sell_price, 2),
                        'shares': h['shares'],
                        'amount': round(sell_amount, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl / (h['buy_price'] * h['shares']) * 100, 2) if h['buy_price'] > 0 else 0,
                        'drawdown_from_high': drawdown_pct,
                        'held_days': held_days_val,
                    })
                    del holdings[ci]

        # ── 止盈检查：持仓收益超过阈值时止盈 ──
        if cfg.take_profit_pct > 0:
            for ci in list(holdings.keys()):
                h = holdings[ci]
                cur = h.get('cur_price', h['buy_price'])
                buy_p = h['buy_price']
                if buy_p > 0:
                    held_return = (cur / buy_p - 1) * 100
                    if held_return >= cfg.take_profit_pct:
                        sell_price = cur
                        sell_amount = h['shares'] * sell_price
                        cost = sell_amount * cfg.cost_bps / 10000.0
                        pnl = (sell_price - buy_p) * h['shares'] - cost
                        cash += sell_amount - cost
                        code = panel.stock_codes[ci]
                        entry_str = h.get('entry_date', date_str)[:10]
                        try:
                            held_days_val = int((np.datetime64(date_str[:10]) - np.datetime64(entry_str)) / np.timedelta64(1, 'D'))
                        except Exception:
                            held_days_val = 0
                        trades.append({
                            'date': date_str, 'stock_code': code, 'side': 'take_profit',
                            'buy_price': round(buy_p, 2),
                            'high_price': round(h.get('high_since_entry', buy_p), 2),
                            'sell_price': round(sell_price, 2),
                            'shares': h['shares'],
                            'amount': round(sell_amount, 2),
                            'pnl': round(pnl, 2),
                            'pnl_pct': round(held_return, 2),
                            'held_days': held_days_val,
                            'drawdown_from_high': 0,
                        })
                        del holdings[ci]

        # 最大持仓天数检查
        if cfg.max_holding_days > 0 and (t - start_idx) % cfg.rebalance_days != 0:
            for ci in list(holdings.keys()):
                h = holdings[ci]
                entry_t = h.get('entry_t')
                if entry_t is not None:
                    held_days = int(t - entry_t)
                else:
                    entry = h.get('entry_date', '')
                    if entry and len(entry) >= 10:
                        try:
                            entry_dt = np.datetime64(entry[:10])
                            cur_dt = np.datetime64(date_str[:10])
                            held_days = int((cur_dt - entry_dt) / np.timedelta64(1, 'D'))
                        except Exception:
                            continue
                    else:
                        continue
                if held_days >= cfg.max_holding_days:
                    cur = h.get('cur_price', h['buy_price'])
                    sell_price = cur
                    sell_amount = h['shares'] * sell_price
                    cost = sell_amount * cfg.cost_bps / 10000.0
                    pnl = (sell_price - h['buy_price']) * h['shares'] - cost
                    cash += sell_amount - cost
                    code = panel.stock_codes[ci]
                    trades.append({
                        'date': date_str, 'stock_code': code, 'side': 'expire_sell',
                        'buy_price': round(h['buy_price'], 2),
                        'high_price': round(h.get('high_since_entry', h['buy_price']), 2),
                        'sell_price': round(sell_price, 2),
                        'shares': h['shares'],
                        'amount': round(sell_amount, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl / (h['buy_price'] * h['shares']) * 100, 2) if h['buy_price'] > 0 else 0,
                        'held_days': held_days,
                        'drawdown_from_high': round((h.get('high_since_entry', h['buy_price']) - sell_price) / h.get('high_since_entry', h['buy_price']) * 100, 2) if h.get('high_since_entry', 0) > 0 else 0,
                    })
                    del holdings[ci]

        # 计算净值
        total_mv = sum(h['market_value'] for h in holdings.values())
        nav = cash + total_mv
        cur_total_weight = float(np.sum([h['market_value'] for h in holdings.values()]) / nav) if nav > 0 else 0.0
        total_weight_sum += cur_total_weight
        total_weight_count += 1

        # 调仓日
        if (t - start_idx) % cfg.rebalance_days == 0:
            rebalance_positions.append((date_str, cur_total_weight))
            valid_scores = scores.copy()
            valid_scores[~mask] = -np.inf
            sorted_idx = np.argsort(valid_scores)[::-1]
            effective_top_k = meta_top_k if (meta_env_type in ('portfolio_weight_lstm', 'wavehunter_lstm') and meta_top_k > 0) else 20
            target_idx = set()
            for si in sorted_idx:
                if len(target_idx) >= effective_top_k:
                    break
                if np.isfinite(valid_scores[si]):
                    target_idx.add(si)

            # ── 信号过滤：弱信号时保持当前持仓 ──
            if cfg.signal_threshold > 0 and len(target_idx) > 0:
                valid_mask = np.isfinite(scores) & (scores > -np.inf)
                if valid_mask.any():
                    valid_scores_only = scores[valid_mask]
                    sorted_valid = np.sort(valid_scores_only)[::-1]
                    top_k_scores = sorted_valid[:min(effective_top_k, len(sorted_valid))]
                    median_score = float(np.median(valid_scores_only))
                    signal_strength = float(np.mean(top_k_scores)) - median_score
                    if signal_strength < cfg.signal_threshold:
                        if (t - start_idx) % (cfg.rebalance_days * 5) == 0:
                            _log(f"   ⚡ 信号过滤: strength={signal_strength:.4f} < threshold={cfg.signal_threshold:.4f} → 保持持仓")
                        total_mv = sum(h['market_value'] for h in holdings.values())
                        nav = cash + total_mv
                        cur_total_weight = float(total_mv / nav) if nav > 0 else 0.0
                        total_weight_sum += cur_total_weight
                        total_weight_count += 1
                        rebalance_positions.append((date_str, cur_total_weight))
                        daily_ret = (nav / prev_nav - 1) if prev_nav > 0 else 0
                        bench_ret = bench.get(date_str, 0) if has_bench else 0
                        if (t - start_idx) == 0:
                            bench_nav = 1.0
                        bench_nav *= (1 + bench_ret)
                        nav_curve.append({
                            'date': date_str,
                            'nav': round(nav, 2),
                            'benchmark_nav': round(bench_nav, 4),
                            'daily_return': round(daily_ret * 100, 4),
                        })
                        prev_nav = nav
                        if (t - start_idx) % 50 == 0:
                            pct = (t - start_idx) / (len(dates) - start_idx) * 100
                            _state._sim_result.progress = f"{pct:.0f}% ({date_str}) nav={nav:.0f}"
                        continue

            # 卖出: 不在目标中的
            for ci in list(set(holdings.keys()) - target_idx):
                h = holdings[ci]
                cur = day_prices.get(ci, h.get('cur_price', h['buy_price']))
                sell_price = cur
                sell_amount = h['shares'] * sell_price
                cost = sell_amount * cfg.cost_bps / 10000.0
                pnl = (sell_price - h['buy_price']) * h['shares'] - cost
                cash += sell_amount - cost
                code = panel.stock_codes[ci]
                entry_str = h.get('entry_date', date_str)[:10]
                try:
                    held_days_val = int((np.datetime64(date_str[:10]) - np.datetime64(entry_str)) / np.timedelta64(1, 'D'))
                except Exception:
                    held_days_val = 0
                high_val = h.get('high_since_entry', h['buy_price'])
                trades.append({
                    'date': date_str, 'stock_code': code, 'side': 'sell',
                    'buy_price': round(h['buy_price'], 2),
                    'high_price': round(high_val, 2),
                    'sell_price': round(sell_price, 2),
                    'shares': h['shares'],
                    'amount': round(sell_amount, 2),
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl / (h['buy_price'] * h['shares']) * 100, 2) if h['buy_price'] > 0 else 0,
                    'held_days': held_days_val,
                    'drawdown_from_high': round((high_val - sell_price) / high_val * 100, 2) if high_val > 0 else 0,
                })
                del holdings[ci]

            # 买入: 新增的目标股票
            new_buys = target_idx - set(holdings.keys())
            if new_buys:
                nav_now = cash + sum(h['market_value'] for h in holdings.values())
                _alloc = {}
                _total_score = 0
                for ci in new_buys:
                    sc = scores[ci] if scores[ci] > 0 else 0
                    if meta_env_type in ('portfolio_weight_lstm', 'wavehunter_lstm') and meta_top_k > 0:
                        _alloc[ci] = nav_now * sc
                    else:
                        _alloc[ci] = nav_now / 20 if 20 > 0 else 0
                    _total_score += _alloc[ci]
                if _total_score > 0 and cash > 0:
                    _scale = min(1.0, cash / _total_score)
                else:
                    _scale = 0
                for ci in sorted(new_buys):
                    target_amount = _alloc.get(ci, 0) * _scale
                    if target_amount <= 0:
                        continue
                    price = day_prices.get(ci)
                    if not price or price <= 0:
                        continue
                    buy_price = price
                    shares = max(100, int(target_amount / buy_price / 100) * 100)
                    cost_amount = shares * buy_price
                    fee = cost_amount * cfg.cost_bps / 10000.0
                    if cash >= cost_amount + fee:
                        cash -= cost_amount + fee
                        cost_basis_per_share = (cost_amount + fee) / shares if shares > 0 else buy_price
                        holdings[ci] = {
                            'shares': shares,
                            'buy_price': cost_basis_per_share,
                            'raw_buy_price': buy_price,
                            'cur_price': price,
                            'market_value': shares * price,
                            'entry_date': date_str,
                            'entry_t': t,
                            'high_since_entry': price,
                        }
                        code = panel.stock_codes[ci]
                        trades.append({
                            'date': date_str, 'stock_code': code, 'side': 'buy',
                            'buy_price': round(cost_basis_per_share, 2),
                            'sell_price': 0,
                            'shares': shares,
                            'amount': round(cost_amount, 2),
                            'pnl': 0,
                            'pnl_pct': 0,
                        })

            # 调仓后重算净值
            total_mv = sum(h['market_value'] for h in holdings.values())
            nav = cash + total_mv
            cur_post = float(total_mv / nav) if nav > 0 else 0.0
            if rebalance_positions and rebalance_positions[-1][0] == date_str:
                pre_w = rebalance_positions[-1][1]
                rebalance_positions[-1] = (date_str, cur_post)
                total_weight_sum += (cur_post - pre_w)

            # 调仓快照
            hold_list = []
            for ci, h in holdings.items():
                hold_list.append({
                    'code': panel.stock_codes[ci],
                    'shares': h['shares'],
                    'buy_price': round(h['buy_price'], 2),
                    'cur_price': round(h.get('cur_price', h['buy_price']), 2),
                    'market_value': round(h['market_value'], 2),
                    'weight': round(h['market_value'] / nav * 100, 2) if nav > 0 else 0,
                    'pnl_pct': round((h.get('cur_price', h['buy_price']) / h['buy_price'] - 1) * 100, 2) if h['buy_price'] > 0 else 0,
                    'entry_date': h.get('entry_date', date_str),
                    'action': 'hold',
                })
            today_sells = [s for s in trades if s['date'] == date_str and s['side'] != 'buy']
            for s in today_sells:
                hold_list.append({
                    'code': s['stock_code'],
                    'shares': s['shares'],
                    'buy_price': s['buy_price'],
                    'cur_price': s['sell_price'],
                    'market_value': 0,
                    'weight': 0,
                    'pnl_pct': s.get('pnl_pct', 0),
                    'entry_date': '',
                    'action': s['side'],
                })
            holdings_snapshots.append({
                'date': date_str,
                'stocks': hold_list,
                'total_market_value': round(total_mv, 2),
                'cash': round(cash, 2),
                'nav': round(nav, 2),
            })

        # 日净值
        daily_ret = (nav / prev_nav - 1) if prev_nav > 0 else 0
        bench_ret = bench.get(date_str, 0) if has_bench else 0
        # ── hedge_ratio: 模拟时不应用 ──
        # 训练时已应用hedge，模拟时不再应用，避免双重计算
        # if cfg.hedge_ratio > 0 and has_bench:
        #     portfolio_beta = 1.0  # 简化：假设beta=1
        #     hedge_pnl = -cfg.hedge_ratio * portfolio_beta * bench_ret
        #     # 对冲PnL计入现金（模拟做空市场基准的损益）
        #     cash += nav * hedge_pnl
        #     # 重算净值
        #     total_mv = sum(h['market_value'] for h in holdings.values())
        #     nav = cash + total_mv
        #     daily_ret = (nav / prev_nav - 1) if prev_nav > 0 else 0
        if (t - start_idx) == 0:
            bench_nav = 1.0
        bench_nav *= (1 + bench_ret)

        nav_curve.append({
            'date': date_str,
            'nav': round(nav, 2),
            'benchmark_nav': round(bench_nav, 4),
            'daily_return': round(daily_ret * 100, 4),
        })
        prev_nav = nav

        if (t - start_idx) % 50 == 0:
            pct = (t - start_idx) / (len(dates) - start_idx) * 100
            _state._sim_result.progress = f"{pct:.0f}% ({date_str}) nav={nav:.0f}"

    return {
        'config': asdict(cfg),
        'nav_curve': nav_curve,
        'trades': trades,
        'holdings_snapshots': holdings_snapshots,
    }


def _compute_metrics(result: dict, cfg: SimConfig) -> dict:
    """计算模拟指标。"""
    nav_curve = result['nav_curve']
    trades = result['trades']

    navs = np.array([n['nav'] for n in nav_curve]) if nav_curve else np.array([cfg.initial_capital])
    bench_navs = np.array([n['benchmark_nav'] for n in nav_curve]) if nav_curve else np.array([1.0])

    total_return = (navs[-1] / navs[0] - 1) * 100 if len(navs) > 1 else 0
    bench_total = (bench_navs[-1] / bench_navs[0] - 1) * 100 if len(bench_navs) > 1 else 0
    excess = total_return - bench_total

    n_days = len(navs)
    ann_factor = 252 / max(n_days, 1)
    ann_return = ((1 + total_return / 100) ** ann_factor - 1) * 100

    daily_rets = np.array([n['daily_return'] for n in nav_curve]) / 100 if nav_curve else np.array([0])
    sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0

    peak = np.maximum.accumulate(navs) if len(navs) > 1 else np.array([1.0])
    dd = (navs - peak) / peak if len(navs) > 1 else np.array([0.0])
    max_dd = float(dd.min() * 100) if len(navs) > 1 else 0

    # 胜率 + 盈亏分析
    sell_trades = [t for t in trades if t['side'] in ('sell', 'stop_loss', 'take_profit', 'expire_sell') and t.get('pnl', 0) != 0]
    wins = [t for t in sell_trades if t['pnl'] > 0]
    losses = [t for t in sell_trades if t['pnl'] <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / len(sell_trades) * 100 if sell_trades else 0
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
    profit_factor = (sum(t['pnl'] for t in wins) / sum(abs(t['pnl']) for t in losses)) if losses and sum(abs(t['pnl']) for t in losses) > 0 else 0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

    stop_count = sum(1 for t in trades if t['side'] == 'stop_loss')
    expire_count = sum(1 for t in trades if t['side'] == 'expire_sell')
    volatility = float(np.std(daily_rets) * np.sqrt(252) * 100) if len(daily_rets) > 1 else 0

    max_dd_days = 0
    if len(navs) > 1:
        in_dd = False
        dd_start = 0
        for i in range(len(navs)):
            if dd[i] < 0:
                if not in_dd:
                    dd_start = i
                    in_dd = True
            else:
                if in_dd:
                    max_dd_days = max(max_dd_days, i - dd_start)
                    in_dd = False
        if in_dd:
            max_dd_days = max(max_dd_days, len(navs) - dd_start)

    calmar = (ann_return / abs(max_dd)) if max_dd != 0 else 0

    return {
        'total_return': round(total_return, 2),
        'benchmark_return': round(bench_total, 2),
        'excess_return': round(excess, 2),
        'annualized_return': round(ann_return, 2),
        'volatility': round(volatility, 2),
        'sharpe_ratio': round(sharpe, 3),
        'sharpe': round(sharpe, 3),
        'calmar_ratio': round(calmar, 3),
        'max_drawdown': round(max_dd, 2),
        'max_dd_days': max_dd_days,
        'total_trades': len(trades),
        'buy_trades': sum(1 for t in trades if t['side'] == 'buy'),
        'sell_trades': sum(1 for t in trades if t['side'] in ('sell', 'take_profit')),
        'stop_loss_count': stop_count,
        'expire_count': expire_count,
        'take_profit_count': sum(1 for t in trades if t['side'] == 'take_profit'),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'avg_win': round(float(avg_win), 2),
        'avg_loss': round(float(avg_loss), 2),
        'win_loss_ratio': round(win_loss_ratio, 2),
        'final_nav': round(float(navs[-1]), 2),
        'n_trading_days': n_days,
        'n_rebalance_periods': 0,
    }


def _compute_weekly_nav(nav_curve: list) -> list:
    """计算每周市值汇总。"""
    weekly_nav = []
    if not nav_curve:
        return weekly_nav

    week_start = nav_curve[0]
    week_max_nav = nav_curve[0]['nav']
    week_max_bench = nav_curve[0]['benchmark_nav']

    for i, point in enumerate(nav_curve):
        d = datetime.date.fromisoformat(point['date'])
        is_friday = d.weekday() == 4
        is_last = (i == len(nav_curve) - 1)

        if is_friday or is_last:
            week_return = (point['nav'] / week_start['nav'] - 1) * 100
            bench_return = (point['benchmark_nav'] / week_start['benchmark_nav'] - 1) * 100 if week_start['benchmark_nav'] > 0 else 0
            weekly_nav.append({
                'week_end': point['date'],
                'week_start': week_start['date'],
                'nav': round(point['nav'], 2),
                'benchmark_nav': round(point['benchmark_nav'], 4),
                'week_return': round(week_return, 2),
                'bench_return': round(bench_return, 2),
                'max_nav': round(week_max_nav, 2),
            })
            week_start = point
            week_max_nav = point['nav']
            week_max_bench = point['benchmark_nav']
        else:
            week_max_nav = max(week_max_nav, point['nav'])
            week_max_bench = max(week_max_bench, point['benchmark_nav'])

    return weekly_nav
