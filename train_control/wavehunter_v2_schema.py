"""WaveHunter v2 parameter single source of truth."""
from __future__ import annotations

V2_PARAMETER_SCHEMA = [
    {"key":"panel","cli":"--panel","type":"panel","default":"wavehunter_hs300_20040102_20260804.parquet","label":"因子面板","unit":""},
    {"key":"start_date","cli":"--start-date","type":"date","default":"2023-01-01","label":"训练开始日期","unit":""},
    {"key":"end_date","cli":"--end-date","type":"date","default":"2023-12-31","label":"训练结束日期","unit":""},
    {"key":"total_timesteps","cli":"--total-timesteps","type":"int","default":10000,"min":10000,"max":1000000,"step":10000,"label":"训练步数","unit":"步"},
    {"key":"window","cli":"--window","type":"int","default":20,"min":5,"max":120,"step":1,"label":"时序窗口","unit":"天"},
    {"key":"top_k","cli":"--top-k","type":"int","default":20,"min":5,"max":100,"step":5,"label":"Top-K","unit":"只"},
    {"key":"forward_period","cli":"--forward-period","type":"int","default":20,"min":1,"max":60,"step":1,"label":"前瞻收益窗口","unit":"天"},
    {"key":"max_position_pct","cli":"--max-position-pct","type":"float","default":0.02,"min":0.005,"max":0.2,"step":0.005,"label":"单股上限","unit":"比例"},
    {"key":"rebalance_days","cli":"--rebalance-days","type":"int","default":20,"min":1,"max":60,"step":1,"label":"调仓周期","unit":"天"},
    {"key":"cost_bps","cli":"--cost-bps","type":"float","default":15.0,"min":0,"max":100,"step":1,"label":"交易成本","unit":"bps"},
    {"key":"learning_rate","cli":"--learning-rate","type":"float","default":0.00003,"min":0.000001,"max":0.01,"step":0.00001,"label":"学习率","unit":""},
    {"key":"lstm_hidden","cli":"--lstm-hidden","type":"int","default":64,"min":32,"max":128,"step":32,"label":"LSTM隐层","unit":"维"},
    {"key":"lstm_layers","cli":"--lstm-layers","type":"int","default":1,"min":1,"max":3,"step":1,"label":"LSTM层数","unit":"层"},
    {"key":"universe_filter","cli":"--universe-filter","type":"select","default":"main_board_non_st","options":["main_board_non_st","all_a","hs300","zz500"],"label":"股票池筛选","unit":""},
    {"key":"seed","cli":"--seed","type":"int","default":42,"min":0,"max":999999,"step":1,"label":"随机种子","unit":""},
    {"key":"n_factors","cli":"--n-factors","type":"int","default":0,"min":0,"max":1000,"step":10,"label":"因子数量","unit":"0=全部,留空用全因子"},
    {"key":"max_grad_norm","cli":"--max-grad-norm","type":"float","default":0.5,"min":0.1,"max":5,"step":0.1,"label":"梯度裁剪","unit":""},
    {"key":"target_kl","cli":"--target-kl","type":"float","default":0.02,"min":0,"max":0.5,"step":0.01,"label":"KL阈值","unit":""},
    {"key":"batch_size","cli":"--batch-size","type":"int","default":64,"min":16,"max":64,"step":16,"label":"Batch","unit":""},
    {"key":"n_steps","cli":"--n-steps","type":"int","default":64,"min":16,"max":64,"step":16,"label":"Rollout步数","unit":""},
    {"key":"aux_lambda","cli":"--aux-lambda","type":"float","default":0.1,"min":0,"max":5,"step":0.05,"label":"Aux权重","unit":"λ"},
    {"key":"a1_lambda","cli":"--a1-lambda","type":"float","default":1.0,"min":0,"max":5,"step":0.1,"label":"A1 BCE权重","unit":"λ"},
    {"key":"a1_pos_weight","cli":"--a1-pos-weight","type":"float","default":20.0,"min":1,"max":100,"step":1,"label":"A1正样本权重","unit":":1"},
    {"key":"a2_lambda","cli":"--a2-lambda","type":"float","default":1.0,"min":0,"max":5,"step":0.1,"label":"A2 BCE权重","unit":"λ"},
    {"key":"a2_pos_weight","cli":"--a2-pos-weight","type":"float","default":40.0,"min":1,"max":100,"step":1,"label":"A2正样本权重","unit":":1"},
    {"key":"peak_lambda","cli":"--peak-lambda","type":"float","default":1.5,"min":0,"max":5,"step":0.1,"label":"Peak BCE权重","unit":"λ"},
    {"key":"peak_pos_weight","cli":"--peak-pos-weight","type":"float","default":80.0,"min":1,"max":100,"step":1,"label":"Peak正样本权重","unit":":1"},
    {"key":"b1_lambda","cli":"--b1-lambda","type":"float","default":0.3,"min":0,"max":5,"step":0.1,"label":"B1 BCE权重","unit":"λ"},
    {"key":"b1_pos_weight","cli":"--b1-pos-weight","type":"float","default":10.0,"min":1,"max":100,"step":1,"label":"B1正样本权重","unit":":1"},
    {"key":"label_window","cli":"--label-window","type":"int","default":20,"min":5,"max":120,"step":5,"label":"A1标签窗口","unit":"天"},
    {"key":"label_entry_days","cli":"--label-entry-days","type":"int","default":1,"min":0,"max":20,"step":1,"label":"标签入场延迟","unit":"天"},
    {"key":"label_n_bins","cli":"--label-n-bins","type":"int","default":5,"min":2,"max":10,"step":1,"label":"A1分档数","unit":"档"},
    {"key":"label_a2_threshold","cli":"--label-a2-threshold","type":"float","default":0.10,"min":0.01,"max":1,"step":0.01,"label":"A2启动阈值","unit":"比例"},
    {"key":"label_a2_window","cli":"--label-a2-window","type":"int","default":3,"min":1,"max":20,"step":1,"label":"A2确认窗口","unit":"天"},
    {"key":"label_min_amount_pct","cli":"--label-min-amount-pct","type":"float","default":0.05,"min":0,"max":1,"step":0.01,"label":"低量过滤","unit":"均值比例"},
    # v4 reward 组件
    {"key":"hit_rate_bonus_weight","cli":"--hit-rate-bonus-weight","type":"float","default":0.0,"min":0,"max":5,"step":0.1,"label":"胜率奖励权重","unit":"调仓日正收益比例奖励"},
    {"key":"continuation_bonus_weight","cli":"--continuation-bonus-weight","type":"float","default":0.0,"min":0,"max":5,"step":0.1,"label":"持仓延续奖励","unit":"≥3天正收益平均奖励"},
    {"key":"drawdown_penalty","cli":"--drawdown-penalty","type":"float","default":0.0,"min":0,"max":10,"step":0.1,"label":"回撤惩罚","unit":">5%平方惩罚"},
    {"key":"excess_weight","cli":"--excess-weight","type":"float","default":0.5,"min":0,"max":2,"step":0.05,"label":"超额辅助权重","unit":"0=纯绝对收益"},
    # ── 市场中性对冲 ──
    {"key":"hedge_ratio","cli":"--hedge-ratio","type":"float","default":0.0,"min":0,"max":1,"step":0.1,"label":"对冲比例","unit":"0=不对冲,1=完全对冲beta"},
    # ── 信号过滤 ──
    {"key":"signal_threshold","cli":"--signal-threshold","type":"float","default":0.0,"min":0,"max":1,"step":0.01,"label":"信号强度阈值","unit":"0=不过滤"},
    # ── 动态止损 ──
    {"key":"dynamic_stop_loss","cli":"--dynamic-stop-loss","type":"bool","default":False,"label":"动态止损","unit":"根据波动率调整"},
    # ── 止盈 ──
    {"key":"take_profit_pct","cli":"--take-profit-pct","type":"float","default":0.0,"min":0,"max":100,"step":1,"label":"止盈阈值","unit":"0=不止盈"},
    # ── v9 综合 Reward ──
    {"key":"reward_w_abs","cli":"--reward-w-abs","type":"float","default":0.60,"min":0,"max":1,"step":0.05,"label":"Reward 绝对收益权重","unit":"0-1"},
    {"key":"reward_w_dd","cli":"--reward-w-dd","type":"float","default":0.25,"min":0,"max":1,"step":0.05,"label":"Reward 回撤权重","unit":"0-1"},
    {"key":"reward_w_hit","cli":"--reward-w-hit","type":"float","default":0.15,"min":0,"max":1,"step":0.05,"label":"Reward 胜率权重","unit":"0-1"},
    # ── v9 ZigZag + KNN ──
    {"key":"zigzag_threshold","cli":"--zigzag-threshold","type":"float","default":0.10,"min":0.03,"max":0.20,"step":0.01,"label":"ZigZag 阈值","unit":"回撤比例"},
    {"key":"zigzag_pullback","cli":"--zigzag-pullback","type":"int","default":5,"min":3,"max":20,"step":1,"label":"ZigZag 最小回调","unit":"天"},
    {"key":"knn_enabled","cli":"--knn-enabled","type":"bool","default":False,"label":"启用KNN","unit":""},
    {"key":"knn_k","cli":"--knn-k","type":"int","default":20,"min":5,"max":100,"step":5,"label":"KNN K","unit":""},
    {"key":"knn_metric","cli":"--knn-metric","type":"select","default":"cosine","options":["cosine","euclidean"],"label":"KNN 距离","unit":""},
    {"key":"label_version","cli":"--label-version","type":"select","default":"v9","options":["v9","v2","auto"],"label":"标签版本","unit":""},
    {"key":"reward_norm_vol","cli":"--reward-norm-vol","type":"float","default":0.02,"min":0.005,"max":0.1,"step":0.005,"label":"Reward 归一化波动率 (EMA 自适应)","unit":"EMA_vol"},
]

FIXED_V2 = {"reward_type":"absolute_return_v3","n_envs":1,"vec_normalize":False}


def schema() -> dict:
    return {"version":"wavehunter_v2","reward":{"value":"absolute_return_v3","editable":False},"fixed":FIXED_V2,"parameters":V2_PARAMETER_SCHEMA}
