"""统一 WebUI 参数 schema 注册表。

Schema 是前端动态控件、API 请求模型和 CLI 审计的产品目录；迁移期间先覆盖
所有现有功能模块的字段，不改变现有业务默认值或训练逻辑。
"""
from __future__ import annotations

from typing import Any


def f(key: str, typ: str, default: Any, label: str, cli: str | None = None, **kw: Any) -> dict:
    row = {"key": key, "type": typ, "default": default, "label": label}
    if cli is not None:
        row["cli"] = cli
    row.update(kw)
    return row


def schema(*fields: dict, fixed: dict | None = None) -> dict:
    return {"parameters": list(fields), "fixed": fixed or {}}


MODULE_SCHEMAS: dict[str, dict] = {
    "train": schema(
        f("panel", "panel", "", "因子面板"), f("algorithm", "select", "PPO", "算法", options=["PPO"]),
        f("total_timesteps", "int", 100000, "训练步数", "--total-timesteps", min=10000, max=10000000, step=10000),
        f("start_date", "date", "2023-01-01", "训练开始"), f("end_date", "date", "2025-06-30", "训练结束"),
        f("universe_filter", "select", "main_board_non_st", "股票池", options=["main_board_non_st", "all_a", "hs300", "zz500"]),
        f("n_envs", "int", 4, "并行环境", min=1, max=16, step=1), f("n_factors", "int", 0, "因子数", min=0, max=1000, step=10),
        f("top_k", "int", 30, "Top-K", min=1, max=100, step=1), f("forward_period", "int", 10, "前瞻窗口", min=1, max=60, step=1),
        f("cost_bps", "float", 30.0, "交易成本", min=0, max=100, step=1), f("learning_rate", "float", 0.0003, "学习率", min=1e-6, max=0.01, step=1e-5),
        f("target_kl", "float", 0.05, "KL阈值", min=0, max=1, step=0.01), f("n_steps", "int", 0, "Rollout", min=0, max=2048, step=64),
        f("batch_size", "int", 0, "Batch", min=0, max=2048, step=64), f("seed", "int", 42, "随机种子", min=0, max=999999, step=1),
    ),
    "simulation": schema(
        f("model_dir", "model", "", "模型"), f("panel", "panel", "", "因子面板"), f("initial_capital", "float", 100000.0, "初始资金", min=1),
        f("start_date", "date", "2024-01-01", "开始日期"), f("end_date", "date", "", "结束日期"), f("top_k", "int", 30, "Top-K", min=1, max=100, step=1),
        f("cost_bps", "float", -1.0, "成本", min=-1, max=100, step=1), f("slippage_bps", "float", 10.0, "滑点", min=0, max=100, step=1),
        f("rebalance_days", "int", -1, "调仓周期", min=-1, max=100, step=1), f("stop_loss_pct", "float", 8.0, "止损", min=0, max=100, step=0.5),
        f("max_position_pct", "float", -1.0, "单股上限", min=-1, max=1, step=0.005), f("max_holding_days", "int", -1, "最大持仓", min=-1, max=1000, step=1),
    ),
    "p22c": schema(
        f("panel", "panel", "", "因子面板"), f("total_timesteps", "int", 200000, "训练步数", min=10000, max=10000000, step=10000),
        f("start_date", "date", "2005-01-04", "训练开始"), f("end_date", "date", "2026-12-31", "训练结束"),
        f("universe_filter", "select", "main_board_non_st", "股票池", options=["main_board_non_st", "all_a", "hs300", "zz500"]),
        f("n_envs", "int", 16, "并行环境", min=1, max=32, step=1), f("top_k", "int", 3, "Top-K", min=1, max=100, step=1),
        f("forward_period", "int", 10, "前瞻窗口", min=1, max=60, step=1), f("cost_bps", "float", 30.0, "交易成本", min=0, max=100, step=1),
        f("seed", "int", 0, "随机种子", min=0, max=999999, step=1), f("learning_rate", "float", 1e-4, "学习率", min=1e-6, max=0.01, step=1e-5),
        f("n_steps", "int", 128, "Rollout", min=16, max=2048, step=16), f("n_epochs", "int", 10, "Epoch", min=1, max=100, step=1),
        f("batch_size", "int", 512, "Batch", min=16, max=4096, step=16), f("encoder_hidden", "text", "128,64", "编码器隐层"),
        f("encoder_out_dim", "int", 32, "编码器输出", min=8, max=512, step=8), f("mwl_hold_window", "int", 5, "持仓窗口", min=1, max=60, step=1),
        f("mwl_vol_window", "int", 20, "波动窗口", min=1, max=120, step=1), f("mwl_sigma_multiplier", "float", 2.0, "波动倍数", min=0, max=10, step=0.1),
        f("mwl_absolute_threshold", "float", 0.06, "绝对阈值", min=0, max=1, step=0.01), f("mwl_amount_ma_min", "float", 1e8, "成交额下限", min=0, step=1e7),
    ),
    "ml": schema(
        f("path", "select", "path1", "ML路径", options=["path1", "path2", "path3", "path4", "path5"]), f("bundle", "bundle", "", "P3数据包"),
        f("train_start", "date", "2018-01-02", "训练开始"), f("train_end", "date", "2024-12-31", "训练结束"), f("valid_start", "date", "2025-01-02", "验证开始"), f("valid_end", "date", "2025-06-30", "验证结束"), f("test_start", "date", "2025-07-01", "测试开始"), f("test_end", "date", "2026-07-24", "测试结束"),
        f("top_k", "int", 5, "Top-K", min=1, max=100, step=1), f("seed", "int", 42, "随机种子", min=0, max=999999, step=1),
        f("num_leaves", "int", 63, "LightGBM叶子", min=2, max=1024, step=1), f("learning_rate", "float", 0.03, "学习率", min=1e-6, max=1, step=1e-3), f("min_data_in_leaf", "int", 31, "叶节点最小样本", min=1, max=10000, step=1), f("num_boost_round", "int", 2000, "Boost轮数", min=1, max=100000, step=100), f("feat_fraction", "float", 0.7, "特征采样", min=0, max=1, step=0.05), f("grid_search", "bool", True, "网格搜索"), f("gpu", "bool", False, "使用GPU"),
        f("iterations", "int", 2000, "CatBoost轮数", min=1, max=100000, step=100), f("depth", "int", 8, "CatBoost深度", min=1, max=16, step=1), f("l2_leaf_reg", "float", 3.0, "L2正则", min=0, max=100, step=0.1), f("n_d", "int", 64, "TabNet宽度", min=8, max=512, step=8), f("n_steps", "int", 5, "TabNet步数", min=1, max=20, step=1), f("lambda_sparse", "float", 0.001, "稀疏正则", min=0, max=1, step=0.0001), f("epochs", "int", 200, "Epoch", min=1, max=10000, step=10),
    ),
    "knn_lstm": schema(
        f("panel", "panel", "", "因子面板"), f("out_dir", "text", "", "输出目录"), f("window", "int", 20, "窗口", min=5, max=120, step=1), f("forward_period", "int", 1, "前瞻窗口", min=1, max=60, step=1), f("k_neighbors", "int", 20, "KNN邻居", min=1, max=200, step=1), f("lstm_hidden", "int", 256, "LSTM隐层", min=32, max=1024, step=32), f("lstm_layers", "int", 2, "LSTM层数", min=1, max=4, step=1), f("lstm_dropout", "float", 0.3, "Dropout", min=0, max=0.9, step=0.05), f("lstm_lr", "float", 0.001, "学习率", min=1e-6, max=0.1, step=1e-4), f("epochs", "int", 50, "Epoch", min=1, max=10000, step=10), f("batch_size", "int", 2048, "Batch", min=16, max=16384, step=16), f("num_workers", "int", 3, "Workers", min=0, max=32, step=1), f("max_samples_per_stock", "int", 2000, "每股样本上限", min=1, max=100000, step=100), f("lstm_weight", "float", 0.6, "LSTM权重", min=0, max=1, step=0.05), f("knn_weight", "float", 0.4, "KNN权重", min=0, max=1, step=0.05), f("confidence", "float", 0.55, "置信度", min=0, max=1, step=0.01), f("train_end", "date", "2024-12-31", "训练结束"), f("test_start", "date", "2025-01-01", "测试开始"), f("device", "select", "cuda", "设备", options=["cuda", "cpu"]), f("seed", "int", 42, "随机种子", min=0, max=999999, step=1),
    ),
    "joint": schema(
        f("panel", "panel", "", "因子面板"), f("out_dir", "text", "", "输出目录"), f("start_date", "date", "2024-01-01", "训练开始"), f("end_date", "date", "2024-06-30", "训练结束"), f("total_timesteps", "int", 2000000, "训练步数", min=10000, max=10000000, step=10000), f("top_k", "int", 20, "Top-K", min=1, max=100, step=1), f("forward_period", "int", 10, "前瞻窗口", min=1, max=60, step=1), f("window", "int", 10, "窗口", min=5, max=120, step=1), f("lstm_hidden", "int", 128, "LSTM隐层", min=32, max=128, step=32), f("lstm_layers", "int", 1, "LSTM层数", min=1, max=3, step=1), f("n_envs", "int", 1, "环境数", min=1, max=1, step=1), f("learning_rate", "float", 3e-4, "学习率", min=1e-6, max=0.01, step=1e-5), f("n_factors", "int", 0, "因子数", min=0, max=1000, step=10), f("cost_bps", "float", 30.0, "交易成本", min=0, max=100, step=1), f("max_position_pct", "float", 0.05, "单股上限", min=0.005, max=0.2, step=0.005), f("max_industry_pct", "float", 0.30, "行业上限", min=0, max=1, step=0.05), f("reward_type", "select", "return", "Reward", options=["return", "excess", "sharpe", "sortino", "mean_variance"]), f("seed", "int", 42, "随机种子", min=0, max=999999, step=1), f("max_grad_norm", "float", 0.5, "梯度裁剪", min=0.1, max=5, step=0.1), f("target_kl", "float", 0, "KL阈值", min=0, max=1, step=0.01), f("batch_size", "int", 64, "Batch", min=16, max=64, step=16), f("n_steps", "int", 128, "Rollout", min=16, max=128, step=16), f("n_epochs", "int", 10, "Epoch", min=1, max=100, step=1), f("use_knn", "bool", False, "启用KNN"), f("knn_k", "int", 20, "KNN K", min=1, max=200, step=1), f("knn_ref_lookback", "int", 750, "KNN回看", min=50, max=5000, step=50), f("knn_max_ref_days", "int", 12, "KNN最大参考日", min=1, max=100, step=1), f("rebalance_days", "int", 20, "调仓周期", min=1, max=60, step=1),
    ),
    "walkforward": schema(
        f("panel", "panel", "", "因子面板"), f("steps", "int", 100000, "训练步数", min=10000, max=10000000, step=10000),
        f("train_years", "int", 3, "训练年数", min=1, max=20, step=1), f("val_years", "int", 1, "验证年数", min=1, max=10, step=1),
    ),
    "matrix": schema(
        f("panel", "panel", "", "因子面板"), f("steps", "int", 100000, "训练步数", min=10000, max=10000000, step=10000),
        f("max_position_pcts", "json", "[0.01,0.02,0.05,0.08]", "仓位候选"), f("top_ks", "json", "[10,20,30,50]", "Top-K候选"),
        f("rebalance_days_list", "json", "[10,20,40]", "调仓候选"), f("lstm_hiddens", "json", "[64,128]", "隐层候选"),
    ),
    "panel": schema(
        f("pool", "select", "hs300", "股票池", options=["hs300", "csi500", "cs800", "sse50"]),
        f("start_date", "date", "2004-01-02", "开始日期"), f("end_date", "date", "", "结束日期"), f("include_fundamental", "bool", False, "包含基本面"),
    ),
    "p3": schema(
        f("factor_panel", "panel", "", "因子面板"), f("out_dir", "text", "p3_hs300", "输出目录"), f("forward_period", "int", 10, "前瞻窗口", min=1, max=60, step=1),
        f("start_date", "date", "2004-01-01", "开始日期"), f("end_date", "date", "2026-07-24", "结束日期"),
    ),
    "wavehunter": {"version": "wavehunter_v1", "parameters": [], "fixed": {"env_type": "wavehunter_lstm"}},
    "wavehunter_v2": {"version": "wavehunter_v2"},
}


def get_schema(name: str) -> dict:
    result = dict(MODULE_SCHEMAS.get(name, {"parameters": [], "fixed": {}}))
    result["module"] = name
    result["version"] = result.get("version", "schema_v1")
    return result
