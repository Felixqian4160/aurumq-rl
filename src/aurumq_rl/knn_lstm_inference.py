"""KNN+LSTM 模型推理引擎（供模拟交易使用）。

与 RlAgentInference 的差异：
- RL 模型：每时间步输入当天截面因子 (n_stocks, n_factors) → 输出每只股票 action
- KNN+LSTM：每只股票需要 window 天历史窗口 (window, n_factors) 才能预测

因此本推理器维护内部窗口历史，逐日累积，够 window 天后开始输出分数。
分数 = lstm_weight * LSTM概率 + knn_weight * KNN概率（与训练时 fuse_predictions 一致）。

特征标准化：推理窗口用窗口内自身数据做 z-score（与训练时每股票 z-score 近似，
因子序列相对平稳，rolling 标准化保持同一分布尺度）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

_ORT_PROVIDERS = ["CPUExecutionProvider"]


def _rolling_zscore(seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对 (window, n_factors) 序列按列 z-score（用序列自身统计量）。"""
    mu = seq.mean(axis=0, keepdims=True)
    sigma = seq.std(axis=0, keepdims=True)
    sigma[sigma < eps] = 1.0
    out = (seq - mu) / sigma
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class KnnLstmInference:
    """加载 KNN+LSTM 模型目录（lstm_model.onnx + knn_train_X.npy + metadata.json）。"""

    def __init__(self, model_dir: Path | str) -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        onnx_path = model_dir / "lstm_model.onnx"
        knn_x_path = model_dir / "knn_train_X.npy"
        knn_y_path = model_dir / "knn_train_y.npy"
        meta_path = model_dir / "metadata.json"

        if not onnx_path.exists():
            raise FileNotFoundError(f"LSTM ONNX not found: {onnx_path}")
        if not knn_x_path.exists() or not knn_y_path.exists():
            raise FileNotFoundError(f"KNN 训练数据缺失: {model_dir}")
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {meta_path}")

        self._lstm = ort.InferenceSession(str(onnx_path), providers=_ORT_PROVIDERS)
        self._input_name = self._lstm.get_inputs()[0].name
        self._output_name = self._lstm.get_outputs()[0].name

        self.knn_X = np.load(knn_x_path)  # (n_train, window*n_factors) float32
        self.knn_y = np.load(knn_y_path)  # (n_train,)
        self._nn = None  # NearestNeighbors 惰性构建

        meta = json.loads(meta_path.read_text())
        self.window = int(meta.get("window", 20))
        self.k = int(meta.get("k_neighbors", 20))
        self.lstm_weight = float(meta.get("lstm_weight", 0.6))
        self.knn_weight = float(meta.get("knn_weight", 0.4))
        factor_names = list(meta.get("factor_names") or [])
        self.factor_names = factor_names
        self.n_factors = len(factor_names)
        if self.n_factors <= 0:
            raise ValueError(f"metadata.factor_names 为空: {model_dir}")
        if self.knn_X.ndim != 2:
            raise ValueError(f"knn_train_X 必须是二维数组，实际 shape={self.knn_X.shape}")
        if self.knn_y.ndim != 1 or self.knn_y.shape[0] != self.knn_X.shape[0]:
            raise ValueError(
                f"KNN X/Y 样本数不一致: X={self.knn_X.shape}, y={self.knn_y.shape}"
            )
        expected_knn_dim = self.window * self.n_factors
        if self.knn_X.shape[1] != expected_knn_dim:
            raise ValueError(
                f"KNN 特征维度不匹配: X.shape[1]={self.knn_X.shape[1]}, "
                f"expected window*n_factors={self.window}*{self.n_factors}={expected_knn_dim}"
            )

        # 窗口历史：每步存 (n_stocks, n_factors)
        self._history: list[np.ndarray] = []

        # 兼容模拟引擎读取的 metadata 字段
        self._metadata = SimpleNamespace(
            algorithm="knn_lstm",
            training_timesteps=int(meta.get("train_samples", 0)),
            stock_codes=None,  # 不按训练股票池对齐（面板即同池）
            factor_names=factor_names,
            obs_shape=(0, self.n_factors),  # 模拟引擎 prod 检查用，knn_lstm 分支不走该检查
        )
        # 调试：确认 factor_names 非空
        assert factor_names, f"metadata.json 缺少 factor_names: {model_dir}"
        logger.info("KnnLstmInference ready: window=%d k=%d factors=%d",
                    self.window, self.k, self.n_factors)

    @property
    def metadata(self):
        return self._metadata

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """输入 (n_stocks, n_factors) 当天截面，输出 (n_stocks,) 融合分数。

        历史不足 window 天时返回全零（模拟引擎不交易）。
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"KNN+LSTM obs 必须是二维 (n_stocks,n_factors)，实际 shape={obs.shape}")
        if obs.shape[1] != self.n_factors:
            raise ValueError(
                f"KNN+LSTM obs 因子数不匹配: obs.shape[1]={obs.shape[1]}, "
                f"metadata.factor_names={self.n_factors}"
            )
        n_stocks = obs.shape[0]
        if n_stocks == 0:
            return np.zeros(0, dtype=np.float32)

        self._history.append(obs)
        if len(self._history) < self.window:
            return np.zeros(n_stocks, dtype=np.float32)

        hist = np.stack(self._history[-self.window:])  # (window, n_stocks, n_factors)

        # 批量 z-score: 每只股票独立标准化
        seqs = np.transpose(hist, (1, 0, 2))  # (n_stocks, window, n_factors)
        seqs_z = np.stack([_rolling_zscore(s) for s in seqs])  # (n_stocks, window, n_factors)

        # 批量 LSTM 推理（一次 session.run 处理全部股票）
        out = self._lstm.run([self._output_name], {self._input_name: seqs_z.astype(np.float32)})[0]
        lstm_probs = np.asarray(out).reshape(n_stocks)

        # 批量 KNN 推理（一次 kneighbors 查全部股票）
        knn_probs = self._knn_proba_batch(seqs_z)

        scores = self.lstm_weight * lstm_probs + self.knn_weight * knn_probs

        # 截断历史（只保留 window 天）
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

        return scores.astype(np.float32)

    def _knn_proba_batch(self, seqs_z: np.ndarray) -> np.ndarray:
        """批量 KNN 概率：一次查询所有股票窗口的邻居。"""
        from sklearn.neighbors import NearestNeighbors

        if self._nn is None:
            self._nn = NearestNeighbors(
                n_neighbors=min(self.k, self.knn_X.shape[0]),
                algorithm="auto", metric="euclidean", n_jobs=-1,
            )
            self._nn.fit(self.knn_X)

        flat = seqs_z.reshape(len(seqs_z), -1)  # (n_stocks, window*n_factors)
        _, idx = self._nn.kneighbors(flat)
        return np.array([self.knn_y[i].mean() for i in idx], dtype=np.float32)
