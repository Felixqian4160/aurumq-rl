"""v9_v2 — 多头 ONNX 推理包装 (5 输出: action + a1 + a2 + peak + b1)

修复:
  - 旧 inference.py 只解单输出 'action' → 模拟端拿不到 4 头概率
  - v9_v2 解析 5 个输出，按类名后缀区分: action/a1/a2/peak/b1
  - 兼容旧 1 头 ONNX (无 4 头): 自动退化为 sigmoid(weights) 选股
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# 四头概率阈值 (默认 0.5)
DEFAULT_THRESHOLDS = {
    'a1_buy': 0.55,
    'a2_hold': 0.55,
    'peak_sell': 0.55,
    'b1_wait': 0.50,
}

class MultiHeadInference:
    """v9_v2 ONNX 推理：返回 (action, a1, a2, peak, b1) 五个 (S,) 概率/权重。"""

    def __init__(self, model_dir: str | Path, providers: Optional[list] = None):
        if ort is None:
            raise ImportError("onnxruntime 未安装")
        model_dir = Path(model_dir)
        onnx_path = model_dir / "policy.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if providers is None:
            providers = ort.get_available_providers()
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]
        # 缓存：识别 4 头存在
        names_lower = [n.lower() for n in self._output_names]
        self.has_a1 = any('a1' in n for n in names_lower)
        self.has_a2 = any('a2' in n for n in names_lower)
        self.has_peak = any('peak' in n for n in names_lower)
        self.has_b1 = any('b1' in n for n in names_lower)
        self.has_4heads = self.has_a1 and self.has_a2 and self.has_peak and self.has_b1
        # 元数据
        meta_path = model_dir / "metadata.json"
        self.metadata = None
        if meta_path.exists():
            try:
                import json
                self.metadata = json.loads(meta_path.read_text())
            except Exception:
                pass

    @property
    def input_name(self) -> str:
        return self._input_name

    def predict(self, observation: np.ndarray) -> dict:
        """返回 {action, a1, a2, peak, b1}，每项 shape (S,) 的 np.float32 数组。
        action: 旧版是 sigmoid(weights)，新版直接 sigmoid 输出
        a1/a2/peak/b1: sigmoid(logit) 概率；缺失时 None
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs_input = obs[np.newaxis, ...]
        else:
            obs_input = obs
        outputs = self.session.run(None, {self._input_name: obs_input})
        # 5 输出按名字归位
        result = {'action': None, 'a1': None, 'a2': None, 'peak': None, 'b1': None}
        for name, arr in zip(self._output_names, outputs):
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            arr = np.squeeze(arr)
            n = name.lower()
            if 'a1' in n:
                result['a1'] = 1.0 / (1.0 + np.exp(-arr))
            elif 'a2' in n:
                result['a2'] = 1.0 / (1.0 + np.exp(-arr))
            elif 'peak' in n:
                result['peak'] = 1.0 / (1.0 + np.exp(-arr))
            elif 'b1' in n:
                result['b1'] = 1.0 / (1.0 + np.exp(-arr))
            elif 'action' in n or 'output' in n:
                # 默认 1 头时已是 sigmoid 过权重；若是 logit 需 sigmoid
                result['action'] = 1.0 / (1.0 + np.exp(-arr)) if arr.min() < 0 or arr.max() > 1 else arr
        # 兼容旧 1 头 ONNX
        if result['action'] is None and outputs:
            arr = np.asarray(outputs[0], dtype=np.float32).squeeze()
            result['action'] = 1.0 / (1.0 + np.exp(-arr)) if arr.min() < 0 or arr.max() > 1 else arr
        return result

# 兼容旧 RlAgentInference 入口
class RlAgentV10Inference(MultiHeadInference):
    """上层 API: 与 wavehunter_v2_api 一致的 predict() 接口, 返回 scores 给 simulation 选股。"""
    def predict_scores(self, observation: np.ndarray) -> np.ndarray:
        """返回 sigmoid(action) 作为选股权重；4 头概率另存 self._last_probs"""
        out = self.predict(observation)
        self._last_probs = out
        return out['action']
