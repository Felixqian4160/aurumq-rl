"""v10.1 ONNX inference wrapper."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None


class MultiHeadV10_1Inference:
    def __init__(self, model_dir: str | Path, providers: Optional[list] = None):
        if ort is None: raise ImportError('onnxruntime 未安装')
        self.model_dir=Path(model_dir); self.version='v10.1'
        meta=self.model_dir/'metadata.json'; self.metadata=json.loads(meta.read_text()) if meta.is_file() else {}
        if self.metadata.get('version') != 'v10.1' or self.metadata.get('model') != 'wavehunter_v10_1_4heads':
            raise ValueError('模型 metadata 不是 v10.1 独立模型')
        panel=str(self.metadata.get('panel',''))
        if 'wavehunter_v10_1_' not in panel: raise ValueError('v10.1 模型 panel 不是 v10.1 面板')
        onnx=self.model_dir/'policy.onnx'
        if not onnx.is_file(): raise FileNotFoundError(onnx)
        self.session=ort.InferenceSession(str(onnx),providers=providers or ort.get_available_providers())
        self.input_name=self.session.get_inputs()[0].name
        self.output_names=[x.name for x in self.session.get_outputs()]
        required={'action','a1','a2','peak','b1'}
        if not required.issubset(set(self.output_names)): raise ValueError(f'v10.1 ONNX outputs mismatch: {self.output_names}')

    def predict(self, observation: np.ndarray) -> dict[str,np.ndarray]:
        x=np.asarray(observation,dtype=np.float32)
        if x.ndim==1:x=x[None,:]
        x=np.nan_to_num(x,nan=0.,posinf=0.,neginf=0.)
        vals=self.session.run(None,{self.input_name:x})
        out={n:np.asarray(v,dtype=np.float32).squeeze(0) for n,v in zip(self.output_names,vals)}
        for k in ('a1','a2','peak','b1'):
            out[k]=1./(1.+np.exp(-out[k]))
        out['action']=np.clip(out['action'],0.,1.)
        return out
