"""
onnx_export_v2.py — PerStockEncoderPolicy 专用 ONNX 导出器
=========================================================

为 GPU-v2 自定义策略写一个最小化 ONNX 包装：
  obs (B, S, F) → features_extractor → latent_pi
                → mlp_extractor.forward_actor → action_net
                → action_mean (B, S, 1) [deterministic mean]

为什么不用标准 onnx_export.py：
  标准的 DeterministicPolicy 期望 ActorCriticPolicy 有 mlp_extractor.forward_actor()
  返回 latent_pi，但 PerStockEncoderPolicy 的 mlp_extractor 是 _IdentityMlpExtractor，
  forward 已被 monkeypatch 成返回 (features, features) — 而 features_extractor 返回的
  是 dict 而非 tensor，所以标准路径的 features_extractor.forward(obs) 在 ONNX trace 时
  会因为输入已经是 dict 而炸 (F.linear 收到 dict)。

替代方案：直接手工 trace，绕开 mlp_extractor。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def export_perstock_policy_to_onnx(
    model_path: Path,
    output_dir: Path,
    obs_shape: tuple[int, int],          # (n_stocks, n_factors)
    training_timesteps: int = 0,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """把 PerStockEncoderPolicy 训练的 PPO 导出为 ONNX。

    Parameters
    ----------
    model_path : Path
        ppo_final.zip 路径
    output_dir : Path
        输出目录（写入 policy.onnx + metadata.json）
    obs_shape : (n_stocks, n_factors)
        与训练时一致
    training_timesteps : int
    extra_metadata : dict
        写入 metadata.json 的额外字段（stock_codes / factor_names / ...）

    Returns
    -------
    onnx_path : Path
        生成的 policy.onnx 路径
    """
    import torch
    from stable_baselines3 import PPO

    from aurumq_rl.policy import PerStockEncoderPolicy
    from aurumq_rl.gpu_env import GPUStockPickingEnv

    n_stocks, n_factors = obs_shape
    print(f"[onnx_export_v2] loading model from {model_path}")
    # 注册 custom_objects 以兼容 PerStockEncoderPolicy
    # 必须 cuda 加载（GPURolloutBuffer 要求）
    load_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[onnx_export_v2] device={load_device}")
    model = PPO.load(str(model_path), device=load_device, custom_objects={
        "PerStockEncoderPolicy": PerStockEncoderPolicy,
    })
    policy = model.policy
    policy.eval()

    # 检查策略类型
    if not isinstance(policy, PerStockEncoderPolicy):
        raise TypeError(
            f"Expected PerStockEncoderPolicy, got {type(policy).__name__}. "
            "Use aurumq_rl.onnx_export.export_sb3_policy_to_onnx for standard policies."
        )

    # 包装一个最小 module: obs → action_mean
    class PerStockDeterministic(torch.nn.Module):
        def __init__(self, p: PerStockEncoderPolicy):
            super().__init__()
            self.features_extractor = p.features_extractor
            self.action_net = p.action_net
            self.log_std = p.log_std  # 兼容 future；这里只用 mean

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            # obs: (B, S, F)  fp32
            features = self.features_extractor(obs)
            # PerStockExtractor 返回 dict: {"per_stock": (B,S,out), "pooled": (B,2*out)}
            # action_net 用的是 per_stock
            latent_pi = features["per_stock"]
            action_mean = self.action_net(latent_pi)  # (B, S, 1)
            return action_mean.squeeze(-1)             # (B, S)

    wrapper = PerStockDeterministic(policy)
    wrapper = wrapper.to(load_device)
    wrapper.eval()

    # 构造 dummy input trace
    dummy_obs = torch.zeros(1, n_stocks, n_factors, dtype=torch.float32, device=load_device)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "policy.onnx"
    print(f"[onnx_export_v2] tracing + exporting → {onnx_path}")
    torch.onnx.export(
        wrapper,
        (dummy_obs,),
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={
            "obs": {0: "batch", 1: "n_stocks"},
            "action": {0: "batch", 1: "n_stocks"},
        },
        verbose=False,
    )
    print(f"[onnx_export_v2] ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Build metadata.json
    metadata = {
        "algorithm": "PPO",
        "framework": "gpu_v2",
        "policy_class": "PerStockEncoderPolicy",
        "training_timesteps": int(training_timesteps),
        "obs_shape": [int(n_stocks), int(n_factors)],
        "action_shape": [int(n_stocks)],
        "obs_normalized": False,
        "git_sha": _get_git_sha(),
        "onnx_opset": 17,
        **(extra_metadata or {}),
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[onnx_export_v2] metadata saved: {meta_path}")

    return onnx_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("用法: python onnx_export_v2.py <ppo_final.zip> <output_dir> <n_stocks> <n_factors>")
        sys.exit(1)
    zip_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    n_s = int(sys.argv[3])
    n_f = int(sys.argv[4]) if len(sys.argv) > 4 else 298
    export_perstock_policy_to_onnx(
        zip_path, out_dir, (n_s, n_f), training_timesteps=0,
    )