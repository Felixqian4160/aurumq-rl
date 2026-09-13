"""Custom Stable-Baselines3 callbacks for wandb + JSONL co-logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# Optional dependency — SB3 only needed for training
try:
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from aurumq_rl.wandb_integration import WandbLogger


if SB3_AVAILABLE:

    class WandbMetricsCallback(BaseCallback):  # type: ignore[misc]
        """Push SB3 internal logger metrics to wandb + JSONL every log_freq steps.

        Parameters
        ----------
        wandb_logger:
            WandbLogger instance (no-op when disabled).
        jsonl_path:
            JSONL file path for offline metric records.
        log_freq:
            Push frequency in timesteps. Default 1000.
        """

        def __init__(
            self,
            wandb_logger: WandbLogger,
            jsonl_path: Path,
            log_freq: int = 1000,
        ) -> None:
            super().__init__(verbose=0)
            self._wandb = wandb_logger
            self._jsonl = Path(jsonl_path)
            self._log_freq = log_freq
            self._t_start: float | None = None
            self._steps_at_start = 0
            self._recent_rewards: list[float] = []  # rollout 内的每步 reward
            # P2: divergence 检测 — KL 连续 N 次超过阈值则自动终止
            self._kl_history: list[float] = []
            self._divergence_threshold: float = 10.0  # approx_kl > 10 视为发散
            self._divergence_patience: int = 5  # 连续 5 次超阈值则停止

        def _on_step(self) -> bool:
            # 收当前步的 reward（VecEnv 一次性给 n_envs 个）
            try:
                rewards = self.locals.get('rewards', [])
                if hasattr(rewards, '__iter__'):
                    self._recent_rewards.extend(float(r) for r in rewards)
            except Exception:
                pass

            # P2: divergence 检测 — 监控 approx_kl，连续超阈值则终止训练
            # 🔴 FIX (2026-08-14): SB3 logger 的 train/approx_kl 只在 PPO update 后
            # 更新一次，同一个 rollout 内（n_steps 步）每次 _on_step 读到的都是
            # 同一个旧值 → 一旦某次 KL>阈值，后面 5 步都读到相同值 → 误判连续发散。
            # 修复: KL 值与上一次相同时不计数（去重），只有真正连续发散才终止。
            try:
                kl = float(self.logger.name_to_value.get('train/approx_kl', 0))
                # 去重: 只有 KL 值变化才记录（新 update 的 KL）
                if not self._kl_history or abs(kl - self._kl_history[-1]) > 1e-6:
                    self._kl_history.append(kl)
                    if len(self._kl_history) > 20:
                        self._kl_history = self._kl_history[-20:]
                recent_kl = self._kl_history[-5:] if len(self._kl_history) >= 5 else self._kl_history
                if len(recent_kl) >= 5 and all(k > self._divergence_threshold for k in recent_kl):
                    print(f"\n[ALERT] 🚨 训练发散检测: 近 5 次 KL={recent_kl} > 阈值 {self._divergence_threshold}")
                    print(f"[ALERT] 自动终止训练以避免浪费 GPU 时间 (step {self.num_timesteps})")
                    return False  # 返回 False 终止训练
            except Exception:
                pass

            if self.num_timesteps % self._log_freq != 0:
                return True

            raw_metrics: dict[str, Any] = dict(self.logger.name_to_value)
            # 用我们自己收集的 reward 覆盖 SB3 logger 里过时的 0.0
            # P1 fix: 这是 per-step reward 均值（非 episode reward），改名避免误导
            if self._recent_rewards:
                raw_metrics['rollout/step_rew_mean'] = sum(self._recent_rewards) / len(self._recent_rewards)
                # 保持向后兼容：同时写入旧 key，前端/summary 读取不受影响
                raw_metrics['rollout/ep_rew_mean'] = raw_metrics['rollout/step_rew_mean']
                self._recent_rewards = []
            if not raw_metrics:
                return True

            self._wandb.log_metrics(raw_metrics, step=self.num_timesteps)
            self._append_jsonl(raw_metrics)
            return True

        def _append_jsonl(self, metrics: dict[str, Any]) -> None:
            """Append metrics to JSONL file.

            Writes a record that conforms to the canonical
            :class:`aurumq_rl.metrics.TrainingMetrics` schema (so
            ``summarize_metrics`` can read it back), with the raw SB3 keys
            preserved under ``extra`` for debugging.
            """
            self._jsonl.parent.mkdir(parents=True, exist_ok=True)

            def _f(key: str, default: float = 0.0) -> float:
                v = metrics.get(key)
                if v is None:
                    return default
                try:
                    return float(v.item() if hasattr(v, "item") else v)
                except (TypeError, ValueError):
                    return default

            algo_name = type(self.model).__name__ if self.model is not None else "PPO"
            if algo_name not in {"PPO", "A2C", "SAC"}:
                algo_name = "PPO"

            # Compute fps from elapsed wall time. SB3 only emits time/fps in
            # the rollout-summary frame, so most callback flushes wouldn't
            # see it and would default to 0 → mean_fps = 0 in summary.
            if self._t_start is None:
                self._t_start = time.monotonic()
                self._steps_at_start = self.num_timesteps
                fps_now = 0
            else:
                elapsed = time.monotonic() - self._t_start
                steps_since_start = self.num_timesteps - self._steps_at_start
                fps_now = int(steps_since_start / elapsed) if elapsed > 0 else 0

            sb3_fps = _f("time/fps")
            record_fps = int(sb3_fps) if sb3_fps > 0 else fps_now

            record = {
                "timestep": self.num_timesteps,
                "episode_reward_mean": _f("rollout/ep_rew_mean"),
                "policy_loss": _f("train/policy_gradient_loss") or _f("train/loss"),
                "value_loss": _f("train/value_loss"),
                "entropy": -_f("train/entropy_loss"),  # SB3 reports entropy_loss = -E
                "explained_variance": _f("train/explained_variance"),
                "learning_rate": _f("train/learning_rate", default=1e-9),
                "fps": record_fps,
                "algorithm": algo_name,
                "extra": metrics,
            }

            try:
                with self._jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            except OSError:
                pass

    class CheckpointArtifactCallback(BaseCallback):  # type: ignore[misc]
        """Save checkpoint locally and upload to wandb artifact registry.

        只保留最新的 1 个 checkpoint（用于续训），旧的自动删除。
        单个 checkpoint 约 2GB（[1024,1024] 策略网络），保留太多会撑爆磁盘。

        Parameters
        ----------
        wandb_logger:
            WandbLogger (no-op when disabled — local save still happens).
        save_path:
            Local checkpoint directory.
        name_prefix:
            Filename prefix (e.g. "ppo").
        save_freq:
            Save frequency in timesteps. Default 100k.
        artifact_type:
            Wandb artifact type label.
        keep_latest:
            保留最新 N 个 checkpoint。默认 1（只保留最新）。
        """
        def __init__(
            self,
            wandb_logger: WandbLogger,
            save_path: Path,
            name_prefix: str = "model",
            save_freq: int = 100_000,
            artifact_type: str = "model",
            keep_latest: int = 1,
        ) -> None:
            super().__init__(verbose=0)
            self._wandb = wandb_logger
            self._save_path = Path(save_path)
            self._name_prefix = name_prefix
            self._save_freq = save_freq
            self._artifact_type = artifact_type
            self._keep_latest = keep_latest

        def _on_step(self) -> bool:
            if self.num_timesteps % self._save_freq != 0:
                return True

            self._save_path.mkdir(parents=True, exist_ok=True)
            ckpt_name = f"{self._name_prefix}_{self.num_timesteps}_steps"
            ckpt_path = self._save_path / f"{ckpt_name}.zip"

            try:
                self.model.save(str(ckpt_path))
            except Exception:
                return True

            # 删除旧 checkpoint，只保留最新的 N 个
            self._cleanup_old_checkpoints()

            self._wandb.log_artifact(
                path=ckpt_path,
                name=ckpt_name,
                artifact_type=self._artifact_type,
                metadata={"timestep": self.num_timesteps},
            )

            return True

        def _cleanup_old_checkpoints(self) -> None:
            """删除旧 checkpoint，只保留最新的 keep_latest 个。"""
            ckpts = sorted(
                self._save_path.glob(f"{self._name_prefix}_*_steps.zip"),
                key=lambda p: p.stat().st_mtime,
            )
            if len(ckpts) <= self._keep_latest:
                return
            for old in ckpts[: -self._keep_latest]:
                try:
                    old.unlink()
                except OSError:
                    pass

    class ProgressLogCallback(BaseCallback):  # type: ignore[misc]
        """定期输出训练进度到 stdout（带 tqdm 风格进度条）。

        train_runner 逐行读 stdout，tqdm 的 \\r 不会被捕获，
        所以用 print(..., flush=True) 输出带 \\n 的进度行。
        """
        def __init__(self, total_timesteps: int, log_every: int = 5000) -> None:
            super().__init__(verbose=0)
            self._total = total_timesteps
            self._log_every = log_every
            self._last_log = 0
            self._start_time: float | None = None

        def _on_training_start(self) -> None:
            import time as _time
            self._start_time = _time.time()

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_log < self._log_every:
                return True
            self._last_log = self.num_timesteps

            import time as _time
            elapsed = _time.time() - (self._start_time or _time.time())
            pct = self.num_timesteps / self._total * 100
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            remaining = (self._total - self.num_timesteps) / fps if fps > 0 else 0

            # 进度条
            bar_len = 30
            filled = int(bar_len * self.num_timesteps / self._total)
            bar = '█' * filled + '░' * (bar_len - filled)

            # 关键指标（logger 可能还没初始化）
            kl = rew = ev = 0.0
            try:
                kl = self.logger.name_to_value.get('train/approx_kl', 0) or 0
                rew = self.logger.name_to_value.get('rollout/ep_rew_mean', 0) or 0
                ev = self.logger.name_to_value.get('train/explained_variance', 0) or 0
            except Exception:
                pass

            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            h2, rem2 = divmod(int(remaining), 3600)
            m2, s2 = divmod(rem2, 60)
            print(
                f"[{bar}] {pct:5.1f}% | "
                f"step {self.num_timesteps:>9,}/{self._total:,} | "
                f"fps {fps:.0f} | "
                f"rew {rew:+.4f} | kl {kl:.4f} | ev {ev:.4f} | "
                f"elapsed {h}h{m:02d}m | ETA {h2}h{m2:02d}m",
                flush=True,
            )
            return True

else:
    # Stubs when SB3 not installed
    class WandbMetricsCallback:  # type: ignore[no-redef]
        """Placeholder when stable-baselines3 is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "stable-baselines3 not installed. Install with: pip install aurumq-rl[train]"
            )

    class CheckpointArtifactCallback:  # type: ignore[no-redef]
        """Placeholder when stable-baselines3 is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "stable-baselines3 not installed. Install with: pip install aurumq-rl[train]"
            )


__all__ = [
    "SB3_AVAILABLE",
    "WandbMetricsCallback",
    "CheckpointArtifactCallback",
    "ProgressLogCallback",
]
