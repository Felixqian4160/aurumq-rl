"""矩阵工作线程模块 — 训练/模拟协调逻辑。"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from matrix_config import (
    _AURUMQ_ROOT, FIXED_SIM, POOL_TRAIN_PROFILES, _build_runs_for_phase,
)
from matrix_registry import (
    MATRIX_DIR, REGISTRY_FILE, LEDGER_DIR,
    _load_registry, _save_registry, _save_summary_cache, _append_result,
    _latest_ledger, _event,
)

_lock = threading.Lock()
_matrix_stop = threading.Event()
_matrix_session_id: Optional[str] = None
_matrix_state = {
    "running": False, "status": "idle", "current": 0, "total": 24,
    "run_id": None, "progress": "", "error": "", "completed": 0,
}


def _restore_state_from_registry() -> None:
    """WebUI重启后恢复状态。"""
    global _matrix_session_id
    try:
        d = _load_registry()
        runs = d["runs"]
        changed = False
        for run in runs:
            if run.get("status") == "running":
                run.update(status="pending", error="WebUI重启后自动恢复，重新排队", finished_at=None)
                changed = True
        if changed:
            d["runs"] = runs
            _save_registry(d)
        done = [r for r in runs if r.get("status") == "done"]
        next_run = next((r for r in runs if r.get("status") not in ("done",)), None)
        _matrix_session_id = d.get("session_id")
        _matrix_state.update(
            current=(runs.index(next_run) + 1 if next_run else len(runs)),
            completed=len(done),
            progress=(f"{len(done)}/{len(runs)} {next_run['run_id']}" if next_run else f"{len(done)}/{len(runs)}"),
            status="stopped" if done else "idle",
        )
    except Exception:
        pass


def _effective_train(run: dict) -> dict:
    """获取有效的训练配置。"""
    requested = dict(run.get("train") or {})
    effective = dict(POOL_TRAIN_PROFILES[run["pool"]], total_timesteps=run["steps"])
    run["requested_config"] = requested
    run["effective_config"] = effective
    run["train"] = effective
    return effective


def _find_model_artifacts(out_dir: Path, run_id: str) -> tuple[Path, bool]:
    """定位已有模型。"""
    model_dir = out_dir
    return model_dir, (model_dir / "ppo_final.zip").exists()


def _recover_model_artifacts(run: dict, model_dir: Path) -> None:
    """从已完成但半成品的 ppo_final.zip 恢复 ONNX/metadata。"""
    import datetime
    from aurumq_rl.data_loader import FactorPanelLoader, UniverseFilter
    from aurumq_rl.onnx_export import export_sb3_policy_to_onnx
    train = run.get("train", {})
    panel_path = _AURUMQ_ROOT / "data" / run["panel"]
    uf = {
        "main_board_non_st": UniverseFilter.MAIN_BOARD_NON_ST,
        "all_a": UniverseFilter.ALL_A,
        "hs300": UniverseFilter.HS300,
        "zz500": UniverseFilter.ZZ500,
    }.get(train.get("universe_filter"), UniverseFilter.MAIN_BOARD_NON_ST)
    panel = FactorPanelLoader(parquet_path=panel_path).load_panel(
        datetime.date.fromisoformat(run["train_start"]),
        datetime.date.fromisoformat(run["train_end"]),
        n_factors=None,
        forward_period=int(train.get("forward_period", 20)),
        universe_filter=uf,
    )
    n_stocks, n_factors = len(panel.stock_codes), len(panel.factor_names)
    extra = {
        "universe": train.get("universe_filter"), "env_type": "wavehunter_lstm",
        "top_k": train.get("top_k", 20), "forward_period": train.get("forward_period", 20),
        "window": train.get("window", 20), "lstm_hidden": train.get("lstm_hidden", 64),
        "lstm_layers": train.get("lstm_layers", 1), "max_position_pct": train.get("max_position_pct", 0.02),
        "rebalance_days": train.get("rebalance_days", 20), "cost_bps": train.get("cost_bps", 15.0),
        "learning_rate": train.get("learning_rate", 1e-4), "max_grad_norm": train.get("max_grad_norm", 1.0),
        "n_steps": train.get("n_steps", 32), "batch_size": train.get("batch_size", 32),
        "factor_count": n_factors, "factor_names": list(panel.factor_names),
        "stock_codes": list(panel.stock_codes),
        "train_start_date": run["train_start"], "train_end_date": run["train_end"],
    }
    run_version = run.get("version", "v2")
    if run_version == "v3":
        import torch as _t
        _t.set_num_threads(1)
        from stable_baselines3 import PPO
        model = PPO.load(str(model_dir / "ppo_final.zip"), device="cpu")
        model.policy.eval()
        obs_shape = (n_stocks * int(train.get("window", 20)) * n_factors,)
        dummy = _t.zeros(1, *obs_shape, dtype=_t.float32)
        _t.onnx.export(
            model.policy, dummy, str(model_dir / "policy.onnx"),
            opset_version=14, input_names=["observation"], output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            export_params=True, dynamo=False,
        )
        import json as _json
        obs_shape_v3 = (n_stocks * int(train.get("window", 20)) * n_factors,)
        v3_meta = {
            "algorithm": "PPO", "model": "wavehunter_v3_double_head",
            "model_version": "wavehunter_v3",
            "reward_type": "absolute_return_v3", "reward_version": "wavehunter_v3_absolute_return",
            "training_timesteps": int(run["steps"]),
            "obs_shape": list(obs_shape_v3), "action_shape": [n_stocks],
            "factor_count": n_factors,
            **extra, "train_start_date": run["train_start"], "train_end_date": run["train_end"],
        }
        (model_dir / "metadata.json").write_text(_json.dumps(v3_meta, indent=2, ensure_ascii=False))
    else:
        export_sb3_policy_to_onnx(
            model_dir / "ppo_final.zip", model_dir,
            obs_shape=(n_stocks * int(train.get("window", 20)) * n_factors,),
            training_timesteps=int(run["steps"]), extra_metadata=extra,
        )
    _event("postprocess_recovered", run["run_id"], model_dir=str(model_dir), n_stocks=n_stocks, n_factors=n_factors)


def _external_train_identity() -> Optional[dict]:
    """读取真实训练子进程的 Run 身份。

    systemd-run 会同时留下 bash wrapper 和 Python 子进程；只认 Python
    训练进程，避免把 wrapper PID 当成训练身份，导致下一 run 提前抢锁。
    """
    try:
        out = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    except Exception:
        return None
    for line in out.splitlines():
        if 'train_wavehunter_v2.py' not in line and 'train_wavehunter_v3.py' not in line:
            continue
        if '/bin/bash -lc' in line or line.lstrip().startswith(('bash ', '/bin/bash ')):
            continue
        m = re.search(r'--out-dir\s+\S*matrix_(hs300_[^\s]+)', line)
        if not m:
            continue
        rid = m.group(1)
        steps = re.search(r'--total-timesteps\s+(\d+)', line)
        version = 'v3' if 'train_wavehunter_v3.py' in line else 'v2'
        parts = line.strip().split(None, 1)
        if not parts or not parts[0].isdigit():
            continue
        return {
            'pid': int(parts[0]),
            'run_id': rid, 'cmd': line.strip(),
            'steps': int(steps.group(1)) if steps else None,
            'version': version,
        }
    return None


def _run_processes(run_id: str, version: str = 'v3') -> list[int]:
    """返回指定矩阵 run 的真实 Python 训练 PID，不把 shell wrapper 算进去。"""
    script = 'train_wavehunter_v3.py' if version == 'v3' else 'train_wavehunter_v2.py'
    marker = f'matrix_{run_id}'
    try:
        out = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        if script not in line or marker not in line:
            continue
        if '/bin/bash -lc' in line or line.lstrip().startswith(('bash ', '/bin/bash ')):
            continue
        parts = line.strip().split(None, 1)
        if parts and parts[0].isdigit():
            pids.append(int(parts[0]))
    return pids


def _wait_external_train(identity: dict, timeout: float = 60 * 60 * 8) -> bool:
    """等待真实 Python 训练进程完成。"""
    start = time.time()
    pid = identity["pid"]
    run_id = identity.get("run_id", "")
    version = identity.get("version", "v3")
    while time.time() - start < timeout:
        if _matrix_stop.is_set():
            return False
        if _run_processes(run_id, version):
            time.sleep(5)
            continue
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(5)
    _event("train_timeout", run_id)
    return False


def _wait_train(run_id: str, timeout: float = 60 * 60 * 8, version: str = "v2") -> bool:
    """等待本次 run 的真实训练终态，不能把旧的 finished/idle 状态当成功。"""
    t0 = time.time()
    expected_out = f"matrix_{run_id}"
    model_dir = _AURUMQ_ROOT / "models" / expected_out
    while time.time() - t0 < timeout:
        if _matrix_stop.is_set():
            return False
        if version == "v3":
            from wavehunter_v2_api import wavehunter_v3_status
            st = wavehunter_v3_status()
        else:
            from wavehunter_v2_api import wavehunter_v2_status
            st = wavehunter_v2_status()
        status = st.get("status", "")
        out_name = st.get("out_name")
        active_pids = _run_processes(run_id, version)
        if active_pids or status in ("running", "starting"):
            time.sleep(5)
            continue
        # 只接受本 run 的 done/finished；旧 run 的终态不能放行。
        if status in ("done", "finished") and out_name in (None, expected_out):
            return (model_dir / "ppo_final.zip").exists()
        if status in ("failed", "stopped", "stale"):
            return False
        # API 状态文件可能短暂落后，但真实产物已经完整生成时允许进入后处理。
        if (model_dir / "ppo_final.zip").exists() and not active_pids:
            return True
        time.sleep(5)
    _event("train_timeout", run_id)
    return False


def _wait_sim(run_id: str, runtime_path: str | None = None, timeout: float = 60 * 30) -> bool:
    """等待本 run 的持久化模拟任务完成，不读取旧的全局 sim 状态。"""
    t0 = time.time()
    runtime = Path(runtime_path) if runtime_path else None
    from app import sim_status
    while time.time() - t0 < timeout:
        if _matrix_stop.is_set():
            return False
        if runtime and runtime.exists():
            try:
                state = json.loads(runtime.read_text())
                status = state.get("status")
                if status in ("done", "finished"):
                    return state.get("returncode", 0) == 0
                if status in ("failed", "stopped", "blocked", "stale"):
                    return False
            except Exception:
                pass
        else:
            # 兼容旧 runner 没有 runtime_path 的情况。
            st = sim_status()
            if not st.get("running", False):
                return st.get("status") in ("done", "finished")
        time.sleep(3)
    _event("sim_timeout", run_id)
    return False


def _has_completed_artifacts(run_id: str) -> bool:
    """检查是否有完整的训练产物。"""
    model = _AURUMQ_ROOT / 'models' / f'matrix_{run_id}'
    return all((model / name).exists() for name in ('ppo_final.zip', 'policy.onnx', 'metadata.json', 'training_summary.json'))


def _hydrate_run_from_ledger(run: dict) -> bool:
    """从已存在的本 run ledger 回填业务指标和 gate。"""
    ledger_value = (run.get("artifacts") or {}).get("ledger")
    candidates = []
    if ledger_value:
        candidates.append(Path(ledger_value))
    candidates.extend(sorted(LEDGER_DIR.glob(f"*{run.get('run_id', '')}*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True))
    for ledger in candidates:
        try:
            if not ledger.exists():
                continue
            data = json.loads(ledger.read_text())
            if data.get("status") != "done" or not isinstance(data.get("metrics"), dict):
                continue
            run.setdefault("artifacts", {})["ledger"] = str(ledger)
            run["metrics"] = data["metrics"]
            run["gates"] = {
                "model_artifacts": "pass" if _has_completed_artifacts(run["run_id"]) else "pending",
                "ledger": "pass",
                "signal_audit": "pending",
                "trade_audit": "pending",
            }
            return True
        except Exception:
            continue
    return False


def _recover_completed_runs(registry: dict) -> dict:
    """WebUI 重启或中断后按真实产物/ledger恢复，不把旧失败覆盖成成功。"""
    changed = False
    for run in registry.get('runs', []):
        rid = run.get('run_id', '')
        model = _AURUMQ_ROOT / 'models' / f"matrix_{rid}"
        status = run.get('status')
        # 完整产物 + ledger 是可恢复的唯一证据；训练错误状态也可恢复。
        before_metrics = dict(run.get('metrics') or {})
        before_gates = dict(run.get('gates') or {})
        before_ledger = (run.get('artifacts') or {}).get('ledger')
        if _has_completed_artifacts(rid) and _hydrate_run_from_ledger(run):
            if (
                status != 'done'
                or run.get('error')
                or before_metrics != run.get('metrics')
                or before_gates != run.get('gates')
                or before_ledger != (run.get('artifacts') or {}).get('ledger')
            ):
                run['status'] = 'done'
                run['error'] = ''
                if not run.get('finished_at'):
                    run['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                changed = True
            continue
        # 完整训练产物但没有 ledger：恢复后处理，不重新训练。
        recovery_hint = (
            'WebUI重启' in str(run.get('error', ''))
            or '训练结束但缺少' in str(run.get('error', ''))
            or 'external_process' in (run.get('artifacts') or {})
        )
        if status in ('running', 'pending', 'failed') and (model / 'ppo_final.zip').exists() and recovery_hint:
            run['status'] = 'pending'
            run['error'] = '训练产物已存在，恢复 ONNX/metadata/模拟/ledger 后处理'
            run.setdefault('artifacts', {})['recovery_required'] = True
            changed = True
    if changed:
        _save_registry(registry)
        _save_summary_cache(registry)
    return registry


def _worker(resume: bool, pause_after_run: bool = True) -> None:
    """矩阵工作线程主函数。"""
    global _matrix_state
    setattr(threading.current_thread(), "_wavehunter_v2_matrix_worker", True)
    registry = _recover_completed_runs(_load_registry())
    runs = registry["runs"]
    for index, run in enumerate(runs, 1):
        if _matrix_stop.is_set():
            break
        if resume and run.get("status") == "done":
            continue
        pool, rid = run["pool"], run["run_id"]
        panel_path = _AURUMQ_ROOT / "data" / run["panel"]
        with _lock:
            _matrix_state.update(current=index, total=len(runs), run_id=rid, progress=f"{index}/{len(runs)} {rid}")
        if not panel_path.exists():
            run.update(status="blocked", error=f"v3面板不存在: {panel_path}")
            _event("blocked", rid, reason=run["error"])
            _append_result(run)
            continue
        out_name = f"matrix_{rid}"
        effective_train = _effective_train(run)
        recovery_required = bool(run.get("artifacts", {}).get("recovery_required"))
        registry["runs"] = runs
        REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2))
        if recovery_required:
            run["status"] = "running"
            run["error"] = "恢复后处理：ONNX/metadata/模拟/ledger"
            _event("recovery_start", rid, index=index, total=len(runs), config=run)
        else:
            run["status"] = "running"
            run["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            external = _external_train_identity()
            if external:
                run["status"] = "running"
                run["artifacts"]["external_process"] = external
                run["error"] = ""
            registry["runs"] = runs
            REGISTRY_FILE.write_text(json.dumps(registry, ensure_ascii=False, indent=2))
            _event("run_start", rid, index=index, total=len(runs), config=run)
        try:
            from wavehunter_v2_api import wavehunter_v2_start, WaveHunterTrainRequest
            import wavehunter_v2_api as _v2_api
            req = WaveHunterTrainRequest(
                panel=run["panel"], start_date=run["train_start"], end_date=run["train_end"],
                total_timesteps=run["steps"], out_dir=out_name, matrix_internal=True,
                **{k: v for k, v in effective_train.items()
                   if k not in ("total_timesteps", "memory_profile")},
            )
            external = _external_train_identity()
            if external and external["run_id"] == rid:
                start_resp = {"status": "external_recovery", "pid": external["pid"], "cmd": external["cmd"]}
                _event("external_recovery", rid, pid=external["pid"])
            elif (_AURUMQ_ROOT / "models" / out_name / "ppo_final.zip").exists():
                start_resp = {"status": "recovery_only", "reason": "ppo_final.zip exists; skip duplicate training"}
                _event("recovery_only", rid, reason="existing ppo_final.zip")
            else:
                _v2_api.set_matrix_internal(True)
                try:
                    run_version = run.get("version", "v3")
                    if run_version == "v3":
                        from wavehunter_v2_api import wavehunter_v3_start, WaveHunterV3TrainRequest
                        v3_req = WaveHunterV3TrainRequest(**req.model_dump())
                        start_resp = wavehunter_v3_start(v3_req)
                    else:
                        start_resp = wavehunter_v2_start(req)
                finally:
                    _v2_api.set_matrix_internal(False)
            run["artifacts"]["train_response"] = start_resp
            if start_resp.get("status") == "external_recovery":
                if not _wait_external_train(external or {}):
                    raise RuntimeError("外部训练失败、停止或超时")
            elif start_resp.get("status") != "recovery_only" and not _wait_train(rid, version=run.get("version", "v2")):
                stopped = _matrix_stop.is_set()
                run.update(
                    status="stopped" if stopped else "failed",
                    error="用户主动停止" if stopped else "训练失败、停止或超时",
                    finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                )
                _event("run_stopped" if stopped else "run_failed", rid,
                       reason="user_stop" if stopped else "train_exit_without_completion")
                _save_registry(registry)
                _append_result(run)
                continue
            model_dir = _AURUMQ_ROOT / "models" / out_name
            run["artifacts"]["model_dir"] = str(model_dir)
            if not (model_dir / "ppo_final.zip").exists():
                raise RuntimeError("训练结束但缺少 ppo_final.zip")
            if not (model_dir / "policy.onnx").exists() or not (model_dir / "metadata.json").exists():
                _recover_model_artifacts(run, model_dir)
            before = max([p.stat().st_mtime for p in LEDGER_DIR.glob("*.json")], default=0.0)
            from app import start_sim, SimRequest
            sim_req = SimRequest(
                model_dir=str(model_dir.relative_to(_AURUMQ_ROOT / "models")),
                panel=run["panel"], start_date=run["oos_start"], end_date=run["oos_end"],
                **FIXED_SIM,
            )
            sim_resp = start_sim(sim_req)
            run["artifacts"]["sim_response"] = sim_resp
            if not _wait_sim(rid, sim_resp.get("runtime_path")):
                runtime = sim_resp.get("runtime_path")
                detail = "模拟失败、停止或超时"
                if runtime and Path(runtime).exists():
                    try:
                        persisted_sim = json.loads(Path(runtime).read_text())
                        if persisted_sim.get('status') == 'finished':
                            run["artifacts"]["ledger"] = str(_latest_ledger(before) or '')
                            run["status"] = "done"
                            run["error"] = ""
                            run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                            _event("run_done", rid, recovery=True)
                            _append_result(run)
                            _save_registry(registry)
                            continue
                    except Exception:
                        pass
                if runtime and Path(runtime).exists():
                    try:
                        detail = json.loads(Path(runtime).read_text()).get("error") or detail
                    except Exception:
                        pass
                raise RuntimeError(detail)
            ledger = _latest_ledger(before)
            if ledger is None:
                raise RuntimeError("模拟完成但未找到新 ledger")
            run["artifacts"]["ledger"] = str(ledger)
            data = json.loads(ledger.read_text())
            run["metrics"] = data.get("metrics", {})
            run["gates"] = {"model_artifacts": "pass", "ledger": "pass", "signal_audit": "pending", "trade_audit": "pending"}
            run["status"] = "done"
            run["error"] = ""
            run["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _event("run_done", rid, metrics=run["metrics"], ledger=str(ledger))
        except Exception as e:
            run.update(status="failed", error=str(e))
            _event("run_failed", rid, error=str(e))
        _append_result(run)
        registry["runs"] = runs
        _save_registry(registry)
        try:
            _save_summary_cache(registry)
        except Exception:
            pass
        if pause_after_run and run.get("status") == "done":
            _matrix_stop.set()
            _event("matrix_paused_after_run", rid, reason="pause_after_run")
            break
    with _lock:
        _matrix_state.update(
            running=False,
            status="done" if not _matrix_stop.is_set() else "stopped",
            run_id=None,
            total=len(runs),
            completed=sum(1 for r in runs if r.get("status") == "done"),
        )
