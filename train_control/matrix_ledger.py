#!/usr/bin/env python3
"""矩阵账本管理器 - 生成和查询参数矩阵测试账本"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from core.config import DATA_DIR

MATRIX_LEDGER_DIR = DATA_DIR / "matrix_ledger"
MATRIX_LEDGER_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = MATRIX_LEDGER_DIR / "index.json"


def _load_index() -> dict:
    """加载账本索引"""
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            return json.load(f)
    return {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "tests": [],
        "summary": {
            "total_tests": 0,
            "completed": 0,
            "running": 0,
            "failed": 0,
            "best_sharpe": None,
            "best_excess": None
        }
    }


def _save_index(index: dict):
    """保存账本索引"""
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def create_test_entry(test_id: str, name: str, config: dict) -> dict:
    """创建新的测试条目"""
    entry = {
        "test_id": test_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "training": {
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "elapsed_s": 0,
            "a1_acc": None,
            "a2_acc": None,
            "reward_curve": []
        },
        "simulation": None,
        "comparison": {
            "rank_by_sharpe": None,
            "rank_by_excess": None
        }
    }

    # 保存单测试文件
    test_file = MATRIX_LEDGER_DIR / f"{test_id}.json"
    with open(test_file, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    # 更新索引
    index = _load_index()
    index["tests"].append({
        "test_id": test_id,
        "name": name,
        "status": "pending",
        "created_at": entry["created_at"]
    })
    index["summary"]["total_tests"] += 1
    index["summary"]["pending"] = index["summary"]["total_tests"] - index["summary"]["completed"] - index["summary"]["running"] - index["summary"]["failed"]
    _save_index(index)

    return entry


def update_training(test_id: str, training_data: dict):
    """更新训练结果"""
    test_file = MATRIX_LEDGER_DIR / f"{test_id}.json"
    if not test_file.exists():
        return

    with open(test_file) as f:
        entry = json.load(f)

    entry["training"].update(training_data)
    entry["training"]["end_time"] = datetime.now().isoformat()

    with open(test_file, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    # 更新索引状态
    index = _load_index()
    for t in index["tests"]:
        if t["test_id"] == test_id:
            t["status"] = entry["training"]["status"]
            break
    index["summary"]["completed"] = sum(1 for t in index["tests"] if t["status"] == "completed")
    index["summary"]["running"] = sum(1 for t in index["tests"] if t["status"] == "running")
    index["summary"]["failed"] = sum(1 for t in index["tests"] if t["status"] == "failed")
    _save_index(index)


def update_simulation(test_id: str, sim_data: dict):
    """更新模拟结果"""
    test_file = MATRIX_LEDGER_DIR / f"{test_id}.json"
    if not test_file.exists():
        return

    with open(test_file) as f:
        entry = json.load(f)

    entry["simulation"] = sim_data

    with open(test_file, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)

    # 更新排名
    _update_rankings()


def _update_rankings():
    """更新所有测试的排名"""
    index = _load_index()

    # 加载所有有模拟结果的测试
    tests_with_sim = []
    for t in index["tests"]:
        test_file = MATRIX_LEDGER_DIR / f"{t['test_id']}.json"
        if test_file.exists():
            with open(test_file) as f:
                data = json.load(f)
            if data.get("simulation") and data["simulation"].get("metrics"):
                tests_with_sim.append({
                    "test_id": t["test_id"],
                    "sharpe": data["simulation"]["metrics"].get("sharpe_ratio", 0),
                    "excess": data["simulation"]["metrics"].get("excess_return", 0)
                })

    # 按夏普排名
    tests_with_sim.sort(key=lambda x: x["sharpe"], reverse=True)
    for i, t in enumerate(tests_with_sim):
        test_file = MATRIX_LEDGER_DIR / f"{t['test_id']}.json"
        with open(test_file) as f:
            data = json.load(f)
        data["comparison"]["rank_by_sharpe"] = i + 1
        with open(test_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # 按超额收益排名
    tests_with_sim.sort(key=lambda x: x["excess"], reverse=True)
    for i, t in enumerate(tests_with_sim):
        test_file = MATRIX_LEDGER_DIR / f"{t['test_id']}.json"
        with open(test_file) as f:
            data = json.load(f)
        data["comparison"]["rank_by_excess"] = i + 1
        with open(test_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # 更新索引汇总
    if tests_with_sim:
        best_sharpe = max(tests_with_sim, key=lambda x: x["sharpe"])
        best_excess = max(tests_with_sim, key=lambda x: x["excess"])
        index["summary"]["best_sharpe"] = {
            "test_id": best_sharpe["test_id"],
            "value": best_sharpe["sharpe"]
        }
        index["summary"]["best_excess"] = {
            "test_id": best_excess["test_id"],
            "value": best_excess["excess"]
        }
    _save_index(index)


def get_test_entry(test_id: str) -> Optional[dict]:
    """获取单测试详情"""
    test_file = MATRIX_LEDGER_DIR / f"{test_id}.json"
    if test_file.exists():
        with open(test_file) as f:
            return json.load(f)
    return None


def load_reward_curve(model_path: str) -> dict:
    """从 tensorboard 加载 reward 曲线数据"""
    import glob
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        # 查找 tensorboard 事件文件
        tb_pattern = f"{model_path}/tb_logs/PPO_1/events.out.tfevents.*"
        files = glob.glob(tb_pattern)
        if not files:
            return {"avg_excess": [], "win_rate": [], "loss": []}

        ea = EventAccumulator(files[0])
        ea.Reload()

        result = {"avg_excess": [], "win_rate": [], "loss": []}

        # 提取 reward/avg_excess
        if 'reward/avg_excess' in ea.Tags()['scalars']:
            vals = ea.Scalars('reward/avg_excess')
            result["avg_excess"] = [{"step": v.step, "value": v.value} for v in vals]

        # 提取 reward/win_rate
        if 'reward/win_rate' in ea.Tags()['scalars']:
            vals = ea.Scalars('reward/win_rate')
            result["win_rate"] = [{"step": v.step, "value": v.value} for v in vals]

        # 提取 train/loss (采样，避免数据量太大)
        if 'train/loss' in ea.Tags()['scalars']:
            vals = ea.Scalars('train/loss')
            # 每10个点采样一次
            sampled = vals[::max(1, len(vals)//100)]
            result["loss"] = [{"step": v.step, "value": v.value} for v in sampled]

        return result
    except Exception as e:
        print(f"Failed to load reward curve: {e}")
        return {"avg_excess": [], "win_rate": [], "loss": []}


def get_index() -> dict:
    """获取账本索引"""
    return _load_index()


def list_tests() -> list:
    """列出所有测试"""
    index = _load_index()
    return index["tests"]


def get_summary() -> dict:
    """获取汇总信息"""
    index = _load_index()
    return index["summary"]


if __name__ == "__main__":
    # 测试
    print("Matrix Ledger Dir:", MATRIX_LEDGER_DIR)
    print("Index:", get_index())
