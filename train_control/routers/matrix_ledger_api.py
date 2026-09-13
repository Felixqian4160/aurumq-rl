"""矩阵账本 API - 查询参数矩阵测试结果"""
from fastapi import APIRouter, HTTPException
from typing import Optional
import json
from pathlib import Path

router = APIRouter()

from core.config import AURUMQ_ROOT, DATA_DIR

MATRIX_LEDGER_DIR = DATA_DIR / "matrix_ledger"

# 导入 load_reward_curve
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from matrix_ledger import load_reward_curve


@router.get("/api/matrix-ledger/index")
def get_ledger_index():
    """获取账本索引（所有测试列表+汇总）"""
    index_file = MATRIX_LEDGER_DIR / "index.json"
    if not index_file.exists():
        return {"tests": [], "summary": {"total_tests": 0}}
    with open(index_file) as f:
        return json.load(f)


@router.get("/api/matrix-ledger/test/{test_id}")
def get_test_detail(test_id: str):
    """获取单测试详情（含 reward 曲线）"""
    test_file = MATRIX_LEDGER_DIR / f"{test_id}.json"
    if not test_file.exists():
        raise HTTPException(404, f"Test {test_id} not found")

    with open(test_file) as f:
        data = json.load(f)

    # 加载 reward 曲线
    model_path = data.get("config", {}).get("model_path")
    if model_path:
        # Resolve model paths relative to the configured project root and reject traversal.
        candidate = (AURUMQ_ROOT / "models" / model_path).resolve()
        models_root = (AURUMQ_ROOT / "models").resolve()
        if models_root not in candidate.parents and candidate != models_root:
            raise HTTPException(400, "非法模型路径")
        data["reward_curves"] = load_reward_curve(str(candidate))

    return data


@router.get("/api/matrix-ledger/summary")
def get_summary():
    """获取汇总信息"""
    index_file = MATRIX_LEDGER_DIR / "index.json"
    if not index_file.exists():
        return {"total_tests": 0}
    with open(index_file) as f:
        data = json.load(f)
    return data.get("summary", {})


@router.get("/api/matrix-ledger/rankings")
def get_rankings():
    """获取排名（按夏普/超额收益）"""
    index_file = MATRIX_LEDGER_DIR / "index.json"
    if not index_file.exists():
        return {"by_sharpe": [], "by_excess": []}

    with open(index_file) as f:
        index = json.load(f)

    tests_with_sim = []
    for t in index.get("tests", []):
        test_file = MATRIX_LEDGER_DIR / f"{t['test_id']}.json"
        if test_file.exists():
            with open(test_file) as f:
                data = json.load(f)
            if data.get("simulation") and data["simulation"].get("metrics"):
                tests_with_sim.append({
                    "test_id": t["test_id"],
                    "name": t["name"],
                    "config": data.get("config", {}),
                    "metrics": data["simulation"]["metrics"]
                })

    # 按夏普排名
    by_sharpe = sorted(tests_with_sim, key=lambda x: x["metrics"].get("sharpe_ratio", 0), reverse=True)
    # 按超额收益排名
    by_excess = sorted(tests_with_sim, key=lambda x: x["metrics"].get("excess_return", 0), reverse=True)

    return {
        "by_sharpe": [{"rank": i+1, **t} for i, t in enumerate(by_sharpe[:10])],
        "by_excess": [{"rank": i+1, **t} for i, t in enumerate(by_excess[:10])]
    }
