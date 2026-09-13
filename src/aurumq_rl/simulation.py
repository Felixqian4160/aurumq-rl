"""模拟交易引擎 v2 — 兼容层，导入所有子模块接口。"""
from __future__ import annotations

# 导入所有子模块的公共接口
from .simulation_config import (
    SimConfig,
    SimResult,
    get_sim_result,
    stop_sim,
    _sim_result,
    _sim_stop,
    _sanitize,
    _reset_sim,
)

from .simulation_ledger import (
    save_ledger,
    list_ledgers,
    load_ledger,
    delete_ledger,
)

from .simulation_benchmark import (
    load_csi300_benchmark,
)

from .simulation_core import (
    run_simulation,
)

# 保持向后兼容：重导出所有接口
__all__ = [
    'SimConfig',
    'SimResult',
    'get_sim_result',
    'stop_sim',
    'save_ledger',
    'list_ledgers',
    'load_ledger',
    'delete_ledger',
    'load_csi300_benchmark',
    'run_simulation',
    '_sim_result',
    '_sim_stop',
    '_sanitize',
    '_reset_sim',
]

# 内部路径常量（保持兼容）
from pathlib import Path
_AURUMQ_ROOT = Path(__file__).resolve().parent.parent.parent
_LEDGER_DIR = _AURUMQ_ROOT / 'data' / 'ledgers'
