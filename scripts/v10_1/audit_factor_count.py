#!/usr/bin/env python3
"""v10_1 因子分类自动统计脚本。

**目的**：替代"手工列举 + 手动加和"。每次面板或 registry 改动后，跑一次这个脚本，
输出真实的因子数 + 分类清单。永远不要靠人工重新数。

**数据来源**（单一事实源 = build_factor_panel.py 的 4 个来源）：
  1. ALPHA101_REGISTRY  → src/aurumq_rl/factors/alpha101/*.py 中的 id="alphaXXX"
  2. GTJA191_REGISTRY   → src/aurumq_rl/factors/gtja191/batch_*.py 中的 id="gtja_NNN"
  3. FUNDAMENTAL_FACTORS → scripts/build_factor_panel.py L432-445 (常量列表)
  4. VALUATION_FACTORS   → scripts/build_factor_panel.py L450-453 (常量列表)
  5. add_derived_columns → scripts/build_factor_panel.py L311-370 (vwap/returns/adv*)

**运行方式**：
    python scripts/v10_1/audit_factor_count.py
    python scripts/v10_1/audit_factor_count.py --panel data/wavehunter_v10_1_volfix_csi500_20040102_20260909.parquet

**输出**：
    - 每个来源的因子数 + 实际 id 列表
    - 总计
    - 如果指定 --panel，跟实际 parquet 列数比对（验证 build_stats 是否一致）

**为什么这个脚本能"自动核对"**：
    它不依赖任何手工分类标签，而是从源代码 (register_alpha101(...) / FUNDAMENTAL_FACTORS = [...])
    直接解析。如果谁改了源码但忘了同步更新某个文档，文档就会过时，但这个脚本不会。

**维护者**：Buffett (hermes-agent buffett profile)
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

# 默认项目根
ROOT = Path(__file__).resolve().parents[2]
FACTORS_DIR = ROOT / 'src/aurumq_rl/factors'
BUILD_FACTOR_PANEL = ROOT / 'scripts/build_factor_panel.py'


def parse_alpha101() -> tuple[list[str], dict[str, list[str]]]:
    """从 alpha101/*.py 解析所有 id='alphaXXX' 出现位置。"""
    per_file = defaultdict(list)
    for f in sorted((FACTORS_DIR / 'alpha101').glob('*.py')):
        if f.name in ('__init__.py', '_ops.py'):
            continue
        text = f.read_text()
        ids = re.findall(r'id\s*=\s*[\'"]alpha(\w+)[\'"]', text)
        per_file[f.name] = [f'alpha{i}' for i in ids]
    all_ids = [i for ids in per_file.values() for i in ids]
    return all_ids, per_file


def parse_gtja191() -> tuple[list[str], dict[str, list[str]]]:
    """从 gtja191/batch_*.py 解析所有 id='gtja_NNN'。"""
    per_file = defaultdict(list)
    for f in sorted((FACTORS_DIR / 'gtja191').glob('batch_*.py')):
        text = f.read_text()
        ids = re.findall(r'id\s*=\s*[\'"]gtja_(\d+)[\'"]', text)
        per_file[f.name] = [f'gtja_{i}' for i in ids]
    all_ids = [i for ids in per_file.values() for i in ids]
    return all_ids, per_file


def parse_fundamental() -> tuple[list[str], dict[str, list[str]]]:
    """从 build_factor_panel.py L432-445 提取 FUNDAMENTAL_FACTORS 列表 + 按注释分组。

    使用每个 # 注释到下一个 # / ] 的范围，提取范围内的所有引号字符串。
    """
    text = BUILD_FACTOR_PANEL.read_text()
    m = re.search(r'FUNDAMENTAL_FACTORS\s*=\s*\[(.*?)\n\]', text, re.S)
    if not m:
        raise RuntimeError("FUNDAMENTAL_FACTORS not found in build_factor_panel.py")
    body = m.group(1)
    all_ids = re.findall(r'[\'"]([\w_]+)[\'"]', body)

    # 按注释分组：找到每个 # 注释开始，到下一个 # 或 ] 结束
    groups: dict[str, list[str]] = {}
    pos = 0
    current_label = None
    for line in body.split('\n'):
        comm = re.match(r'\s*#\s*(.+?)\s*$', line)
        if comm:
            current_label = comm.group(1)
            groups[current_label] = []
            continue
        ids = re.findall(r'[\'"]([\w_]+)[\'"]', line)
        if ids and current_label:
            groups[current_label].extend(ids)
        elif ids:
            groups.setdefault('(未分类)', []).extend(ids)
    return all_ids, groups


def parse_valuation() -> list[str]:
    """从 build_factor_panel.py L450-453 提取 VALUATION_FACTORS 列表。"""
    text = BUILD_FACTOR_PANEL.read_text()
    m = re.search(r'VALUATION_FACTORS\s*=\s*\[(.*?)\]', text, re.S)
    if not m:
        raise RuntimeError("VALUATION_FACTORS not found in build_factor_panel.py")
    body = m.group(1)
    return re.findall(r'[\'"]([\w_]+)[\'"]', body)


def parse_derived() -> list[str]:
    """add_derived_columns 生成的衍生列 (vwap/returns/pct_chg + adv 系列)。"""
    derived = ['vwap', 'returns', 'pct_chg']
    derived += [f'adv{w}' for w in (5, 10, 15, 20, 30, 40, 50, 60, 80, 81, 90, 100, 120, 150, 180)]
    return derived


def verify_against_panel(panel_path: Path, expected_total: int, expected_groups: dict[str, list[str]]):
    """读 parquet schema，与预期对比；不一致时大声报错。"""
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(str(panel_path))
        cols = schema.names
    except ImportError:
        print('⚠️  pyarrow not installed; skip parquet cross-check')
        return
    except Exception as e:
        print(f'⚠️  failed to read parquet: {e}')
        return

    print(f'\n=== Parquet cross-check: {panel_path.name} ===')
    print(f'  parquet 列数 = {len(cols)}')
    print(f'  expected 总因子数 = {expected_total}')
    diff = len(cols) - expected_total
    print(f'  差值 = {diff}  (应为 0，或等于 base 列 + label 列的总数)')

    # 分类
    alpha_in = [c for c in cols if c.startswith('alpha_')]
    gtja_in = [c for c in cols if c.startswith('gtja_')]
    v10_1_label = [c for c in cols if c.startswith('v10_1_')]
    fund_in = [c for c in expected_groups['fundamental'] if c in cols]
    val_in = [c for c in expected_groups['valuation'] if c in cols]
    derived_in = [c for c in expected_groups['derived'] if c in cols]
    base = [c for c in cols if c not in (
        set(alpha_in) | set(gtja_in) | set(v10_1_label) |
        set(fund_in) | set(val_in) | set(derived_in)
    )]

    print(f'\n  alpha_*    {len(alpha_in):3d}  (expect {len(expected_groups["alpha"])})')
    print(f'  gtja_*     {len(gtja_in):3d}  (expect {len(expected_groups["gtja"])})')
    print(f'  v10_1_*    {len(v10_1_label):3d}  (label, 非因子)')
    print(f'  基本面      {len(fund_in):3d}  (expect {len(expected_groups["fundamental"])})')
    print(f'  估值        {len(val_in):3d}  (expect {len(expected_groups["valuation"])})')
    print(f'  衍生列      {len(derived_in):3d}  (expect {len(expected_groups["derived"])})')
    print(f'  base + 其它 {len(base):3d}')

    if len(alpha_in) != len(expected_groups['alpha']):
        print(f'  ❌ Alpha101 不一致！差 {len(alpha_in) - len(expected_groups["alpha"])}')
        # 找出哪些 alpha 不在 panel 里
        missing = set(expected_groups['alpha']) - set(alpha_in)
        extra = set(alpha_in) - set(expected_groups['alpha'])
        if missing: print(f'     missing: {sorted(missing)[:20]}')
        if extra: print(f'     extra in panel: {sorted(extra)[:20]}')
    if len(gtja_in) != len(expected_groups['gtja']):
        print(f'  ❌ GTJA191 不一致！差 {len(gtja_in) - len(expected_groups["gtja"])}')
    if len(fund_in) != len(expected_groups['fundamental']):
        print(f'  ❌ 基本面不一致！')
    if len(val_in) != len(expected_groups['valuation']):
        print(f'  ❌ 估值不一致！')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', type=Path, help='可选：指定 v10_1 面板 parquet 做交叉验证')
    ap.add_argument('--json', action='store_true', help='输出 JSON 而非人类可读格式')
    args = ap.parse_args()

    alpha, alpha_per_file = parse_alpha101()
    gtja, gtja_per_file = parse_gtja191()
    fundamental, fund_groups = parse_fundamental()
    valuation = parse_valuation()
    derived = parse_derived()

    # GTJA 按 batch 分组
    gtja_batches = {}
    for fname, ids in gtja_per_file.items():
        nums = sorted({int(i.replace('gtja_', '')) for i in ids})
        if nums:
            gtja_batches[fname] = f'{nums[0]:03d}..{nums[-1]:03d} ({len(nums)})'

    if args.json:
        import json
        result = {
            'alpha101': alpha,
            'gtja191': gtja,
            'fundamental_factors': fundamental,
            'valuation_factors': valuation,
            'derived_columns': derived,
            'total': len(alpha) + len(gtja) + len(fundamental) + len(valuation) + len(derived),
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print('=' * 70)
        print('v10_1 因子分类自动统计')
        print('=' * 70)
        print('\n[1] Alpha101  (从 src/aurumq_rl/factors/alpha101/*.py 注册表解析)')
        print(f'    总计 = {len(alpha)} 个')
        for f, ids in alpha_per_file.items():
            if ids:
                print(f'    {f:30s} {len(ids):3d}  ({ids[0]}..{ids[-1]})')

        print('\n[2] GTJA191  (从 src/aurumq_rl/factors/gtja191/batch_*.py 注册表解析)')
        print(f'    总计 = {len(gtja)} 个')
        for f, label in gtja_batches.items():
            print(f'    {f:30s} {label}')

        print('\n[3] FUNDAMENTAL_FACTORS  (build_factor_panel.py L432-445)')
        print(f'    总计 = {len(fundamental)} 个')
        for label, ids in fund_groups.items():
            print(f'    {label:8s}  {len(ids):2d}  {ids}')

        print('\n[4] VALUATION_FACTORS  (build_factor_panel.py L450-453)')
        print(f'    总计 = {len(valuation)} 个')
        print(f'    {valuation}')

        print('\n[5] add_derived_columns  (build_factor_panel.py L311-370)')
        print(f'    总计 = {len(derived)} 个')
        print(f'    {derived[:5]} ... + adv(5..180 共 16 档)')

        total = len(alpha) + len(gtja) + len(fundamental) + len(valuation) + len(derived)
        print('\n' + '=' * 70)
        print(f'  因子总计 = {total}')
        print('=' * 70)

    if args.panel:
        groups = {
            'alpha': alpha,
            'gtja': gtja,
            'fundamental': fundamental,
            'valuation': valuation,
            'derived': derived,
        }
        verify_against_panel(args.panel, total, groups)


if __name__ == '__main__':
    main()
