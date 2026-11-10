"""Frozen 3-policy x 4-postprocessor attribution and confirmation protocol."""
from __future__ import annotations
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in ('p2j_validation', 'p2l_cached_test', 'p2k_formal'):
    sys.path.insert(0, str(ROOT / p))
import run_validation as v
import run_cached as cached
import run_formal as formal

POLICIES = ('MAPPO', 'kNN', 'Qwen')
MODES = ('raw', 'link', 'node', 'both')
ARMS = tuple(p + '_' + m for p in POLICIES for m in MODES)
OLD_SEEDS = (6601, 6701, 6801)
NEW_SEEDS = (2026110511, 2026110512, 2026110513)
COUNT = 512
TOTAL = 1536
CALLS = TOTAL * 4
OLD = ROOT / 'p2l_cached_test/run_20261105T023605_900720Z'
LOCK = ROOT / 'p2j_validation/run_20261105T014642_470733Z/protocol.json'
OLD_ARMS = {'MAPPO': 'MAPPO_raw', 'Qwen_P2A_Standard1000': 'Qwen_raw',
            'Frozen_link': 'Qwen_link', 'Frozen_link_node': 'Qwen_both'}
CLAIMS = {'candidate_vs_MAPPO': ('Qwen_both', 'MAPPO_raw'),
          'matched_MAPPO': ('Qwen_both', 'MAPPO_both'),
          'matched_kNN': ('Qwen_both', 'kNN_both')}


def read(path, cap=1_000_000):
    return cached.read_json(Path(path), cap)


def lines(path, cap=300_000_000):
    path = Path(path)
    assert path.is_file() and path.stat().st_size < cap, str(path)
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def fingerprint(path):
    path = Path(path)
    assert path.is_file(), str(path)
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'format': path.suffix, 'sha256': v.replay.sha256(path)}


def frozen_sources():
    old = read(LOCK)
    hashes = dict(old['source_sha256'])
    hashes[str(LOCK.relative_to(ROOT))] = v.replay.sha256(LOCK)
    for name in ('common.py', 'run_cached.py', 'run_fresh.py', 'audit.py', 'run.ps1', 'test_contract.py'):
        p = Path(__file__).resolve().with_name(name)
        hashes[str(p.relative_to(ROOT))] = v.replay.sha256(p)
    hashes['p2k_formal/run_formal.py'] = v.replay.sha256(ROOT / 'p2k_formal/run_formal.py')
    for rel, digest in hashes.items():
        assert v.replay.sha256(ROOT / rel) == digest, rel
    selected = read(v.SELECTED)
    assert selected['max_ratio_increment'] == .025 and selected['extra_projected_overload_budget'] == .01
    return hashes


def verify_sources(hashes):
    for rel, digest in hashes.items():
        assert v.replay.sha256(ROOT / rel) == digest, rel


def create_run(kind):
    out = ROOT / 'p2m_paper' / (kind + '_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    return out


def protocol(kind, hashes, ram):
    return {'schema': 'paper_3x4_supplement_v1', 'kind': kind, 'arms': ARMS,
            'policies': POLICIES, 'postprocessors': MODES, 'scenes_per_seed': COUNT,
            'old_seeds': OLD_SEEDS, 'new_seeds': NEW_SEEDS, 'n_paired_each_stage': TOTAL,
            'link': {'target': 'link_mean3', 'alpha': .5, 'public_TRAIN_only': True},
            'node': {'max_ratio_increment': .025, 'extra_projected_overload_budget': .01},
            'claim_family': CLAIMS, 'claim_family_size': 3,
            'multiplicity': 'Bonferroni: each of3 meanV claims has98.333333% interval; other95% intervals descriptive.',
            'bootstrap': '10000 paired scene resamples stratified by generation seed,RNG20261105',
            'source_hashes': hashes, 'available_RAM_GiB': ram,
            'memory_plan': 'One512-scene env at a time; minimum8GiB free. New samples about5GB,require12GiB disk.',
            'new_Qwen_calls_fresh': CALLS, 'cached_stage_API_calls': 0,
            'model': v.MODEL, 'temperature': .1, 'max_tokens': 4096, 'workers': 8,
            'automatic_retries': 0, 'fallback': False, 'parameters_changed': False,
            'optional_stopping': False, 'GPT_calls': 0, 'API_cost_usd_estimate': 0,
            'cost_basis': 'User-provided unlimited internal Qwen quota,not independently verified.',
            'all12_arms_run_in_both_stages_regardless_of_old_results': True,
            'information_boundary': 'Each postprocessor sees only its role own240D observation,own4D action,own publicTRAIN link target; never other policy actions or outcome feedback.',
            'limitations': ['Old tests previously inspected.', 'Fresh tests same GEANT generator/config,not cross-topology.',
                            'One Qwen draw per scene,shared among4 postprocessors.'],
            'original_environment_reward_metrics_migration_routing_unchanged': True}


def knn_action(library, obs):
    """Identical original RetrievalPolicy(knn), including float32 conversion."""
    from p1a.policy import to_env_action
    return {r: to_env_action(r, library.query(r, obs[r], top_k=1)[0]['ratios']).tolist() for r in v.ROLES}


def variants(raw, obs, targets):
    original = {r: list(raw[r]) for r in v.ROLES}
    link, both, _ = v.derived_actions(original, obs, targets)
    node = {r: list(original[r]) for r in v.ROLES}
    for r in v.K:
        node[r], _ = v.adjust_local(obs[r], original[r], .025, .01)
        assert node[r][2:] == original[r][2:]
    assert original['agent_g'] == node['agent_g'] == link['agent_g'] == both['agent_g']
    assert all(node[r][:2] == both[r][:2] for r in v.K)
    return dict(zip(MODES, (original, link, node, both)))


def actions(bases, obs, targets):
    assert set(bases) == set(POLICIES)
    return {p + '_' + m: a for p in POLICIES for m, a in variants(bases[p], obs, targets).items()}


def stats(methods, seeds):
    expected = [(s, i) for s in seeds for i in range(COUNT)]
    assert set(methods) == set(ARMS)
    assert all(set(rows) == set(expected) for rows in methods.values())
    ordered = {arm: [rows[k] for k in expected] for arm, rows in methods.items()}
    pairs = dict(CLAIMS)
    for p in POLICIES:
        for mode in ('link', 'node', 'both'):
            pairs[p + '_' + mode + '_minus_raw'] = (p + '_' + mode, p + '_raw')
        pairs[p + '_both_minus_link'] = (p + '_both', p + '_link')
        pairs[p + '_both_minus_node'] = (p + '_both', p + '_node')
    pairs['raw_Qwen_minus_MAPPO'] = ('Qwen_raw', 'MAPPO_raw')
    pairs['raw_Qwen_minus_kNN'] = ('Qwen_raw', 'kNN_raw')
    idx = v.bootstrap_indices([k[0] for k in expected])
    comparisons = {}
    for label, (a, b) in pairs.items():
        result = {}
        for field in ('V', 'cost_sum', 'V_g', 'g_done', 'cooperative_valid'):
            x = np.array([r['metrics'][field] for r in ordered[a]], dtype=float)
            y = np.array([r['metrics'][field] for r in ordered[b]], dtype=float)
            d = x - y; boot = d[idx].mean(axis=1)
            result[field] = {'mean_delta': float(d.mean()), 'CI95': np.quantile(boot, [.025, .975]).tolist()}
            if field == 'V' and label in CLAIMS:
                result[field]['Bonferroni_family3_CI98_333333'] = np.quantile(boot, [1/120, 119/120]).tolist()
            if field in ('g_done', 'cooperative_valid'):
                result[field].update(wins=int(((x == 1) & (y == 0)).sum()), losses=int(((x == 0) & (y == 1)).sum()))
        comparisons[label] = {'arms': [a, b], **result}
    interactions = {}
    for p in POLICIES:
        d = np.array([methods[p+'_both'][k]['metrics']['V'] - methods[p+'_link'][k]['metrics']['V']
                      - methods[p+'_node'][k]['metrics']['V'] + methods[p+'_raw'][k]['metrics']['V'] for k in expected])
        interactions[p] = {'V_mean_interaction': float(d.mean()), 'CI95': np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist()}
    return {'summaries': {a: v.summarize(rows) for a, rows in ordered.items()},
            'per_seed': {str(s): {a: v.summarize([methods[a][s, i] for i in range(COUNT)]) for a in ARMS} for s in seeds},
            'comparisons': comparisons, 'V_factorial_interactions': interactions}


def report(out, result):
    text = ['# 论文补充实验：3种决策器 × 4种校正', '',
            '阶段：' + result['kind'], '全部参数预先固定；原始基线保留，校正版是新增消融方法。', '',
            '| 方法 | 平均V | 接纳率 | 合作有效率 | 平均成本 |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        s = result['summaries'][arm]
        text.append(f"| {arm} | {s['V_mean']:.4f} | {s['acceptance_rate']:.2%} | {s['cooperative_rate']:.2%} | {s['cost_mean']:.4f} |")
    text += ['', '| 预声明比较 | ΔV | 配对95%区间 | 三重比较Bonferroni区间 |', '|---|---:|---|---|']
    for label in CLAIMS:
        c = result['comparisons'][label]['V']; lo, hi = c['CI95']; blo, bhi = c['Bonferroni_family3_CI98_333333']
        text.append(f"| {label} | {c['mean_delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | [{blo:+.4f}, {bhi:+.4f}] |")
    text += ['', '## 逐种子', '', '| 种子 | 方法 | V | 接纳数 | 合作有效数 |', '|---|---|---:|---:|---:|']
    for seed, methods in result['per_seed'].items():
        for a, s in methods.items():
            text.append(f"| {seed} | {a} | {s['V_mean']:.4f} | {s['admitted']} | {s['cooperative']} |")
    text += ['', '## 解释边界', '',
             '旧测试追加分析不能当成新的最终测试；新测试只支持同GEANT生成分布内的结论。',
             '相同校正只读取当前决策器的自身动作和角色局部观测，不能使用MAPPO动作替换Qwen动作。',
             '三个预声明V比较使用Bonferroni区间，其余指标和消融比较按描述性分析报告。',
             '模型没有智能体间通信；合作有效性不等于实现了通信协商。',
             '平均V提升不意味着逐场、逐参与者提升，也不是端到端吞吐量指标。',
             'API调用数：' + str(result['API_calls']) + '；新增API费用估计$0；GPT调用0。',
             '源文件、数据哈希、动作重建、控制评分及汇总审计全部通过。']
    with (out / 'RESULTS.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(text) + '\n')
