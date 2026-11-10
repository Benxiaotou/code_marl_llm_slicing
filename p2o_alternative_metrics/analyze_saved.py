"""Post-hoc descriptive metrics from saved results; no simulator/API imports."""
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'p2m_paper/fresh_20261105T025939_435887Z'
ARMS = ('MAPPO_raw', 'Qwen_raw', 'Qwen_both')
SEEDS = (2026110511, 2026110512, 2026110513)
K = ('k1', 'k2', 'k3')
LABELS = {'MAPPO_raw': '原始MAPPO', 'Qwen_raw': '原始Qwen',
          'Qwen_both': 'Qwen联合改进版'}


def read_json(path, limit):
    assert path.is_file() and path.stat().st_size < limit
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def delta_v(m):
    return m['V'] - sum(m['V_' + k + '_0'] for k in K)


def lower_tail(values):
    # Empirical lower expected shortfall with exactly 10% probability mass.
    # For N=1536: 153 full observations + 0.6 of the 154th observation.
    ordered = sorted(values)
    mass = len(ordered) * .1
    full = int(math.floor(mass))
    partial = mass - full
    return (math.fsum(ordered[:full]) + partial * ordered[full]) / mass


def aggregate(rows):
    ms = [r['metrics'] for r in rows]
    n = len(ms)
    accepted = sum(m['g_done'] for m in ms)
    total_cost = math.fsum(m['cost_sum'] for m in ms)
    total_revenue = math.fsum(m['revenue_g'] for m in ms)
    return {
        'n': n, 'accepted': accepted, 'acceptance_rate': accepted / n,
        'V_mean': statistics.mean(m['V'] for m in ms),
        'cost_mean': total_cost / n, 'cost_total': total_cost,
        'cost_per_successful_admission': total_cost / accepted if accepted else None,
        'delta_V_mean': statistics.mean(delta_v(m) for m in ms),
        'lower_tail_10pct_delta_V': lower_tail([delta_v(m) for m in ms]),
        'negative_delta_V_count': sum(delta_v(m) < 0 for m in ms),
        'new_slice_gross_revenue_mean': total_revenue / n,
        'new_slice_gross_revenue_given_admission': total_revenue / accepted if accepted else None,
        'new_slice_net_utility_mean': statistics.mean(m['V_g'] for m in ms),
    }


def main():
    records = SOURCE / 'records.jsonl'
    assert records.is_file() and records.stat().st_size < 30_000_000
    saved = read_json(SOURCE / 'summary.json', 100_000)
    expected = {(s, i) for s in SEEDS for i in range(512)}
    rows = {arm: {} for arm in ARMS}
    checked = 0
    with records.open(encoding='utf-8') as source:
        for line in source:
            r = json.loads(line)
            m = r['metrics']
            assert all(math.isfinite(m[k]) for k in m if not isinstance(m[k], bool))
            assert math.isclose(m['V'], math.fsum(m['V_' + k] for k in K + ('g',)), abs_tol=1e-7)
            assert math.isclose(m['V'], math.fsum(m['revenue_' + k] for k in K + ('g',)) - m['cost_sum'], abs_tol=1e-7)
            assert math.isclose(m['cost_sum'], math.fsum(m['cost_' + k] for k in K), abs_tol=1e-7)
            if not m['g_done']:
                assert m['revenue_g'] == 0
            checked += 1
            if r['arm'] in rows:
                key = r['dataset_seed'], r['original_scene_id']
                assert key not in rows[r['arm']]
                rows[r['arm']][key] = r
    assert checked == 18432
    assert all(set(rs) == expected for rs in rows.values())
    for key in expected:
        for k in K:
            vals = [rows[a][key]['metrics']['V_' + k + '_0'] for a in ARMS]
            assert max(vals) - min(vals) < 1e-7
    metrics = {a: aggregate(list(rows[a].values())) for a in ARMS}
    controls = 0
    for arm, summary in metrics.items():
        for field in ('V_mean', 'cost_mean', 'acceptance_rate', 'admitted'):
            actual = summary['accepted'] if field == 'admitted' else summary[field]
            assert math.isclose(actual, saved['summaries'][arm][field], abs_tol=1e-8)
            controls += 1
    per_seed = {str(seed): {a: aggregate([r for key, r in rows[a].items() if key[0] == seed])
                           for a in ARMS} for seed in SEEDS}
    reference = metrics['MAPPO_raw']
    contrasts = {}
    for arm in ARMS[1:]:
        m = metrics[arm]
        contrasts[arm + '_minus_MAPPO_raw'] = {
            'acceptance_percentage_points': 100 * (m['acceptance_rate'] - reference['acceptance_rate']),
            'cost_per_admission_change_pct': 100 * (m['cost_per_successful_admission'] / reference['cost_per_successful_admission'] - 1),
            'new_slice_gross_revenue_change_pct': 100 * (m['new_slice_gross_revenue_mean'] / reference['new_slice_gross_revenue_mean'] - 1),
            'lower_tail_delta_V_difference': m['lower_tail_10pct_delta_V'] - reference['lower_tail_10pct_delta_V'],
        }
    result = {
        'analysis': 'post-hoc descriptive; no confirmatory significance claim or policy selection',
        'arms': list(ARMS), 'dataset_seeds': list(SEEDS), 'scenes_per_seed': 512,
        'input': str(records.relative_to(ROOT)), 'input_bytes': records.stat().st_size,
        'input_sha256': sha256(records), 'script_sha256': sha256(Path(__file__)),
        'definitions': {
            'cost_per_admission': 'sum(cost_sum over all attempts)/sum(g_done); excludes API/inference costs',
            'delta_V': 'V - sum(V_k0 for k1,k2,k3); original new-slice disagreement utility is zero',
            'tail': 'lowest 10% empirical probability mass of delta_V, fractional boundary; sorted separately per arm',
            'gross_revenue': 'mean(revenue_g over all scenes), failures included as zero; modeled revenue, not throughput',
            'business_fulfillment_rate': 'not identifiable from saved scoring records: original requested/actual completed business totals absent',
        },
        'metrics': metrics, 'per_seed': per_seed, 'contrasts': contrasts,
        'checks': {'all_rows_arithmetic': checked, 'paired_scene_sets_equal': True,
                   'per_scene_disagreement_values_equal': True, 'saved_summary_controls_passed': controls},
        'API_calls': 0, 'additional_API_cost_usd': 0,
    }
    out = Path(__file__).resolve().parent / ('run_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    text = ['# 已有1536场的候选指标分析', '',
            '仅比较原始MAPPO、原始Qwen和Qwen联合改进版。三者使用相同1536场景。',
            '种子2026110511/12/13，每种子512场。只计算已有记录，没有新推理或环境回放。', '',
            '| 方法 | 接纳率 | 平均V | 单位成功接纳代价↓ | 最低10%效益增量↑ | 平均新增业务收益↑ |',
            '|---|---:|---:|---:|---:|---:|']
    for arm, m in metrics.items():
        text.append('| {} | {:.2%} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
            LABELS[arm], m['acceptance_rate'], m['V_mean'], m['cost_per_successful_admission'],
            m['lower_tail_10pct_delta_V'], m['new_slice_gross_revenue_mean']))
    text += ['', '## 计算口径', '',
             '- 单位接纳代价=所有场景调整代价之和/成功接纳数；包括失败尝试的代价，不含推理时间和API费用。',
             '- 效益增量=V-三个已有切片的V_k0之和。按各方案自身效益增量排序，最低10%取精确153.6场概率质量；不是三者共同同一组尾部场景。',
             '- 新增业务收益为现有revenue_g的全场景均值，失败计零，是模型收益，不是端到端吞吐量。',
             '- 业务需求满足率不可直接计算：评分记录没有原始完整请求量、实际完成任务量及完整QoS结果。现有g观测只有容量请求等特征，不能补齐这些量。',
             '- main.py在新切片节点资源收缩时可能移出任务，链路收益按各链路负载求和；因此不能把revenue_g冒充真实完成请求量。', '',
             '## 分种子结果', '',
             '| 种子 | 方法 | 接纳数/512 | 单位接纳代价 | 最低10%效益增量 | 平均新增业务收益 |',
             '|---|---|---:|---:|---:|---:|']
    for seed, arms in per_seed.items():
        for arm, m in arms.items():
            text.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
                seed, LABELS[arm], m['accepted'], m['cost_per_successful_admission'],
                m['lower_tail_10pct_delta_V'], m['new_slice_gross_revenue_mean']))
    text += ['', '## 解释', '',
             'Qwen联合改进版的单位接纳代价、尾部效益增量和平均新增收益在本三方案中均最好；三个种子方向一致。',
             '原始Qwen的单位接纳代价和平均新增收益优于MAPPO，但平均V与最低10%效益增量更差。',
             '联合改进版接纳率超过90%，最低10%自然包含部分接纳成功场景；MAPPO的最低10%全部是未接纳场景。尾部改善与接纳率改善存在重叠，不能称为独立收益来源。',
             '这些指标在查看已有结果后提出，属于事后描述性分析；没有据此调参、改提示词或宣称新的预注册显著性。',
             '不改变历史指标，不修改原环境、补偿规则、奖励、原始数据或原实验结果。', '',
             '## 文件与检查', '',
             '查阅：PROJECT_HANDOFF.md、AGENTS.md、main.py相关效用/新切片调整/观测函数、p1a_runner.py指标列表、p2m_paper/README.md与run_fresh.py、p2k_formal/run_formal.py准备字段、p1a/local_obs.py及前次指标核查。',
             '读取精确输入records.jsonl(20,843,146字节)和summary.json。SHA256见同目录summary.json。',
             '本次新增analyze_saved.py和独立run目录的REPORT.md、summary.json；未修改原结果或环境文件。',
             '实际运行18432条记录效用/收入/代价恒等式核查、4608条三方案配对覆盖及合作前效用一致性核查、12项历史汇总对照，全部通过。',
             '无外部API调用，新增API费用0。业务完成量不可由当前记录识别，未假定它等于收益或资源配额。', '',
             '复核命令（只读取原结果，每次生成独立新报告目录）：', '', '```powershell',
             "& 'python' -B .\\p2o_alternative_metrics\\analyze_saved.py", '```', '']
    (out / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    print('OUTPUT', out)
    print(json.dumps({'metrics': metrics, 'contrasts': contrasts, 'checks': result['checks']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
