"""Offline audit, paired estimates and report for the full-policy Qwen run."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

import qwen_full200 as experiment

ROOT = experiment.ROOT
ROLES = experiment.replay.ROLES


def load_jsonl(path):
    assert path.is_file() and path.stat().st_size < 50_000_000, str(path)
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows):
    out = experiment.replay.aggregate(rows)
    metrics = [r['metrics'] for r in rows]
    out['accepted'] = sum(m['g_done'] for m in metrics)
    out['cooperative'] = sum(m['cooperative_valid'] for m in metrics)
    out['effective_V_g'] = float(np.mean([m['V_g'] * m['cooperative_valid'] for m in metrics]))
    out['IR_shortfall'] = float(np.mean([sum(max(0, m['V_' + k + '_0'] - m['V_' + k]) for k in ('k1', 'k2', 'k3')) for m in metrics]))
    count = math.ceil(.1 * len(rows))
    out['V_lower10pct_mean'] = float(np.mean(sorted(m['V'] for m in metrics)[:count]))
    out['cost_upper10pct_mean'] = float(np.mean(sorted(m['cost_sum'] for m in metrics)[-count:]))
    return out


def paired(first, second):
    assert [r['original_scene_id'] for r in first] == [r['original_scene_id'] for r in second]
    n = len(first)
    rng = np.random.default_rng(20261104)
    indices = rng.integers(0, n, (10000, n))
    out = {'n_paired': n, 'difference': 'first minus second', 'metrics': {}}
    for key in ('V', 'V_g', 'cost_sum', 'g_done', 'cooperative_valid'):
        a = np.array([float(r['metrics'][key]) for r in first])
        b = np.array([float(r['metrics'][key]) for r in second])
        delta = a - b
        metric = {'mean_difference': float(delta.mean()),
                  'exploratory_paired_bootstrap_CI95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()}
        if key in ('g_done', 'cooperative_valid'):
            wins = int(((a == 1) & (b == 0)).sum())
            losses = int(((a == 0) & (b == 1)).sum())
            metric['wins'] = wins
            metric['losses'] = losses
            d = wins + losses
            metric['exact_McNemar_two_sided_p'] = min(1., 2 * sum(math.comb(d, k) for k in range(min(wins, losses) + 1)) / 2 ** d)
        out['metrics'][key] = metric
    out['cooperative_win_ids'] = [a['original_scene_id'] for a, b in zip(first, second) if a['metrics']['cooperative_valid'] and not b['metrics']['cooperative_valid']]
    out['cooperative_loss_ids'] = [a['original_scene_id'] for a, b in zip(first, second) if b['metrics']['cooperative_valid'] and not a['metrics']['cooperative_valid']]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    folder = Path(args.run_dir).resolve()
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    finished = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    preflight = json.loads((folder / 'preflight.json').read_text(encoding='utf-8'))
    assert finished['status'] == 'COMPLETE' and protocol['mode'] == 'run'
    assert protocol['model'] == experiment.MODEL and protocol['scene_ids'] == list(range(200))
    assert protocol['split'] == 'validation' and protocol['expected_calls_total'] == 1600
    assert protocol['full_actions_fresh'] and not protocol['cached_components_applied']
    assert preflight['passed'] and preflight['base_messages_reconstructed'] == 800
    assert preflight['historical_controls_checked'] == 400
    for relative, value in protocol['source_sha256'].items():
        assert experiment.replay.sha256(ROOT / relative) == value, relative
    source = experiment.replay.read_jsonl(ROOT / 'p2b_runs/val5601_standard1000_p2a200/records.jsonl')
    rows = {name: load_jsonl(folder / (name + '_replay.jsonl')) for name in ('MAPPO', 'historical_Qwen')}
    events = load_jsonl(folder / 'requests.jsonl')
    starts = {r['request_id']: r for r in events if r['event'] == 'started'}
    ends = {r['request_id']: r for r in events if r['event'] == 'finished'}
    assert len(events) == 3200 and len(starts) == len(ends) == 1600 and starts.keys() == ends.keys()
    expected_requests = {f'{arm}:{scene}:{role}' for arm in experiment.ARMS for scene in range(200) for role in ROLES}
    assert set(starts) == expected_requests
    assert not any(r['event'] == 'failed' for r in events)
    totals = collections.Counter()
    missing = 0
    usage_by_arm = {}
    for arm in experiment.ARMS:
        tokens = collections.Counter()
        missing_arm = 0
        for rid, event in ends.items():
            if not rid.startswith(arm + ':'):
                continue
            assert event['model'] == experiment.MODEL
            usage = event.get('usage')
            if not usage:
                missing += 1
                missing_arm += 1
            else:
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    if k in usage:
                        totals[k] += int(usage[k])
                        tokens[k] += int(usage[k])
        usage_by_arm[arm] = {'reported_tokens': dict(tokens), 'requests_without_usage': missing_arm}
        records = load_jsonl(folder / (arm + '_records.jsonl'))
        assert [r['original_scene_id'] for r in records] == list(range(200))
        for row in records:
            assert row['status'] == 'ok' and not row['fallback_used'] and row['all_four_role_actions_fresh']
            scene = row['original_scene_id']
            for role in ROLES:
                from p1a.policy import parse_ratios, to_env_action
                entry = row['per_role'][role]
                expected = experiment.altered_messages(source[scene]['per_role'][role]['messages'], role, arm, protocol['frozen_prompt'])
                assert entry['messages'] == expected
                rid = f'{arm}:{scene}:{role}'
                assert entry['prompt_sha256'] == starts[rid]['messages_sha256'] == experiment.messages_sha(expected)
                assert entry['raw_model_output'] == ends[rid]['response']
                ratios = parse_ratios(entry['raw_model_output'], role)
                action = to_env_action(role, ratios).tolist()
                assert action == row['env_actions'][role] == entry['env_action']
                assert entry['retrieved_ids_for_audit_only'] == source[scene]['per_role'][role]['retrieved_ids_for_audit_only']
            m = row['metrics']
            margins = [m['V_' + k] - m['V_' + k + '_0'] for k in ('k1', 'k2', 'k3')]
            ir = all(x >= -1e-8 for x in margins) and m['V_g'] >= -1e-8
            assert ir == m['individual_rationality']
            assert m['cooperative_valid'] == (m['g_done'] and ir)
            assert math.isclose(m['V'], sum(m['revenue_' + k] for k in ('k1', 'k2', 'k3', 'g')) - m['cost_sum'], abs_tol=1e-8)
            for k in ('V_k1_0', 'V_k2_0', 'V_k3_0'):
                assert math.isclose(m[k], source[scene]['metrics'][k], abs_tol=1e-8)
        rows[arm] = records
    summaries = {name: summarize(values) for name, values in rows.items()}
    for name, values in rows.items():
        assert [r['original_scene_id'] for r in values] == list(range(200))
        for key, expected in finished['summaries'][name].items():
            assert math.isclose(float(summaries[name][key]), float(expected), abs_tol=1e-8)
    comparisons = {
        'candidate_minus_fresh_original': paired(rows[experiment.ARMS[1]], rows[experiment.ARMS[0]]),
        'candidate_minus_MAPPO': paired(rows[experiment.ARMS[1]], rows['MAPPO']),
        'fresh_original_minus_MAPPO': paired(rows[experiment.ARMS[0]], rows['MAPPO']),
        'fresh_original_minus_historical_Qwen': paired(rows[experiment.ARMS[0]], rows['historical_Qwen']),
    }
    subgroups = {}
    for label, lo, hi in (('pilot_overlap_0_39', 0, 40), ('additional_development_40_199', 40, 200)):
        subgroups[label] = {'summaries': {name: summarize(values[lo:hi]) for name, values in rows.items()},
                            'candidate_minus_fresh_original': paired(rows[experiment.ARMS[1]][lo:hi], rows[experiment.ARMS[0]][lo:hi]),
                            'note': 'Descriptive development subgroup, not independent held-out testing.'}
    g_equal = sum(a['env_actions']['agent_g'] == b['env_actions']['agent_g'] for a, b in zip(rows[experiment.ARMS[0]], rows[experiment.ARMS[1]]))
    result = {'checks_passed': True, 'source_hashes_unchanged': True,
              'preflight_original_controls': 400, 'preflight_role_messages': 800,
              'fresh_records_checked': 400, 'fresh_role_messages_and_full_actions_checked': 1600,
              'requests_finished': 1600, 'failed_or_retried': 0,
              'model': experiment.MODEL, 'API_cost_usd_estimate': 0,
              'token_usage_reported': dict(totals), 'requests_without_usage': missing,
              'usage_by_arm': usage_by_arm, 'summaries': summaries,
              'paired_comparisons': comparisons, 'development_subgroups': subgroups,
              'identical_g_actions_between_arms': g_equal,
              'limitations': protocol['limitations'], 'python': sys.version,
              'numpy_version': np.__version__}
    experiment.save_new(folder / 'audit_and_paired.json', result)
    table = [{'method': name, **value} for name, value in summaries.items()]
    with (folder / 'comparison.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    per_scene = []
    for name, values in rows.items():
        for row in values:
            m = row['metrics']
            per_scene.append({'method': name, 'scene': row['original_scene_id'],
                              **{k: m[k] for k in ('g_done', 'cooperative_valid', 'individual_rationality', 'V', 'V_g', 'cost_sum')}})
    with (folder / 'paired_scene_metrics.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_scene[0]))
        writer.writeheader()
        writer.writerows(per_scene)
    lines = ['# Qwen 完整策略 200 场景实验', '',
             '固定 val5601 验证场景 0..199。两组四角色均重新推理并完整应用动作；候选仅追加已有切片 system 提示，g 消息不变但动作重新采样。', '',
             '| 方法 | 接纳 | 合作有效 | 平均 V | 平均 V_g | 平均成本 | 接纳后 IR 违规 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['accepted']}/200 | {s['cooperative']}/200 | {s['V_mean']:.4f} | {s['V_g_mean']:.4f} | {s['cost_mean']:.4f} | {s['accepted_ir_invalid_n']} |")
    lines += ['', '## 配对差异', '', '全部区间为探索性、未经多重比较校正的同场景 Bootstrap 95% 区间（seed20261104，10000次）。']
    for name, comparison in comparisons.items():
        lines += ['', name + ':']
        for key in ('cooperative_valid', 'V', 'cost_sum', 'V_g'):
            s = comparison['metrics'][key]
            lines.append(f"- {key}: {s['mean_difference']:.6f}, CI={s['exploratory_paired_bootstrap_CI95']}")
    lines += ['', '## 审计与解释范围', '',
              f'400 个历史控制、800 份基准消息、400 条新记录、1600 份新角色消息与完整动作均通过核验。1600 次内网 Qwen 调用成功，无失败或重试。g 的跨组动作恰好相同 {g_equal}/200，不能将差异归因于固定 g 下的链路效果。', '',
              f'接口未返回 usage 的请求数为 {missing}/1600；报告的 token 合计为 {dict(totals)}。缺失不能解释为零消耗。新增 API 费用估计 $0，依据用户内网无限额度说明，未经独立账单验证。没有 GPT 调用。', '',
              '这是已有开发验证集上的单次采样；前40场景曾用于小实验，后160场景也经过离线开发诊断，均不是新独立测试。未使用 P2-C 进行调参或验证。原环境、奖励、指标、迁移/重路由、数据、checkpoint、原提示与既有结果均未修改。', '',
              '原 V 的链路收入受按链路流量求和及路由长度影响，不等于端到端吞吐量。保留合作有效率、V 为主要指标，不用事后更换指标掩盖退化。', '',
              '子组、原始请求 journal、完整角色动作和消息审计见 audit_and_paired.json、protocol.json、requests.jsonl、两个 *_records.jsonl。所有输出为新目录。', '',
              '安全查看命令（不调用 API）：', '', '```powershell',
              f'Import-Csv .\\{folder.relative_to(ROOT)}\\comparison.csv | Format-Table -AutoSize', '```', '',
              'run_qwen_full200.ps1 -Mode run 会发起新的1600次请求；本实验已完成，不要为查看结果重跑。', '']
    with (folder / 'RESULTS.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('AUDIT_PASSED', folder, flush=True)
    print(json.dumps({'summaries': summaries, 'paired': comparisons,
                      'usage_missing': missing, 'identical_g_actions': g_equal}, ensure_ascii=False))


if __name__ == '__main__':
    main()
