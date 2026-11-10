"""Offline development diagnostics; no inference, environment step, or test tuning.

Read exact completed-run files only. Keep P2-C to its published aggregates;
compute newly proposed metrics on validation data only. Never overwrite output.
"""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SHA = '7d92e97808f3269dcf11b3275c1cf85f6daae50d1678f412b7c29e347ad79c86'
INPUTS = {}
ROLES = ('k1', 'k2', 'k3')


def source(relative):
    path = ROOT / relative
    assert path.is_file(), str(path)
    # Check metadata before parsing. Hashes identify the existing records used.
    size = path.stat().st_size
    assert size < 25_000_000, (path, size)
    if str(relative) not in INPUTS:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        INPUTS[str(relative)] = {'bytes': size, 'sha256': digest.hexdigest()}
    return path


def read_csv(relative):
    with source(relative).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load_records(paths, count):
    rows = {}
    for path in paths:
        with source(path).open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                assert record['status'] == 'ok' and not record.get('fallback_used')
                scene = int(record['original_scene_id'])
                assert scene not in rows
                m = record['metrics']
                base = sum(m['V_' + k + '_0'] for k in ROLES)
                margins = [m['V_' + k] - m['V_' + k + '_0'] for k in ROLES]
                relative = [x / max(abs(m['V_' + k + '_0']), 1e-8)
                            for x, k in zip(margins, ROLES)]
                loss = sum(max(0., -x) for x in margins)
                ir = all(x >= -1e-8 for x in margins) and m['V_g'] >= -1e-8
                assert ir == m['individual_rationality']
                assert m['cooperative_valid'] == (m['g_done'] and ir)
                revenue = sum(m['revenue_' + k] for k in (*ROLES, 'g'))
                assert math.isclose(m['V'], revenue - m['cost_sum'], abs_tol=1e-8)
                rows[scene] = {
                    'scene_id': scene, 'accept': int(m['g_done']),
                    'coop': int(m['cooperative_valid']), 'V': m['V'],
                    'V_g': m['V_g'], 'cost': m['cost_sum'], 'base': base,
                    'gain': m['V'] - base, 'gross_revenue': revenue,
                    'existing_utility': sum(m['V_' + k] for k in ROLES),
                    'effective_V_g': m['V_g'] * int(m['cooperative_valid']),
                    'IR_shortfall': loss,
                    'admitted_IR_shortfall': loss * int(m['g_done']),
                    'minimum_relative_IR_margin': min(relative),
                    'latency_s': record['policy_latency_s'],
                }
                assert all(math.isfinite(float(v)) for v in rows[scene].values())
    assert sorted(rows) == list(range(count))
    return rows


def summarize(rows):
    values = list(rows.values())
    n = len(values)
    out = {'n': n}
    for field in ('accept', 'coop', 'V', 'V_g', 'cost', 'base', 'gain',
                  'gross_revenue', 'existing_utility', 'effective_V_g',
                  'IR_shortfall', 'admitted_IR_shortfall',
                  'minimum_relative_IR_margin', 'latency_s'):
        out[field] = float(np.mean([r[field] for r in values]))
    out['admitted_IR_invalid_n'] = sum(r['accept'] - r['coop'] for r in values)
    out['IR_invalid_fraction_of_admissions'] = (
        out['admitted_IR_invalid_n'] / sum(r['accept'] for r in values)
        if out['accept'] else None)
    tail_n = max(1, math.ceil(.1 * n))
    out['V_lower_tail_10pct_mean'] = float(np.mean(sorted(r['V'] for r in values)[:tail_n]))
    out['cost_upper_tail_10pct_mean'] = float(np.mean(sorted(r['cost'] for r in values)[-tail_n:]))
    return out


def paired(first, second):
    assert set(first) == set(second)
    keys = sorted(first)
    assert all(math.isclose(first[i]['base'], second[i]['base'], abs_tol=1e-8) for i in keys)
    rng = np.random.default_rng(20261104)
    indices = rng.integers(0, len(keys), size=(10000, len(keys)))
    out = {'n_paired': len(keys), 'difference': 'first minus second', 'metrics': {}}
    for field in ('accept', 'coop', 'V', 'V_g', 'cost', 'effective_V_g',
                  'IR_shortfall', 'admitted_IR_shortfall', 'minimum_relative_IR_margin'):
        diff = np.array([first[i][field] - second[i][field] for i in keys])
        ci = np.quantile(diff[indices].mean(axis=1), [.025, .975])
        out['metrics'][field] = {'mean_difference': float(diff.mean()),
                                'exploratory_paired_bootstrap_CI95': ci.tolist()}
    shared = [i for i in keys if first[i]['coop'] and second[i]['coop']]
    out['both_cooperative_n'] = len(shared)
    out['cost_difference_on_both_cooperative_scenes'] = (
        float(np.mean([first[i]['cost'] - second[i]['cost'] for i in shared])) if shared else None)
    out['V_wins'] = sum(first[i]['V'] > second[i]['V'] + 1e-8 for i in keys)
    out['V_losses'] = sum(first[i]['V'] < second[i]['V'] - 1e-8 for i in keys)
    return out


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p2b_names = ('mappo', 'standard1000_p2a', 'standard500_p2a')
    runs = {}
    for name in p2b_names:
        directory = 'p2b_runs/val5601_' + name + '200'
        meta = json.loads(source(directory + '/manifest.json').read_text(encoding='utf-8'))
        assert meta['eval_dataset_split'] == 'validation'
        assert meta['eval_dataset_id'] == 'p2b_val_seed5601'
        assert meta['eval_dataset_sha256'] == SHA
        runs[name] = load_records([directory + '/records.jsonl'], 200)
        csv_rows = read_csv(directory + '/metrics.csv')
        assert len(csv_rows) == 200
        for row in csv_rows:
            cached = runs[name][int(row['original_scene_id'])]
            for dest, src in (('V', 'V'), ('V_g', 'V_g'), ('cost', 'cost_sum')):
                assert math.isclose(cached[dest], float(row[src]), abs_tol=1e-8)
    gpt_root = 'gpt61_runs/val5601_prompt100_v1'
    comp_dir = gpt_root + '/comparison_20261104T103753_832997Z'
    protocol = json.loads(source(comp_dir + '/comparison.json').read_text(encoding='utf-8'))['protocol']
    assert protocol['dataset_split'] == 'validation' and protocol['dataset_id'] == 'p2b_val_seed5601'
    assert protocol['scene_ids'] == list(range(100))
    for name in ('semantic_balance', 'joint_feasibility'):
        paths = [gpt_root + '/' + name + '/chunks/{:04d}_005/records.jsonl'.format(i)
                 for i in range(0, 100, 5)]
        runs[name] = load_records(paths, 100)
    paired_csv = read_csv(comp_dir + '/paired_100scene_metrics.csv')
    assert len(paired_csv) == 300
    for row in paired_csv:
        r = runs[row['method']][int(row['scene_id'])]
        for field in ('V', 'V_g', 'cost', 'accept'):
            assert math.isclose(r[field], float(row[field]), abs_tol=1e-8)

    # Historical final-test aggregates only; do not open test scene records.
    test_summary = read_csv('p2c_runs/qwen8_test_v1/p2c_summary_by_seed_and_pooled.csv')
    test_paired = read_csv('p2c_runs/qwen8_test_v1/p2c_paired_comparisons.csv')
    test_summary = [r for r in test_summary if r['seed'] == 'pooled']
    assert len(test_summary) == 8 and all(int(r['n_scenes']) == 1536 for r in test_summary)

    summaries = []
    for n in (100, 200):
        for name, records in runs.items():
            if len(records) < n:
                continue
            subset = {i: records[i] for i in range(n)}
            for i in subset:
                assert math.isclose(subset[i]['base'], runs['mappo'][i]['base'], abs_tol=1e-8)
            summaries.append({'method': name, **summarize(subset)})
    pairs = {}
    for a, b, n in (('standard1000_p2a', 'mappo', 200),
                    ('standard500_p2a', 'mappo', 200),
                    ('standard1000_p2a', 'mappo', 100),
                    ('semantic_balance', 'mappo', 100),
                    ('joint_feasibility', 'mappo', 100),
                    ('standard1000_p2a', 'semantic_balance', 100)):
        pairs['{}_minus_{}_n{}'.format(a, b, n)] = paired(
            {i: runs[a][i] for i in range(n)}, {i: runs[b][i] for i in range(n)})
    out = ROOT / 'p2f_analysis' / ('results_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    report = {
        'dataset': 'p2b_val_seed5601', 'split': 'validation', 'dataset_sha256': SHA,
        'bootstrap_seed': 20261104, 'bootstrap_replicates': 10000,
        'limitations': [
            'New metrics are post-hoc development diagnostics, not confirmatory test endpoints.',
            'Bootstrap intervals are unadjusted exploratory intervals; one LLM draw per scene.',
            'Qwen P2-A and GPT IR-aware differ in retrieval and prompts as well as model.',
            'Each method uses its own tail scenes; common-success cost is a conditional diagnostic.',
            'Effective V_g is cooperative_valid * V_g per scene, averaged over ALL scenes.',
            'IR shortfall sums max(0, V_k0 - V_k) over existing roles, averaged over ALL scenes.',
            'Raw V, cost and cooperative_valid definitions are unchanged.',
            'No test scene files, model inference, environment replay, or independent data were used.',
        ],
        'validation_summaries': summaries, 'exploratory_paired_results': pairs,
        'historical_test_pooled_summary': test_summary,
        'historical_test_pooled_predefined_comparisons': [r for r in test_paired if r['seed'] == 'pooled'],
        'checks': {'raw_records_validated': sum(len(r) for r in runs.values()),
                   'p2b_csv_rows_checked': 600, 'gpt_comparison_rows_checked': 300,
                   'revenue_identity_checked': True, 'IR_definitions_checked': True,
                   'same_scene_initial_utilities_checked': True,
                   'historical_and_latest_MAPPO_first100_match': True},
        'inputs': INPUTS,
    }
    with (out / 'diagnosis.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    write_csv(out / 'validation_metrics.csv', summaries)
    write_csv(out / 'historical_p2c_pooled.csv', test_summary)
    print('OFFLINE_OUTPUT', out)
    print('CHECKS', json.dumps(report['checks']))
    for row in summaries:
        print(json.dumps({k: row[k] for k in ('method', 'n', 'accept', 'coop', 'V', 'V_g',
              'effective_V_g', 'cost', 'IR_shortfall', 'admitted_IR_shortfall',
              'V_lower_tail_10pct_mean', 'cost_upper_tail_10pct_mean')}))
    for key, result in pairs.items():
        print('PAIR', key, json.dumps({k: result['metrics'][k] for k in
              ('V', 'effective_V_g', 'IR_shortfall')}))


if __name__ == '__main__':
    main()
