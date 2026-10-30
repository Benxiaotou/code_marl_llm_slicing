"""Matched-scene P1-A base vs semantic comparison with paired bootstrap CIs.

Descriptive development analysis, not publication-ready confirmatory inference.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

KEYS = ('admission', 'cooperative', 'V', 'V_g', 'cost_sum', 'policy_latency_s')
ROLES = ('agent_k1', 'agent_k2', 'agent_k3', 'agent_g')
AUDIT_FIELDS = ('model', 'dataset', 'scene_ids', 'seed', 'config',
                'max_tokens', 'temperature', 'stream', 'main_py_sha256')


def load_run(path):
    folder = Path(path)
    if folder.is_file() and folder.name == 'records.jsonl':
        folder = folder.parent
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    records = {}
    with (folder / 'records.jsonl').open('r', encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get('status') != 'ok' or rec.get('fallback_used', False):
                raise ValueError('Run contains failed, unscored or fallback-contaminated scene')
            scene = int(rec['original_scene_id'])
            if scene in records:
                raise ValueError('Duplicate scene {}'.format(scene))
            for k in ('g_done', 'cooperative_valid', 'V', 'V_g', 'cost_sum'):
                if k not in rec['metrics']:
                    raise ValueError('Missing metric {}'.format(k))
            for k in ('V', 'V_g', 'cost_sum'):
                if not math.isfinite(float(rec['metrics'][k])):
                    raise ValueError('Non-finite metric {}'.format(k))
            for role in ROLES:
                if role not in rec['per_role'] or 'ratios' not in rec['per_role'][role]:
                    raise ValueError('Missing role action: {}'.format(role))
            records[scene] = rec
    if not records or set(records) != set(int(x) for x in manifest['scene_ids']):
        raise ValueError('Records do not match manifest scene IDs')
    return manifest, records


def _as_values(rec):
    metrics = rec['metrics']
    return {
        'admission': int(metrics['g_done']),
        'cooperative': int(metrics['cooperative_valid']),
        'V': float(metrics['V']),
        'V_g': float(metrics['V_g']),
        'cost_sum': float(metrics['cost_sum']),
        'policy_latency_s': float(rec['policy_latency_s']),
    }


def _bootstrap_delta(x, rng, replicates):
    if len(x) < 2:
        return [None, None]
    idx = rng.integers(0, len(x), size=(replicates, len(x)))
    draws = np.mean(x[idx], axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def compare_runs(baseline, enhanced, *, bootstrap=2000, seed=2026):
    man_a, rec_a = load_run(baseline)
    man_b, rec_b = load_run(enhanced)
    if man_a['policy'] != 'qwen_zeroshot_local':
        raise ValueError('Baseline run must be original qwen_zeroshot_local')
    if man_b['policy'] != 'qwen_semantic_local_v1':
        raise ValueError('Enhanced run must be qwen_semantic_local_v1')
    for field in AUDIT_FIELDS:
        if man_a.get(field) != man_b.get(field):
            raise ValueError('Different experimental conditions: {}'.format(field))
    if set(rec_a) != set(rec_b):
        raise ValueError('Unmatched scene IDs')
    scenes = sorted(rec_a)
    rows = []
    for scene in scenes:
        a, b = _as_values(rec_a[scene]), _as_values(rec_b[scene])
        row = {'scene_id': scene}
        for key in KEYS:
            row['base_' + key] = a[key]
            row['semantic_' + key] = b[key]
            row['delta_' + key] = b[key] - a[key]
        action_delta = []
        for role in ROLES:
            x = np.asarray(rec_a[scene]['per_role'][role]['ratios'], dtype=float)
            y = np.asarray(rec_b[scene]['per_role'][role]['ratios'], dtype=float)
            if len(x) != (3 if role == 'agent_g' else 4) or x.shape != y.shape:
                raise ValueError('Mismatched action dimensions for {}'.format(role))
            action_delta.extend(np.abs(x - y).tolist())
        row['mean_abs_action_delta'] = float(np.mean(action_delta))
        rows.append(row)
    rng = np.random.default_rng(seed)
    summary = {'n_paired': len(scenes), 'scene_ids': scenes,
               'baseline_label': man_a['policy'], 'semantic_label': man_b['policy'],
               'warning': ('Development paired estimates. Bootstrap intervals assume scenes independent; '
                           'do not claim final significance without untouched test samples and repeat runs.'),
               'mean_abs_action_delta': float(np.mean([r['mean_abs_action_delta'] for r in rows]))}
    for key in KEYS:
        delta = np.asarray([row['delta_' + key] for row in rows], dtype=float)
        summary[key] = {
            'baseline_mean': float(np.mean([row['base_' + key] for row in rows])),
            'semantic_mean': float(np.mean([row['semantic_' + key] for row in rows])),
            'paired_delta_semantic_minus_baseline': float(np.mean(delta)),
            'paired_bootstrap_95pct_interval': _bootstrap_delta(delta, rng, bootstrap),
        }
    diversity = {}
    for run_name, records in (('baseline', rec_a), ('semantic', rec_b)):
        per_role = {}
        for role in ROLES:
            xs = np.asarray([records[s]['per_role'][role]['ratios'] for s in scenes], dtype=float)
            per_role[role] = {
                'per_dimension_std_across_scenes': [float(v) for v in xs.std(axis=0)],
                'unique_rounded_actions_3dp': len(set(tuple(np.round(v, 3)) for v in xs)),
            }
        diversity[run_name] = per_role
    summary['action_diversity'] = diversity
    return summary, rows


def main(argv=None):
    p = argparse.ArgumentParser(description='Compare original Qwen P1-A vs semantic P1-A+')
    p.add_argument('--baseline', required=True, help='Directory holding P1-A manifest.json and records.jsonl')
    p.add_argument('--semantic', required=True, help='Directory holding P1-A+ manifest.json and records.jsonl')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    if args.bootstrap <= 0 or args.bootstrap > 20000:
        raise ValueError('bootstrap repetitions must be between 1 and 20000')
    summary, rows = compare_runs(args.baseline, args.semantic, bootstrap=args.bootstrap)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'comparison.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (out / 'paired_scenes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
