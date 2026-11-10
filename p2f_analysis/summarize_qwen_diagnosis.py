"""Summarize completed Qwen/MAPPO cached-action interventions, without API."""
import argparse
import collections
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASE1 = ROOT / 'p2f_analysis/val5601_counterfactual/run_20261104T125542_548095Z'
TRACE = ROOT / 'p2f_analysis/val5601_trace/run_20261104T131408_665359Z'


def records(folder, case):
    path = folder / (case + '_records.jsonl')
    assert path.is_file() and path.stat().st_size < 2_000_000
    with path.open(encoding='utf-8') as f:
        rows = {r['original_scene_id']: r for r in map(json.loads, f)}
    assert sorted(rows) == list(range(200))
    return rows


def stats(rows):
    ms = [r['metrics'] for r in rows.values()]
    return {'n': len(ms), 'accepted': sum(m['g_done'] for m in ms),
            'cooperative': sum(m['cooperative_valid'] for m in ms),
            'admitted_IR_invalid': sum(m['g_done'] and not m['individual_rationality'] for m in ms),
            **{k: float(np.mean([m[k] for m in ms])) for k in ('V', 'V_g', 'cost_sum')}}


def differences(first, second):
    keys = sorted(first)
    rng = np.random.default_rng(20261104)
    bootstrap = rng.integers(0, len(keys), (10000, len(keys)))
    out = {}
    for metric in ('V', 'V_g', 'cost_sum', 'g_done', 'cooperative_valid'):
        diff = np.array([float(first[i]['metrics'][metric]) - float(second[i]['metrics'][metric]) for i in keys])
        out[metric] = {'mean_difference': float(diff.mean()),
                       'exploratory_paired_bootstrap_CI95': np.quantile(diff[bootstrap].mean(axis=1), [.025, .975]).tolist()}
    out['cooperative_wins'] = [i for i in keys if first[i]['metrics']['cooperative_valid'] and not second[i]['metrics']['cooperative_valid']]
    out['cooperative_losses'] = [i for i in keys if second[i]['metrics']['cooperative_valid'] and not first[i]['metrics']['cooperative_valid']]
    out['admission_changed'] = [i for i in keys if first[i]['metrics']['g_done'] != second[i]['metrics']['g_done']]
    return out


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--pilot-dir')
    opt = args.parse_args()
    p1 = json.loads((PHASE1 / 'summary.json').read_text(encoding='utf-8'))
    p2 = json.loads((TRACE / 'summary.json').read_text(encoding='utf-8'))
    assert p1['status'] == p2['status'] == 'VALIDATED'
    cases = {c: records(PHASE1, c) for c in ('MM', 'QQ', 'MQ', 'QM', 'Q_NODE_M', 'Q_LINK_M')}
    traces = {c: records(TRACE, c) for c in ('MM', 'QQ', 'Q_LINK_M', 'Q_LINK_k1_M', 'Q_LINK_k2_M', 'Q_LINK_k3_M')}
    for c in ('MM', 'QQ', 'Q_LINK_M'):
        for scene in cases[c]:
            for k, v in cases[c][scene]['metrics'].items():
                assert math.isclose(float(v), float(traces[c][scene]['metrics'][k]), abs_tol=1e-8)
    cases.update({c: traces[c] for c in ('Q_LINK_k1_M', 'Q_LINK_k2_M', 'Q_LINK_k3_M')})
    q, m = cases['QQ'], cases['MM']
    cohorts = {
        'Q_admitted_IR_invalid': [i for i in q if q[i]['metrics']['g_done'] and not q[i]['metrics']['individual_rationality']],
        'Q_rejected': [i for i in q if not q[i]['metrics']['g_done']],
        'M_cooperative_Q_not': [i for i in q if m[i]['metrics']['cooperative_valid'] and not q[i]['metrics']['cooperative_valid']],
        'Q_cooperative_M_not': [i for i in q if q[i]['metrics']['cooperative_valid'] and not m[i]['metrics']['cooperative_valid']],
    }
    summaries = {c: stats(rows) for c, rows in cases.items()}
    paired = {c + '_minus_QQ': differences(rows, q) for c, rows in cases.items() if c != 'QQ'}
    components = {}
    gates = {}
    for case, rows in traces.items():
        components[case] = {key: float(np.mean([sum(r['passive_trace']['roles'][k][key] for k in ('k1', 'k2', 'k3'))
                                               for r in rows.values()]))
                            for key in ('node_cost', 'link_cost', 'node_revenue', 'link_revenue', 'sum_link_load')}
        gates[case] = dict(collections.Counter((r['passive_trace']['admission_gate']['first_failure'] or {'stage': 'accepted'})['stage']
                                               for r in rows.values()))
    cohort_results = {}
    for name, ids in cohorts.items():
        cohort_results[name] = {'scene_ids': ids, 'n': len(ids),
                                'cases': {c: stats({i: r[i] for i in ids}) for c, r in cases.items()}}
    for i in q:
        assert q[i]['metrics']['g_done'] == cases['Q_LINK_M'][i]['metrics']['g_done']
        assert math.isclose(q[i]['metrics']['revenue_g'], cases['Q_LINK_M'][i]['metrics']['revenue_g'], abs_tol=1e-8)
        for k in ('k1', 'k2', 'k3'):
            for field in ('node_cost', 'node_revenue'):
                assert math.isclose(traces['QQ'][i]['passive_trace']['roles'][k][field],
                                    traces['Q_LINK_M'][i]['passive_trace']['roles'][k][field], abs_tol=1e-8)
    pilot = None
    if opt.pilot_dir:
        pp = Path(opt.pilot_dir).resolve()
        pilot = json.loads((pp / 'summary.json').read_text(encoding='utf-8'))
        assert pilot['status'] == 'COMPLETE'
    result = {'split': 'validation', 'n_scenes': 200, 'summaries': summaries,
              'paired_exploratory': paired, 'cohorts': cohort_results,
              'passive_components': components, 'admission_gates': gates,
              'bootstrap_seed': 20261104, 'bootstrap_repetitions': 10000,
              'pilot': pilot, 'source_folders': [str(PHASE1), str(TRACE)],
              'checks': {'phase1_steps': 1200, 'trace_steps': 1200,
                         'phase1_MM_QQ_control_scenes': 400,
                         'trace_MM_QQ_Q_LINK_M_reference_scenes': 600,
                         'passive_cost_decomposition_role_checks': 3600,
                         'link_swap_node_cost_revenue_unchanged': True,
                         'link_swap_g_revenue_and_admission_unchanged': True},
              'limitations': 'Cached current MAPPO actions are diagnostic interventions only. These results are not a deployable method, independent testing, or evidence that uniformly increasing link release is beneficial.'}
    out = ROOT / 'p2f_analysis/qwen_diagnosis' / ('report_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tab = [{'case': c, **v} for c, v in summaries.items()]
    with (out / 'comparison.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(tab[0]))
        writer.writeheader()
        writer.writerows(tab)
    print('OUTPUT', out)
    print('CHECKS', json.dumps(result['checks']))
    for c in ('Q_LINK_M', 'Q_LINK_k1_M', 'Q_LINK_k2_M', 'Q_LINK_k3_M'):
        print(c, json.dumps(paired[c + '_minus_QQ']))


if __name__ == '__main__':
    main()
