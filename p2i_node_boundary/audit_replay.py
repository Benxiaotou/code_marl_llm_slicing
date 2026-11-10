"""Independent saved-output checks; no environment or model construction."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_replay import ROOT, MULTI, MANIFEST, K, ARMS, replay, stats, validate_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    folder = Path(args.run_dir).resolve()
    assert folder.parent == ROOT / 'p2i_node_boundary'
    for name in ('protocol.json', 'summary.json', 'records.jsonl', 'controls.jsonl'):
        p = folder / name
        assert p.is_file() and p.stat().st_size < 30 * 1024**2
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'COMPLETE' and summary['API_calls'] == 0
    for relative, value in protocol['source_hashes'].items():
        assert replay.sha256(ROOT / relative) == value
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert replay.sha256(Path(manifest['sample_file'])) == manifest['sample_sha256']
    cached = {}
    with MULTI.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['method'] == 'link_mean3_a0.5':
                cached[row['cohort'], row['original_scene_id']] = row
    arms = {name: (delta, budget) for name, delta, budget in ARMS}
    cases = collections.defaultdict(dict)
    changed_groups = collections.Counter()
    for line in (folder / 'records.jsonl').open(encoding='utf-8'):
        r = json.loads(line)
        label, name, scene = r['cohort'], r['method'], r['original_scene_id']
        assert scene not in cases[label, name]
        cases[label, name][scene] = r
        source = cached[label, scene]
        delta, budget = arms[name]
        assert r['env_actions']['agent_g'] == source['env_actions']['agent_g']
        for role in K:
            a, b = r['env_actions'][role], source['env_actions'][role]
            assert a[2:] == b[2:] and all(-1.000001 <= x <= 1.000001 for x in a)
            for j, detail in enumerate(r['local_adjustment'][role]):
                inc = (a[j] - b[j]) / 2
                assert -1e-12 <= inc <= delta + 1e-12
                assert math.isclose(inc, detail['ratio_increment'], abs_tol=1e-12)
                assert detail.get('max_extra_projected_overload', 0) <= budget + 1e-10
                changed_groups[label, name] += int(inc > 1e-8)
        validate_metrics(r['metrics'])
    assert len(cases) == 15 and sum(map(len, cases.values())) == 3000
    for (label, name), rows in cases.items():
        assert sorted(rows) == list(range(200))
        ordered = [rows[i] for i in range(200)]
        actual = stats(ordered)
        saved = summary['summaries'][label][name]
        for key in actual:
            assert math.isclose(actual[key], saved[key], abs_tol=1e-9)
        p = summary['comparisons'][label][name]['versus_frozen']
        for field in ('g_done', 'cooperative_valid'):
            for direction, positive in (('win_ids', True), ('loss_ids', False)):
                expected = [i for i in range(200) if
                    bool(rows[i]['metrics'][field]) == positive and
                    bool(cached[label, i]['metrics'][field]) != positive]
                assert expected == p[field][direction]
        actual_delta = sum(rows[i]['metrics']['V'] - cached[label, i]['metrics']['V']
                           for i in range(200)) / 200
        assert math.isclose(actual_delta, p['V']['mean_delta'], abs_tol=1e-9)
    controls = [json.loads(line) for line in (folder / 'controls.jsonl').open(encoding='utf-8')]
    assert len(controls) == 812
    mappo = replay.read_jsonl(ROOT / 'p2b_runs/val5601_mappo200/records.jsonl')
    for row in controls:
        source = mappo[row['original_scene_id']] if row['case'] == 'MAPPO' else cached[row['case'], row['original_scene_id']]
        assert not replay.control_diff(source['metrics'], row['metrics'])
    eligible = [name for name, _, _ in ARMS if all(
        summary['comparisons'][label][name]['versus_frozen']['V']['mean_delta'] > 0 and
        not summary['comparisons'][label][name]['versus_frozen']['g_done']['loss_ids'] and
        summary['comparisons'][label][name]['versus_frozen']['cooperative_valid']['mean_delta'] >= 0
        for label in protocol['cohorts'])]
    assert eligible == summary['eligible_under_preregistered_rule']
    audit = {'passed': True, 'intervention_rows': 3000, 'controls_reproduced': 812,
             'source_and_dataset_hashes_unchanged': True, 'API_calls': 0,
             'eligible': eligible, 'changed_control_groups': {
                 label + '/' + name: count for (label, name), count in changed_groups.items()}}
    with (folder / 'audit.json').open('x', encoding='utf-8') as f:
        json.dump(audit, f, indent=2); f.write('\n')
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
