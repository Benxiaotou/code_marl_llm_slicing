"""Audit saved cached-test replay without an API or environment construction."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cached as run


def audit(folder):
    from p1c.retrieval import LocalHistory
    from teacher_link_multicohort import teacher_targets
    folder = Path(folder).resolve()
    assert folder.parent == run.ROOT / 'p2l_cached_test'
    def lines(name):
        p = folder / name
        assert p.is_file() and p.stat().st_size < 100 * 1024**2
        with p.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    protocol = run.read_json(folder / 'protocol.json', 5_000_000)
    summary = run.read_json(folder / 'scored_summary.json', 200000)
    assert summary['status'] == 'SCORED_AWAITING_AUDIT' and summary['API_calls'] == 0
    assert summary['parameters_changed'] is False and summary['test_previously_inspected'] is True
    for relative, digest in protocol['source_hashes'].items():
        assert run.v.replay.sha256(run.ROOT / relative) == digest, relative
    for seed in run.SEEDS:
        m = run.read_json(run.ROOT / 'p1b_data' / ('p2c_test_seed' + str(seed) + '.json'))
        assert run.v.replay.sha256(Path(m['sample_file'])) == m['sample_sha256']
    history_rows = lines('historical_metrics.jsonl'); records = lines('records.jsonl')
    assert len(history_rows) == 12288 and len(records) == 6144
    methods = collections.defaultdict(dict); replayed = collections.defaultdict(dict)
    for r in history_rows:
        key = r['dataset_seed'], r['original_scene_id']
        assert key not in methods[r['arm']]
        methods[r['arm']][key] = r
        run.v.validate_metrics(r['metrics'])
    assert len(methods) == 8
    for r in records:
        key = r['dataset_seed'], r['original_scene_id']
        assert key not in replayed[r['arm']]
        replayed[r['arm']][key] = r
        run.v.validate_metrics(r['metrics'])
    assert len(replayed) == 4
    keys = {(s, i) for s in run.SEEDS for i in range(run.COUNT)}
    assert all(source.keys() == keys for source in methods.values())
    assert all(source.keys() == keys for source in replayed.values())
    library = LocalHistory(run.v.LIBRARY)
    for key in sorted(keys):
        node = replayed[run.NEW][key]
        original = replayed[run.NAMES['p2a_std1000']][key]['env_actions']
        targets = {}
        for role in run.v.K:
            t, info = teacher_targets(library, role, node['own_observations'][role])
            assert np.allclose(t['link_mean3'], node['link_targets'][role], atol=1e-12)
            assert info['link_mean3_ids'] == node['link_TRAIN_ids'][role]
            targets[role] = t['link_mean3']
        link, expected, adjustments = run.v.derived_actions(original, node['own_observations'], targets)
        assert expected == node['env_actions']
        assert link == replayed[run.LINK][key]['env_actions']
        assert adjustments == node['node_adjustment']
        for name in ('MAPPO', run.NAMES['p2a_std1000']):
            assert not run.v.replay.control_diff(methods[name][key]['metrics'], replayed[name][key]['metrics'])
    prior_checks = 0
    for seed in run.SEEDS:
        for line in (run.PRIOR / ('records_' + str(seed) + '.jsonl')).open(encoding='utf-8'):
            r = json.loads(line)
            if r['arm'] == 'Frozen_link_mean3_a0.5':
                assert not run.v.replay.control_diff(r['metrics'], replayed[run.LINK][seed, r['original_scene_id']]['metrics'])
                prior_checks += 1
    assert prior_checks == 1536
    methods[run.LINK] = replayed[run.LINK]; methods[run.NEW] = replayed[run.NEW]
    for name, source in methods.items():
        computed = run.v.summarize(run.ordered(source))
        for field, value in computed.items():
            assert math.isclose(value, summary['summaries'][name][field], abs_tol=1e-9)
        if name != run.NEW:
            delta = sum(methods[run.NEW][key]['metrics']['V'] - source[key]['metrics']['V'] for key in keys) / 1536
            assert math.isclose(delta, summary['comparisons'][name]['V']['mean_delta'], abs_tol=1e-9)
    post = lines('post_controls.jsonl'); assert len(post) == 12
    for r in post:
        key = r['dataset_seed'], r['original_scene_id']
        assert not run.v.replay.control_diff(r['metrics'], replayed[r['arm']][key]['metrics'])
    result = {'passed': True, 'API_calls': 0, 'archived_rows_checked': 12288,
              'replayed_rows_checked': 6144, 'MAPPO_Qwen_link_controls': 4608,
              'post_controls': 12, 'reconstructed_node_actions': 1536,
              'source_and_sample_hashes_unchanged': True,
              'test_previously_inspected': True, 'parameters_changed': False}
    run.v.write_json(folder / 'audit.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-dir', required=True)
    print(json.dumps(audit(p.parse_args().run_dir)))
