"""Read-only reconstruction and journal audit; creates only a new audit file."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as run


def audit(folder):
    from p1a.policy import parse_ratios, to_env_action
    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    from p2a.policy import IRRiskRerankPolicy
    from teacher_link_multicohort import teacher_targets
    folder = Path(folder).resolve()
    assert folder.parent == run.ROOT / 'p2j_validation'
    def read(name):
        p = folder / name
        assert p.is_file() and p.stat().st_size < 100 * 1024**2
        return json.loads(p.read_text(encoding='utf-8'))
    def lines(name):
        p = folder / name
        assert p.is_file() and p.stat().st_size < 100 * 1024**2
        with p.open(encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    protocol, summary = read('protocol.json'), read('summary.json')
    assert summary['status'] == 'COMPLETE' and summary['API_calls'] == 800
    assert not summary['parameters_changed'] and not summary['fallback_or_retry']
    assert tuple(protocol['new_generation_seeds']) == run.SEEDS
    assert protocol['primary_endpoint'] == 'Frozen_link_node mean V minus MAPPO on all200 fixed new scenes'
    for relative, digest in protocol['source_sha256'].items():
        assert run.replay.sha256(run.ROOT / relative) == digest
    manifests = read('datasets.json')
    assert len(manifests) == 2
    for m in manifests:
        assert m['split'] == 'validation' and m['n_scenes'] == 100 and m['seed'] in run.SEEDS
        assert run.replay.sha256(Path(m['sample_file'])) == m['sample_sha256']
    assert len({m['sample_sha256'] for m in manifests}) == 2
    library = LocalHistory(run.LIBRARY); ir = IRHistory(run.IR_LIBRARY, library)
    policy = IRRiskRerankPolicy(library, ir, client=object(), examples=3, seed=1)
    for m in manifests:
        library.assert_independent(m)
    prepared = lines('prepared.jsonl'); responses = lines('responses.jsonl')
    assert len(prepared) == 200 and len(responses) == 800
    response_lookup = {r['request_id']: r for r in responses}
    assert len(response_lookup) == 800
    events = lines('requests.jsonl')
    assert len(events) == 1600 and {r['event'] for r in events} == {'started', 'finished'}
    starts = {e['request_id']: e for e in events if e['event'] == 'started'}
    ends = {e['request_id']: e for e in events if e['event'] == 'finished'}
    assert starts.keys() == ends.keys() == response_lookup.keys() and len(starts) == 800
    records = lines('records.jsonl'); assert len(records) == 800
    by_scene = collections.defaultdict(dict)
    for row in records:
        key = (row['dataset_seed'], row['original_scene_id'])
        assert row['arm'] not in by_scene[key]
        by_scene[key][row['arm']] = row
        run.validate_metrics(row['metrics'])
    assert len(by_scene) == 200
    seen = set()
    for item in prepared:
        seed, scene = item['dataset_seed'], item['original_scene_id']
        assert seed in run.SEEDS and scene in range(100) and (seed, scene) not in seen
        seen.add((seed, scene)); qa = {}; targets = {}
        for role in run.ROLES:
            obs = item['observations'][role]
            matches = policy.library.query(role, obs, top_k=3)
            messages = policy._make_messages(role, obs, matches)
            assert messages == item['per_role'][role]['messages']
            assert [x['experience_id'] for x in matches] == item['per_role'][role]['retrieved_ids']
            rid = str(seed) + ':' + str(scene) + ':' + role
            r = response_lookup[rid]
            assert run.message_hash(messages) == r['prompt_sha256'] == starts[rid]['prompt_sha256']
            assert r['model'] == run.MODEL and r['finish_reason'] == 'stop'
            ratios = parse_ratios(r['raw_model_output'], role)
            assert ratios == r['ratios']
            qa[role] = to_env_action(role, ratios).tolist()
            assert qa[role] == r['env_action']
            if role in run.K:
                t, info = teacher_targets(library, role, obs)
                assert np.allclose(t['link_mean3'], item['link_targets'][role], atol=1e-12)
                assert info['link_mean3_ids'] == item['link_target_TRAIN_ids'][role]
                targets[role] = t['link_mean3']
        link, node, _ = run.derived_actions(qa, item['observations'], targets)
        rows = by_scene[seed, scene]
        assert set(rows) == set(run.ARMS)
        for label, expected in zip(run.ARMS, (item['MAPPO_actions'], qa, link, node)):
            assert rows[label]['env_actions'] == expected
        assert not run.replay.control_diff(item['MAPPO_metrics'], rows['MAPPO']['metrics'])
    for name in run.ARMS:
        rows = [r for r in records if r['arm'] == name]
        for key, value in run.summarize(rows).items():
            assert math.isclose(value, summary['summaries'][name][key], abs_tol=1e-9)
    assert read('historical_preflight.json')['passed'] and read('preflight.json')['passed']
    output = {'passed': True, 'new_paired_scenes': 200, 'fresh_requests': 800,
              'journal_events': 1600, 'reconstructed_prompts': 800,
              'reconstructed_postprocessed_actions': 400, 'scored_rows_checked': 800,
              'MAPPO_control_metrics_reproduced': 200,
              'sources_and_dataset_hashes_unchanged': True, 'API_calls_during_audit': 0}
    run.write_json(folder / 'audit.json', output)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-dir', required=True)
    print(json.dumps(audit(p.parse_args().run_dir)))
