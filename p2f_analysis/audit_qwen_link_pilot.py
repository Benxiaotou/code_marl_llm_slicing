"""Audit completed link pilot and compare both arms, without model calls."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_rows(path):
    assert path.is_file() and path.stat().st_size < 25_000_000
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    folder = Path(args.run_dir).resolve()
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'COMPLETE' and protocol['expected_calls'] == 240
    assert protocol['model'] == 'qwen3-next-80b-a3b-instruct'
    assert protocol['scene_ids'] == list(range(40)) and protocol['split'] == 'validation'
    source = ROOT / 'p2b_runs/val5601_standard1000_p2a200/records.jsonl'
    cached = {r['original_scene_id']: r for r in read_rows(source)}
    arms = protocol['arms']
    events = read_rows(folder / 'requests.jsonl')
    starts = {e['request_id']: e for e in events if e['event'] == 'started'}
    finishes = {e['request_id']: e for e in events if e['event'] == 'finished'}
    assert len(events) == 480 and len(starts) == len(finishes) == 240
    assert starts.keys() == finishes.keys() and not any(e['event'] == 'failed' for e in events)
    totals = collections.Counter()
    usage_missing = 0
    for event in finishes.values():
        assert event['model'] == protocol['model'] and event['API_cost_usd'] == 0
        usage = event.get('usage')
        if not usage:
            usage_missing += 1
        else:
            for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                if k in usage:
                    totals[k] += int(usage[k])
    by = {}
    for arm in arms:
        rows = read_rows(folder / (arm + '_records.jsonl'))
        assert [r['original_scene_id'] for r in rows] == list(range(40))
        by[arm] = rows
        for r in rows:
            scene = r['original_scene_id']
            original = cached[scene]
            assert r['env_actions']['agent_g'] == original['env_actions']['agent_g']
            for role in ('agent_k1', 'agent_k2', 'agent_k3'):
                assert r['env_actions'][role][:2] == original['env_actions'][role][:2]
                decision = r['per_role_decisions'][role]
                expected = [dict(x) for x in original['per_role'][role]['messages']]
                if arm == 'link_calibration':
                    expected[0]['content'] += '\n' + protocol['prompt']
                assert decision['messages'] == expected
                event = starts[f'{arm}:{scene}:{role}']
                assert event['messages_sha256'] == sha(json.dumps(expected, ensure_ascii=False))
                assert decision['link_components_applied'] == r['env_actions'][role][2:]
        for key, raw_key in (('V_mean', 'V'), ('cost_mean', 'cost_sum'),
                             ('acceptance_rate', 'g_done'), ('cooperative_rate', 'cooperative_valid')):
            assert abs(np.mean([float(r['metrics'][raw_key]) for r in rows]) - summary['results'][arm][key]) < 1e-8
    first, second = by[arms[1]], by[arms[0]]
    rng = np.random.default_rng(20261104)
    ids = rng.integers(0, 40, (10000, 40))
    paired = {}
    for key in ('V', 'V_g', 'cost_sum', 'g_done', 'cooperative_valid'):
        delta = np.array([float(a['metrics'][key]) - float(b['metrics'][key]) for a, b in zip(first, second)])
        paired[key] = {'candidate_minus_control_mean': float(delta.mean()),
                       'exploratory_CI95': np.quantile(delta[ids].mean(axis=1), [.025, .975]).tolist()}
    paired['cooperative_wins'] = [a['original_scene_id'] for a, b in zip(first, second) if a['metrics']['cooperative_valid'] and not b['metrics']['cooperative_valid']]
    paired['cooperative_losses'] = [a['original_scene_id'] for a, b in zip(first, second) if b['metrics']['cooperative_valid'] and not a['metrics']['cooperative_valid']]
    report = {'checks_passed': True, 'requests_finished': 240, 'failed_or_retried': 0,
              'role_messages_and_fixed_action_checks': 240,
              'API_cost_usd': 0, 'model': protocol['model'], 'token_usage_reported': dict(totals),
              'requests_without_usage': usage_missing, 'summaries': summary['results'],
              'paired_candidate_minus_fresh_control': paired,
              'limitations': '40 development scenes, one draw per arm; only new link components are applied. Not a full joint policy or independent test.'}
    out = folder / 'audit_and_paired.json'
    with out.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('AUDIT_PASSED', out)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
