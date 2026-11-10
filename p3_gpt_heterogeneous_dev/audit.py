"""No-API audit and replay of a completed P3 development experiment."""
import argparse
import contextlib
import json
from pathlib import Path
import time

import numpy as np
import run_dev20 as m


def read_lines(path, limit=30_000_000):
    assert path.is_file() and path.stat().st_size < limit, str(path)
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def audit(folder):
    assert folder.is_dir() and folder.parent == m.ROOT / 'p3_gpt_heterogeneous_dev'
    protocol = m.read_json(folder / 'protocol.json')
    summary = m.read_json(folder / 'summary.json')
    assert summary['status'] == 'COMPLETE'
    assert protocol['scene_ids'] == list(m.SCENES) and protocol['arms'] == m.ASSIGNMENTS
    assert all(m.sha(m.ROOT / p) == h for p, h in protocol['source_hashes'].items())
    assert all(m.sha(m.ROOT / p) == d['sha256'] for p, d in protocol['input_files'].items())
    prepared = {row['scene']: row for row in read_lines(folder / 'prepared.jsonl')}
    assert set(prepared) == set(m.SCENES)
    raw = {}; counts = {}; timing = {}; model_ids = {}
    for name in ('qwen', 'gpt'):
        events = read_lines(folder / f'{name}_requests.jsonl')
        starts = {e['request_id']: e for e in events if e['event'] == 'started'}
        ends = {e['request_id']: e for e in events if e['event'] == 'finished'}
        replies = read_lines(folder / f'{name}_responses.jsonl')
        assert len(events) == 160 and len(starts) == len(ends) == len(replies) == 80
        assert set(starts) == set(ends) and all(e['status'] == 'ok' for e in ends.values())
        raw[name] = {row['request_id']: row for row in replies}
        assert set(raw[name]) == set(starts)
        for row in replies:
            scene, role = row['scene'], row['role']
            messages = prepared[scene]['roles'][role]['messages']
            assert row['prompt_sha256'] == m.msg_hash(messages) == starts[row['request_id']]['prompt_sha256']
            ratios = m.parse_ratios(row['raw_model_output'], role)
            assert ratios == row['ratios']
            assert m.to_env_action(role, ratios).tolist() == row['env_action']
            assert row['total_tokens'] == ends[row['request_id']]['total_tokens']
            if name == 'gpt':
                assert row['total_tokens'] == row['usage']['input_tokens'] + row['usage']['output_tokens']
                assert abs(row['estimated_units'] - row['total_tokens'] / m.TOKENS_PER_UNIT) < 1e-12
        counts[name] = {'calls': len(replies), 'known_tokens': sum(r['total_tokens'] or 0 for r in replies),
                        'unknown_usage': sum(r['total_tokens'] is None for r in replies)}
        timing[name] = {'role_latency_mean_s': float(np.mean([r['latency_s'] for r in replies])),
                        'role_latency_median_s': float(np.median([r['latency_s'] for r in replies]))}
        model_ids[name] = sorted({r['response_model'] for r in replies if r['response_model'] is not None})
    assert counts['gpt']['known_tokens'] == summary['gpt_total_tokens']
    assert counts['gpt']['unknown_usage'] == 0
    assert counts['gpt']['known_tokens'] <= m.TOKEN_BUDGET
    report = folder / ('audit_' + m.dt.datetime.now(m.dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    report.mkdir(exist_ok=False)
    replayed = 0
    by_arm = {}
    with (report / 'env_stdout.log').open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        env = m.make_env(m.read_json(m.DATASET), report)
        for arm, assignment in m.ASSIGNMENTS.items():
            rows = read_lines(folder / f'{arm}_records.jsonl')
            assert len(rows) == 20 and [row['original_scene_id'] for row in rows] == list(m.SCENES)
            by_arm[arm] = rows
            for row in rows:
                scene = row['original_scene_id']
                item = prepared[scene]
                expected = {r: raw[assignment[r]][f'{assignment[r]}:{scene}:{r}']['env_action'] for r in m.ROLES}
                assert row['raw_actions'] == expected and row['role_models'] == assignment
                final, post = m.fixed_postprocess(expected, item['observations'], item['targets'])
                assert row['env_actions'] == final and row['postprocess'] == post
                assert not m.replay.control_diff(row['metrics'], m.score(env, final, scene))
                replayed += 1
            aggregate = m.summarize(rows)
            assert all(aggregate[k] == summary['summaries'][arm][k] for k in aggregate)
        for arm, rows in by_arm.items():
            if arm != 'all_qwen':
                assert m.paired(rows, by_arm['all_qwen']) == summary['comparisons'][arm + '_minus_all_qwen']
        # Repeat first/last after all arms to verify scene restoration.
        for scene in (0, 19):
            reference = by_arm['all_qwen'][scene]
            assert not m.replay.control_diff(reference['metrics'], m.score(env, reference['env_actions'], scene))
    # A paired 2x2 diagnostic separates g assignment from the three k roles as a block.
    values = {a: np.array([r['metrics']['V'] for r in rows]) for a, rows in by_arm.items()}
    g_effect = .5 * (values['gpt_g_qwen_k'] - values['all_qwen'] + values['all_gpt'] - values['qwen_g_gpt_k'])
    k_effect = .5 * (values['qwen_g_gpt_k'] - values['all_qwen'] + values['all_gpt'] - values['gpt_g_qwen_k'])
    effect = {'g_GPT_minus_Qwen_marginal_mean_V': float(g_effect.mean()),
              'k_block_GPT_minus_Qwen_marginal_mean_V': float(k_effect.mean())}
    for key, vector in (('g', g_effect), ('k_block', k_effect)):
        idx = np.random.default_rng(560120).integers(0, 20, size=(10000, 20))
        effect[key + '_CI95_exploratory'] = np.quantile(vector[idx].mean(axis=1), [.025, .975]).tolist()
    result = {'status': 'PASSED', 'API_calls': 0, 'additional_API_units': 0,
              'reconstructed_role_messages': 80, 'role_responses_checked': 160,
              'request_journal_events_checked': 320, 'postprocessor_checks': 80,
              'metric_replays': replayed, 'after_replay_controls': 2,
              'counts': counts, 'timing': timing, 'response_model_ids': model_ids,
              'factorial_effects_exploratory': effect, 'sources_and_inputs_unchanged': True}
    m.write_json(report / 'audit.json', result)
    print(json.dumps({'audit_directory': str(report), **result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True)
    args = ap.parse_args()
    audit((m.ROOT / args.folder).resolve())
