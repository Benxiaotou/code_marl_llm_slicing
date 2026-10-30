"""P1-A+ semantic-only local policy on the UNMODIFIED original GEANT environment.

Requires existing p1a_runner.py and p1a/ package. It does not alter them.
"""
import csv
import datetime
import json
import os
from pathlib import Path

from p1a.policy import ChatCompletionsClient, MockClient, PolicyFailure
from p1aplus.policy import SemanticDecentralizedPolicy
from p1aplus.preflight import verify_main_rules

import p1a_runner as base


LABEL = 'qwen_semantic_local_v1'


def get_args(argv=None):
    return base.get_args(argv)


def run(args, *, env_override=None, mapping_override=None, client_override=None):
    root = Path.cwd()
    if env_override is None:
        verify_main_rules(root / 'main.py')
    out = Path(args.output_dir).resolve() if args.output_dir else (
        root / 'p1aplus_runs' / ('{}_{}_'.format(args.policy, args.dataset)
                                 + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite existing run dir: {}'.format(out))
    out.mkdir(parents=True, exist_ok=True)
    env, mapping = ((env_override, mapping_override) if env_override is not None
                    else base.create_env(args, root))
    scenes = base.select_indices(mapping, args)
    client = client_override or (MockClient() if args.policy == 'mock' else
       ChatCompletionsClient(base_url=args.base_url, model=args.model,
                             max_tokens=args.max_tokens, temperature=args.temperature,
                             read_timeout=args.api_timeout, stream=not args.no_stream))
    policy = SemanticDecentralizedPolicy(client)
    label = 'semantic_mock_not_an_llm_result' if args.policy == 'mock' else LABEL
    manifest = {
        'policy': label,
        'model': getattr(client, 'model', args.model),
        'dataset': args.dataset,
        'scene_ids': [x[1] for x in scenes],
        'seed': args.seed, 'config': args.config,
        'max_tokens': args.max_tokens, 'temperature': args.temperature,
        'stream': not args.no_stream, 'on_error': args.on_error,
        'information_boundary': 'each call receives only its original per-agent observation; semantic derived features computed from that observation',
        'guard_used': False, 'cross_agent_communication': False,
        'task_finetuning': False, 'icl_examples': 0, 'fallback_used': False,
        'main_py_sha256': base._hash_file(root / 'main.py'),
        'p1a_policy_sha256': base._hash_file(root / 'p1a' / 'policy.py'),
        'p1a_runner_sha256': base._hash_file(root / 'p1a_runner.py'),
        'semantic_module_sha256': base._hash_file(root / 'p1aplus' / 'semantics.py'),
        'prompt_treatment': 'original raw local observation + deterministic local action semantics',
        'g_legacy_check_warning': 'Original g compute/cache physical admission uses pre-compression demand; NOT silently repaired.',
        'notes': 'P1-A+ development run only. Train/test partitions must be tracked independently.',
    }
    base._json_write(out / 'manifest.json', manifest)
    records = []
    try:
        with (out / 'records.jsonl').open('w', encoding='utf-8') as f:
            for internal, original in scenes:
                try:
                    record = base.run_one_scene(env, policy, internal, original, args.seed)
                except PolicyFailure as exc:
                    record = {
                        'status': 'policy_error', 'original_scene_id': original,
                        'internal_scene_index': internal, 'failed_role': exc.role,
                        'error': str(exc), 'partial_decisions': exc.partial,
                        'fallback_used': False,
                    }
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                f.flush()
                if record['status'] == 'ok':
                    m = record['metrics']
                    print('scene={} admission={} IR+admit={} V={:.3f} policy_time={:.2f}s'.format(
                        original, int(m['g_done']), int(m['cooperative_valid']),
                        m['V'], record['policy_latency_s']), flush=True)
                else:
                    print('scene={} FAILED role={} (no env.step)'.format(
                        original, record['failed_role']), flush=True)
                    if args.on_error == 'abort':
                        raise RuntimeError(record['error'])
    finally:
        summary = base.summarize(records, label)
        base._json_write(out / 'summary.json', summary)
        fields = ['original_scene_id', 'status', 'g_done', 'cooperative_valid',
                  'individual_rationality', 'V', 'V_g', 'V_k1', 'V_k2',
                  'V_k3', 'cost_sum', 'reward_mean', 'policy_latency_s',
                  'env_step_s', 'failed_role']
        with (out / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in records:
                row = {'original_scene_id': rec['original_scene_id'],
                       'status': rec['status'], 'failed_role': rec.get('failed_role', '')}
                if rec['status'] == 'ok':
                    row.update({key: rec['metrics'].get(key) for key in
                                ('g_done', 'cooperative_valid', 'individual_rationality',
                                 'V', 'V_g', 'V_k1', 'V_k2', 'V_k3',
                                 'cost_sum', 'reward_mean')})
                    row['policy_latency_s'] = rec['policy_latency_s']
                    row['env_step_s'] = rec['env_step_s']
                writer.writerow(row)
        print('Output directory: {}'.format(out), flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    run(get_args())
