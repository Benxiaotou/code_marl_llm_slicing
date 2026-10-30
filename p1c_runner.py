"""P1-C role-local kNN / frozen-Qwen ICL / optional frozen MAPPO evaluation.

Real GEANT and private Qwen endpoint must run in the user's local environment.
This runner never trains, edits main.py, invokes Guard or accesses global state
for action selection. Only original environment.step evaluates the joint action.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path

from p1a.local_obs import AGENTS
from p1a.policy import ChatCompletionsClient, MockClient, PolicyFailure
from p1b.core import ContractError, file_sha256, read_json, save_json
from p1c.policy import LABELS, RetrievalPolicy, FrozenMAPPOEvalPolicy, ZeroShotEvalPolicy
from p1c.retrieval import LocalHistory
from p1d.ir import IRHistory
from p1d.policy import IRRetrievalPolicy, IRAblationPolicy
from p2a.policy import IRRiskRerankPolicy
from p2a.rerank import RerankConfig

import p1a_runner as base


def get_args(argv=None):
    p = argparse.ArgumentParser(description='P1-C independent role-local experience evaluation')
    p.add_argument('--library', required=True, help='P1-B audited selected library directory')
    p.add_argument('--dataset-manifest', required=True,
                   help='NEW P1-B validation/test manifest (never training data)')
    p.add_argument('--method', choices=['knn', 'icl', 'ir_icl', 'ir_prompt_only', 'ir_data_only', 'ir_rerank', 'random_icl', 'mappo', 'zero'], required=True)
    p.add_argument('--examples', type=int, default=3, help='ICL examples per role; kNN always uses one')
    p.add_argument('--ir-history', default='',
                   help='Training role-only IR history overlay (required for ir_icl and ir_data_only)')
    p.add_argument('--risk-pool', type=int, default=20, help='P2-A candidate pool; default 20')
    p.add_argument('--risk-weight', type=float, default=.35, help='P2-A historical risk weight [0,1]')
    p.add_argument('--risk-max-gap', type=float, default=.10, help='P2-A maximum local distance increase')
    p.add_argument('--risk-max-ratio', type=float, default=1.35, help='P2-A max distance / closest distance')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--config', default='configs_main.yaml')
    p.add_argument('--model', default=os.getenv('P1A_MODEL', 'qwen3-next-80b-a3b-instruct'))
    p.add_argument('--base-url', default=os.getenv('LLM_BASE_URL', 'http://192.168.131.35:30001/v1'))
    p.add_argument('--temperature', type=float, default=.1)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--api-timeout', type=float, default=120)
    p.add_argument('--no-stream', action='store_true')
    p.add_argument('--checkpoint', default='marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth')
    p.add_argument('--on-error', choices=['abort', 'record'], default='abort')
    p.add_argument('--mock-llm', action='store_true', help='Pipeline smoke test ONLY, not scored LLM')
    p.add_argument('--output-dir', required=True)
    return p.parse_args(argv)


def checked_eval_manifest(root, manifest_path, library, config, *, check_files=True):
    manifest = read_json(root / manifest_path)
    library.assert_independent(manifest)
    if not isinstance(manifest.get('n_scenes'), int) or manifest['n_scenes'] < 1:
        raise ContractError('Invalid validation/test scene count')
    # The two independent seeds are a research requirement, not a claim that
    # distinct file hashes alone establish statistical independence.
    if check_files:
        sample = Path(manifest['sample_file']).resolve()
        if not sample.is_file() or file_sha256(sample) != manifest['sample_sha256']:
            raise ContractError('Validation/test sample missing or SHA-256 mismatch')
        for filename, field in (('main.py', 'main_py_sha256'),
                                (config, 'config_sha256')):
            expected = manifest.get(field)
            if expected and file_sha256(root / filename) != expected:
                raise ContractError('Original environment/config changed since dataset generation: ' + filename)
    return manifest


def create_eval_env(root, manifest, config):
    from p1a_runner import _config, _compat_pickle_main_classes, make_mini_env_type
    main, conf = _config(root, root / config)
    _compat_pickle_main_classes(main)
    cls = make_mini_env_type(main.CustomMultiAgentEnv,
                             Path(manifest['sample_file']).resolve(), manifest['n_scenes'])
    env = cls(conf)
    env.is_eval = True
    if (env.sample_cnt != manifest['n_scenes'] or len(env.sample_dict) != manifest['n_scenes']
            or set(env.sample_dict) != set(range(manifest['n_scenes']))):
        raise ContractError('Custom evaluation dataset size or keys inconsistent')
    return env


def build_policy(args, library, root, *, client_override=None, teacher_override=None,
                 ir_history=None):
    if args.method == 'mappo':
        if teacher_override is None:
            from p1b_collect import _prepare_teacher_conf
            from p1b.teacher import P0FrozenTeacher
            teacher = P0FrozenTeacher(_prepare_teacher_conf(root, args.config),
                                      root / args.checkpoint)
        else:
            teacher = teacher_override
        return FrozenMAPPOEvalPolicy(teacher, seed=args.seed), None, teacher
    if args.method == 'knn':
        if args.mock_llm:
            raise ValueError('Mock LLM option is only for ICL')
        return RetrievalPolicy(library, 'knn', examples=1, seed=args.seed), None, None
    client = client_override or (MockClient() if args.mock_llm else
        ChatCompletionsClient(base_url=args.base_url, model=args.model,
                              temperature=args.temperature, max_tokens=args.max_tokens,
                              read_timeout=args.api_timeout, stream=not args.no_stream))
    if args.method == 'zero':
        return ZeroShotEvalPolicy(client), client, None
    if args.method == 'ir_icl':
        if ir_history is None:
            raise ContractError('IR-aware policy requires audited training IR overlay')
        return IRRetrievalPolicy(library, ir_history, client=client,
                                 examples=args.examples, seed=args.seed), client, None
    if args.method == 'ir_rerank':
        if ir_history is None:
            raise ContractError('Risk-aware reranking requires training IR overlay')
        cfg = RerankConfig(pool_size=args.risk_pool, risk_weight=args.risk_weight,
                           max_abs_distance_gap=args.risk_max_gap,
                           max_distance_ratio=args.risk_max_ratio)
        return IRRiskRerankPolicy(library, ir_history, client=client,
                                  examples=args.examples, seed=args.seed,
                                  config=cfg), client, None
    if args.method in ('ir_prompt_only', 'ir_data_only'):
        return IRAblationPolicy(library, args.method, client=client,
                                ir_history=ir_history, examples=args.examples,
                                seed=args.seed), client, None
    return RetrievalPolicy(library, args.method, client=client, examples=args.examples,
                           seed=args.seed), client, None


def run(args, *, root=None, env_override=None, eval_manifest_override=None,
        client_override=None, teacher_override=None):
    root = Path(root or Path.cwd()).resolve()
    if args.start < 0 or args.count < 1 or args.seed < 0:
        raise ValueError('start>=0, count>=1, seed>=0 required')
    if args.method == 'ir_rerank' and args.examples != 3:
        raise ValueError('P2-A v1 is locked to 3 examples for paired P1-D comparison')
    if args.mock_llm and args.method not in ('icl', 'ir_icl', 'ir_prompt_only', 'ir_data_only', 'ir_rerank', 'random_icl', 'zero'):
        raise ValueError('Mock LLM only applies to ICL methods')
    if bool(args.ir_history) != (args.method in ('ir_icl', 'ir_data_only', 'ir_rerank')):
        raise ValueError('--ir-history is required for ir_icl/ir_data_only/ir_rerank and forbidden for others')
    library = LocalHistory(root / args.library)
    ir_history = IRHistory(root / args.ir_history, library) if args.method in ('ir_icl', 'ir_data_only', 'ir_rerank') else None
    manifest = (eval_manifest_override if eval_manifest_override is not None else
                checked_eval_manifest(root, args.dataset_manifest, library, args.config))
    library.assert_independent(manifest)
    if args.start + args.count > manifest['n_scenes']:
        raise ValueError('Requested scenes exceed validation/test size')
    expected_scene_ids = list(range(args.start, args.start + args.count))
    env = (env_override if env_override is not None else
           create_eval_env(root, manifest, args.config))
    policy, client, teacher = build_policy(args, library, root,
                                           client_override=client_override,
                                           teacher_override=teacher_override,
                                           ir_history=ir_history)
    label = (('MOCK_NOT_REAL_' + args.method) if args.mock_llm else LABELS[args.method])
    output = (root / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite existing results: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    experiment_manifest = {
        'schema': 'p1c_run_v1', 'policy': label, 'method': args.method,
        'mock_not_real_llm': bool(args.mock_llm),
        'model': (getattr(client, 'model', args.model) if args.method in ('icl','ir_icl','ir_prompt_only','ir_data_only','ir_rerank','random_icl','zero')
                  else ('frozen_mappo' if args.method == 'mappo' else 'no_llm_knn')),
        'eval_dataset_id': manifest['dataset_id'], 'eval_dataset_split': manifest['split'],
        'eval_dataset_sha256': manifest['sample_sha256'],
        'eval_generation_seed': manifest.get('seed'), 'scene_ids': expected_scene_ids,
        'library_training_id': library.meta['dataset_id'],
        'library_train_sha256': library.meta['training_sample_sha256'],
        'library_manifest_sha256': library.manifest_sha256,
        'library_role_sha256': library.library_hashes,
        'examples': (args.examples if args.method in ('icl','ir_icl','ir_prompt_only','ir_data_only','ir_rerank','random_icl') else
                     (1 if args.method == 'knn' else 0)),
        'retrieval': ('train_fitted_local_numeric_group_features_v1_ir_rerank_p2a'
                      if args.method == 'ir_rerank' else
                      ('train_fitted_local_numeric_group_features_v1'
                       if args.method in ('knn', 'icl', 'ir_icl', 'ir_prompt_only', 'ir_data_only', 'random_icl')
                       else None)),
        'ir_history_manifest_sha256': (ir_history.manifest_sha256 if ir_history else None),
        'ir_history_role_sha256': (ir_history.role_hashes if ir_history else None),
        'ir_feedback_basis': ('historical_self_only_training_joint_outcomes' if ir_history else None),
        'ir_risk_prompt_present': args.method in ('ir_icl', 'ir_prompt_only', 'ir_rerank'),
        'ir_history_data_present': args.method in ('ir_icl', 'ir_data_only', 'ir_rerank'),
        'p2a_rerank_config': (policy.reranker.config.as_dict() if args.method == 'ir_rerank' else None),
        'p2a_nearest_anchor': bool(args.method == 'ir_rerank'),
        'ir_ablation_factorial': (args.method if args.method in ('icl', 'ir_icl',
                                                               'ir_prompt_only', 'ir_data_only') else None),
        'seed': args.seed, 'config': args.config,
        'config_sha256': file_sha256(root / args.config) if (root / args.config).exists() else None,
        'main_py_sha256': file_sha256(root / 'main.py') if (root / 'main.py').exists() else None,
        'checkpoint_sha256': (getattr(teacher, 'source_info', {}).get('checkpoint_sha256')
                              if teacher is not None else None),
        'temperature': args.temperature if client else None,
        'max_tokens': args.max_tokens if client else None,
        'stream': not args.no_stream if client else None,
        'on_error': args.on_error, 'guard_used': False,
        'task_finetuning': False, 'test_time_global_state_access': False,
        'cross_agent_communication': False, 'fallback_used': False,
        'information_boundary': ('each role uses own current observation, own training local examples, '
                                 'and own historical IR feedback only' if ir_history else
                                 'each role uses own current observation and own historical local examples only'),
        'limitations': ('Development / validation statistics are NOT final held-out tests. '
                        'Historical joint success is not local causal optimality. '
                        'Original g compression/admission semantics unchanged.'),
    }
    save_json(output / 'manifest.json', experiment_manifest)
    records = []
    try:
        with (output / 'records.jsonl').open('w', encoding='utf-8') as sink:
            for scene_id in expected_scene_ids:
                policy.set_scene(scene_id)
                try:
                    row = base.run_one_scene(env, policy, scene_id, scene_id, args.seed)
                except PolicyFailure as exc:
                    row = {'status': 'policy_error', 'original_scene_id': scene_id,
                           'internal_scene_index': scene_id, 'failed_role': exc.role,
                           'error': str(exc), 'partial_decisions': exc.partial,
                           'fallback_used': False}
                records.append(row)
                sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                sink.flush()
                if row['status'] == 'ok':
                    m = row['metrics']
                    print('method={} scene={} accept={} cooperative={} V={:.3f} cost={:.3f} decision_s={:.2f}'.format(
                        args.method, scene_id, int(m['g_done']), int(m['cooperative_valid']),
                        m['V'], m['cost_sum'], row['policy_latency_s']), flush=True)
                else:
                    print('method={} scene={} FAILED {}: {}'.format(
                        args.method, scene_id, row['failed_role'], row['error']), flush=True)
                    if args.on_error == 'abort':
                        raise RuntimeError('Stopped on failed scene; see records.jsonl')
    finally:
        if teacher is not None and hasattr(teacher, 'close'):
            teacher.close()
        summary = base.summarize(records, label)
        summary.update({'eval_dataset_id': manifest['dataset_id'],
                        'split': manifest['split'], 'method': args.method,
                        'mock_not_real_llm': bool(args.mock_llm)})
        save_json(output / 'summary.json', summary)
        fields = ['original_scene_id', 'status', 'g_done', 'cooperative_valid',
                  'individual_rationality', 'V', 'V_g', 'V_k1', 'V_k2', 'V_k3',
                  'cost_sum', 'policy_latency_s', 'env_step_s', 'failed_role']
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in records:
                rec = {'original_scene_id': row['original_scene_id'],
                       'status': row['status'], 'failed_role': row.get('failed_role', '')}
                if row['status'] == 'ok':
                    m = row['metrics']
                    rec.update({k: m[k] for k in fields if k in m})
                    rec['policy_latency_s'] = row['policy_latency_s']
                    rec['env_step_s'] = row['env_step_s']
                writer.writerow(rec)
        print('Results: ' + str(output), flush=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    run(get_args())
