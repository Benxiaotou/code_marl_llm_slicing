"""Fixed frozen-candidate validation on two NEW seeds, using internal Qwen only."""
from __future__ import annotations
import concurrent.futures as futures
import contextlib
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'p2f_analysis', ROOT / 'p2g_autonomous/jobs',
             ROOT / 'p2h_v_decomposition', ROOT / 'p2i_node_boundary'):
    sys.path.insert(0, str(path))
import val5601_counterfactual_replay as replay
from bounded_release import adjust_local

SEEDS = (2026110501, 2026110502)
COUNT = 100
MODEL = 'qwen3-next-80b-a3b-instruct'
WORKERS = 8
ROLES = replay.ROLES
K = ROLES[:3]
ARMS = ('MAPPO', 'Qwen_original', 'Frozen_link', 'Frozen_link_node')
SELECTED = ROOT / 'p2i_node_boundary/run_20261105T012338_935948Z/candidate.json'
LIBRARY = ROOT / 'p2b_libraries/p2b_standard_1000_v2'
IR_LIBRARY = ROOT / 'p2b_ir_libraries/p2b_standard_1000_ir_v2'
CHECKPOINT = ROOT / 'marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth'


def write_json(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def journal(f, data):
    f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + '\n')
    f.flush(); os.fsync(f.fileno())


def message_hash(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def derived_actions(original, observations, targets):
    """Own TRAIN-local link targets, then frozen own-local node postprocessor."""
    link = {r: list(original[r]) for r in ROLES}
    node = {r: list(original[r]) for r in ROLES}
    audits = {}
    for role in K:
        action = np.asarray(original[role], dtype=float).copy()
        action[2:] += .5 * (2 * np.asarray(targets[role], dtype=float) - 1 - action[2:])
        link[role] = action.tolist()
        node[role], audits[role] = adjust_local(observations[role], link[role], .025, .01)
        assert link[role][:2] == original[role][:2]
        assert node[role][2:] == link[role][2:]
    assert node['agent_g'] == link['agent_g'] == original['agent_g']
    return link, node, audits


def bootstrap_indices(seed_labels, reps=10000):
    labels = np.asarray(seed_labels)
    rng = np.random.default_rng(20261105)
    groups = [np.flatnonzero(labels == s) for s in sorted(set(seed_labels))]
    # Keep each seed's original weight instead of pooling repeated local IDs.
    return np.concatenate([rng.choice(g, size=(reps, len(g)), replace=True) for g in groups], axis=1)


def paired(a, b):
    keys = lambda rows: [(r['dataset_seed'], r['original_scene_id']) for r in rows]
    assert keys(a) == keys(b) and len(set(keys(a))) == len(a)
    idx = bootstrap_indices([r['dataset_seed'] for r in a])
    out = {}
    for key in ('V', 'cost_sum', 'V_g', 'g_done', 'cooperative_valid'):
        x = np.array([float(r['metrics'][key]) for r in a])
        y = np.array([float(r['metrics'][key]) for r in b])
        delta = x - y
        out[key] = {'mean_delta': float(delta.mean()),
                    'paired_seed_stratified_CI95': np.quantile(delta[idx].mean(axis=1), [.025, .975]).tolist()}
        if key in ('g_done', 'cooperative_valid'):
            out[key]['wins'] = int(((x == 1) & (y == 0)).sum())
            out[key]['losses'] = int(((x == 0) & (y == 1)).sum())
    return out


def summarize(rows):
    out = replay.aggregate(rows)
    out['admitted'] = sum(bool(r['metrics']['g_done']) for r in rows)
    out['cooperative'] = sum(bool(r['metrics']['cooperative_valid']) for r in rows)
    return out


def validate_metrics(m):
    assert math.isclose(m['V'], sum(m['revenue_' + k] for k in ('k1', 'k2', 'k3', 'g')) - m['cost_sum'], abs_tol=1e-7)
    ir = m['V_g'] >= -1e-8 and all(m['V_' + k] >= m['V_' + k + '_0'] - 1e-8 for k in ('k1', 'k2', 'k3'))
    assert m['individual_rationality'] == ir and m['cooperative_valid'] == (m['g_done'] and ir)


def config_for(runtime, sample=None, count=None):
    import p1a_runner as base
    main, conf = base._config(ROOT, ROOT / 'configs_main.yaml')
    base._compat_pickle_main_classes(main)
    for name in ('path_reward_save', 'path_env_save', 'path_model_save', 'path_model_train_log',
                 'path_F_save', 'log_dir', 'model_dir'):
        path = runtime / name; path.mkdir(parents=True, exist_ok=True)
        setattr(conf, name, str(path))
    conf.render = False
    if sample is not None:
        conf.p0_sample_file = str(sample); conf.p0_sample_cnt = count
    return main, conf


def make_env(manifest, runtime):
    import p1a_runner as base
    main, conf = config_for(runtime)
    cls = base.make_mini_env_type(main.CustomMultiAgentEnv, Path(manifest['sample_file']), manifest['n_scenes'])
    env = cls(conf); env.is_eval = True
    assert sorted(env.sample_dict) == list(range(manifest['n_scenes']))
    return env


def score(env, actions, scene):
    import p1a_runner as base
    base.load_scene_local_observations(env, scene)
    random.seed(1 + scene); np.random.seed(1 + scene)
    before = len(env.g_done_list)
    _, rewards, _, _, _ = env.step({r: np.asarray(v, dtype=np.float32) for r, v in actions.items()})
    m = base._get_step_metrics(env, before, rewards)
    validate_metrics(m)
    return m


def historical_preflight(policy, teacher, out, log):
    """Three old development controls, not test data or selection examples."""
    import p1a_runner as base
    from p2e_prompt.val5601_eval import verify_environment
    manifest_path = ROOT / 'p1b_data/p2b_val_seed5601.json'
    assert manifest_path.is_file() and manifest_path.stat().st_size < 10000
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    sample = Path(manifest['sample_file'])
    assert sample.is_file() and sample.stat().st_size == 644537810
    verify_environment(ROOT, manifest)
    selected = (0, 39, 199)
    cached = {}
    for label, path in (('q', ROOT / 'p2b_runs/val5601_standard1000_p2a200/records.jsonl'),
                        ('m', ROOT / 'p2b_runs/val5601_mappo200/records.jsonl')):
        assert path.is_file() and path.stat().st_size < 50_000_000
        cached[label] = {}
        with path.open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r['original_scene_id'] in selected:
                    cached[label][r['original_scene_id']] = r
    with contextlib.redirect_stdout(log):
        env = make_env(manifest, out / 'historical_control_runtime')
        for scene in selected:
            obs = base.load_scene_local_observations(env, scene)
            for role in ROLES:
                matches = policy.library.query(role, obs[role], top_k=3)
                assert policy._make_messages(role, obs[role], matches) == cached['q'][scene]['per_role'][role]['messages']
            actions = teacher.act(obs, original_scene_id=scene, seed=1)
            for role in ROLES:
                assert np.allclose(actions[role], cached['m'][scene]['env_actions'][role], atol=2e-6)
            assert not replay.control_diff(cached['m'][scene]['metrics'], score(env, actions, scene))
            assert not replay.control_diff(cached['q'][scene]['metrics'], score(env, cached['q'][scene]['env_actions'], scene))
        del env
    gc.collect()
    write_json(out / 'historical_preflight.json', {'passed': True, 'reconstructed_prompts': 12,
               'frozen_MAPPO_action_controls': 3, 'historical_metric_controls': 6, 'API_calls': 0})


def infer_all(prepared, out):
    from p1a.policy import ChatCompletionsClient, parse_ratios, to_env_action
    stop = threading.Event(); lock = threading.Lock(); local = threading.local()
    completed = {}
    with (out / 'requests.jsonl').open('x', encoding='utf-8') as ledger, \
            (out / 'responses.jsonl').open('x', encoding='utf-8') as responses:
        def call(item, role):
            if stop.is_set():
                raise RuntimeError('CANCELLED_BEFORE_REQUEST')
            rid = str(item['dataset_seed']) + ':' + str(item['original_scene_id']) + ':' + role
            messages = item['per_role'][role]['messages']
            with lock:
                journal(ledger, {'event': 'started', 'request_id': rid, 'model': MODEL,
                                 'prompt_sha256': message_hash(messages)})
            if not hasattr(local, 'client'):
                local.client = ChatCompletionsClient(base_url=os.environ['LLM_BASE_URL'],
                    api_key=os.environ['LLM_API_KEY'], model=MODEL, temperature=.1,
                    max_tokens=4096, read_timeout=120, stream=True)
            try:
                response = local.client.complete(role, messages)
                if response.finish_reason != 'stop':
                    raise ValueError('Incomplete model response')
                ratios = parse_ratios(response.text, role)
                entry = {'request_id': rid, 'model': MODEL, 'env_action': to_env_action(role, ratios).tolist(),
                         'ratios': ratios, 'raw_model_output': response.text, 'usage': response.usage,
                         'latency_s': response.latency_s, 'finish_reason': response.finish_reason,
                         'prompt_sha256': message_hash(messages)}
                with lock:
                    journal(responses, entry)
                    journal(ledger, {'event': 'finished', 'request_id': rid, 'usage': response.usage,
                                     'latency_s': response.latency_s})
                return entry
            except Exception as exc:
                stop.set()
                with lock:
                    journal(ledger, {'event': 'failed', 'request_id': rid,
                                     'exception_type': type(exc).__name__, 'no_retry': True})
                raise RuntimeError('QWEN_REQUEST_FAILED ' + rid) from None
        with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            pending = [pool.submit(call, item, role) for item in prepared for role in ROLES]
            try:
                for future in futures.as_completed(pending):
                    entry = future.result()
                    assert entry['request_id'] not in completed
                    completed[entry['request_id']] = entry
            except Exception:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
    assert len(completed) == 800
    return completed


def run():
    from trace_service import windows_ram_gib
    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    from p2a.policy import IRRiskRerankPolicy
    from p2a.rerank import RerankConfig
    from teacher_link_multicohort import teacher_targets
    from p1b.teacher import P0FrozenTeacher
    from p1b_generate import generate, get_args
    import p1a_runner as base

    if (not os.environ.get('LLM_BASE_URL') or not os.environ.get('LLM_API_KEY')
            or os.environ.get('P1A_MODEL') != MODEL):
        raise RuntimeError('Configured internal Qwen environment required; no calls made')
    ram = windows_ram_gib()
    assert ram >= 8 and shutil.disk_usage(ROOT).free >= 3 * 1024**3
    assert SELECTED.is_file() and SELECTED.stat().st_size < 10000
    selected = json.loads(SELECTED.read_text(encoding='utf-8'))
    assert selected['method'] == 'buffer01_d025' and selected['max_ratio_increment'] == .025
    assert selected['extra_projected_overload_budget'] == .01
    assert replay.sha256(ROOT / 'p2i_node_boundary/bounded_release.py') == selected['policy_source_sha256']
    assert CHECKPOINT.is_file() and CHECKPOINT.stat().st_size < 10_000_000
    assert replay.sha256(CHECKPOINT) == 'fa36695c78feb28b3506fbd2858e6efbc2c8f44e1f98fd7fe0282acf191ebf03'
    # Exact public TRAIN paths only; never private_offline or test records.
    public = [LIBRARY / 'library_manifest.json', LIBRARY / 'audit_report.json',
              IR_LIBRARY / 'ir_manifest.json', *[LIBRARY / 'role_libraries' / (r + '.jsonl') for r in ROLES],
              *[IR_LIBRARY / 'role_ir_history' / (r + '.jsonl') for r in ROLES]]
    for path in public:
        assert path.is_file() and path.stat().st_size < 50_000_000
    library = LocalHistory(LIBRARY); ir = IRHistory(IR_LIBRARY, library)
    reference = json.loads((ROOT / 'p2b_runs/val5601_standard1000_p2a200/manifest.json').read_text(encoding='utf-8'))
    assert library.manifest_sha256 == reference['library_manifest_sha256']
    assert library.library_hashes == reference['library_role_sha256']
    assert ir.manifest_sha256 == reference['ir_history_manifest_sha256']
    assert ir.role_hashes == reference['ir_history_role_sha256']
    assert RerankConfig().as_dict() == reference['p2a_rerank_config']
    policy = IRRiskRerankPolicy(library, ir, client=object(), examples=3, seed=1)
    out = ROOT / 'p2j_validation' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    sources = [ROOT / p for p in ('main.py', 'configs_main.yaml', 'p1a_runner.py', 'p0_runner.py',
        'p1b_generate.py', 'p1a/policy.py', 'p1a/local_obs.py', 'p1b/teacher.py', 'p1c/retrieval.py',
        'p1c/policy.py', 'p1d/ir.py', 'p1d/policy.py', 'p2a/policy.py', 'p2a/rerank.py',
        'p2g_autonomous/jobs/teacher_link_multicohort.py', 'p2i_node_boundary/bounded_release.py',
        'p2j_validation/run_validation.py', 'p2j_validation/run_validation.ps1')]
    sources += [SELECTED, CHECKPOINT, *public]
    hashes = {str(p.relative_to(ROOT)): replay.sha256(p) for p in sources}
    protocol = {'schema': 'frozen_node_NEW_validation_v1', 'new_generation_seeds': SEEDS,
                'scenes_per_seed': COUNT, 'n_paired': 200, 'split': 'validation', 'arms': ARMS,
                'frozen_candidate': selected, 'model': MODEL, 'temperature': .1, 'max_tokens': 4096,
                'stream': True, 'API_timeout_s': 120, 'max_concurrent_requests': WORKERS,
                'expected_Qwen_calls': 800, 'automatic_retries': 0, 'fallback': False,
                'Qwen_draws_per_scene': 1, 'same_Qwen_actions_shared_across_postprocessors': True,
                'primary_endpoint': 'Frozen_link_node mean V minus MAPPO on all200 fixed new scenes',
                'secondary_comparisons': ['Frozen_link_node minus Frozen_link', 'Frozen_link_node minus Qwen_original'],
                'secondary_endpoints': ['g_done', 'cooperative_valid', 'cost_sum', 'V_g'],
                'bootstrap': '10000 paired samples stratified by generation seed; RNG20261105',
                'optional_stopping': False, 'parameter_changes_during_run': False,
                'seed_environment_policy': 1, 'available_RAM_GiB': ram,
                'memory_plan': 'Generate100 scenes at a time; one evaluation env plus one frozen actor constructor env; require8GiB available.',
                'disk_plan': 'New datasets about650MiB total; structured outputs below100MiB; require3GiB free.',
                'source_sha256': hashes, 'API_cost_usd_estimate': 0,
                'cost_basis': 'User-provided unlimited internal Qwen quota; not independently verified.',
                'information_boundary': 'Own local observation and public TRAIN role history only. MAPPO actions and outcomes never enter Qwen or postprocessing.',
                'limitations': ['New validation, not independent final test.', 'One Qwen draw per scene.',
                                'Generation seeds add independent draws under same original generator, not new traffic topology distribution.']}
    write_json(out / 'protocol.json', protocol)
    print('VALIDATION_OUTPUT', out, flush=True)
    print('PREFLIGHT', json.dumps({'RAM_GiB': ram, 'seeds': SEEDS, 'scenes': 200,
                                 'expected_API_calls': 800, 'estimated_extra_fee_USD': 0}), flush=True)
    phase = 'generation'; teacher = None
    try:
        manifests = []
        with (out / 'env_stdout.log').open('x', encoding='utf-8') as log:
            for seed in SEEDS:
                dataset = 'p2j_val_seed' + str(seed)
                sample = out / 'datasets' / (dataset + '.pkl')
                manifest_dir = out / 'datasets'
                assert not sample.exists() and not (manifest_dir / (dataset + '.json')).exists()
                with contextlib.redirect_stdout(log):
                    main, conf = config_for(out / ('generator_runtime_' + str(seed)))
                    cls = base.make_mini_env_type(main.CustomMultiAgentEnv, sample, COUNT)
                    args = get_args(['--name', dataset, '--count', str(COUNT), '--seed', str(seed),
                        '--split', 'validation', '--sample-file', str(sample), '--manifest-dir', str(manifest_dir)])
                    # Generator override accepts a callable and passes None as config.
                    # Capture the original environment's operational output config.
                    manifest = generate(args, root=ROOT, env_class_override=lambda _: cls(conf))
                assert manifest['main_py_sha256'] == hashes['main.py'] and manifest['config_sha256'] == hashes['configs_main.yaml']
                library.assert_independent(manifest); manifests.append(manifest)
                gc.collect()
                print('DATASET_GENERATED', dataset, COUNT, 'bytes', sample.stat().st_size, flush=True)
            write_json(out / 'datasets.json', manifests)
            with contextlib.redirect_stdout(log):
                _, conf = config_for(out / 'teacher_runtime', Path(manifests[0]['sample_file']), COUNT)
                conf.render = False; conf.test_mode = True; conf.device = 'cpu'; conf.parallels = 1
                teacher = P0FrozenTeacher(conf, CHECKPOINT)
            phase = 'historical_preflight'
            historical_preflight(policy, teacher, out, log)
            print('HISTORICAL_PREFLIGHT_PASSED', '12 prompts,3 MAPPO action controls,6 metric controls', flush=True)
            phase = 'prepare_new_observations_prompts_and_MAPPO'
            prepared = []
            with (out / 'prepared.jsonl').open('x', encoding='utf-8') as sink:
                for manifest in manifests:
                    seed = manifest['seed']
                    with contextlib.redirect_stdout(log):
                        env = make_env(manifest, out / ('prepare_runtime_' + str(seed)))
                        for scene in range(COUNT):
                            obs = base.load_scene_local_observations(env, scene)
                            role_inputs = {}; targets = {}; targets_info = {}
                            for role in ROLES:
                                matches = policy.library.query(role, obs[role], top_k=3)
                                messages = policy._make_messages(role, obs[role], matches)
                                role_inputs[role] = {'messages': messages, 'prompt_sha256': message_hash(messages),
                                    'retrieved_ids': [x['experience_id'] for x in matches],
                                    'rerank_trace': policy.adapter.last_trace[role]}
                                if role in K:
                                    t, info = teacher_targets(library, role, obs[role])
                                    targets[role] = t['link_mean3'].tolist(); targets_info[role] = info['link_mean3_ids']
                            ma = teacher.act(obs, original_scene_id=scene, seed=1)
                            ma = {r: list(map(float, ma[r])) for r in ROLES}
                            item = {'dataset_seed': seed, 'original_scene_id': scene,
                                    'observations': {r: obs[r].tolist() for r in ROLES}, 'per_role': role_inputs,
                                    'link_targets': targets, 'link_target_TRAIN_ids': targets_info,
                                    'MAPPO_actions': ma, 'MAPPO_metrics': score(env, ma, scene)}
                            prepared.append(item); journal(sink, item)
                        del env
                    gc.collect()
            # Only close the constructor's vector env; never call training/finish/save.
            teacher.close(); teacher = None
            write_json(out / 'preflight.json', {'passed': True, 'new_scenes': 200,
                       'prepared_prompts': 800, 'new_MAPPO_scores': 200, 'API_calls': 0})
        print('NEW_PREFLIGHT_PASSED', '200 scenes,800 prompts,200 frozen MAPPO evaluations', flush=True)
        phase = 'Qwen_inference'
        responses = infer_all(prepared, out)
        print('QWEN_INFERENCE_COMPLETE', len(responses), flush=True)
        phase = 'paired_evaluation'
        arms = {name: [] for name in ARMS}; post_controls = 0
        with (out / 'evaluation_stdout.log').open('x', encoding='utf-8') as log, \
                (out / 'records.jsonl').open('x', encoding='utf-8') as sink:
            for manifest in manifests:
                seed = manifest['seed']
                with contextlib.redirect_stdout(log):
                    env = make_env(manifest, out / ('evaluation_runtime_' + str(seed)))
                    for item in (r for r in prepared if r['dataset_seed'] == seed):
                        scene = item['original_scene_id']
                        qa = {role: responses[str(seed) + ':' + str(scene) + ':' + role]['env_action'] for role in ROLES}
                        link, node, audits = derived_actions(qa, item['observations'], item['link_targets'])
                        for label, action in zip(ARMS, (item['MAPPO_actions'], qa, link, node)):
                            metrics = score(env, action, scene)
                            if label == 'MAPPO':
                                assert not replay.control_diff(item['MAPPO_metrics'], metrics)
                            row = {'dataset_seed': seed, 'original_scene_id': scene, 'arm': label,
                                   'metrics': metrics, 'env_actions': action}
                            if label == 'Frozen_link_node':
                                row['node_adjustment'] = audits
                            arms[label].append(row); journal(sink, row)
                    for label in ('MAPPO', 'Qwen_original'):
                        first = next(r for r in arms[label] if r['dataset_seed'] == seed)
                        assert not replay.control_diff(first['metrics'], score(env, first['env_actions'], 0))
                        post_controls += 1
                    del env
                gc.collect()
                print('SEED_EVALUATED', seed, '100 paired scenes,4 arms', flush=True)
        phase = 'audit_and_report'
        for relative, h in hashes.items():
            assert replay.sha256(ROOT / relative) == h, relative
        for manifest in manifests:
            assert replay.sha256(Path(manifest['sample_file'])) == manifest['sample_sha256']
        summaries = {name: summarize(rows) for name, rows in arms.items()}
        per_seed = {str(seed): {name: summarize([r for r in rows if r['dataset_seed'] == seed])
                              for name, rows in arms.items()} for seed in SEEDS}
        comparisons = {label: paired(arms['Frozen_link_node'], arms[base_label]) for label, base_label in
                       (('node_minus_MAPPO', 'MAPPO'), ('node_minus_link', 'Frozen_link'), ('node_minus_Qwen', 'Qwen_original'))}
        per_seed_comparisons = {str(seed): {label: paired([r for r in arms['Frozen_link_node'] if r['dataset_seed'] == seed],
            [r for r in arms[base_label] if r['dataset_seed'] == seed]) for label, base_label in
            (('node_minus_MAPPO', 'MAPPO'), ('node_minus_link', 'Frozen_link'))} for seed in SEEDS}
        usages = [r['usage'] for r in responses.values()]
        token_usage = {'requests_without_usage': sum(not u for u in usages),
                       'reported_token_totals': {key: sum(int(u.get(key, 0)) for u in usages if u)
                         for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}
        result = {'status': 'COMPLETE', 'n_paired_scenes': 200, 'generation_seeds': SEEDS,
                  'summaries': summaries, 'per_seed': per_seed, 'comparisons': comparisons,
                  'per_seed_comparisons': per_seed_comparisons, 'API_calls': len(responses),
                  'API_cost_usd_estimate': 0, 'token_usage': token_usage,
                  'source_and_dataset_hashes_unchanged': True, 'MAPPO_controls_reproduced': 200,
                  'post_controls': post_controls, 'independent_final_test': False,
                  'parameters_changed': False, 'fallback_or_retry': False}
        write_json(out / 'summary.json', result)
        # Save completion only after reconstructing all fresh decisions and journal counts.
        from audit_validation import audit
        audit(out)
        lines = ['# 新场景固定方案验证', '', '两个新种子各100场景，共200场景，参数及提示词固定。',
                 '每场一组真实Qwen动作，原Qwen、链路候选、节点候选共享该组动作。MAPPO使用原冻结检查点。', '',
                 '| 方法 | 平均V | 接纳数 | 合作有效数 | 平均成本 |', '|---|---:|---:|---:|---:|']
        for name, s in summaries.items():
            lines.append(f"| {name} | {s['V_mean']:.4f} | {s['admitted']} | {s['cooperative']} | {s['cost_mean']:.4f} |")
        lines += ['', '| 配对比较（节点方案减参照） | 平均ΔV | 按种子分层配对95%区间 |', '|---|---:|---|']
        for name, comp in comparisons.items():
            ci = comp['V']['paired_seed_stratified_CI95']
            lines.append(f"| {name} | {comp['V']['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        lines += ['', '## 分种子结果', '', '| 种子 | 方法 | V | 接纳 | 合作有效 |', '|---|---|---:|---:|---:|']
        for seed, values in per_seed.items():
            for name, s in values.items():
                lines.append(f"| {seed} | {name} | {s['V_mean']:.4f} | {s['admitted']} | {s['cooperative']} |")
        lines += ['', '主要终点预先指定为节点方案相对MAPPO的平均V；其余比较和指标为次要分析。',
                  '这是新验证集、同一原生成分布、每场一次Qwen抽样；没有使用以前的独立测试，也未训练或更改参数。',
                  'API调用800次，模型qwen3-next-80b-a3b-instruct；按用户内网无限额度配置新增费用预计$0。',
                  'usage缺失请求数：' + str(token_usage['requests_without_usage']) + '；缺失token不能按0宣称。',
                  '6次历史指标、3次MAPPO动作、12条历史提示词、200次新MAPPO复现与4次后控制通过；完整保存结果审计通过。', '',
                  '复现命令（生成全新场景文件和输出；会再次调用Qwen800次）：', '',
                  '```powershell', '& .\\p2j_validation\\run_validation.ps1', '```', '']
        with (out / 'RESULTS.md').open('x', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print('VALIDATION_COMPLETE', out, flush=True)
        print(json.dumps({'summaries': summaries, 'primary': comparisons['node_minus_MAPPO']['V'],
                          'node_minus_link': comparisons['node_minus_link']['V'], 'API_calls': 800}), flush=True)
    except Exception as exc:
        write_json(out / 'STOPPED.json', {'phase': phase, 'error_type': type(exc).__name__,
                   'message': str(exc) if phase != 'Qwen_inference' else 'See request journal; no automatic retries.',
                   'outputs_preserved': True})
        raise
    finally:
        if teacher is not None:
            teacher.close()


if __name__ == '__main__':
    run()
