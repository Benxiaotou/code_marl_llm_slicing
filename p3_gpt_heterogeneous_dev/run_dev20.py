"""Frozen P3 four-arm, 20-scene development comparison. No dataset generation.

One Qwen draw and one GPT draw per local role/scene are shared across arms.
Only exact validation records and public TRAIN role libraries are accessed.
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for directory in ('p2f_analysis', 'p2h_v_decomposition', 'p2g_autonomous/jobs'):
    sys.path.insert(0, str(ROOT / directory))

from p1a.policy import ChatCompletionsClient, parse_ratios, to_env_action
from p1a.local_obs import AGENTS
from p1c.retrieval import LocalHistory
from p1d.ir import IRHistory
from p2a.policy import IRRiskRerankPolicy
from p2i_node_boundary.bounded_release import adjust_local
from teacher_link_multicohort import teacher_targets
from trace_service import windows_ram_gib
import p2e_prompt.val5601_eval as compat
import p2e_prompt.gpt61_sol_dev as adapter
import p1a_runner as base
import val5601_counterfactual_replay as replay

MODEL_QWEN = 'qwen3-next-80b-a3b-instruct'
MODEL_GPT = 'gpt-6.1-sol'
DATASET = ROOT / 'p1b_data/p2b_val_seed5601.json'
LIBRARY = ROOT / 'p2b_libraries/p2b_standard_1000_v2'
IR_LIBRARY = ROOT / 'p2b_ir_libraries/p2b_standard_1000_ir_v2'
HISTORICAL = ROOT / 'p2b_runs/val5601_standard1000_p2a200'
MAPPO = ROOT / 'p2b_runs/val5601_mappo200'
SELECTED = ROOT / 'p2i_node_boundary/run_20261105T012338_935948Z/candidate.json'
SCENES = tuple(range(20))
ROLES = tuple(AGENTS)
K_ROLES = ROLES[:3]
TOKENS_PER_UNIT = 5_000_000
TOKEN_BUDGET = 1_000_000  # 0.2 user-defined monetary units
GPT_OUTPUT_LIMIT = 8192
WORKERS = 4
ASSIGNMENTS = {
    'all_qwen': {r: 'qwen' for r in ROLES},
    'all_gpt': {r: 'gpt' for r in ROLES},
    'gpt_g_qwen_k': {r: ('gpt' if r == 'agent_g' else 'qwen') for r in ROLES},
    'qwen_g_gpt_k': {r: ('qwen' if r == 'agent_g' else 'gpt') for r in ROLES},
}
SOURCES = (
    'main.py', 'configs_main.yaml', 'p1a_runner.py', 'p0_runner.py',
    'p1a/policy.py', 'p1a/local_obs.py', 'p1c/policy.py', 'p1c/retrieval.py',
    'p1d/ir.py', 'p1d/policy.py', 'p2a/policy.py', 'p2a/rerank.py',
    'p2i_node_boundary/bounded_release.py', 'p2j_validation/run_validation.py',
    'p2g_autonomous/jobs/teacher_link_multicohort.py',
    'p2e_prompt/gpt61_sol_dev.py', 'p2e_prompt/val5601_eval.py',
    'p3_gpt_heterogeneous_dev/run_dev20.py',
    'p3_gpt_heterogeneous_dev/run.ps1',
)


def sha(path):
    return replay.sha256(path)


def read_json(path, cap=1_000_000):
    path = Path(path)
    if not path.is_file() or path.stat().st_size >= cap:
        raise RuntimeError('Missing/oversized exact input: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def msg_hash(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                     sort_keys=True).encode('utf-8')).hexdigest()


def usage_total(usage):
    try:
        u = adapter.parse_usage({'usage': usage}, 'chat')
    except (ValueError, TypeError):
        return None
    return u['input_tokens'] + u['output_tokens']


def fixed_postprocess(actions, observations, targets):
    """Exactly the frozen link_mean3 alpha=.5 and bounded node release."""
    link = {r: list(actions[r]) for r in ROLES}
    node = {r: list(actions[r]) for r in ROLES}
    audits = {}
    for role in K_ROLES:
        candidate = np.asarray(actions[role], dtype=float).copy()
        candidate[2:] += .5 * (2 * np.asarray(targets[role], dtype=float) - 1 - candidate[2:])
        link[role] = candidate.tolist()
        node[role], audits[role] = adjust_local(observations[role], link[role], .025, .01)
        assert node[role][2:] == link[role][2:]
    assert node['agent_g'] == link['agent_g'] == actions['agent_g']
    return node, {'link_actions': link, 'node_audit': audits}


def make_env(manifest, out):
    main, conf = base._config(ROOT, ROOT / 'configs_main.yaml')
    base._compat_pickle_main_classes(main)
    for name in ('path_reward_save', 'path_env_save', 'path_model_save',
                 'path_model_train_log', 'path_F_save', 'log_dir', 'model_dir'):
        p = out / 'env_runtime' / name
        p.mkdir(parents=True, exist_ok=True)
        setattr(conf, name, str(p))
    conf.render = False
    sample = Path(manifest['sample_file'])
    cls = base.make_mini_env_type(main.CustomMultiAgentEnv, sample, manifest['n_scenes'])
    env = cls(conf)
    env.is_eval = True
    assert set(env.sample_dict) == set(range(200))
    return env


def score(env, actions, scene):
    base.load_scene_local_observations(env, scene)
    random.seed(1 + scene)
    np.random.seed(1 + scene)
    before = len(env.g_done_list)
    _, rewards, _, _, _ = env.step({r: np.asarray(actions[r], dtype=np.float32) for r in ROLES})
    m = base._get_step_metrics(env, before, rewards)
    assert abs(m['V'] - sum(m['revenue_' + k] for k in ('k1', 'k2', 'k3', 'g')) + m['cost_sum']) < 1e-7
    return m


def selected_history(folder):
    path = folder / 'records.jsonl'
    assert path.is_file() and path.stat().st_size < 30_000_000
    selected = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            scene = row['original_scene_id']
            if scene in SCENES:
                assert scene not in selected and row['status'] == 'ok' and not row.get('fallback_used')
                selected[scene] = row
            if len(selected) == len(SCENES):
                break
    assert set(selected) == set(SCENES)
    return selected


def prepare_contract(out, log):
    """Only this offline coordinator may read snapshots and control metrics."""
    manifest = read_json(DATASET, 10_000)
    sample = Path(manifest['sample_file'])
    assert sample.is_file() and sample.stat().st_size == 644537810
    environment_check = compat.verify_environment(ROOT, manifest)
    ram = windows_ram_gib()
    disk = shutil.disk_usage(ROOT).free / 1024 ** 3
    assert ram >= 5 and disk >= 1, 'Require 5GiB RAM and 1GiB free disk'
    selected = read_json(SELECTED, 10_000)
    assert selected['method'] == 'buffer01_d025'
    assert selected['policy_source_sha256'] == sha(ROOT / 'p2i_node_boundary/bounded_release.py')
    inputs = [DATASET, SELECTED, HISTORICAL / 'manifest.json', MAPPO / 'manifest.json',
              HISTORICAL / 'records.jsonl', MAPPO / 'records.jsonl',
              LIBRARY / 'library_manifest.json', LIBRARY / 'audit_report.json',
              IR_LIBRARY / 'ir_manifest.json']
    inputs += [LIBRARY / 'role_libraries' / (r + '.jsonl') for r in ROLES]
    inputs += [IR_LIBRARY / 'role_ir_history' / (r + '.jsonl') for r in ROLES]
    for path in inputs:
        assert path.is_file() and path.stat().st_size < 30_000_000, str(path)
    library = LocalHistory(LIBRARY)
    library.assert_independent(manifest)
    ir = IRHistory(IR_LIBRARY, library)
    reference = read_json(HISTORICAL / 'manifest.json', 20_000)
    for folder in (HISTORICAL, MAPPO):
        meta = read_json(folder / 'manifest.json', 20_000)
        assert meta['eval_dataset_id'] == manifest['dataset_id'] and meta['seed'] == 1
        assert meta['eval_dataset_sha256'] == compat.SAMPLE_SHA and not meta['mock_not_real_llm']
    assert library.manifest_sha256 == reference['library_manifest_sha256']
    assert library.library_hashes == reference['library_role_sha256']
    assert ir.manifest_sha256 == reference['ir_history_manifest_sha256']
    assert ir.role_hashes == reference['ir_history_role_sha256']
    policy = IRRiskRerankPolicy(library, ir, client=object(), examples=3, seed=1)
    assert policy.reranker.config.as_dict() == reference['p2a_rerank_config']
    cached_q = selected_history(HISTORICAL)
    cached_m = selected_history(MAPPO)
    protocol = {
        'schema': 'p3_model_assignment_dev20_v1', 'dataset_id': manifest['dataset_id'],
        'dataset_split': 'validation', 'dataset_seed': 5601, 'scene_ids': list(SCENES),
        'scene_selection': 'fixed first20 IDs, no outcome filtering; previously used development data',
        'environment': environment_check,
        'models': {'qwen': MODEL_QWEN, 'gpt': MODEL_GPT}, 'arms': ASSIGNMENTS,
        'library': str(LIBRARY.relative_to(ROOT)), 'ir_history': str(IR_LIBRARY.relative_to(ROOT)),
        'library_manifest_sha256': library.manifest_sha256, 'library_role_sha256': library.library_hashes,
        'ir_manifest_sha256': ir.manifest_sha256, 'ir_role_sha256': ir.role_hashes,
        'rerank': policy.reranker.config.as_dict(), 'examples': 3,
        'prompt': 'same exact P2-A IR-aware message content per role/scene for both models',
        'GPT_transport': 'Chat Completions; adapter maps system to developer without text changes',
        'qwen_config': {'temperature': .1, 'max_tokens': 4096, 'stream': True, 'timeout_s': 120},
        'gpt_config': {'reasoning_effort': 'medium', 'max_completion_tokens': GPT_OUTPUT_LIMIT,
                       'stream': False, 'timeout_s': 360},
        'postprocessor': {'link': 'own TRAIN link_mean3 alpha=.5', 'node_max_increment': .025,
                          'node_extra_projected_overload': .01, 'g_unchanged': True},
        'expected_calls': {'qwen': 80, 'gpt': 80}, 'max_concurrent_requests': WORKERS,
        'gpt_budget_units': TOKEN_BUDGET / TOKENS_PER_UNIT, 'tokens_per_unit': TOKENS_PER_UNIT,
        'gpt_accounting': 'input+output tokens; cached/reasoning already included; uniform user proxy price',
        'qwen_incremental_units': 0, 'automatic_retries': 0, 'fallback': False,
        'shared_draws_across_arms': True, 'environment_seed': 1, 'test_set_used': False,
        'available_RAM_GiB': ram, 'free_disk_GiB': disk,
        'memory_plan': 'Read existing644.5MB sample once; one environment, no teacher constructor; require5GiB RAM',
        'source_hashes': {p: sha(ROOT / p) for p in SOURCES},
        'input_files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'format': p.suffix,
                                                'sha256': sha(p)} for p in inputs},
        'limitations': ['Exploratory development,20 scenes and one draw/model/role only.',
                       'Model-specific inference controls differ; base messages and postprocessor are fixed.'],
    }
    write_json(out / 'protocol.json', protocol)
    from p2j_validation.run_validation import derived_actions
    with contextlib.redirect_stdout(log):
        env = make_env(manifest, out)
        prepared = []
        for scene in SCENES:
            obs = base.load_scene_local_observations(env, scene)
            roles = {}; targets = {}; target_ids = {}
            for role in ROLES:
                matches = policy.library.query(role, obs[role], top_k=3)
                messages = policy._make_messages(role, obs[role], matches)
                assert messages == cached_q[scene]['per_role'][role]['messages']
                assert [m['experience_id'] for m in matches] == cached_q[scene]['per_role'][role]['retrieved_ids_for_audit_only']
                # The GPT adapter changes only message role tags, never content.
                transported = adapter.transform_request({'messages': messages}, api='chat', model=MODEL_GPT,
                                                       effort='medium', max_output_tokens=GPT_OUTPUT_LIMIT)
                assert [m['content'] for m in transported['messages']] == [m['content'] for m in messages]
                roles[role] = {'messages': messages, 'prompt_sha256': msg_hash(messages),
                               'retrieved_ids': [m['experience_id'] for m in matches],
                               'rerank_trace': policy.adapter.last_trace[role]}
                if role in K_ROLES:
                    target, info = teacher_targets(library, role, obs[role])
                    targets[role] = target['link_mean3'].tolist()
                    target_ids[role] = info['link_mean3_ids']
            for cached in (cached_q, cached_m):
                assert not replay.control_diff(cached[scene]['metrics'], score(env, cached[scene]['env_actions'], scene))
            final, audit = fixed_postprocess(cached_q[scene]['env_actions'], obs, targets)
            link_ref, both_ref, node_ref = derived_actions(cached_q[scene]['env_actions'], obs, targets)
            assert final == both_ref and audit == {'link_actions': link_ref, 'node_audit': node_ref}
            item = {'scene': scene, 'observations': {r: obs[r].tolist() for r in ROLES},
                    'roles': roles, 'targets': targets, 'link_target_TRAIN_ids': target_ids}
            prepared.append(item)
            append_jsonl(out / 'prepared.jsonl', item)
        # Recheck original controls after all interventions to detect state pollution.
        for scene in (0, 19):
            assert not replay.control_diff(cached_q[scene]['metrics'], score(env, cached_q[scene]['env_actions'], scene))
    write_json(out / 'preflight.json', {'passed': True, 'API_calls': 0, 'reconstructed_prompts': 80,
               'prompt_content_transport_checks': 80, 'historical_metric_controls': 40,
               'postprocess_equivalence_controls': 20, 'post_intervention_controls': 2,
               'historical_MAPPO_reference': summarize([cached_m[s] for s in SCENES])})
    return prepared, env, protocol


class AuditedCalls:
    """Bounded concurrent requests; durable journal, no retries or fallback."""
    def __init__(self, out, model_name):
        self.out = Path(out)
        self.name = model_name
        self.model = MODEL_GPT if model_name == 'gpt' else MODEL_QWEN
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.local = threading.local()
        self.attempts = 0
        self.tokens = 0
        self.reserved = 0
        self.unknown_cost = False
        self.key = os.environ['GPT61_API_KEY' if model_name == 'gpt' else 'LLM_API_KEY']
        if model_name == 'gpt':
            self.url = adapter.upstream_url(os.environ['GPT61_BASE_URL'], 'chat')

    def call(self, item, role):
        scene = item['scene']
        rid = f'{self.name}:{scene}:{role}'
        messages = item['roles'][role]['messages']
        reserve = sum(len(m['content'].encode('utf-8')) for m in messages) + 1000 + GPT_OUTPUT_LIMIT
        with self.lock:
            if self.stop.is_set() or self.unknown_cost:
                raise RuntimeError('CANCELLED_BEFORE_REQUEST')
            if self.attempts >= 80:
                raise RuntimeError('REQUEST_LIMIT')
            if self.name == 'gpt' and self.tokens + self.reserved + reserve > TOKEN_BUDGET:
                self.stop.set()
                raise RuntimeError('TOKEN_BUDGET_RESERVE_EXCEEDED')
            self.attempts += 1
            if self.name == 'gpt':
                self.reserved += reserve
            append_jsonl(self.out / f'{self.name}_requests.jsonl', {
                'event': 'started', 'request_id': rid, 'model': self.model,
                'prompt_sha256': msg_hash(messages), 'token_reserve': reserve if self.name == 'gpt' else None})
        start = time.perf_counter()
        usage = None
        actual = None
        raw_saved = False
        try:
            if self.name == 'gpt':
                body = adapter.transform_request({'messages': messages}, api='chat', model=self.model,
                                                 effort='medium', max_output_tokens=GPT_OUTPUT_LIMIT)
                response = adapter.invoke_upstream(body, url=self.url, api_key=self.key, timeout=360)
                # Preserve every returned response before usage/action parsing.
                with self.lock:
                    append_jsonl(self.out / 'gpt_raw_replies.jsonl', {'request_id': rid, 'response': response})
                raw_saved = True
                usage = adapter.parse_usage(response, 'chat')
                actual = usage['input_tokens'] + usage['output_tokens']
                normalized = adapter.normalise_response(response, 'chat', usage)
                choice = normalized['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ValueError('Incomplete GPT completion')
                text = choice['message']['content']
                response_model = response.get('model')
            else:
                if not hasattr(self.local, 'client'):
                    self.local.client = ChatCompletionsClient(base_url=os.environ['LLM_BASE_URL'],
                        api_key=self.key, model=self.model, temperature=.1, max_tokens=4096,
                        read_timeout=120, stream=True)
                completion = self.local.client.complete(role, messages)
                text = completion.text
                usage = completion.usage
                actual = usage_total(usage)
                response_model = None  # legacy SSE client does not expose server model ID
                with self.lock:
                    append_jsonl(self.out / 'qwen_raw_replies.jsonl', {'request_id': rid,
                                 'text': text, 'usage': usage, 'finish_reason': completion.finish_reason})
                raw_saved = True
                if completion.finish_reason != 'stop':
                    raise ValueError('Incomplete Qwen completion')
            ratios = parse_ratios(text, role)
            entry = {'role': role, 'scene': scene, 'model': self.model,
                     'response_model': response_model, 'request_id': rid,
                     'raw_model_output': text, 'ratios': ratios,
                     'env_action': to_env_action(role, ratios).tolist(), 'usage': usage,
                     'total_tokens': actual, 'latency_s': time.perf_counter() - start,
                     'estimated_units': actual / TOKENS_PER_UNIT if self.name == 'gpt' else 0,
                     'prompt_sha256': msg_hash(messages)}
            with self.lock:
                append_jsonl(self.out / f'{self.name}_responses.jsonl', entry)
                append_jsonl(self.out / f'{self.name}_requests.jsonl', {'event': 'finished',
                             'request_id': rid, 'status': 'ok', 'total_tokens': actual,
                             'usage': usage, 'latency_s': entry['latency_s'],
                             'estimated_units': entry['estimated_units']})
            return entry
        except Exception as exc:
            self.stop.set()
            with self.lock:
                append_jsonl(self.out / f'{self.name}_requests.jsonl', {'event': 'finished',
                             'request_id': rid, 'status': 'error', 'exception_type': type(exc).__name__,
                             'http_status': getattr(exc, 'status', None), 'usage': usage,
                             'total_tokens': actual, 'raw_reply_saved': raw_saved,
                             'cost_unknown': self.name == 'gpt' and actual is None,
                             'no_retry': True})
                if self.name == 'gpt' and actual is None:
                    self.unknown_cost = True
            raise RuntimeError('MODEL_REQUEST_FAILED ' + rid + ' (' + type(exc).__name__ + ')') from None
        finally:
            with self.lock:
                if self.name == 'gpt':
                    self.reserved -= reserve
                    if actual is not None:
                        self.tokens += actual


def run_model(out, name, prepared):
    client = AuditedCalls(out, name)
    responses = {}
    with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = [pool.submit(client.call, item, role) for item in prepared for role in ROLES]
        try:
            for future in futures.as_completed(pending):
                entry = future.result()
                assert entry['request_id'] not in responses
                responses[entry['request_id']] = entry
        except Exception:
            client.stop.set()
            for future in pending:
                future.cancel()
            raise
    assert len(responses) == 80 and client.attempts == 80
    totals = [e['total_tokens'] for e in responses.values()]
    write_json(out / f'{name}_complete.json', {'requests': len(responses),
               'known_total_tokens': sum(t for t in totals if t is not None),
               'unknown_usage_requests': sum(t is None for t in totals),
               'estimated_units': client.tokens / TOKENS_PER_UNIT if name == 'gpt' else 0,
               'elapsed_role_seconds': sum(e['latency_s'] for e in responses.values()),
               'API_errors': 0, 'retries': 0})
    print('MODEL_COMPLETE', name, '80 roles', flush=True)
    return responses


def summarize(rows):
    result = replay.aggregate(rows)
    admitted = sum(bool(r['metrics']['g_done']) for r in rows)
    cost = sum(r['metrics']['cost_sum'] for r in rows)
    increments = np.sort([r['metrics']['V'] - sum(r['metrics']['V_' + k + '_0']
                                               for k in ('k1', 'k2', 'k3')) for r in rows])
    result.update({'admitted': admitted,
                   'cost_per_successful_admission': cost / admitted if admitted else None,
                   'lowest10pct_utility_increment_exploratory': float(increments[:2].mean())})
    return result


def paired(a, b):
    assert [r['original_scene_id'] for r in a] == [r['original_scene_id'] for r in b]
    idx = np.random.default_rng(560120).integers(0, len(a), size=(10000, len(a)))
    result = {}
    for key in ('V', 'g_done', 'cost_sum', 'V_g'):
        x = np.array([float(r['metrics'][key]) for r in a])
        y = np.array([float(r['metrics'][key]) for r in b])
        d = x - y
        result[key] = {'mean_delta': float(d.mean()),
                       'exploratory_paired_CI95_unadjusted': np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist(),
                       'scene_wins': int((d > 1e-8).sum()), 'scene_losses': int((d < -1e-8).sum())}
    return result


def evaluate(out, prepared, env, protocol, log, raw):
    summaries = {}; rows_by_arm = {}
    with contextlib.redirect_stdout(log):
        for arm, assignment in ASSIGNMENTS.items():
            rows = []
            for item in prepared:
                scene = item['scene']
                selected = {r: raw[assignment[r]][f'{assignment[r]}:{scene}:{r}'] for r in ROLES}
                actions = {r: selected[r]['env_action'] for r in ROLES}
                assert all(selected[r]['prompt_sha256'] == item['roles'][r]['prompt_sha256'] for r in ROLES)
                final, audit = fixed_postprocess(actions, item['observations'], item['targets'])
                m = score(env, final, scene)
                row = {'status': 'ok', 'original_scene_id': scene, 'arm': arm, 'metrics': m,
                       'role_models': assignment, 'raw_actions': actions, 'env_actions': final,
                       'postprocess': audit, 'role_request_ids': {r: selected[r]['request_id'] for r in ROLES},
                       'prompt_sha256': {r: item['roles'][r]['prompt_sha256'] for r in ROLES},
                       'estimated_GPT_units': sum(selected[r]['estimated_units'] for r in ROLES),
                       'sum_role_latency_s': sum(selected[r]['latency_s'] for r in ROLES)}
                rows.append(row)
                append_jsonl(out / f'{arm}_records.jsonl', row)
            rows_by_arm[arm] = rows
            summaries[arm] = summarize(rows)
            summaries[arm]['inference_GPT_units_if_deployed_on20'] = sum(r['estimated_GPT_units'] for r in rows)
            summaries[arm]['sum_role_latency_s_mean'] = float(np.mean([r['sum_role_latency_s'] for r in rows]))
    assert all(sha(ROOT / p) == h for p, h in protocol['source_hashes'].items())
    for name in ('qwen', 'gpt'):
        events = [json.loads(line) for line in (out / f'{name}_requests.jsonl').read_text(encoding='utf-8').splitlines()]
        started = {e['request_id'] for e in events if e['event'] == 'started'}
        finished = {e['request_id'] for e in events if e['event'] == 'finished' and e['status'] == 'ok'}
        assert len(events) == 160 and started == finished and len(started) == 80
    comparisons = {arm + '_minus_all_qwen': paired(rows, rows_by_arm['all_qwen'])
                   for arm, rows in rows_by_arm.items() if arm != 'all_qwen'}
    # Factorial interaction is descriptive only: no model/prompt selection during calls.
    v = {arm: np.array([r['metrics']['V'] for r in rows]) for arm, rows in rows_by_arm.items()}
    interaction = v['all_gpt'] - v['qwen_g_gpt_k'] - v['gpt_g_qwen_k'] + v['all_qwen']
    actual_tokens = sum(e['total_tokens'] for e in raw['gpt'].values())
    report = {'status': 'COMPLETE', 'summaries': summaries, 'comparisons': comparisons,
              'V_model_assignment_interaction_mean_exploratory': float(interaction.mean()),
              'actual_requests': {'qwen': len(raw['qwen']), 'gpt': len(raw['gpt'])},
              'gpt_total_tokens': actual_tokens, 'gpt_estimated_units': actual_tokens / TOKENS_PER_UNIT,
              'source_hashes_unchanged': True, 'journal_checks_passed': True,
              'scored_rows': 80, 'paired_prompt_checks': 80, 'new_datasets_generated': 0,
              'limitations': protocol['limitations']}
    write_json(out / 'summary.json', report)
    lines = ['# P3 同提示词异构模型20场开发实验', '',
             '复用 p2b_val_seed5601 前20场；同一P2-A提示词与TRAIN经验，同一固定link+node后处理。',
             '每模型每角色每场只采样一次，四组共享这些动作。不是独立测试或多轮协商。', '',
             '| 方案 | 接纳 | 平均V | 平均V_g | 平均代价 | 单位成功接纳代价 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, s in summaries.items():
        lines.append(f"| {arm} | {s['admitted']}/20 | {s['V_mean']:.4f} | {s['V_g_mean']:.4f} | {s['cost_mean']:.4f} | {s['cost_per_successful_admission']:.4f} |")
    lines += ['', f'实际GPT：80次请求，{actual_tokens} token，按500万token/单位约 {actual_tokens / TOKENS_PER_UNIT:.6f} 单位。',
              'Qwen为内网无限额度，增量费用按0计；本估算不是网关账单核验。',
              '80份历史提示词、40次历史指标回放、20次后处理等价和2次末尾控制通过；160次请求无重试，80条四组评分完成。',
              '20场只用于探索；最低10%指标仅取2场，不适合正式结论。配对区间为未校正探索性区间。',
              '推理时延统计为各角色实际耗时之和；本实验并发执行并共享动作，不能当作四组分别部署的实测墙钟时长。', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('EXPERIMENT_COMPLETE', out, flush=True)
    print(json.dumps({'summary': summaries, 'gpt_total_tokens': actual_tokens,
                      'gpt_estimated_units': actual_tokens / TOKENS_PER_UNIT}, ensure_ascii=False), flush=True)


def validate_credentials():
    for name in ('LLM_BASE_URL', 'LLM_API_KEY', 'GPT61_BASE_URL', 'GPT61_API_KEY'):
        assert os.environ.get(name, '').strip(), 'Missing configuration: ' + name
    assert os.environ.get('P1A_MODEL', MODEL_QWEN) == MODEL_QWEN
    adapter.upstream_url(os.environ['GPT61_BASE_URL'], 'chat')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    args = ap.parse_args()
    if args.mode == 'run':
        validate_credentials()
    out = ROOT / 'p3_gpt_heterogeneous_dev' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT', out, flush=True)
    with (out / 'env_stdout.log').open('x', encoding='utf-8') as log:
        prepared, env, protocol = prepare_contract(out, log)
        print('PREFLIGHT_OK 80 prompts,40 metric controls,20 postprocess controls; no API calls', flush=True)
        if args.mode == 'preflight':
            return
        raw = {}
        try:
            start = time.perf_counter()
            for name in ('qwen', 'gpt'):
                raw[name] = run_model(out, name, prepared)
            evaluate(out, prepared, env, protocol, log, raw)
            write_json(out / 'execution.json', {'wall_seconds_inference_and_scoring': time.perf_counter() - start})
        except Exception as exc:
            write_json(out / 'STOPPED.json', {'status': 'STOPPED', 'exception_type': type(exc).__name__,
                       'message': str(exc)[:200], 'no_retry': True, 'inspect_saved_journals_before_any_continuation': True})
            print('STOPPED', type(exc).__name__, str(exc)[:200], flush=True)
            raise SystemExit(1)


if __name__ == '__main__':
    main()
