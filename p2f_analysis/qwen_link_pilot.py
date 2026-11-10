"""Authorized, zero-monetary-cost Qwen link-only development pilot.

Two predeclared arms, scenes 0..39, three OWN-role calls per scene per arm.
Cached Qwen node and g actions remain fixed. Freshly returned node components
are recorded but not applied. This isolates link decisions; it is not a full
fresh-policy evaluation or an independent test.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

import val5601_counterfactual_replay as replay

ROOT = replay.ROOT
sys.path.insert(0, str(ROOT))
MODEL = 'qwen3-next-80b-a3b-instruct'
SCENES = list(range(40))
ARMS = ('link_refresh_control', 'link_calibration')


def digest_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def journal(stream, entry):
    stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def main():
    if os.environ.get('P1A_MODEL', MODEL) != MODEL:
        raise RuntimeError('This experiment permits only the configured internal Qwen model')
    if not os.environ.get('LLM_API_KEY') or not os.environ.get('LLM_BASE_URL'):
        raise RuntimeError('Qwen configuration is missing; no calls started')
    source = ROOT / 'p2b_runs/val5601_standard1000_p2a200/records.jsonl'
    cached = replay.read_jsonl(source)
    prompt_path = Path(__file__).with_name('qwen_link_calibration.txt')
    prompt = prompt_path.read_text(encoding='utf-8').strip()
    assert prompt and all(len(cached[i]['per_role'][r]['messages']) == 2 for i in SCENES for r in replay.K_ROLES)
    manifest = json.loads((ROOT / 'p1b_data/p2b_val_seed5601.json').read_text(encoding='utf-8'))
    from p2e_prompt.val5601_eval import verify_environment
    check = verify_environment(ROOT, manifest)
    from p1a.policy import ChatCompletionsClient, parse_ratios, to_env_action
    import p1a_runner as base
    import p1c_runner
    out = ROOT / 'p2f_analysis/qwen_link_pilot' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'env_runtime'
    runtime.mkdir()
    protocol = {'schema': 'qwen_conditional_link_pilot_v1', 'model': MODEL,
                'dataset': 'p2b_val_seed5601', 'split': 'validation', 'scene_ids': SCENES,
                'environment_seed': 1, 'temperature': .1, 'max_tokens': 4096,
                'stream': True, 'timeout_s': 120, 'arms': ARMS,
                'expected_calls': 240, 'automatic_retries': 0, 'API_cost_usd': 0,
                'billing_basis': 'User states internal unlimited-quota Qwen deployment',
                'environment': check, 'source_records_sha256': replay.sha256(source),
                'source_script_sha256': replay.sha256(Path(__file__).resolve()),
                'prompt_sha256': digest_text(prompt), 'prompt': prompt,
                'cached_action_policy': 'Keep original Qwen g and k node components; replace ONLY k link components with each arm response.',
                'information_boundary': 'Each request reuses only that role original local observation and own TRAIN examples/IR feedback. No replay results or current MAPPO actions in prompt.',
                'limitations': 'Conditional link-refresh development diagnostic; one draw per arm per scene. All 200 scenes already used in development.'}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (out / 'env_stdout.log').open('x', encoding='utf-8') as log, (out / 'requests.jsonl').open('x', encoding='utf-8') as requests:
        with contextlib.redirect_stdout(log):
            main_module, _ = base._config(ROOT, ROOT / 'configs_main.yaml')
            base._compat_pickle_main_classes(main_module)
            env = p1c_runner.create_eval_env(ROOT, manifest, 'configs_main.yaml')
        env.path_reward_save = str(runtime)
        env.path_env_save = str(runtime)
        # Reproduce all cached pilot controls BEFORE constructing the client.
        controls = []
        for scene in SCENES:
            with contextlib.redirect_stdout(log):
                row = replay.replay_scene(env, (cached, cached), 'QQ', scene, 1)
            diff = replay.control_diff(cached[scene]['metrics'], row['metrics'])
            if diff:
                raise RuntimeError(f'Cached Qwen control mismatch scene {scene}')
            controls.append(row)
        (out / 'cached_control.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in controls), encoding='utf-8')
        client = ChatCompletionsClient(base_url=os.environ['LLM_BASE_URL'],
                                       api_key=os.environ['LLM_API_KEY'], model=MODEL,
                                       temperature=.1, max_tokens=4096, read_timeout=120, stream=True)
        results = {'cached_Qwen': replay.aggregate(controls)}
        for arm in ARMS:
            rows = []
            with (out / (arm + '_records.jsonl')).open('x', encoding='utf-8') as sink:
                for scene in SCENES:
                    actions = {r: list(cached[scene]['env_actions'][r]) for r in replay.ROLES}
                    decisions = {}
                    for role in replay.K_ROLES:
                        messages = [dict(x) for x in cached[scene]['per_role'][role]['messages']]
                        original_hash = digest_text(json.dumps(messages, ensure_ascii=False))
                        if arm == 'link_calibration':
                            messages[0]['content'] += '\n' + prompt
                        request_id = f'{arm}:{scene}:{role}'
                        journal(requests, {'event': 'started', 'request_id': request_id,
                                           'model': MODEL, 'scene': scene, 'role': role,
                                           'base_messages_sha256': original_hash,
                                           'messages_sha256': digest_text(json.dumps(messages, ensure_ascii=False))})
                        start = time.perf_counter()
                        try:
                            response = client.complete(role, messages)
                            ratios = parse_ratios(response.text, role)
                            fresh = to_env_action(role, ratios)
                        except Exception as exc:
                            journal(requests, {'event': 'failed', 'request_id': request_id,
                                               'exception_type': type(exc).__name__, 'no_retry': True})
                            raise RuntimeError(f'Qwen request failed at {request_id}; no retries; inspect preserved journal') from None
                        elapsed = time.perf_counter() - start
                        actions[role][2:] = fresh.tolist()[2:]
                        decisions[role] = {'messages': messages, 'fresh_ratios': ratios,
                                           'raw_response': response.text, 'usage': response.usage,
                                           'finish_reason': response.finish_reason,
                                           'request_seconds': elapsed,
                                           'node_components_applied': 'original cached Qwen',
                                           'link_components_applied': fresh.tolist()[2:]}
                        journal(requests, {'event': 'finished', 'request_id': request_id,
                                           'model': MODEL, 'usage': response.usage,
                                           'request_seconds': elapsed, 'API_cost_usd': 0})
                    intervention = {scene: {'env_actions': actions}}
                    with contextlib.redirect_stdout(log):
                        row = replay.replay_scene(env, (intervention, intervention), 'QQ', scene, 1)
                    row['case'] = arm
                    row['no_model_call'] = False
                    row['conditional_link_refresh'] = True
                    row['per_role_decisions'] = decisions
                    for role in replay.K_ROLES:
                        assert row['env_actions'][role][:2] == cached[scene]['env_actions'][role][:2]
                    assert row['env_actions']['agent_g'] == cached[scene]['env_actions']['agent_g']
                    sink.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    sink.flush()
                    rows.append(row)
            results[arm] = replay.aggregate(rows)
            print('COMPLETE', arm, json.dumps(results[arm]), flush=True)
        (out / 'summary.json').write_text(json.dumps({'status': 'COMPLETE', 'results': results,
                  'calls': 240, 'API_cost_usd': 0, 'conditional_development_diagnostic': True}, indent=2) + '\n', encoding='utf-8')
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
