"""Full fresh four-role Qwen policy comparison on val5601 scenes 0..199.

Two frozen prompts, 1600 serial calls, no retry/fallback. No cached action is
applied in either fresh policy. Historical Qwen and MAPPO are offline controls.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

import val5601_counterfactual_replay as replay

ROOT = replay.ROOT
sys.path.insert(0, str(ROOT))
MODEL = 'qwen3-next-80b-a3b-instruct'
SCENES = list(range(200))
ARMS = ('full_original', 'full_link_calibration')
PROMPT = Path(__file__).with_name('qwen_link_calibration.txt')
FROZEN_PROMPT_SHA = '88555a457fd20b54f5e4110f751c9afa388703f70cf18e56913bba70ffc8ff29'


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def messages_sha(messages):
    return text_sha(json.dumps(messages, ensure_ascii=False, sort_keys=True))


def altered_messages(original, role, arm, prompt):
    if arm not in ARMS or role not in replay.ROLES:
        raise ValueError('Unrecognized arm or role')
    if len(original) != 2 or [m['role'] for m in original] != ['system', 'user']:
        raise ValueError('Unexpected base message structure')
    messages = [dict(m) for m in original]
    if arm == 'full_link_calibration' and role in replay.K_ROLES:
        messages[0]['content'] += '\n' + prompt
    return messages


def journal(stream, entry):
    stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def save_new(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


class AuditedClient:
    def __init__(self, client, stream):
        self.client = client
        self.stream = stream
        self.arm = None
        self.scene = None
        self.count = 0

    def complete(self, role, messages):
        from p1a.policy import parse_ratios
        rid = f'{self.arm}:{self.scene}:{role}'
        journal(self.stream, {'event': 'started', 'request_id': rid, 'model': MODEL,
                              'messages_sha256': messages_sha(messages),
                              'utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        start = time.perf_counter()
        response = None
        try:
            response = self.client.complete(role, messages)
            if response.finish_reason != 'stop':
                raise ValueError('Unfinished completion')
            parse_ratios(response.text, role)
        except Exception as exc:
            journal(self.stream, {'event': 'failed', 'request_id': rid,
                                  'exception_type': type(exc).__name__,
                                  'usage': getattr(response, 'usage', None),
                                  'response': getattr(response, 'text', None),
                                  'no_retry': True, 'API_cost_usd_estimate': 0})
            raise RuntimeError(f'Qwen failed at {rid}; no retry, see journal') from None
        self.count += 1
        journal(self.stream, {'event': 'finished', 'request_id': rid, 'model': MODEL,
                              'response': response.text, 'usage': response.usage,
                              'finish_reason': response.finish_reason,
                              'seconds': time.perf_counter() - start,
                              'API_cost_usd_estimate': 0})
        return response


def verify_source_meta(meta, library, ir, manifest, method):
    assert meta['eval_dataset_id'] == manifest['dataset_id']
    assert meta['eval_dataset_split'] == 'validation'
    assert meta['eval_dataset_sha256'] == replay.SAMPLE_SHA
    assert meta['scene_ids'] == SCENES and meta['seed'] == 1
    assert meta['method'] == method
    assert not any(meta.get(k) for k in ('guard_used', 'fallback_used', 'mock_not_real_llm',
                                        'task_finetuning', 'test_time_global_state_access',
                                        'cross_agent_communication'))
    assert meta['config_sha256'] == manifest['config_sha256']
    if method == 'ir_rerank':
        assert meta['model'] == MODEL and meta['examples'] == 3
        assert meta['temperature'] == .1 and meta['max_tokens'] == 4096 and meta['stream']
        assert meta['library_manifest_sha256'] == library.manifest_sha256
        assert meta['library_role_sha256'] == library.library_hashes
        assert meta['ir_history_manifest_sha256'] == ir.manifest_sha256
        assert meta['ir_history_role_sha256'] == ir.role_hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('check', 'run'), default='check')
    args = ap.parse_args()
    prompt = PROMPT.read_text(encoding='utf-8').strip()
    if text_sha(prompt) != FROZEN_PROMPT_SHA:
        raise RuntimeError('Pilot prompt changed; expansion must use the frozen candidate')
    if args.mode == 'run' and (not os.environ.get('LLM_BASE_URL') or not os.environ.get('LLM_API_KEY')
                               or os.environ.get('P1A_MODEL') != MODEL):
        raise RuntimeError('Internal Qwen configuration incomplete; no calls started')
    manifest_file = ROOT / 'p1b_data/p2b_val_seed5601.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    from p2e_prompt.val5601_eval import verify_environment
    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    from p2a.policy import IRRiskRerankPolicy
    from p2a.rerank import RerankConfig
    import p1a_runner as base
    import p1c_runner
    environment = verify_environment(ROOT, manifest)
    library = LocalHistory(ROOT / 'p2b_libraries/p2b_standard_1000_v2')
    library.assert_independent(manifest)
    ir = IRHistory(ROOT / 'p2b_ir_libraries/p2b_standard_1000_ir_v2', library)
    qfolder = ROOT / 'p2b_runs/val5601_standard1000_p2a200'
    mfolder = ROOT / 'p2b_runs/val5601_mappo200'
    qmeta = json.loads((qfolder / 'manifest.json').read_text(encoding='utf-8'))
    mmeta = json.loads((mfolder / 'manifest.json').read_text(encoding='utf-8'))
    verify_source_meta(qmeta, library, ir, manifest, 'ir_rerank')
    verify_source_meta(mmeta, library, ir, manifest, 'mappo')
    assert qmeta['p2a_rerank_config'] == RerankConfig().as_dict()
    cached = {'historical_Qwen': replay.read_jsonl(qfolder / 'records.jsonl'),
              'MAPPO': replay.read_jsonl(mfolder / 'records.jsonl')}
    class FreshPolicy(IRRiskRerankPolicy):
        def __init__(self, client, arm):
            self.arm = arm
            super().__init__(library, ir, client=client, examples=3, seed=1)

        def _make_messages(self, role, observation, matches):
            original = super()._make_messages(role, observation, matches)
            expected = cached['historical_Qwen'][self.scene_id]['per_role'][role]['messages']
            if original != expected:
                raise RuntimeError(f'Base prompt changed scene {self.scene_id}/{role}')
            return altered_messages(original, role, self.arm, prompt)

    out = ROOT / 'p2f_analysis/qwen_full200' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'env_runtime'
    runtime.mkdir()
    files = [Path(__file__).resolve(), PROMPT, manifest_file, qfolder / 'manifest.json',
             qfolder / 'records.jsonl', mfolder / 'records.jsonl',
             ROOT / 'main.py', ROOT / 'configs_main.yaml', ROOT / 'p1a_runner.py',
             ROOT / 'p1a/local_obs.py', ROOT / 'p1a/policy.py', ROOT / 'p1c/policy.py',
             ROOT / 'p1c/retrieval.py', ROOT / 'p1d/policy.py', ROOT / 'p1d/ir.py',
             ROOT / 'p2a/policy.py', ROOT / 'p2a/rerank.py', ROOT / 'p2e_prompt/val5601_eval.py']
    protocol = {'schema': 'qwen_full_policy_200_v1', 'mode': args.mode, 'model': MODEL,
                'dataset_id': manifest['dataset_id'], 'split': 'validation', 'scene_ids': SCENES,
                'dataset_generation_seed': 5601, 'policy_environment_seed': 1,
                'temperature': .1, 'max_tokens': 4096, 'timeout_s': 120, 'stream': True,
                'arms': list(ARMS), 'arm_order': 'serial baseline then candidate',
                'expected_calls_per_arm': 800, 'expected_calls_total': 1600,
                'automatic_retries': 0, 'fallback': False, 'frozen_prompt': prompt,
                'frozen_prompt_sha256': FROZEN_PROMPT_SHA,
                'full_actions_fresh': True, 'cached_components_applied': False,
                'environment': environment, 'examples': 3,
                'library_manifest_sha256': library.manifest_sha256,
                'library_role_sha256': library.library_hashes,
                'ir_manifest_sha256': ir.manifest_sha256, 'ir_role_sha256': ir.role_hashes,
                'rerank_config': RerankConfig().as_dict(),
                'source_sha256': {str(p.relative_to(ROOT)): replay.sha256(p) for p in files},
                'API_cost_usd_estimate': 0, 'cost_basis': 'User-provided internal unlimited Qwen quota; billing not independently verified',
                'information_boundary': 'Own current local observation and own TRAIN role cases/IR only; no current MAPPO actions or replay outcomes in policy messages.',
                'primary_endpoints': ['cooperative_valid', 'V'],
                'secondary_endpoints': ['g_done', 'V_g', 'cost_sum'],
                'diagnostics': ['effective_V_g', 'IR_shortfall', 'V_lower10pct_mean', 'cost_upper10pct_mean'],
                'bootstrap_seed': 20261104, 'bootstrap_replicates': 10000,
                'limitations': 'Previously used validation data, including pilot scenes0..39; one draw per arm per scene. No independent final-test inference. Serial arm order may include endpoint time drift.'}
    save_new(out / 'protocol.json', protocol)
    with (out / 'env_stdout.log').open('x', encoding='utf-8') as log, (out / 'requests.jsonl').open('x', encoding='utf-8') as ledger:
        with contextlib.redirect_stdout(log):
            main_module, _ = base._config(ROOT, ROOT / 'configs_main.yaml')
            base._compat_pickle_main_classes(main_module)
            env = p1c_runner.create_eval_env(ROOT, manifest, 'configs_main.yaml')
        env.path_reward_save = str(runtime)
        env.path_env_save = str(runtime)
        # No client exists during preflight: all 400 historical controls and
        # all 800 role messages must reconstruct before a single API request.
        probe = FreshPolicy(object(), ARMS[0])
        message_count = 0
        for scene in SCENES:
            with contextlib.redirect_stdout(log):
                obs = base.load_scene_local_observations(env, scene)
            probe.set_scene(scene)
            for role in replay.ROLES:
                matches = probe.library.query(role, obs[role], top_k=3)
                probe._make_messages(role, obs[role], matches)
                assert [x['experience_id'] for x in matches] == cached['historical_Qwen'][scene]['per_role'][role]['retrieved_ids_for_audit_only']
                message_count += 1
        controls = {}
        for name, original in cached.items():
            rows = []
            with (out / (name + '_replay.jsonl')).open('x', encoding='utf-8') as sink:
                for scene in SCENES:
                    with contextlib.redirect_stdout(log):
                        row = replay.replay_scene(env, (original, original), 'QQ', scene, 1)
                    diff = replay.control_diff(original[scene]['metrics'], row['metrics'])
                    if diff:
                        save_new(out / 'CONTROL_FAILED.json', {'method': name, 'scene': scene, 'differences': diff})
                        raise RuntimeError(f'Historical control mismatch {name}/{scene}; no API started')
                    sink.write(json.dumps(row, allow_nan=False) + '\n')
                    rows.append(row)
            controls[name] = replay.aggregate(rows)
        save_new(out / 'preflight.json', {'passed': True, 'base_messages_reconstructed': message_count,
                 'historical_controls_checked': 400, 'controls': controls, 'new_API_calls': 0})
        if args.mode == 'check':
            print('PREFLIGHT_PASSED', out, flush=True)
            return
        from p1a.policy import ChatCompletionsClient
        client = AuditedClient(ChatCompletionsClient(base_url=os.environ['LLM_BASE_URL'],
                               api_key=os.environ['LLM_API_KEY'], model=MODEL,
                               temperature=.1, max_tokens=4096, read_timeout=120, stream=True), ledger)
        summaries = dict(controls)
        start_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            for arm in ARMS:
                policy = FreshPolicy(client, arm)
                client.arm = arm
                rows = []
                with (out / (arm + '_records.jsonl')).open('x', encoding='utf-8') as sink:
                    for scene in SCENES:
                        client.scene = scene
                        policy.set_scene(scene)
                        with contextlib.redirect_stdout(log):
                            row = base.run_one_scene(env, policy, scene, scene, 1)
                        row['arm'] = arm
                        row['all_four_role_actions_fresh'] = True
                        journal(sink, row)
                        rows.append(row)
                summaries[arm] = replay.aggregate(rows)
                save_new(out / (arm + '_summary.json'), summaries[arm])
                print('COMPLETE', arm, json.dumps(summaries[arm]), flush=True)
            assert client.count == 1600
            # Verify sources stayed frozen through the entire experiment.
            for relative, expected in protocol['source_sha256'].items():
                assert replay.sha256(ROOT / relative) == expected, relative
            save_new(out / 'summary.json', {'status': 'COMPLETE', 'summaries': summaries,
                     'successful_calls': client.count, 'API_cost_usd_estimate': 0,
                     'started_utc': start_utc, 'finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                     'source_hashes_unchanged': True, 'full_four_role_actions_fresh': True})
            print('OUTPUT', out, flush=True)
        except Exception as exc:
            save_new(out / 'STOPPED.json', {'error_type': type(exc).__name__,
                     'successful_calls': client.count, 'arm': client.arm, 'scene': client.scene,
                     'automatic_retries': 0, 'message': 'Stopped; preserve records and journal. No automatic restart.'})
            raise RuntimeError(f'Experiment stopped at {client.arm} scene {client.scene}; inspect preserved STOPPED and journal') from None


if __name__ == '__main__':
    main()
