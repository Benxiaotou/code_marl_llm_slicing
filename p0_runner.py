"""Run a controlled P0 evaluation without modifying reward/transition mathematics.

Mini mode uses P0_small_samples.pkl. Full mode loads the original 600-sample data.
For fair comparisons, each scenario is deep-copied immediately before acting.
No MAPPO training occurs; the optional MAPPO policy loads a frozen checkpoint.
"""
import argparse
import csv
from copy import deepcopy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import yaml

from p0_policy import (AGENTS, FixedPolicy, GreedyPolicy, RandomPolicy, LLMPolicy,
                       build_prompt, ratio_actions, check_obs, Decision)


def prepare_config(project_root, sample_file, sample_count, output_dir, n_steps):
    config_path = project_root / 'configs_main.yaml'
    if not config_path.is_file():
        raise FileNotFoundError('Missing configs_main.yaml in ' + str(project_root))
    with config_path.open(encoding='utf-8') as f:
        data = yaml.safe_load(f)
    from argparse import Namespace
    data['root'] = str(project_root)
    data['data_root'] = str(project_root / 'data')
    data['result_root'] = str(output_dir)
    data['path_reward_save'] = str(output_dir / 'raw_reward')
    data['path_env_save'] = str(output_dir / 'raw_env')
    data['path_model_save'] = str(output_dir / 'raw_model')
    data['save_attach_word'] = 'p0_'
    data['traffic_file'] = 'GEANT_TM1'
    data['test_traffic_file'] = 'GEANT_TM2'
    data['train_traffic_dir'] = '2005_04_1st_week'
    data['test_traffic_dir'] = '2005_04_2nd_week'
    data['p0_sample_cnt'] = sample_count
    if sample_file is not None:
        data['p0_sample_file'] = str(sample_file.resolve())
    data['parallels'] = 1
    data['render'] = False
    data['test_mode'] = True
    data['eval_max_step'] = max(1000000, n_steps + 1)  # prevent legacy eval pickle writes
    data['eval_max_episode'] = 1
    data['device'] = 'cpu'
    for d in ('path_reward_save', 'path_env_save', 'path_model_save'):
        Path(data[d]).mkdir(parents=True, exist_ok=True)
    return Namespace(**data)


def stage_scene(env, scene_index):
    """Independently clone the exact original serialized scenario.

    IMPORTANT: original env.step mutates k1,k2,k3,g,physical in place.
    Without deepcopy, a second policy would observe a scenario changed by the first.
    """
    sample = deepcopy(env.sample_dict[scene_index])
    if not isinstance(sample, (list, tuple)) or len(sample) != 5:
        raise RuntimeError('Scene {} does not contain 4 virtual networks and one physical graph'.format(scene_index))
    env.k1_net, env.k2_net, env.k3_net, env.g_net, env.DG = sample
    env.sample_idx = scene_index
    obs = {}
    for key, net in zip(AGENTS[:3], sample[:3]):
        nn, ll, _ = env._get_overlap_node_link(net, env.g_net)
        obs[key] = env._get_v_net_observations(net, sorted(nn), sorted(ll), True)
    obs['agent_g'] = env._get_g_net_observations(True)
    check_obs(obs)
    env.observations = obs
    env.global_state = np.hstack([obs[k] for k in AGENTS])
    return obs


def uncover_raw_env(vec):
    obj = vec
    visited = set()
    while id(obj) not in visited:
        visited.add(id(obj))
        if obj.__class__.__name__ == 'CustomMultiAgentEnv' or hasattr(obj, '_get_g_adjustment'):
            return obj
        if hasattr(obj, 'envs') and obj.envs:
            obj = obj.envs[0]
        elif hasattr(obj, 'env'):
            obj = obj.env
        elif hasattr(obj, 'unwrapped') and obj.unwrapped is not obj:
            obj = obj.unwrapped
        else:
            break
    raise RuntimeError('Cannot find CustomMultiAgentEnv inside XuanCe environment wrappers')


class MAPPOPolicy:
    name = 'mappo'
    def __init__(self, config, checkpoint):
        import torch
        from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
        from main import CustomMultiAgentEnv, CustomMAPPO_Agents
        REGISTRY_MULTI_AGENT_ENV[config.env_name] = CustomMultiAgentEnv
        self.vec = make_envs(config)
        self.env = uncover_raw_env(self.vec)
        self.agents = CustomMAPPO_Agents(config=config, envs=self.vec)
        self.hidden_actor, self.hidden_critic = self.agents.init_rnn_hidden(1)
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError('MAPPO checkpoint: ' + str(checkpoint))
        try:
            state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(str(checkpoint), map_location='cpu')
        self.agents.policy.load_state_dict(state, strict=True)
        self.agents.policy.eval()

    def act(self, observations):
        with __import__('torch').no_grad():
            policy_out = self.agents.action(
                obs_dict=[observations], state=None, avail_actions_dict=None,
                rnn_hidden_actor=self.hidden_actor, rnn_hidden_critic=self.hidden_critic,
                test_mode=True)
        self.hidden_actor = policy_out.get('rnn_hidden_actor')
        self.hidden_critic = policy_out.get('rnn_hidden_critic')
        actions = policy_out['actions']
        if isinstance(actions, list):
            if len(actions) != 1:
                raise ValueError('Expected one vectorized MAPPO environment')
            actions = actions[0]
        if set(actions) != set(AGENTS):
            raise ValueError('MAPPO returned unexpected agent keys')
        result = {}
        for key in AGENTS:
            x = np.asarray(actions[key], dtype=np.float32).reshape(-1)
            if x.shape != (4,) or not np.isfinite(x).all():
                raise ValueError('MAPPO returned invalid action shape/values for ' + key)
            result[key] = np.clip(x, -1., 1.)
        return Decision(result, self.name)

    def close(self):
        try:
            self.agents.finish()
        finally:
            self.vec.close()


def _safe_float(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite experiment metric')
    return value


def _jsonl_append(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
    f.flush()


def run_one(policy, env, indices, original_ids, output_dir, seed, policy_label):
    records = []
    env.is_eval = True
    log_path = output_dir / (policy_label + '_records.jsonl')
    with log_path.open('w', encoding='utf-8') as f:
        for seq, sample_index in enumerate(indices):
            # Reset randomness per original scene, independent of policy ordering.
            this_seed = seed + int(original_ids[sample_index])
            random.seed(this_seed)
            np.random.seed(this_seed)
            try:
                import torch
                torch.manual_seed(this_seed)
            except ImportError:
                pass
            if isinstance(policy, RandomPolicy):
                policy.rng = np.random.RandomState(this_seed)
            obs = stage_scene(env, sample_index)
            # Policy internals may consume Python/NumPy RNG; keep environment
            # migration stochasticity independent of the compared policy.
            env_python_rng = random.getstate()
            env_numpy_rng = np.random.get_state()
            before = time.perf_counter()
            decision = policy.act(obs)
            policy_wall_s = time.perf_counter() - before
            random.setstate(env_python_rng)
            np.random.set_state(env_numpy_rng)
            before_step = time.perf_counter()
            _, reward, _, _, _ = env.step(decision.action)
            env_wall_s = time.perf_counter() - before_step
            if not env.dict_eval_index['V']:
                raise RuntimeError('eval_env_step did not record V; is_eval flag incorrect')
            rec = {
                'policy': policy_label, 'scene_index': sample_index,
                'original_scene_index': int(original_ids[sample_index]),
                'seq': seq, 'g_done': bool(env.g_done_list[-1]),
                'reward_mean': _safe_float(np.mean([reward[k] for k in AGENTS])),
                'action_latency_s': _safe_float(policy_wall_s),
                'api_latency_s': _safe_float(decision.latency_s),
                'env_step_s': _safe_float(env_wall_s),
                'used_fallback': decision.used_fallback,
                'decision_error': decision.error,
                'raw_response': decision.raw_response,
                'prompt_hash': decision.prompt_hash,
                'usage': decision.usage,
                'action': {k: [float(v) for v in decision.action[k]] for k in AGENTS},
                'ratio': {k: [float((v + 1) / 2) for v in decision.action[k]] for k in AGENTS},
            }
            for name, history in env.dict_eval_index.items():
                rec[name] = _safe_float(history[-1])
            rec['cost_sum'] = rec['cost_k1'] + rec['cost_k2'] + rec['cost_k3']
            records.append(rec)
            _jsonl_append(f, rec)
            print('policy={} scene={} g_done={} V={:.3f} latency={:.2f}s{}'.format(
                policy_label, rec['original_scene_index'], int(rec['g_done']), rec['V'],
                policy_wall_s, ' FALLBACK' if decision.used_fallback else ''), flush=True)
    return records


def summarize(records):
    if not records:
        raise ValueError('No records to summarize')
    means = {name: sum(float(r[name]) for r in records) / len(records)
             for name in ('V', 'V_k1','V_k2','V_k3','V_g','cost_sum','cost_k1','cost_k2','cost_k3',
                          'revenue_k1','revenue_k2','revenue_k3','revenue_g',
                          'reward_mean','action_latency_s','api_latency_s','env_step_s')}
    means.update({'policy': records[0]['policy'], 'n': len(records),
                  'acceptance_rate': sum(int(r['g_done']) for r in records) / len(records),
                  'fallback_rate': sum(int(r['used_fallback']) for r in records) / len(records),
                  'api_latency_median_s': statistics.median(r['api_latency_s'] for r in records),
                  'api_latency_max_s': max(r['api_latency_s'] for r in records)})
    return means


def make_args():
    p = argparse.ArgumentParser(description='P0 zero-shot LLM vs frozen MAPPO and non-RL baselines')
    p.add_argument('--project-root', default='.', help='Root with patched main.py, config and data folder')
    p.add_argument('--dataset', choices=['mini', 'full'], default='mini')
    p.add_argument('--sample-file', default=None, help='Override default scene pickle file')
    p.add_argument('--mini-indices-json', default='P0_small_samples.indices.json')
    p.add_argument('--policy', choices=['llm','fixed','rule','random','mappo','mock'], default='mock')
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--indices', default=None, help='Comma-separated CURRENT dataset indices, e.g. 0,1,2')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--checkpoint', default=None, help='Original frozen MAPPO .pth file')
    p.add_argument('--api-url', default=None)
    p.add_argument('--model', default=None)
    p.add_argument('--api-timeout', type=float, default=90,
                   help='Read timeout in seconds; with SSE it applies between data chunks')
    p.add_argument('--api-stream', action='store_true',
                   help='Enable Chat Completions SSE streaming (experimental compatibility option)')
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--fail-mode', choices=['abort','rule'], default='abort')
    return p.parse_args()


def main():
    args = make_args()
    root = Path(args.project_root).resolve()
    if not (root / 'main.py').is_file():
        raise FileNotFoundError('Project root must contain main.py')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sample_count = 600 if args.dataset == 'full' else 5
    sample_path = Path(args.sample_file).resolve() if args.sample_file else (
        (root / 'P0_small_samples.pkl') if args.dataset == 'mini' else None)
    if sample_path is not None and not sample_path.is_file():
        raise FileNotFoundError('Missing mini dataset ' + str(sample_path) +
            ' - copy uploaded P0_small_samples.pkl to project root')
    indices = ([int(s.strip()) for s in args.indices.split(',')]
               if args.indices is not None else list(range(args.count)))
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= sample_count:
        raise ValueError('Selected indices must be distinct and in [0,{})'.format(sample_count))
    original_ids = {i:i for i in range(sample_count)}
    if args.dataset == 'mini':
        index_path = root / args.mini_indices_json
        if index_path.is_file():
            with index_path.open(encoding='utf-8') as f:
                original_ids = {int(k):int(v) for k,v in json.load(f).items()}
            if set(original_ids) != set(range(sample_count)):
                raise ValueError('Mini mapping file keys must cover 0..4')
    when = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest = Path(args.output_dir).resolve() if args.output_dir else (root / 'p0_runs' / (args.policy + '_' + args.dataset + '_' + when))
    dest.mkdir(parents=True, exist_ok=True)
    config = prepare_config(root, sample_path, sample_count, dest, len(indices))
    from main import CustomMultiAgentEnv, InitVnetworkTrain
    # P0_FULL_PICKLE_LEGACY_ALIAS_V1
    # Original dataset was serialized with main.py executed as __main__.
    sys.modules["__main__"].InitVnetworkTrain = InitVnetworkTrain
    if args.policy == 'llm' and not os.environ.get('LLM_API_KEY'):
        raise ValueError('Set LLM_API_KEY before --policy llm; refusing to run as a silent rule policy')
    if args.dataset == 'full' and sample_path is None:
        expected = root / 'data' / 'GEANT' / 'GEANT_3_12_600_sample_save.pkl'
        if not expected.is_file():
            raise FileNotFoundError('Original 600-sample file missing: ' + str(expected))
    if config.topology_file != 'GEANT':
        raise ValueError('This P0 runner is explicitly for GEANT')
    policy = None
    checkpoint_path = None
    vec_policy = None
    if args.policy == 'mappo':
        path = Path(args.checkpoint).resolve() if args.checkpoint else (root / 'marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth')
        if not path.is_file():
            raise FileNotFoundError('MAPPO frozen checkpoint missing: ' + str(path))
        checkpoint_path = path
        vec_policy = MAPPOPolicy(config, path)
        policy, env = vec_policy, vec_policy.env
    else:
        env = CustomMultiAgentEnv(config)
        if args.policy == 'llm':
            policy = LLMPolicy(base_url=args.api_url, model=args.model,
                               timeout=args.api_timeout, max_tokens=args.max_tokens,
                               fail_mode=args.fail_mode, stream=args.api_stream)
        elif args.policy == 'fixed': policy = FixedPolicy()
        elif args.policy == 'rule': policy = GreedyPolicy()
        elif args.policy == 'random': policy = RandomPolicy(args.seed)
        elif args.policy == 'mock':
            class MockPolicy:
                name = 'mock_not_an_llm_result'
                def act(self, observations):
                    check_obs(observations)
                    ratio = {k: [0.35]*(3 if k=='agent_g' else 4) for k in AGENTS}
                    return Decision(ratio_actions(ratio), 'mock_not_an_llm_result')
            policy = MockPolicy()
    name = policy.name
    manifest = {'dataset': args.dataset, 'dataset_sample_count':sample_count,
                'dataset_path':str(sample_path) if sample_path else 'Original 600 scenes',
                'selected_scene_indices':indices,
                'selected_original_scene_indices':[original_ids[x] for x in indices],
                'seed':args.seed, 'policy':name,
                'checkpoint':str(checkpoint_path) if checkpoint_path else None,
                'llm_model':policy.model if args.policy=='llm' else None,
                'llm_fail_mode':args.fail_mode if args.policy=='llm' else None,
                'llm_stream':args.api_stream if args.policy=='llm' else None,
                'llm_base_url':policy.base_url if args.policy=='llm' else None,
                'notes':'Single-step independent scenarios; no RL training; original step/reward unchanged.'}
    manifest['main_sha256'] = hashlib.sha256((root/'main.py').read_bytes()).hexdigest()
    manifest['scene_file_bytes'] = sample_path.stat().st_size if sample_path else None
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False),encoding='utf-8')
    try:
        records = run_one(policy, env, indices, original_ids, dest, args.seed, name)
        (dest / 'summary.json').write_text(json.dumps(summarize(records), indent=2, ensure_ascii=False),encoding='utf-8')
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False),encoding='utf-8')
        print('\nOutputs:', dest)
        print(json.dumps(summarize(records), indent=2, ensure_ascii=False))
    finally:
        if vec_policy is not None:
            vec_policy.close()
        elif hasattr(env, 'close'):
            env.close()


if __name__ == '__main__':
    main()
