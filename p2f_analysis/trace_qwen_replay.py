"""Passive cost/revenue tracing and per-role link interventions; zero API."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

import val5601_counterfactual_replay as replay

ROOT = replay.ROOT
sys.path.insert(0, str(ROOT))
FIRST = ROOT / 'p2f_analysis/val5601_counterfactual/run_20261104T125542_548095Z'
CASES = ('MM', 'QQ', 'Q_LINK_M', 'Q_LINK_k1_M', 'Q_LINK_k2_M', 'Q_LINK_k3_M')


class PassiveTrace:
    def __init__(self, env):
        self.env = env
        self.reset()
        node = env._get_k_node_cost
        link = env._get_k_link_cost
        g = env._get_g_adjustment

        def node_wrapper(*args, **kwargs):
            result = node(*args, **kwargs)
            self.nodes.append(float(result[0]))
            return result

        def link_wrapper(*args, **kwargs):
            result = link(*args, **kwargs)
            self.links.append({'raw_cost': float(result[0]),
                               'sum_link_load': float(result[3]),
                               'old_sum_link_load': float(result[4])})
            return result

        def g_wrapper(*args, **kwargs):
            net = kwargs['g_net']
            action = kwargs['action_g']
            first = None
            gaps = {'compute': [], 'cache': [], 'bandwidth': []}
            # Passive resource-capacity diagnostics; evaluation is unchanged.
            for idx in range(net.num_nodes):
                physical = env.node_id_to_idx[net.node_id_list[idx]]
                pn = env.DG.nodes[physical]
                gn = net.DG.nodes[idx]
                for label, cap, used in (('compute', 'comp_capacity', 'allocated_comp'),
                                         ('cache', 'cach_capacity', 'allocated_cach')):
                    gap = float(pn[cap] - pn[used] - gn[cap])
                    gaps[label].append(gap)
                    if gap < 0 and first is None:
                        first = {'stage': label, 'local_index': idx, 'gap': gap}
            for idx in range(net.num_links):
                s, d = net.link_idx_to_sd[idx]
                ps = env.node_id_to_idx[net.node_id_list[s]]
                pd = env.node_id_to_idx[net.node_id_list[d]]
                pe = env.DG[ps][pd]
                old = net.DG[s][d]['bandwidth']
                needed = .3 * old + .7 * old * (1 - action[2])
                gap = float(pe['bandwidth'] - pe['allocated_bandw'] - needed)
                gaps['bandwidth'].append(gap)
                if gap < 0 and first is None:
                    first = {'stage': 'bandwidth', 'local_index': idx, 'gap': gap}
            result = g(*args, **kwargs)
            if first is not None:
                assert not result[3], ('capacity gate prediction contradicted original result', first)
            if first is None and not result[3]:
                util = max(net.DG[s][d]['used_bandw'] / net.DG[s][d]['bandwidth']
                           for s, d in net.link_idx_to_sd.values())
                assert util > .7
                first = {'stage': 'g_link_utilization', 'max_utilization': float(util)}
            self.gate = {'first_failure': first,
                         'minimum_gaps': {k: min(v) for k, v in gaps.items()},
                         'successful': bool(result[3])}
            return result

        env._get_k_node_cost = node_wrapper
        env._get_k_link_cost = link_wrapper
        env._get_g_adjustment = g_wrapper

    def reset(self):
        self.nodes = []
        self.links = []
        self.gate = None

    def finish(self, metrics):
        assert len(self.nodes) == len(self.links) == 3 and self.gate is not None
        out = {'admission_gate': self.gate, 'roles': {}}
        for i, k in enumerate(('k1', 'k2', 'k3')):
            node_cost = .005 * self.nodes[i]
            link_cost = .2 * self.links[i]['raw_cost']
            assert math.isclose(node_cost + link_cost, metrics['cost_' + k], abs_tol=1e-8)
            link_revenue = 10 * self.links[i]['sum_link_load']
            node_revenue = metrics['revenue_' + k] - link_revenue
            out['roles'][k] = {'node_cost': node_cost, 'link_cost': link_cost,
                               'node_revenue': node_revenue, 'link_revenue': link_revenue,
                               'sum_link_load': self.links[i]['sum_link_load']}
        return out


def main():
    import p1a_runner as base
    import p1c_runner
    from p2e_prompt.val5601_eval import verify_environment
    manifest = json.loads((ROOT / 'p1b_data/p2b_val_seed5601.json').read_text(encoding='utf-8'))
    checked = verify_environment(ROOT, manifest)
    m = replay.read_jsonl(ROOT / 'p2b_runs/val5601_mappo200/records.jsonl')
    q = replay.read_jsonl(ROOT / 'p2b_runs/val5601_standard1000_p2a200/records.jsonl')
    references = {}
    for case in ('MM', 'QQ', 'Q_LINK_M'):
        with (FIRST / (case + '_records.jsonl')).open(encoding='utf-8') as f:
            references[case] = {x['original_scene_id']: x['metrics'] for x in map(json.loads, f)}
        assert sorted(references[case]) == list(range(200))
    out = ROOT / 'p2f_analysis/val5601_trace' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'env_runtime'
    runtime.mkdir()
    protocol = {'schema': 'qwen_passive_trace_v1', 'dataset': 'p2b_val_seed5601',
                'split': 'validation', 'seed': 1, 'scene_ids': list(range(200)),
                'no_API': True, 'no_checkpoint': True, 'environment': checked,
                'phase1': str(FIRST), 'cases': CASES,
                'instrumentation': 'Read-only wrappers return each original method result unchanged; no extra RNG calls.',
                'source_hashes': {str(p.relative_to(ROOT)): replay.sha256(p) for p in
                                  (Path(__file__).resolve(), ROOT / 'main.py',
                                   Path(replay.__file__).resolve(), ROOT / 'p1a_runner.py')}}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    with (out / 'env_stdout.log').open('x', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            main_module, _ = base._config(ROOT, ROOT / 'configs_main.yaml')
            base._compat_pickle_main_classes(main_module)
            env = p1c_runner.create_eval_env(ROOT, manifest, 'configs_main.yaml')
        env.path_reward_save = str(runtime)
        env.path_env_save = str(runtime)
        trace = PassiveTrace(env)
        all_rows = {}
        for case in CASES:
            rows = []
            with (out / (case + '_records.jsonl')).open('x', encoding='utf-8') as f:
                for scene in range(200):
                    trace.reset()
                    if case in ('MM', 'QQ', 'Q_LINK_M'):
                        actions = replay.chosen_actions(m, q, case, scene)
                    else:
                        actions = replay.chosen_actions(m, q, 'QQ', scene)
                        role = 'agent_' + case.split('_')[2]
                        actions[role][2:] = m[scene]['env_actions'][role][2:]
                    # Pass the exact cached intervention through the same
                    # original scene loader/step/metric path as phase 1.
                    cached = {scene: {'env_actions': actions}}
                    with contextlib.redirect_stdout(log):
                        row = replay.replay_scene(env, (cached, cached), 'QQ', scene, 1)
                    row['case'] = case
                    row['passive_trace'] = trace.finish(row['metrics'])
                    if case in references:
                        diff = replay.control_diff(references[case][scene], row['metrics'])
                        if diff:
                            (out / 'CONTROL_FAILED.json').write_text(json.dumps({'case': case, 'scene': scene, 'diff': diff}), encoding='utf-8')
                            raise RuntimeError(f'Trace altered {case} scene {scene}')
                    rows.append(row)
                    f.write(json.dumps(row, allow_nan=False) + '\n')
            all_rows[case] = rows
            print('COMPLETE', case, json.dumps(replay.aggregate(rows)), flush=True)
        result = {'status': 'VALIDATED', 'summary': {k: replay.aggregate(v) for k, v in all_rows.items()},
                  'control_match_MM_QQ_Q_LINK_M': True, 'no_API': True}
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('OUTPUT', out)


if __name__ == '__main__':
    main()
