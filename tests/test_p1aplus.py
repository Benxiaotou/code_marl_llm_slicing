import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import p1a_runner as base

import numpy as np

from p1a.local_obs import AGENTS, make_messages
from p1a.policy import Completion, PolicyFailure
from p1aplus.semantics import (describe_action_semantics, make_semantic_messages,
                               _group_indices)
from p1aplus.preflight import RULE_MARKERS, verify_main_rules
from p1aplus.policy import SemanticDecentralizedPolicy
from p1aplus_runner import get_args, run
from p1aplus_compare import compare_runs


def valid_obs(role):
    nodes = np.zeros((12, 5), dtype=np.float32)
    nodes[:, 1:] = 0.2
    if role == 'agent_g':
        nodes[:, 1] = 0.7  # physical allocated compute fraction
        nodes[:, 2] = 0.4  # g requested compute fraction
        nodes[:, 3] = 0.4  # physical allocated cache
        nodes[:, 4] = 0.3  # g requested cache
    else:
        nodes[:, 1] = 0.3  # local compute capacity fraction
        nodes[:, 2] = 0.3  # local compute utilization
        nodes[:, 3] = 0.3  # local cache capacity fraction
        nodes[:, 4] = 0.25  # local cache utilization
    links = np.zeros((36, 3), dtype=np.float32)
    if role == 'agent_g':
        links[:5, 1] = 0.6
        links[:5, 2] = 0.5
    else:
        links[:5, 1] = 0.4
        links[:5, 2] = 0.2
    adj = np.zeros((36, 2), dtype=np.float32)
    adj[:5, 1] = 1.0
    return nodes, links, adj


def compact(role, overlap_nodes=(), overlap_links=()):
    node, link, adj = valid_obs(role)
    if role != 'agent_g':
        for i in overlap_nodes:
            node[i, 0] = 1.0
        for i in overlap_links:
            link[i, 0] = 1.0
    return np.concatenate((node.ravel(), link.ravel(), adj.ravel()))


class CaptureClient:
    model = 'test-mock'
    def __init__(self):
        self.calls = []

    def complete(self, agent, messages):
        self.calls.append((agent, messages))
        return Completion(json.dumps({'ratios':[0.4] * (3 if agent == 'agent_g' else 4)}), 0.01)


class SemanticsTests(unittest.TestCase):
    def test_contiguous_repeat_grouping(self):
        self.assertEqual(_group_indices(0), ([], []))
        self.assertEqual(_group_indices(1), ([0], []))
        self.assertEqual(_group_indices(2), ([0], [1]))
        self.assertEqual(_group_indices(3), ([0, 1], [2]))
        self.assertEqual(_group_indices(5), ([0, 1, 2], [3, 4]))

    def test_group_target_mapping(self):
        o = compact('agent_k1', [1, 3, 6, 9, 10], [0, 1, 4])
        s = describe_action_semantics('agent_k1', o)
        self.assertEqual([x['affected_local_indices'] for x in s['controls']],
                         [[1, 3, 6], [9, 10], [0, 1], [4]])
        self.assertEqual(len(s['controls']), 4)

    def test_upper_bound_uses_both_compute_and_cache(self):
        o = compact('agent_k1', [1, 3, 5], [])
        o[1*5 + 2] = 0.8
        o[3*5 + 4] = 0.5
        s = describe_action_semantics('agent_k1', o)
        self.assertAlmostEqual(s['controls'][0]['no_recorded_usage_shortfall_reference_max_ratio'], 0.25)
        self.assertEqual(s['controls'][0]['example_consequences'][3]['elements_below_current_recorded_usage'], 2)
        self.assertFalse(s['controls'][2]['active'])

    def test_link_utilization_warning_and_inactive_controls(self):
        o = compact('agent_k3', [], [0])
        s = describe_action_semantics('agent_k3', o)
        self.assertFalse(s['controls'][0]['active'])
        self.assertTrue(s['controls'][2]['active'])
        self.assertIn('NOT a guarantee', s['controls'][2]['warning'])
        self.assertFalse(s['controls'][2]['zero_link_utilization_snapshot'])

    def test_zero_bandwidth_util_is_not_a_safe_ratio(self):
        o = compact('agent_k3', [], [0, 1])
        o[12*5 + 2] = 0.0
        o[12*5 + 3 + 2] = 0.0
        s = describe_action_semantics('agent_k3', o)
        for control in s['controls'][2:]:
            self.assertIsNone(control['no_recorded_usage_shortfall_reference_max_ratio'])
            self.assertTrue(control['zero_link_utilization_snapshot'])

    def test_g_legacy_precompression_warning(self):
        o = compact('agent_g')
        s = describe_action_semantics('agent_g', o)
        self.assertEqual(s['controls'][0]['nodes_currently_failing_precompression_admission'], 12)
        self.assertEqual(s['controls'][1]['nodes_currently_failing_precompression_admission'], 0)
        self.assertIn('ORIGINAL', s['controls'][0]['legacy_node_admission_test'])

    def test_g_bandwidth_minimum_compression(self):
        o = compact('agent_g')
        s = describe_action_semantics('agent_g', o)
        self.assertAlmostEqual(s['controls'][2]['snapshot_required_ratio_for_all_links'], 0.28571, places=4)
        self.assertTrue(s['controls'][2]['snapshot_feasible_within_ratio_0_to_1'])

    def test_g_bandwidth_impossible_case(self):
        o = compact('agent_g')
        link_start = 12*5
        o[link_start + 1] = .95
        o[link_start + 2] = .6
        s = describe_action_semantics('agent_g', o)
        self.assertFalse(s['controls'][2]['snapshot_feasible_within_ratio_0_to_1'])
        self.assertEqual(s['controls'][2]['links_unfillable_at_max_compression_before_peer_actions'], [0])

    def test_semantic_messages_extend_but_do_not_replace_base(self):
        o = compact('agent_k1', [1, 2, 3], [0, 1])
        original = make_messages('agent_k1', o)
        enhanced = make_semantic_messages('agent_k1', o)
        self.assertTrue(enhanced[1]['content'].startswith(original[1]['content']))
        self.assertTrue(enhanced[0]['content'].startswith(original[0]['content']))
        self.assertIn('node_control_0', enhanced[1]['content'])

    def test_information_isolation(self):
        x = {r: compact(r) for r in AGENTS}
        one = make_semantic_messages('agent_k1', x['agent_k1'])
        x['agent_k2'] = compact('agent_k2', [2, 3, 4])
        self.assertEqual(one, make_semantic_messages('agent_k1', x['agent_k1']))
        self.assertNotIn('agent_k2', json.dumps(one))

    def test_does_not_output_an_action_or_force_zero(self):
        messages = make_semantic_messages('agent_k1', compact('agent_k1', [3]))
        self.assertIn('Reply as', messages[1]['content'])
        self.assertNotIn('FINAL_ACTION', messages[1]['content'])

    def test_invalid_obs_is_rejected(self):
        with self.assertRaises(ValueError):
            describe_action_semantics('agent_k1', np.zeros(239))
        with self.assertRaises(ValueError):
            describe_action_semantics('agent_g', np.ones(240) * float('nan'))

    def test_four_calls_and_actions(self):
        x = {r: compact(r, [1, 4] if r != 'agent_g' else (), [0] if r != 'agent_g' else ())
             for r in AGENTS}
        client = CaptureClient()
        acts, log = SemanticDecentralizedPolicy(client).act(x)
        self.assertEqual([x[0] for x in client.calls], list(AGENTS))
        self.assertEqual(len(acts['agent_g']), 4)
        self.assertEqual(acts['agent_g'][-1], 0.0)
        self.assertAlmostEqual(acts['agent_k2'][0], -0.2, places=6)
        self.assertIn('LOCAL ACTION SEMANTICS', log['agent_k3']['messages'][1]['content'])

    def test_no_fallback_on_failed_role(self):
        class Bad(CaptureClient):
            def complete(self, agent, messages):
                if agent == 'agent_k3':
                    return Completion('wrong', 0.01)
                return super().complete(agent, messages)
        x = {r: compact(r) for r in AGENTS}
        with self.assertRaises(PolicyFailure) as cm:
            SemanticDecentralizedPolicy(Bad()).act(x)
        self.assertEqual(cm.exception.role, 'agent_k3')

    def test_rule_preflight(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'main.py'
            p.write_text('\n'.join(RULE_MARKERS.values()), encoding='utf-8')
            self.assertEqual(len(verify_main_rules(p)), len(RULE_MARKERS))
            p.write_text('changed main.py', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                verify_main_rules(p)


class CompareTests(unittest.TestCase):
    def _write_run(self, root, label, value, *, seed=1):
        root.mkdir()
        m = {'policy': label, 'model': 'qwen', 'dataset': 'full',
             'scene_ids': [512, 513], 'seed': seed, 'config': 'configs_main.yaml',
             'max_tokens': 4096, 'temperature': .1, 'stream': True,
             'main_py_sha256': 'dummyhash'}
        (root/'manifest.json').write_text(json.dumps(m), encoding='utf-8')
        with (root/'records.jsonl').open('w', encoding='utf-8') as f:
            for scene in (512,513):
                d = {'status':'ok', 'original_scene_id':scene,'fallback_used':False,
                     'metrics':{'g_done': bool(value), 'cooperative_valid': bool(value),
                                'V': value, 'V_g': value, 'cost_sum': 10-value},
                     'policy_latency_s': 2.0,
                     'per_role':{r:{'ratios':[float(value)]*(3 if r == 'agent_g' else 4)} for r in AGENTS}}
                f.write(json.dumps(d) + '\n')

    def test_matched_comparison(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'base',Path(t)/'sem'
            self._write_run(a,'qwen_zeroshot_local',0)
            self._write_run(b,'qwen_semantic_local_v1',1)
            s, rows = compare_runs(a,b, bootstrap=300)
            self.assertEqual(s['n_paired'],2)
            self.assertEqual(s['admission']['paired_delta_semantic_minus_baseline'], 1.0)
            self.assertEqual(s['mean_abs_action_delta'], 1.0)
            self.assertEqual(len(rows), 2)

    def test_rejects_different_seed(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'base',Path(t)/'sem'
            self._write_run(a,'qwen_zeroshot_local',0)
            self._write_run(b,'qwen_semantic_local_v1',1,seed=2)
            with self.assertRaises(ValueError):
                compare_runs(a,b)




class SyntheticEnv:
    def __init__(self):
        self.sample_dict = {0: tuple(SimpleNamespace(tag=a) for a in
                                      ('agent_k1','agent_k2','agent_k3','agent_g','graph'))}
        self.sample_idx = 0
        self.observations = None
        self.infos = None
        self.g_done_list = []
        self.dict_eval_index = {name: [] for name in base.METRICS}

    def _get_overlap_node_link(self, k_net, g_net):
        return [1, 3, 4], [0, 2], []

    def _get_v_net_observations(self, net, nodes, links, pad):
        return compact(net.tag, nodes, links)

    def _get_g_net_observations(self, pad):
        return compact('agent_g')

    def reset(self):
        return self.observations, self.infos

    def step(self, actions):
        assert set(actions) == set(AGENTS)
        assert all(len(x) == 4 for x in actions.values())
        self.g_done_list.append(True)
        for name in base.METRICS:
            self.dict_eval_index[name].append({
                'V': 320, 'V_g':20, 'V_k1':100, 'V_k2':100, 'V_k3':100,
                'V_k1_0': 99, 'V_k2_0':99, 'V_k3_0':99,
                'cost_k1':3, 'cost_k2':4, 'cost_k3':5,
            }.get(name, 0.0))
        return {}, {a: 1.0 for a in AGENTS}, {}, False, {}


class FullPipelineTests(unittest.TestCase):
    def test_mock_pipeline_and_output_artifacts(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'semantic_mock'
            args = get_args(['--dataset','mini','--policy','mock','--count','1',
                             '--output-dir',str(out)])
            res = run(args, env_override=SyntheticEnv(), mapping_override=[512])
            self.assertEqual(res['n_scored'],1)
            self.assertEqual(res['policy'], 'semantic_mock_not_an_llm_result')
            rows = [json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['original_scene_id'], 512)
            self.assertIn('LOCAL ACTION SEMANTICS', rows[0]['per_role']['agent_k2']['messages'][1]['content'])
            self.assertEqual(json.loads((out/'manifest.json').read_text())['scene_ids'], [512])
            self.assertTrue((out/'metrics.csv').is_file())


if __name__ == '__main__':
    unittest.main()
