import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p2e_prompt_3way as p
import gpt61_sol_dev as adapter


class RuntimeSafetyTests(unittest.TestCase):
    def state(self, folder, **changes):
        options = dict(upstream='https://unused.invalid/v1/chat/completions',
                       key='OFFLINE_TEST_SECRET', api='chat', model='gpt-6.1-sol',
                       effort='medium', max_output_tokens=8192,
                       ledger=Path(folder) / 'usage.jsonl', prices=adapter.PRICES,
                       budget=2, max_calls=2, timeout=30)
        options.update(changes)
        return adapter.ProxyState(**options)

    def request(self):
        return {'messages': [{'role': 'system', 'content': 'offline test'}], 'stream': False}

    def response(self, **changes):
        response = {'choices': [{'message': {'content': '{"ratios":[0.2,0.2,0.2,0.2]}'}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
        response.update(changes)
        return response

    def test_unknown_cost_blocks_without_calling_upstream(self):
        with tempfile.TemporaryDirectory() as folder:
            state = self.state(folder)
            adapter.append_jsonl(state.ledger, {'status': 'error', 'cost_unknown': True})
            with patch.object(adapter, 'invoke_upstream') as invoke:
                with self.assertRaisesRegex(RuntimeError, 'unknown cost'):
                    state.handle(self.request())
                invoke.assert_not_called()

    def test_crash_leaves_durable_attempt_and_blocks_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            state = self.state(folder)
            def crash(*args, **kwargs):
                self.assertEqual(adapter.read_events(state.attempt_ledger)[0]['status'], 'started')
                raise KeyboardInterrupt()
            with patch.object(adapter, 'invoke_upstream', side_effect=crash):
                with self.assertRaises(KeyboardInterrupt):
                    state.handle(self.request())
            restarted = self.state(folder)
            with patch.object(adapter, 'invoke_upstream') as invoke:
                with self.assertRaisesRegex(RuntimeError, 'Unresolved API attempt'):
                    restarted.handle(self.request())
                invoke.assert_not_called()

    def test_known_usage_survives_unusable_response(self):
        with tempfile.TemporaryDirectory() as folder:
            state = self.state(folder)
            with patch.object(adapter, 'invoke_upstream', return_value=self.response(choices=[])):
                with self.assertRaisesRegex(RuntimeError, 'Chat response missing'):
                    state.handle(self.request())
            usage = adapter.totals(adapter.read_events(state.ledger))
            self.assertEqual(usage['requests_ok'], 0)
            self.assertEqual(usage['cost_unknown_requests'], 0)
            self.assertEqual((usage['input_tokens'], usage['output_tokens']), (100, 20))
            self.assertGreater(usage['cost_usd_estimate'], 0)

    def test_success_attempt_limit_and_budget_prevent_extra_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            state = self.state(folder, max_calls=1)
            with patch.object(adapter, 'invoke_upstream', return_value=self.response()) as invoke:
                state.handle(self.request())
                with self.assertRaisesRegex(RuntimeError, 'Maximum 1'):
                    state.handle(self.request())
                self.assertEqual(invoke.call_count, 1)
            self.assertNotIn(state.key, state.ledger.read_text() + state.attempt_ledger.read_text())
        with tempfile.TemporaryDirectory() as folder:
            state = self.state(folder, budget=0.001)
            with patch.object(adapter, 'invoke_upstream') as invoke:
                with self.assertRaisesRegex(RuntimeError, 'Budget safety reserve'):
                    state.handle(self.request())
                invoke.assert_not_called()
                self.assertFalse(state.attempt_ledger.exists())

    def test_proxy_token_only_changes_child_environment(self):
        with patch.dict(os.environ, {'LLM_API_KEY': 'QWEN_SAVED_TEST_VALUE'}):
            child = p.runner_environment()
            self.assertEqual(os.environ['LLM_API_KEY'], 'QWEN_SAVED_TEST_VALUE')
            self.assertEqual(child['LLM_API_KEY'], 'p2e-local-proxy-token')

    def test_comparison_exports_real_metric_fields_and_preserves_reports(self):
        rows = []
        for scene in range(10):
            rows.append({'status': 'ok', 'original_scene_id': scene, 'policy_latency_s': 1,
                         'metrics': {'g_done': True, 'cooperative_valid': True,
                                     'individual_rationality': True, 'V': 100, 'V_g': 10,
                                     'cost_sum': 3, 'V_k1': 12, 'V_k1_0': 10,
                                     'V_k2': 15, 'V_k2_0': 10, 'V_k3': 11, 'V_k3_0': 10},
                         'per_role': {role: {'ratios': [0.2, 0.4, 0.6, 0.8]}
                                      for role in ('agent_k1', 'agent_k2', 'agent_k3')}})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(p, 'collect_baseline', return_value=rows), \
                 patch.object(p, 'collect_mappo', return_value=rows), \
                 patch.object(p, 'collect_new', return_value=None):
                p.compare(root, 'report')
                with (root / 'report/paired_10scene_metrics.csv').open(encoding='utf-8-sig') as fp:
                    metrics = list(csv.DictReader(fp))
                self.assertEqual(len(metrics), 20)
                self.assertEqual(float(metrics[0]['k1_ir_margin']), 2)
                self.assertAlmostEqual(float(metrics[0]['node_ratio_mean']), 0.3)
                result = json.loads((root / 'report/comparison.json').read_text())
                self.assertEqual(result['results'][0]['IR_violation_scenes'], 0)
                with self.assertRaises(FileExistsError):
                    p.compare(root, 'report')


if __name__ == '__main__':
    unittest.main()
