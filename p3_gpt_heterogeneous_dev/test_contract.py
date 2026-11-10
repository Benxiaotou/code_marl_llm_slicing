"""Offline checks for paid-call accounting and four-arm attribution."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import run_dev20 as m


class Contracts(unittest.TestCase):
    def item(self):
        return {'scene': 0, 'roles': {r: {'messages': [
            {'role': 'system', 'content': 'same base prompt'},
            {'role': 'user', 'content': 'local data'}]} for r in m.ROLES}}

    def response(self, text='{"ratios":[0.2,0.3,0.4,0.5]}'):
        return {'model': m.MODEL_GPT, 'choices': [{'finish_reason': 'stop',
                'message': {'content': text}}], 'usage': {'prompt_tokens': 200,
                'completion_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 80},
                'completion_tokens_details': {'reasoning_tokens': 60}}}

    def test_usage_includes_cached_and_reasoning_once(self):
        self.assertEqual(m.usage_total(self.response()['usage']), 300)
        self.assertIsNone(m.usage_total(None))
        self.assertIsNone(m.usage_total({'total_tokens': 4}))

    def test_assignments_are_factorial(self):
        self.assertEqual([m.ASSIGNMENTS[a]['agent_g'] for a in m.ASSIGNMENTS],
                         ['qwen', 'gpt', 'gpt', 'qwen'])
        self.assertEqual([m.ASSIGNMENTS[a]['agent_k1'] for a in m.ASSIGNMENTS],
                         ['qwen', 'gpt', 'qwen', 'gpt'])

    def test_g_padding_and_strict_dimensions(self):
        self.assertEqual(m.to_env_action('agent_g', [.1, .2, .3]).shape, (4,))
        self.assertEqual(m.to_env_action('agent_g', [.1, .2, .3])[3], 0)
        with self.assertRaises(ValueError):
            m.parse_ratios('{"ratios":[0.1,0.2,0.3,0.4]}', 'agent_g')

    def test_postprocessor_exact_match(self):
        from p2j_validation.run_validation import derived_actions
        actions = {r: [-.6, -.3, .2, .6] for r in m.ROLES}
        actions['agent_g'][3] = 0
        obs = {r: np.zeros(240) for r in m.ROLES}
        for r in m.K_ROLES:
            obs[r][:60] = np.tile([1, .7, .5, .7, .6], 12)
        targets = {r: [.1, .3] for r in m.K_ROLES}
        final, audit = m.fixed_postprocess(actions, obs, targets)
        link, both, node = derived_actions(actions, obs, targets)
        self.assertEqual(final, both)
        self.assertEqual(audit, {'link_actions': link, 'node_audit': node})
        self.assertEqual(final['agent_g'], actions['agent_g'])
        for r in m.K_ROLES:
            delta = (np.array(final[r][:2]) - actions[r][:2]) / 2
            self.assertTrue((delta >= 0).all())
            self.assertTrue((delta <= .025 + 1e-12).all())

    def test_prompt_content_survives_transport(self):
        msgs = self.item()['roles']['agent_k1']['messages']
        body = m.adapter.transform_request({'messages': msgs}, api='chat', model=m.MODEL_GPT,
                                           effort='medium', max_output_tokens=8192)
        self.assertEqual([x['content'] for x in body['messages']], [x['content'] for x in msgs])
        self.assertEqual(body['messages'][0]['role'], 'developer')

    @patch.dict(os.environ, {'GPT61_API_KEY': 'offline-test-only', 'GPT61_BASE_URL': 'https://example.invalid/v1'})
    def test_paid_usage_recorded_and_no_double_count(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m.adapter, 'invoke_upstream', return_value=self.response()) as invoke:
            client = m.AuditedCalls(d, 'gpt')
            result = client.call(self.item(), 'agent_k1')
            self.assertEqual(invoke.call_count, 1)
            self.assertEqual(client.tokens, 300)
            self.assertEqual(client.reserved, 0)
            self.assertAlmostEqual(result['estimated_units'], 300 / 5_000_000)

    @patch.dict(os.environ, {'GPT61_API_KEY': 'offline-test-only', 'GPT61_BASE_URL': 'https://example.invalid/v1'})
    def test_invalid_action_preserves_billed_usage_and_raw_reply(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m.adapter, 'invoke_upstream', return_value=self.response('invalid')) as invoke:
            client = m.AuditedCalls(d, 'gpt')
            with self.assertRaises(RuntimeError):
                client.call(self.item(), 'agent_k1')
            self.assertEqual(invoke.call_count, 1)
            self.assertEqual(client.tokens, 300)
            self.assertTrue(client.stop.is_set())
            self.assertTrue((Path(d) / 'gpt_raw_replies.jsonl').is_file())
            events = [json.loads(x) for x in (Path(d) / 'gpt_requests.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['total_tokens'], 300)
            self.assertFalse(events[-1]['cost_unknown'])

    @patch.dict(os.environ, {'GPT61_API_KEY': 'offline-test-only', 'GPT61_BASE_URL': 'https://example.invalid/v1'})
    def test_budget_reserves_inflight_before_network(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m.adapter, 'invoke_upstream') as invoke:
            client = m.AuditedCalls(d, 'gpt')
            client.reserved = m.TOKEN_BUDGET - 100
            with self.assertRaisesRegex(RuntimeError, 'TOKEN_BUDGET'):
                client.call(self.item(), 'agent_k1')
            invoke.assert_not_called()
            self.assertEqual(client.attempts, 0)

    @patch.dict(os.environ, {'GPT61_API_KEY': 'offline-test-only', 'GPT61_BASE_URL': 'https://example.invalid/v1'})
    def test_unknown_usage_stops_without_retry(self):
        reply = self.response()
        del reply['usage']
        with tempfile.TemporaryDirectory() as d, patch.object(m.adapter, 'invoke_upstream', return_value=reply) as invoke:
            client = m.AuditedCalls(d, 'gpt')
            with self.assertRaises(RuntimeError):
                client.call(self.item(), 'agent_k1')
            with self.assertRaises(RuntimeError):
                client.call(self.item(), 'agent_k2')
            self.assertEqual(invoke.call_count, 1)
            self.assertTrue(client.unknown_cost)


if __name__ == '__main__':
    unittest.main()
