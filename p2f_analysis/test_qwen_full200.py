"""Offline contracts: g unchanged, complete fresh actions, fail-stop journaling."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location('full200', HERE / 'qwen_full200.py')
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class FullPolicyContracts(unittest.TestCase):
    def test_only_existing_role_system_prompt_changes(self):
        original = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'own state'}]
        for role in run.replay.ROLES:
            baseline = run.altered_messages(original, role, run.ARMS[0], 'candidate')
            candidate = run.altered_messages(original, role, run.ARMS[1], 'candidate')
            self.assertEqual(baseline, original)
            self.assertEqual(candidate[1], original[1])
            self.assertEqual(candidate[0]['content'], 'rules' if role == 'agent_g' else 'rules\ncandidate')
        self.assertEqual(original[0]['content'], 'rules')

    def test_candidate_is_exact_pilot_prompt(self):
        prompt = run.PROMPT.read_text(encoding='utf-8').strip()
        self.assertEqual(run.text_sha(prompt), run.FROZEN_PROMPT_SHA)

    def test_audited_success_validates_full_dimensions_and_preserves_usage(self):
        class FakeClient:
            def complete(self, role, messages):
                ratios = [.91, .82, .73] if role == 'agent_g' else [.91, .82, .73, .64]
                return SimpleNamespace(text=json.dumps({'ratios': ratios}), finish_reason='stop',
                                       usage={'prompt_tokens': 20, 'completion_tokens': 8})
        from p1a.policy import parse_ratios, to_env_action
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as f:
            client = run.AuditedClient(FakeClient(), f)
            client.arm, client.scene = run.ARMS[1], 0
            for role in run.replay.ROLES:
                response = client.complete(role, [{'role': 'user', 'content': 'own'}])
                ratios = parse_ratios(response.text, role)
                action = to_env_action(role, ratios)
                self.assertAlmostEqual(float(action[0]), .82, places=6)
                self.assertAlmostEqual(float(action[1]), .64, places=6)
                if role == 'agent_g':
                    self.assertEqual(float(action[3]), 0)
                else:
                    self.assertAlmostEqual(float(action[3]), .28, places=6)
            self.assertEqual(client.count, 4)
            f.seek(0)
            rows = [json.loads(s) for s in f]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r['usage']['completion_tokens'] == 8 for r in rows if r['event'] == 'finished'))

    def test_invalid_response_is_not_retried_and_usage_is_preserved(self):
        class FakeClient:
            calls = 0
            def complete(self, role, messages):
                self.calls += 1
                return SimpleNamespace(text='not JSON', finish_reason='stop', usage={'total_tokens': 17})
        fake = FakeClient()
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as f:
            client = run.AuditedClient(fake, f)
            client.arm, client.scene = run.ARMS[0], 4
            with self.assertRaises(RuntimeError):
                client.complete('agent_k1', [{'role': 'user', 'content': 'own'}])
            self.assertEqual(fake.calls, 1)
            self.assertEqual(client.count, 0)
            f.seek(0)
            rows = [json.loads(s) for s in f]
            self.assertEqual([r['event'] for r in rows], ['started', 'failed'])
            self.assertEqual(rows[-1]['usage']['total_tokens'], 17)
            self.assertTrue(rows[-1]['no_retry'])


if __name__ == '__main__':
    unittest.main()
