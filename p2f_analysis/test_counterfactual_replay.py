"""Contract tests for cached-action interventions; no GEANT or API calls."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('qwen_replay', Path(__file__).with_name('val5601_counterfactual_replay.py'))
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


class ReplayContracts(unittest.TestCase):
    def setUp(self):
        self.m = {0: {'env_actions': {r: [-.8, -.6, -.4, -.2] for r in replay.ROLES}}}
        self.q = {0: {'env_actions': {r: [.1, .3, .5, .7] for r in replay.ROLES}}}

    def test_node_link_interventions_preserve_qwen_g_and_source_values(self):
        node = replay.chosen_actions(self.m, self.q, 'Q_NODE_M', 0)
        link = replay.chosen_actions(self.m, self.q, 'Q_LINK_M', 0)
        for r in replay.K_ROLES:
            self.assertEqual(node[r], [-.8, -.6, .5, .7])
            self.assertEqual(link[r], [.1, .3, -.4, -.2])
        self.assertEqual(node['agent_g'], [.1, .3, .5, .7])
        self.assertEqual(link['agent_g'], [.1, .3, .5, .7])
        node['agent_k1'][0] = 0
        self.assertEqual(self.m[0]['env_actions']['agent_k1'][0], -.8)

    def test_role_swap_preserves_complete_cached_g_vector(self):
        for case, k, g in [('MQ', self.m, self.q), ('QM', self.q, self.m)]:
            got = replay.chosen_actions(self.m, self.q, case, 0)
            for r in replay.K_ROLES:
                self.assertEqual(got[r], k[0]['env_actions'][r])
            self.assertEqual(got['agent_g'], g[0]['env_actions']['agent_g'])

    def test_controls_reject_ir_flip_even_when_V_matches(self):
        original = {k: True for k in replay.BOOLEANS}
        original.update({k: 1. for k in replay.NUMERIC})
        self.assertEqual(replay.control_diff(original, dict(original)), [])
        wrong = dict(original, individual_rationality=False)
        self.assertTrue(replay.control_diff(original, wrong))
        wrong = dict(original, V_k2=.99)
        self.assertTrue(replay.control_diff(original, wrong))


if __name__ == '__main__':
    unittest.main()
