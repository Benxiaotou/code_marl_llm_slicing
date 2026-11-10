import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_validation as run


class ValidationContractTest(unittest.TestCase):
    def test_shared_draw_keeps_g_and_separates_node_link_changes(self):
        obs = np.zeros(240)
        obs[:60].reshape(12, 5)[:] = [1, .3, .7, .2, .8]
        observations = {r: obs for r in run.ROLES}
        original = {r: [0., 0., -.4, .4] for r in run.ROLES}
        targets = {r: [.8, .2] for r in run.K}
        link, node, _ = run.derived_actions(original, observations, targets)
        self.assertEqual(original['agent_k1'], [0., 0., -.4, .4])
        self.assertEqual(node['agent_g'], original['agent_g'])
        self.assertEqual(link['agent_k1'][:2], original['agent_k1'][:2])
        self.assertTrue(np.allclose(link['agent_k1'][2:], [.1, -.1]))
        self.assertEqual(node['agent_k1'][2:], link['agent_k1'][2:])
        self.assertTrue(all(0 < x <= .05 for x in node['agent_k1'][:2]))

    def test_bootstrap_preserves_seed_counts_with_repeated_scene_indices(self):
        labels = [20] * 7 + [10] * 3
        idx = run.bootstrap_indices(labels, reps=30)
        self.assertEqual(idx.shape, (30, 10))
        selected = np.asarray(labels)[idx]
        self.assertTrue(np.all((selected == 20).sum(axis=1) == 7))
        self.assertTrue(np.all((selected == 10).sum(axis=1) == 3))
        self.assertTrue(np.array_equal(idx, run.bootstrap_indices(labels, reps=30)))

    def test_pairing_requires_seed_and_scene_identity(self):
        a = [{'dataset_seed': 1, 'original_scene_id': 0}]
        b = [{'dataset_seed': 2, 'original_scene_id': 0}]
        with self.assertRaises(AssertionError):
            run.paired(a, b)

    def test_response_parser_and_g_padding_match_original_contract(self):
        from p1a.policy import parse_ratios, to_env_action
        ratios = parse_ratios('{"ratios":[0.2,0.3,0.4]}', 'agent_g')
        self.assertTrue(np.allclose(to_env_action('agent_g', ratios), [-.6, -.4, -.2, 0.]))


if __name__ == '__main__':
    unittest.main()
