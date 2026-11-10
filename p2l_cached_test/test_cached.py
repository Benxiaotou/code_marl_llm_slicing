import copy
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cached as run


class CachedReplayContractTest(unittest.TestCase):
    def test_exact_existing_paths_cover_every_scene_once(self):
        job = next(j for j in run.JOBS if j.tag == 'p2a_std1000')
        paths = run.record_paths(6601, job)
        self.assertEqual(len(paths), 16)
        self.assertEqual(paths[0].name, 'chunk_0000_032')
        self.assertEqual(paths[-1].name, 'chunk_0480_032')
        self.assertEqual(len(set(paths)), 16)

    def test_metadata_rejects_unpaired_dataset(self):
        dataset = {'dataset_id': 'p2c_test_seed6601', 'sample_sha256': 'abc', 'seed': 6601}
        job = next(j for j in run.JOBS if j.tag == 'knn_std1000')
        meta = {'method': job.method, 'eval_dataset_id': 'different', 'eval_dataset_split': 'test',
                'eval_dataset_sha256': 'abc', 'eval_generation_seed': 6601,
                'scene_ids': list(range(32)), 'seed': 1,
                'config_sha256': run.v.replay.sha256(run.ROOT / 'configs_main.yaml'),
                'main_py_sha256': run.v.replay.sha256(run.ROOT / 'main.py'), 'mock_not_real_llm': False}
        with self.assertRaises(AssertionError):
            run.frozen_meta_matches(meta, dataset, job, 0)

    def test_frozen_postprocessor_shared_actions_are_not_mutated(self):
        obs = np.zeros(240); obs[:60].reshape(12, 5)[:] = [1, .3, .7, .2, .8]
        observations = {r: obs for r in run.v.K}
        raw = {r: [0., 0., -.4, .4] for r in run.v.ROLES}; before = copy.deepcopy(raw)
        link, node, _ = run.v.derived_actions(raw, observations, {r: [.8, .2] for r in run.v.K})
        self.assertEqual(raw, before)
        self.assertEqual(node['agent_g'], raw['agent_g'])
        for role in run.v.K:
            self.assertEqual(node[role][2:], link[role][2:])
            self.assertEqual(link[role][:2], raw[role][:2])
            self.assertTrue(all(0 <= node[role][j] - raw[role][j] <= .05 for j in range(2)))


if __name__ == '__main__':
    unittest.main()
