"""Offline P0-v4 tests (no LLM API, no training). Run in user's marl environment.

Includes REAL source GEANT sample tests using original main.py and the
P0_small_samples.pkl already present in the project root.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
from p0_guard import HardAdmissionGuard,NoFeasibleAction,_group_lookup,_verify_with_untouched_env,_solve_projected_lp,_vec_from_action
from p0_guard_runner import _get_record_actions
from p0_policy import AGENTS,ratio_actions
from p0_runner import prepare_config,stage_scene

REAL_RAW = {
    0: {'agent_k1':[0.8,0.3,0.,0.], 'agent_k2':[0.7,0.4,0.,0.],
        'agent_k3':[0.6,0.5,0.,0.], 'agent_g':[0.3,0.4,0.2]},
    1: {'agent_k1':[0.6,0.3,0.,0.], 'agent_k2':[0.7,0.4,0.,0.],
        'agent_k3':[0.8,0.5,0.,0.], 'agent_g':[0.3,0.2,0.]},
}

class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent
        from main import CustomMultiAgentEnv
        from tempfile import mkdtemp
        import atexit,shutil
        cls.tempdir=Path(mkdtemp(prefix='p0_guard_test_'))
        atexit.register(shutil.rmtree,str(cls.tempdir),ignore_errors=True)
        cfg=prepare_config(cls.root,cls.root/'P0_small_samples.pkl',5,cls.tempdir,10)
        cls.env=CustomMultiAgentEnv(cfg)
        cls.env.is_eval=True

    def test_overlap_slot_group_matches_source(self):
        self.assertEqual(_group_lookup([2,4,6,9,10]),{2:0,4:0,6:0,9:1,10:1})
        self.assertEqual(_group_lookup([6]),{6:0})
        self.assertEqual(_group_lookup([]),{})

    def test_real_scenes_are_repaired_and_shadow_does_not_mutate_live_environment(self):
        for scene,ratios in REAL_RAW.items():
            with self.subTest(scene=scene):
                stage_scene(self.env,scene)
                phys=self.env.DG
                before_alloc_comp=[float(phys.nodes[n]['allocated_comp']) for n in phys.nodes]
                before_alloc_bw=[float(phys[u][v]['allocated_bandw']) for u,v in phys.edges]
                source_sample=self.env.sample_dict[scene][4]
                before_source_comp=[float(source_sample.nodes[n]['allocated_comp']) for n in source_sample.nodes]
                raw=ratio_actions(ratios)
                result=HardAdmissionGuard().check_and_repair(self.env,scene,raw,1+scene)
                self.assertFalse(result.raw_check['accepted'])
                self.assertTrue(result.checked['accepted'])
                self.assertTrue(result.guard_changed)
                self.assertGreater(result.checked['V'],0)
                self.assertFalse(any(abs(x-y)>1e-9 for x,y in zip(
                    before_alloc_comp,[float(phys.nodes[n]['allocated_comp']) for n in phys.nodes])))
                self.assertFalse(any(abs(x-y)>1e-9 for x,y in zip(
                    before_alloc_bw,[float(phys[u][v]['allocated_bandw']) for u,v in phys.edges])))
                self.assertEqual(before_source_comp,[float(source_sample.nodes[n]['allocated_comp']) for n in source_sample.nodes])
                print('REAL GEANT SCENE',scene,'raw=FAIL repaired=PASS V={:.3f}'.format(result.checked['V']))

    def test_already_feasible_action_is_unchanged(self):
        stage_scene(self.env,0)
        action=ratio_actions({a:[0.35]*(3 if a=='agent_g' else 4) for a in AGENTS})
        result=HardAdmissionGuard().check_and_repair(self.env,0,action,seed=1)
        self.assertTrue(result.raw_check['accepted'])
        self.assertFalse(result.guard_changed)
        self.assertEqual(result.dry_runs,1)
        for k in AGENTS:
            np.testing.assert_array_equal(result.action[k],action[k])

    def test_fail_closed_when_no_solution(self):
        stage_scene(self.env,0)
        action=ratio_actions(REAL_RAW[0])
        # Fabricate a provably impossible constraint A x <= -1 with A = 0.
        with patch('p0_guard._original_scene_constraints',return_value=(
                np.zeros((1,13)),np.array([-1.0]),['UNSAT'],1.0)):
            with self.assertRaises(NoFeasibleAction):
                HardAdmissionGuard().check_and_repair(self.env,0,action,seed=1)

    def test_fail_closed_when_lp_and_original_step_disagree(self):
        stage_scene(self.env,0)
        action=ratio_actions(REAL_RAW[0])
        # Never execute original infeasible action when solver prediction is wrong.
        with patch('p0_guard._solve_projected_lp',return_value=(action,{'status':'fake_lp'})):
            with self.assertRaises(NoFeasibleAction):
                HardAdmissionGuard().check_and_repair(self.env,0,action,seed=1)

    def test_logs_cannot_replay_heuristic_fallback(self):
        with TemporaryDirectory() as td:
            file=Path(td)/'records.jsonl'
            file.write_text(json.dumps({'scene_index':0,'raw_response':'{}','used_fallback':True})+'\n')
            with self.assertRaises(ValueError):
                _get_record_actions(file)

if __name__=='__main__':
    unittest.main(verbosity=2)
