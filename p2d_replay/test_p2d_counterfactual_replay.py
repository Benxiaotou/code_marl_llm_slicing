"""Run: python -m unittest -v test_p2d_counterfactual_replay.py
Tests do not import GEANT, teacher checkpoints or make network requests.
"""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
import p2d_counterfactual_replay as replay

ZIP='/mnt/data/GPT61_MAPPO_10scene_diagnosis.zip'  # available in analysis workspace only

class FakeEnv:
    def __init__(self):
        self.calls=[]
    def step(self,joint):
        self.calls.append(joint)
        return {'ok':True}

class FakePolicy:
    def set_scene(self,scene):self.scene=scene

class FakeRunner:
    @staticmethod
    def run_one_scene(env,policy,scene,index,seed):
        incoming={role:np.asarray([0,0,0,0],dtype=np.float32) for role in replay.ROLES}
        output=env.step(incoming)
        assert output['ok']
        return {'status':'ok','original_scene_id':scene,'env_actions':{
            role:list(v) for role,v in incoming.items()},
            'per_role':{},'metrics':{'g_done':False,'cooperative_valid':False}}

class ReplayUnitTests(unittest.TestCase):
    def test_joint_action_dict_numpy(self):
        sample={r:np.array([.1,.2,.3,.4],dtype=np.float32) for r in replay.ROLES}
        override={r:[-.1,.0,.2,.8] for r in replay.ROLES}
        received=replay.rewrite_joint_action(sample,override)
        self.assertTrue(np.allclose(received['agent_g'],override['agent_g']))
        self.assertEqual(received['agent_k1'].dtype,np.float32)
        self.assertNotEqual(float(received['agent_k1'][0]),float(sample['agent_k1'][0]))
    def test_joint_action_list(self):
        sample=[[0.,0.,0.,0.]]*4
        override={r:[.1,.2,.3,.4] for r in replay.ROLES}
        received=replay.rewrite_joint_action(sample,override)
        self.assertEqual(received[3],[.1,.2,.3,.4])
    def test_joint_action_array(self):
        sample=np.ones((4,4),dtype=np.float32)
        override={r:[.1,.2,.3,.4] for r in replay.ROLES}
        received=replay.rewrite_joint_action(sample,override)
        self.assertTrue(np.allclose(received[3],[.1,.2,.3,.4]))
    def test_bad_shape_blocks(self):
        with self.assertRaises(ValueError):
            replay.rewrite_joint_action({'agent_k1':np.zeros(4)},
                                         {r:[.1,.2,.3,.4] for r in replay.ROLES})
    def test_control_diff(self):
        expected={'g_done':True,'cooperative_valid':True,'individual_rationality':True,'V':100.,'cost_sum':20.}
        self.assertFalse(replay.control_diff(expected,dict(expected)))
        wrong=dict(expected,V=99.,cooperative_valid=False)
        self.assertEqual(len(replay.control_diff(expected,wrong)),2)
    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'x.csv'
            replay.csv_write(path,[{'scene':1,'V':5.2}],['scene','V'])
            self.assertIn('5.2',path.read_text(encoding='utf-8-sig'))
    def test_fake_environment_injection_and_metrics_provenance(self):
        m={r:[0.,.1,.2,.3] for r in replay.ROLES}
        g={r:[.9,.8,.7,.6] for r in replay.ROLES}
        def fake_record(actions):
            return {'env_actions':actions}
        mm={0:fake_record(m)}
        gg={0:fake_record(g)}
        env=FakeEnv()
        policy=FakePolicy()
        result=replay.replay_one(env,policy,0,1,(mm,gg),'MG',FakeRunner)
        self.assertEqual(len(env.calls),1)
        self.assertTrue(np.allclose(env.calls[0]['agent_k2'],m['agent_k2']))
        self.assertTrue(np.allclose(env.calls[0]['agent_g'],g['agent_g']))
        self.assertEqual(result['action_sources']['agent_g'],'GPT')
        self.assertEqual(result['action_sources']['agent_k1'],'MAPPO')
        self.assertIsNone(result['policy_latency_s'])
    @unittest.skipUnless(Path(ZIP).is_file(),'User-provided archive is only available in analysis workspace')
    def test_real_duplicate_zip_entries_are_parsed_by_zipinfo(self):
        mm,gg,info=replay.load_archive(Path(ZIP))
        self.assertEqual(len(mm),10)
        self.assertEqual(len(gg),10)
        self.assertEqual(info['recorded_mappo_accept_count'],6)
        self.assertEqual(info['recorded_gpt_accept_count'],0)
        self.assertEqual(len(set(info['gpt_all_agent_g_fourth_values'])),1)
        self.assertEqual(info['gpt_all_agent_g_fourth_values'][0],0.0)

    def test_compatibility_hook_runs_on_legacy_main_before_teacher(self):
        import types
        from unittest import mock
        parent = sys.modules['__main__']
        old_present = hasattr(parent,'InitVnetworkTrain')
        old_value = getattr(parent,'InitVnetworkTrain',None)
        fake = types.ModuleType('p1a_runner')
        observed = []
        def fake_config(root, config_file):
            observed.append(('config',str(config_file)))
            return types.SimpleNamespace(id='env_main'), object()
        def fake_compat(env_main):
            observed.append(('shim',env_main.id))
            setattr(parent,'InitVnetworkTrain',object())
        fake._config = fake_config
        fake._compat_pickle_main_classes = fake_compat
        try:
            with mock.patch.dict(sys.modules,{'p1a_runner':fake}):
                replay.ensure_legacy_pickle_compat(Path('/project'),'configs_main.yaml')
            self.assertEqual(observed,[('config',str(Path('/project')/'configs_main.yaml')),
                                       ('shim','env_main')])
        finally:
            if old_present:setattr(parent,'InitVnetworkTrain',old_value)
            elif hasattr(parent,'InitVnetworkTrain'):delattr(parent,'InitVnetworkTrain')

if __name__=='__main__':unittest.main()
