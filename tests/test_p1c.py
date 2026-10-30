import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from p1a.local_obs import AGENTS
from p1a.policy import MockClient, Completion
from p1b.core import ContractError, save_json
from p1c.retrieval import LocalHistory, local_numeric_summary
from p1c.policy import RetrievalPolicy, make_icl_messages
from p1c_runner import get_args, run
from p1c_compare import compare


def sample_obs(role, v, overlap=5):
    nodes = np.zeros((12,5),dtype=float)
    nodes[:,1:] = v
    links = np.zeros((36,3),dtype=float)
    edges = np.zeros((36,2),dtype=float)
    if role != 'agent_g':
        nodes[:overlap,0]=1
        links[:7,1] = .3 + v / 5
        links[:7,2] = v
        links[:3,0] = 1
    else:
        links[:7,1] = .1 + v / 4
        links[:7,2] = v
    return np.concatenate((nodes.flatten(),links.flatten(),edges.flatten())).tolist()


def sample_library(root, *, n=6):
    lib = root / 'p1b_libraries' / 'train_v1'
    (lib/'role_libraries').mkdir(parents=True)
    for role in AGENTS:
        with (lib / 'role_libraries' / (role + '.jsonl')).open('w',encoding='utf-8') as f:
            for i in range(n):
                ratios = [round(0.1 + 0.12*i, 6)] * (3 if role=='agent_g' else 4)
                r = {'schema':'p1b_local_case_v1', 'role':role,
                     'experience_id':'p1b_train_seed2601::{:06d}'.format(i),
                     'observation_240': sample_obs(role, .15 + i*.10),
                     'own_teacher_ratios':ratios,
                     'own_teacher_env_action':[2*x-1 for x in ratios],
                     'source':'frozen_MAPPO_selected_joint_cooperation'}
                f.write(json.dumps(r)+'\n')
    save_json(lib/'library_manifest.json', {
        'schema':'p1b_library_manifest_v1', 'dataset_id':'p1b_train_seed2601',
        'training_sample_sha256':'traindata_hash', 'development_only':False,
        'teacher_provenance':{'verified_mappo':True},
        'policy_used_during_collection':'frozen_MAPPO_only',
        'no_current_global_information':True, 'n_experiences_per_role':n})
    save_json(lib/'audit_report.json',{'audit_passed':True, 'dataset_id':'p1b_train_seed2601',
        'no_global_fields_in_role_libraries':True,'matched_joint_case_ids':True})
    return lib


class FakeEnv:
    def __init__(self):
        self.sample_dict={i:tuple(SimpleNamespace(tag=a,sc=i) for a in (*AGENTS,'DG')) for i in range(8)}
        self.sample_idx=0
        self.g_done_list=[]
        self.dict_eval_index={k:[] for k in ('V','V_k1','V_k2','V_k3','V_g',
            'V_k1_0','V_k2_0','V_k3_0','X_k1','X_k2','X_k3',
            'cost_k1','cost_k2','cost_k3','revenue_k1','revenue_k2','revenue_k3','revenue_g')}
    def _get_overlap_node_link(self, k_net, g_net):
        return [0,1,2],[0,1],[]
    def _get_v_net_observations(self, k, nodes, links, padding):
        return np.array(sample_obs(k.tag, .15 + k.sc * .06), dtype=np.float32)
    def _get_g_net_observations(self, padding):
        return np.array(sample_obs('agent_g',.15 + self.sample_idx*.06),dtype=np.float32)
    def reset(self):
        return self.observations,self.infos
    def step(self, actions):
        assert set(actions) == set(AGENTS)
        mean_ratio = float(np.mean([(float(v[0])+1)/2 for v in actions.values()]))
        admit = mean_ratio > .19
        m = {'V':300+10*mean_ratio,'V_k1':100,'V_k2':101,'V_k3':102,
             'V_g':20 if admit else 0,'V_k1_0':95,'V_k2_0':95,'V_k3_0':95,
             'cost_k1':mean_ratio,'cost_k2':mean_ratio,'cost_k3':mean_ratio}
        self.g_done_list.append(admit)
        for k in self.dict_eval_index:
            self.dict_eval_index[k].append(m.get(k,0.0))
        return {},{r:.1 for r in AGENTS},{},False,{}


class FakeClient:
    model = 'qwen3-next-80b-a3b-instruct'
    def complete(self, role, messages):
        if role == 'agent_g':
            ratios = [0.2, 0.3, 0.4]
        else:
            ratios = [0.15, 0.25, 0.35, 0.45]
        return Completion(json.dumps({'ratios':ratios}), 0.005, 'stop', None)


class FakeTeacher:
    source_info = {'checkpoint_sha256':'synthetic_test'}
    def act(self, obs, *, original_scene_id, seed):
        return {r: np.array([-.4,-.3,-.2,-.1],dtype=np.float32) for r in AGENTS}
    def close(self):
        pass


class TestP1CLocalRetrieval(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.lib=sample_library(self.root)
    def tearDown(self): self.tmp.cleanup()

    def test_numeric_retrieval_uses_only_own_observation(self):
        h=LocalHistory(self.lib)
        for role in AGENTS:
            xs=h.query(role, sample_obs(role,.35),top_k=3)
            self.assertEqual(len(xs),3)
            self.assertEqual(xs[0]['experience_id'],'p1b_train_seed2601::000002')
            self.assertLess(xs[0]['distance'],1e-6)
        self.assertFalse((self.lib/'private_offline').exists())

    def test_random_retrieval_is_repeatable_and_unique(self):
        h=LocalHistory(self.lib)
        a=h.random_cases('agent_k1',3,seed=2,scene_id=3)
        b=h.random_cases('agent_k1',3,seed=2,scene_id=3)
        self.assertEqual([x['experience_id'] for x in a], [x['experience_id'] for x in b])
        self.assertEqual(len(set(x['experience_id'] for x in a)),3)

    def test_independent_validation_or_test_only(self):
        h=LocalHistory(self.lib)
        good={'schema':'p1b_dataset_v1','split':'validation','dataset_id':'val_new',
              'sample_sha256':'different','n_scenes':5}
        h.assert_independent(good)
        with self.assertRaises(ContractError):h.assert_independent(dict(good,split='train'))
        with self.assertRaises(ContractError):h.assert_independent(dict(good,dataset_id='p1b_train_seed2601'))
        with self.assertRaises(ContractError):h.assert_independent(dict(good,sample_sha256='traindata_hash'))

    def test_no_private_fields_or_unchecked_library(self):
        role_file=self.lib/'role_libraries'/'agent_k1.jsonl'
        rows=[json.loads(line) for line in role_file.read_text().splitlines()]
        rows[0]['global_state']=[9]*240
        role_file.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
        with self.assertRaises(ContractError):LocalHistory(self.lib)
        self.assertFalse((self.lib/'private_offline').exists())

    def test_icl_prompts_contain_only_whitelisted_examples(self):
        h=LocalHistory(self.lib)
        role='agent_k1'
        obs=sample_obs(role,.26)
        matches=h.query(role,obs,3)
        messages=make_icl_messages(role,obs,matches)
        joined=json.dumps(messages)
        self.assertNotIn('experience_id',joined)
        self.assertNotIn('global_state',joined)
        self.assertNotIn('teacher_provenance',joined)
        self.assertNotIn('V_k1',joined)
        self.assertIn('historical_own_action_ratios',joined)
        self.assertIn('Your CURRENT role-local summary', joined)
        self.assertNotIn('static_no_migration_limit',joined)
        self.assertIn('compute_utilization',joined)

    def test_four_agent_knn_no_policy_fallback(self):
        h=LocalHistory(self.lib)
        policy=RetrievalPolicy(h,'knn')
        policy.set_scene(2)
        actions, entries=policy.act({r:sample_obs(r,.35) for r in AGENTS})
        self.assertEqual(set(actions),set(AGENTS))
        self.assertTrue(all(len(actions[r])==4 for r in AGENTS))
        self.assertAlmostEqual(float(actions['agent_k2'][0]),2*(.1+2*.12)-1,places=5)
        self.assertEqual(entries['agent_k2']['retrieved_ids_for_audit_only'],
                         ['p1b_train_seed2601::000002'])
        self.assertEqual(entries['agent_k2']['model_latency_s'],0)

    def test_mock_icl_and_random_icl_are_structurally_valid(self):
        h=LocalHistory(self.lib)
        for method in ('icl','random_icl'):
            policy=RetrievalPolicy(h,method,client=MockClient(),examples=3)
            policy.set_scene(2)
            actions,entries=policy.act({r:sample_obs(r,.36) for r in AGENTS})
            self.assertEqual(len(actions),4)
            self.assertEqual(len(entries['agent_g']['ratios']),3)
            self.assertEqual(len(entries['agent_k1']['retrieved_ids_for_audit_only']),3)


class TestP1CPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        sample_library(self.root)
        (self.root/'configs_main.yaml').write_text('#test')
        (self.root/'main.py').write_text('#test')
        self.eval_manifest={'schema':'p1b_dataset_v1','split':'validation',
                            'dataset_id':'p1c_val_seed4601',
                            'sample_sha256':'valhash', 'n_scenes':8,'seed':4601}
    def tearDown(self): self.tmp.cleanup()
    def run_method(self,method,name, *, real_stub=False):
        argv=['--library','p1b_libraries/train_v1',
              '--dataset-manifest','not_used.json','--method',method,
              '--start','0','--count','4','--seed','1','--output-dir',name]
        if method in ('icl','random_icl','zero') and not real_stub:
            argv += ['--mock-llm']
        args=get_args(argv)
        return run(args,root=self.root,env_override=FakeEnv(),
                   eval_manifest_override=self.eval_manifest,
                   teacher_override=(FakeTeacher() if method=='mappo' else None),
                   client_override=(FakeClient() if real_stub and method in ('icl','random_icl','zero') else None))
    def test_end_to_end_knn_icl_random_and_teacher(self):
        for method in ('knn','icl','random_icl','mappo','zero'):
            s=self.run_method(method,method)
            self.assertEqual(s['n_scored'],4)
            self.assertEqual(s['n_failed_policy'],0)
        self.assertTrue((self.root/'knn/records.jsonl').exists())
        # A mock model is never admitted to a real performance comparison.
        with self.assertRaises(ContractError):
            compare({'knn':self.root/'knn','icl':self.root/'icl'})

    def test_comparison_matched_scenes_and_audit_mismatch(self):
        for method in ('knn','icl','random_icl','mappo','zero'):
            self.run_method(method,method,real_stub=True)
        result, rows=compare({'knn':self.root/'knn', 'icl':self.root/'icl',
            'random_icl':self.root/'random_icl', 'mappo':self.root/'mappo',
            'zero':self.root/'zero'},bootstrap=100)
        self.assertEqual(result['n_paired'],4)
        self.assertIn('icl_vs_knn',result['pairwise_comparisons'])
        self.assertIn('icl_vs_zero',result['pairwise_comparisons'])
        self.assertEqual(len(rows),4)
        path=self.root/'icl/manifest.json'
        meta=json.loads(path.read_text())
        meta['eval_dataset_sha256']='tampered'
        save_json(path,meta)
        with self.assertRaises(ContractError):
            compare({'knn':self.root/'knn','icl':self.root/'icl'})

    def test_refuse_training_scenes_and_overwrite(self):
        args=get_args(['--library','p1b_libraries/train_v1','--dataset-manifest','foo',
                       '--method','knn','--count','2','--output-dir','knn'])
        with self.assertRaises(ContractError):
            run(args,root=self.root,env_override=FakeEnv(),
                eval_manifest_override=dict(self.eval_manifest,split='train'))
        run(args,root=self.root,env_override=FakeEnv(),
            eval_manifest_override=self.eval_manifest)
        with self.assertRaises(FileExistsError):
            run(args,root=self.root,env_override=FakeEnv(),
                eval_manifest_override=self.eval_manifest)


if __name__ == '__main__': unittest.main()
