import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import (ContractError, make_experience, normalize_teacher_actions,
                      read_jsonl, save_json)
from p1b.select import analyze_and_select, joint_features
from p1b.teacher import RecordedMAPPO
from p1b_collect import collect, get_args
from p1b_select import get_args as select_args, select
from p1b_audit import audit


def fake_obs(role, value=0.25):
    nodes = np.zeros((12, 5), dtype=np.float32)
    nodes[:, 1:] = value
    links = np.zeros((36, 3), dtype=np.float32)
    edges = np.zeros((36, 2), dtype=np.float32)
    if role != 'agent_g':
        nodes[:4, 0] = 1
        links[:3, 0] = 1
        links[:5, 1] = 0.4
        links[:5, 2] = value
    else:
        links[:5, 2] = 0.2
    return np.concatenate((nodes.flatten(), links.flatten(), edges.flatten()))


def fake_metrics(ok=True, ir=True, utility=340.0, cost=20.0):
    m = {'V':utility, 'V_k1':105 if ir else 95, 'V_k2':105, 'V_k3':105,
         'V_g':25 if ok else 0, 'V_k1_0':100, 'V_k2_0':100, 'V_k3_0':100,
         'cost_k1':cost/3, 'cost_k2':cost/3,'cost_k3':cost/3,
         'cost_sum':cost, 'g_done':ok, 'individual_rationality':ir,
         'cooperative_valid':ok and ir}
    return m


class Teacher:
    def __init__(self):
        self.source_info={'kind':'live_frozen_mappo','verified_mappo':True,'checkpoint_sha256':'synthetic_test'}
        self.calls=[]
    def act(self, obs, *, original_scene_id, seed):
        self.calls.append((original_scene_id,seed, sorted(obs)))
        return {a:np.array([-0.4,-0.2,-0.1,0.2],dtype=np.float32) for a in AGENTS}
    def close(self):pass


class FakeEnv:
    def __init__(self):
        self.sample_dict={i:tuple(SimpleNamespace(tag=a,sc=i) for a in (*AGENTS,'physical'))
                          for i in range(4)}
        self.sample_idx=0
        self.g_done_list=[]
        self.dict_eval_index={k:[] for k in ('V','V_k1','V_k2','V_k3','V_g',
             'V_k1_0','V_k2_0','V_k3_0','X_k1','X_k2','X_k3',
             'cost_k1','cost_k2','cost_k3','revenue_k1','revenue_k2','revenue_k3','revenue_g')}
    def _get_overlap_node_link(self,k_net,g_net):
        return [1,2],[0],[]
    def _get_v_net_observations(self,k, nodes,links, padding):
        return fake_obs(k.tag,0.25 + k.sc/20.)
    def _get_g_net_observations(self,padding):
        return fake_obs('agent_g',0.25 + self.sample_idx/20.)
    def reset(self):return self.observations,self.infos
    def step(self,actions):
        assert set(actions)==set(AGENTS)
        self.g_done_list.append(True)
        vals=fake_metrics(ok=True, ir=self.sample_idx != 1,
                          utility=320+self.sample_idx*10, cost=5+self.sample_idx*5)
        for key in self.dict_eval_index:
            self.dict_eval_index[key].append(vals.get(key,0))
        return {},{r:0.1 for r in AGENTS},{},False,{}


class TestContracts(unittest.TestCase):
    def test_action_normalize(self):
        a=normalize_teacher_actions({r:[-1,0,0.5,1] for r in AGENTS})
        self.assertEqual(len(a['agent_g']),4)
        self.assertAlmostEqual(float(a['agent_k1'][0]),-1)
        b=normalize_teacher_actions({'actions_01':{r:[0,.5,.75,1] for r in AGENTS}})
        self.assertAlmostEqual(float(b['agent_k1'][1]),0)
        c=normalize_teacher_actions({'env_actions':{r:[0,-.5,.5,1] for r in AGENTS}})
        self.assertAlmostEqual(float(c['agent_k2'][1]),-.5)
        z=normalize_teacher_actions(list(range(16)),assumed_space='env') if False else None
        with self.assertRaises(ContractError):
            normalize_teacher_actions([0]*14)
        with self.assertRaises(ContractError):
            normalize_teacher_actions({'agent_k1':[0]*4})
        with self.assertRaises(ContractError):
            normalize_teacher_actions({r:[float('nan')]*4 for r in AGENTS})
        with self.assertRaises(ContractError):
            normalize_teacher_actions({'actions_01':{r:[1.4]*4 for r in AGENTS}})

    def test_g_3_dim_padding_and_joint_quality(self):
        a={r:[-1,-.5,0,1] for r in AGENTS}
        a['agent_g']=[-.5,0,.5]
        e=make_experience('train_seed11',12,{r:fake_obs(r) for r in AGENTS},a,
                          fake_metrics(), {'kind':'live_frozen_mappo'})
        self.assertEqual(e['joint_id'],'train_seed11::000012')
        self.assertEqual(len(e['teacher_actions_ratio']['agent_g']),3)
        self.assertEqual(len(e['teacher_actions_env']['agent_g']),4)
        self.assertEqual(e['metrics']['cost_sum'],20)
        with self.assertRaises(ContractError):
            make_experience('train_seed11',12,{r:fake_obs(r) for r in AGENTS},a,
                            dict(fake_metrics(), cooperative_valid=False), {})

    def test_quality_selection_strata_and_deterministic(self):
        rows=[]
        for i in range(50):
            good = i%5 != 0
            ir = i%9 != 0
            metric=fake_metrics(ok=good,ir=ir,utility=300+i*1.5,cost=2+i)
            x=make_experience('trainset', i,{r:fake_obs(r,.1+.8*i/50) for r in AGENTS},
                              {r:[-.5,-.3,.1,.1] for r in AGENTS},metric, {})
            x['status']='ok'
            rows.append(x)
        a,report=analyze_and_select(rows,max_keep=12,cost_quantile=.9)
        b,report2=analyze_and_select(rows,max_keep=12,cost_quantile=.9)
        self.assertEqual([x['row']['joint_id'] for x in a],
                         [x['row']['joint_id'] for x in b])
        self.assertEqual(len(a),12)
        self.assertGreaterEqual(len(report['strata_selected']),2)
        self.assertEqual(report['n_selected'],12)
        self.assertTrue(all(x['row']['metrics']['cooperative_valid'] for x in a))

    def test_zero_good_does_not_fabricate(self):
        r=make_experience('train',1,{a:fake_obs(a) for a in AGENTS},
                          {a:[0]*4 for a in AGENTS},fake_metrics(ok=False),{})
        r['status']='ok'
        chosen,report=analyze_and_select([r])
        self.assertEqual(chosen,[])
        self.assertEqual(report['hard_filter_rejections']['admission_failed'],1)

    def test_recorded_teacher_provenance_and_actions(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            save_json(p/'summary.json',{'policy':'mappo','n':1})
            (p/'mappo_records.jsonl').write_text(json.dumps({
                'original_scene_index':3,
                'env_actions':{a:[-.4,0,.2,.6] for a in AGENTS},'status':'ok'})+'\n')
            teacher=RecordedMAPPO(p)
            self.assertEqual(teacher.source_info['kind'],'recorded_MAPPO_replay')
            acts=teacher.act(None,original_scene_id=3)
            self.assertAlmostEqual(float(acts['agent_k1'][0]),-.4,places=6)
            with self.assertRaises(ContractError):teacher.act(None,original_scene_id=4)

    def test_replay_rejects_unverified(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            (p/'records.jsonl').write_text(json.dumps({'original_scene_id':0,
                'env_actions':{a:[-.4,0,.2,.6] for a in AGENTS}})+'\n')
            with self.assertRaises(ContractError):
                RecordedMAPPO(p)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'main.py').write_text('#fake\n')
        (self.root/'configs_main.yaml').write_text('#fake\n')
        self.dataset={'schema':'p1b_dataset_v1','dataset_id':'new_train','split':'train',
                      'sample_file':str(self.root/'samples.pkl'),'sample_sha256':'fakehash',
                      'n_scenes':4}
        self.teacher=Teacher()
    def tearDown(self):self.temp.cleanup()

    def test_capture_select_audit(self):
        args=get_args(['--dataset','custom','--count','4','--output-dir','collection',
                       '--teacher','live'])
        summary=collect(args,root=self.root,env_override=FakeEnv(),
                        dataset_override=self.dataset,teacher_override=self.teacher)
        self.assertEqual(summary['n_ok'],4)
        self.assertEqual(summary['n_cooperative_valid'],3)
        self.assertEqual(len(self.teacher.calls),4)
        sel=select_args(['--collection',str(self.root/'collection'),
                         '--output-dir',str(self.root/'library'),'--max-keep','3'])
        lib,r=select(sel)
        self.assertEqual(lib['n_experiences_per_role'],3)
        report=audit(self.root/'library')
        self.assertEqual(report['n_per_role'],3)
        rows=list(read_jsonl(self.root/'library/role_libraries/agent_k1.jsonl'))
        self.assertTrue(rows)
        self.assertNotIn('metrics',rows[0]);self.assertNotIn('observations',rows[0])
        self.assertNotIn('agent_g',rows[0]);self.assertNotIn('teacher_actions_ratio',rows[0])
        # Detect accidental insertion of global fields / current other role details.
        with (self.root/'library/role_libraries/agent_k1.jsonl').open('w') as f:
            f.write(json.dumps(dict(rows[0],metrics={'V':320}))+'\n')
        with self.assertRaises(ContractError):audit(self.root/'library')

    def test_resume_and_modified_manifest_reject(self):
        args=get_args(['--count','2','--output-dir','collection'])
        collect(args,root=self.root,env_override=FakeEnv(),dataset_override=self.dataset,
                teacher_override=self.teacher)
        args.resume=True
        result=collect(args,root=self.root,env_override=FakeEnv(),dataset_override=self.dataset,
                       teacher_override=self.teacher)
        self.assertEqual(result['n_ok'],2)
        args.seed=90
        with self.assertRaises(ContractError):
            collect(args,root=self.root,env_override=FakeEnv(),dataset_override=self.dataset,
                    teacher_override=self.teacher)

    def test_split_and_holdout_safety(self):
        args=get_args(['--count','4','--output-dir','collection'])
        collect(args,root=self.root,env_override=FakeEnv(),dataset_override=self.dataset,
                teacher_override=self.teacher)
        holdout=self.root/'holdout.json'
        save_json(holdout,{'split':'test','dataset_id':'new_train','sample_sha256':'other'})
        sargs=select_args(['--collection',str(self.root/'collection'),
                          '--output-dir',str(self.root/'library'),
                          '--holdout-manifest',str(holdout)])
        with self.assertRaises(ContractError):select(sargs)
        # Legacy P0 samples cannot silently become training experience.
        manifest=json.loads((self.root/'collection/manifest.json').read_text())
        manifest['development_only']=True
        save_json(self.root/'collection/manifest.json',manifest)
        sargs=select_args(['--collection',str(self.root/'collection'),
                          '--output-dir',str(self.root/'library')])
        with self.assertRaises(ContractError):select(sargs)


if __name__=='__main__':unittest.main()
