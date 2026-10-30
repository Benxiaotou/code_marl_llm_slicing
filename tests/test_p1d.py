"""P1-D offline IR extraction, role isolation, strict matched-scene evaluation."""
import json
import tempfile
import unittest
from pathlib import Path

from p1a.local_obs import AGENTS
from p1b.core import ContractError, read_json, read_jsonl, save_json, file_sha256
from p1c.retrieval import LocalHistory
from p1d.ir import IRHistory, build_ir_overlay
from p1d.policy import IRRetrievalPolicy
from p1c.policy import RetrievalPolicy
from p1c_runner import get_args, run
from p1c_compare import compare
from test_p1c import sample_library, sample_obs, FakeEnv, FakeClient, FakeTeacher


def fake_joint_source(lib):
    d = lib / 'private_offline'
    d.mkdir(parents=True, exist_ok=True)
    r = {role: list(read_jsonl(lib / 'role_libraries' / (role+'.jsonl'))) for role in AGENTS}
    out=[]
    for i in range(len(r['agent_g'])):
        metrics = {'V_k1_0':100, 'V_k2_0':101, 'V_k3_0':102,
                   'V_k1':104.2+i, 'V_k2':103.5+i, 'V_k3':107.1+i,
                   'V_g':20+i, 'cost_k1':i+2, 'cost_k2':i+3, 'cost_k3':i+4,
                   'g_done':True, 'individual_rationality':True,'cooperative_valid':True}
        out.append({'status':'ok', 'dataset_id':'p1b_train_seed2601',
                    'joint_id':'p1b_train_seed2601::{:06d}'.format(i),
                    'metrics':metrics,
                    'observations':{role:r[role][i]['observation_240'] for role in AGENTS},
                    'teacher_actions_ratio':{role:r[role][i]['own_teacher_ratios'] for role in AGENTS}})
    source=d/'selected_joint_records.jsonl'
    source.write_text(''.join(json.dumps(row)+'\n' for row in out),encoding='utf-8')
    return source


class TestOfflineIRHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lib = sample_library(self.root)
        self.source = fake_joint_source(self.lib)
        self.dest = self.root / 'ir_overlay'
    def tearDown(self):
        self.tmp.cleanup()

    def test_export_152_style_four_role_self_only_records(self):
        original_digest=file_sha256(self.lib/'role_libraries'/'agent_k1.jsonl')
        meta=build_ir_overlay(self.lib,self.dest)
        self.assertEqual(meta['n_selected_training_cases'],6)
        self.assertEqual(file_sha256(self.lib/'role_libraries'/'agent_k1.jsonl'),original_digest)
        h=IRHistory(self.dest, LocalHistory(self.lib))
        a=h.for_examples('agent_k1',LocalHistory(self.lib).query('agent_k1',sample_obs('agent_k1',.35),3))
        self.assertAlmostEqual(a[0]['own_historical_ir_margin'],6.2)
        self.assertEqual(a[0]['own_historical_adjustment_cost'],4.)
        g=list(read_jsonl(self.dest/'role_ir_history'/'agent_g.jsonl'))
        self.assertEqual(set(g[0]),{'schema','role','experience_id','own_historical_new_slice_utility'})
        self.assertEqual(len(g),6)
        self.assertNotIn('metrics',json.dumps(g))
        self.assertFalse((self.dest/'private_offline').exists())
        self.assertTrue(read_json(self.dest/'ir_audit.json')['audit_passed'])

    def test_runtime_does_not_need_private_joint_source(self):
        build_ir_overlay(self.lib,self.dest)
        self.source.unlink()
        h=IRHistory(self.dest,LocalHistory(self.lib))
        self.assertEqual(len(h.rows['agent_k1']),6)

    def test_no_mutation_tampering_or_mismatched_training_library(self):
        build_ir_overlay(self.lib,self.dest)
        with self.assertRaises(FileExistsError):build_ir_overlay(self.lib,self.dest)
        p=self.dest/'role_ir_history'/'agent_k2.jsonl'
        p.write_text(p.read_text(encoding='utf-8').replace('own_historical_ir_margin','global_state'))
        with self.assertRaises(ContractError):IRHistory(self.dest,LocalHistory(self.lib))

    def test_negative_training_outcome_rejected(self):
        rows=list(read_jsonl(self.source))
        rows[0]['metrics']['V_k1']=90
        self.source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaises(ContractError):build_ir_overlay(self.lib,self.dest)

    def test_historical_action_mismatch_rejected(self):
        rows=list(read_jsonl(self.source))
        rows[1]['teacher_actions_ratio']['agent_k1'][0]=.99
        self.source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaises(ContractError):build_ir_overlay(self.lib,self.dest)

    def test_overlay_cannot_live_inside_original_library(self):
        with self.assertRaises(ContractError):build_ir_overlay(self.lib,self.lib/'overlay')

    def test_prompt_role_only_and_retrieval_same_as_icl(self):
        build_ir_overlay(self.lib,self.dest)
        lib=LocalHistory(self.lib)
        overlay=IRHistory(self.dest,lib)
        obs={r:sample_obs(r,.35) for r in AGENTS}
        normal=RetrievalPolicy(lib,'icl',client=FakeClient(),examples=3,seed=1)
        ir=IRRetrievalPolicy(lib,overlay,client=FakeClient(),examples=3,seed=1)
        normal.set_scene(0)
        ir.set_scene(0)
        _,base_entries=normal.act(obs)
        _,ir_entries=ir.act(obs)
        for role in AGENTS:
            self.assertEqual(base_entries[role]['retrieved_ids_for_audit_only'],
                             ir_entries[role]['retrieved_ids_for_audit_only'])
            self.assertEqual(base_entries[role]['distances_for_audit_only'],
                             ir_entries[role]['distances_for_audit_only'])
            msg=json.dumps(ir_entries[role]['messages'])
            self.assertNotIn('experience_id',msg)
            self.assertNotIn('global_state',msg)
            self.assertNotIn('V_k1_0',msg)
            self.assertNotIn('cost_sum',msg)
            self.assertIn('own_historical_',msg)
            if role=='agent_g':
                self.assertNotIn('own_historical_ir_margin',msg)
                self.assertIn('own_historical_new_slice_utility',msg)
            else:
                self.assertIn('own_historical_ir_margin',msg)
                self.assertNotIn('own_historical_new_slice_utility',msg)
        self.assertNotEqual(base_entries['agent_k1']['prompt_sha256'],
                            ir_entries['agent_k1']['prompt_sha256'])


class TestIRMatchedComparison(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.lib=sample_library(self.root)
        fake_joint_source(self.lib)
        self.overlay=self.root/'ir_overlay'
        build_ir_overlay(self.lib,self.overlay)
        (self.root/'configs_main.yaml').write_text('#test')
        (self.root/'main.py').write_text('#test')
        self.eval_manifest={'schema':'p1b_dataset_v1','split':'validation',
                            'dataset_id':'p1c_val_seed4601',
                            'sample_sha256':'valhash','n_scenes':8,'seed':4601}
    def tearDown(self):self.tmp.cleanup()

    def run_method(self, method):
        argv=['--library','p1b_libraries/train_v1', '--dataset-manifest','fake.json',
              '--method',method,'--start','0','--count','4','--seed','1',
              '--output-dir',method]
        if method=='ir_icl':argv.extend(['--ir-history','ir_overlay'])
        args=get_args(argv)
        return run(args,root=self.root,env_override=FakeEnv(),
                   eval_manifest_override=self.eval_manifest, client_override=FakeClient(),
                   teacher_override=(FakeTeacher() if method=='mappo' else None))

    def test_real_stub_pipeline_compare_with_original_six_methods(self):
        for method in ('knn','icl','ir_icl','mappo','random_icl','zero'):
            self.run_method(method)
        methods={m:self.root/m for m in ('knn','icl','ir_icl','mappo','random_icl','zero')}
        summary,rows=compare(methods,bootstrap=100)
        self.assertEqual(summary['n_paired'],4)
        self.assertIn('ir_icl_vs_icl', summary['pairwise_comparisons'])
        self.assertIn('ir_icl_vs_mappo', summary['pairwise_comparisons'])
        self.assertIn('ir_icl_vs_knn', summary['pairwise_comparisons'])
        self.assertIn('ir_icl_vs_random_icl', summary['pairwise_comparisons'])
        self.assertIn('ir_icl_vs_zero', summary['pairwise_comparisons'])
        self.assertEqual(len(rows),4)
        self.assertEqual(read_json(self.root/'ir_icl/manifest.json')['ir_feedback_basis'],
                         'historical_self_only_training_joint_outcomes')
        # Audit refuses any mismatch in retrieved experience IDs.
        path=self.root/'ir_icl/records.jsonl'
        data=list(read_jsonl(path))
        data[0]['per_role']['agent_k1']['retrieved_ids_for_audit_only']=['bad_case']
        path.write_text(''.join(json.dumps(x)+'\n' for x in data),encoding='utf-8')
        with self.assertRaises(ContractError):compare(methods,bootstrap=100)

    def test_reject_ir_history_when_not_ir_method_and_training_eval(self):
        with self.assertRaises(ValueError):
            self.run_invalid('--method','knn','--ir-history','ir_overlay')
        with self.assertRaises(ValueError):
            self.run_invalid('--method','ir_icl')
        self.eval_manifest['split']='train'
        with self.assertRaises(ContractError):self.run_method('ir_icl')

    def run_invalid(self,*method_opts):
        argv=['--library','p1b_libraries/train_v1','--dataset-manifest','fake.json',
              '--output-dir','invalid','--start','0','--count','2']+list(method_opts)
        args=get_args(argv)
        run(args,root=self.root,env_override=FakeEnv(),
            eval_manifest_override=self.eval_manifest,client_override=FakeClient())


if __name__=='__main__':unittest.main()
