import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import os

SCRIPT = Path(__file__).resolve().parents[1] / 'p2c_qwen8_queue.py'
spec = importlib.util.spec_from_file_location('p2c_qwen8_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = queue
spec.loader.exec_module(queue)

GENERATOR = r'''
import argparse, json, pathlib, hashlib
p = argparse.ArgumentParser()
p.add_argument('--name',required=True)
p.add_argument('--split',required=True)
p.add_argument('--seed',type=int,required=True)
p.add_argument('--count',type=int,required=True)
a=p.parse_args()
root=pathlib.Path.cwd()
f=root/'p1b_data'/f'{a.name}.pkl'
f.parent.mkdir(exist_ok=True)
assert not f.exists()
f.write_bytes(f'{a.seed} {a.count} test'.encode())
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(root/'p1b_data'/f'{a.name}.json').write_text(json.dumps(dict(dataset_id=a.name,split=a.split,seed=a.seed,n_scenes=a.count,sample_file=str(f),sample_sha256=h(f),main_py_sha256=h(root/'main.py'),config_sha256=h(root/'configs_main.yaml'))))
'''

RUNNER = r'''
import argparse,csv,json,pathlib,hashlib
p=argparse.ArgumentParser()
for flag in ('library','dataset-manifest','method','start','count','seed','on-error','output-dir','ir-history','model','temperature','max-tokens','api-timeout','examples','risk-pool','risk-weight','risk-max-gap','risk-max-ratio'):
 p.add_argument('--'+flag)
a=p.parse_args()
root=pathlib.Path.cwd()
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=pathlib.Path(a.output_dir)
out.mkdir(parents=True,exist_ok=True)
d=json.loads(pathlib.Path(a.dataset_manifest).read_text())
lib=root/a.library
meta=json.loads((lib/'library_manifest.json').read_text())
ish= bool(a.ir_history)
model=a.model if a.method in ('icl','ir_icl','ir_rerank') else ('frozen_mappo' if a.method=='mappo' else 'no_llm_knn')
rr={'pool_size':20,'risk_weight':0.35,'max_abs_distance_gap':0.10,'max_distance_ratio':1.35,'anchor_nearest':True,'g_retrieval_unchanged':True}
m=dict(method=a.method,mock_not_real_llm=False,eval_dataset_id=d['dataset_id'],eval_dataset_split='test',eval_dataset_sha256=d['sample_sha256'],eval_generation_seed=d['seed'],scene_ids=list(range(int(a.start),int(a.start)+int(a.count))),seed=int(a.seed),on_error=a.on_error,fallback_used=False,guard_used=False,task_finetuning=False,test_time_global_state_access=False,cross_agent_communication=False,config_sha256=h(root/'configs_main.yaml'),main_py_sha256=h(root/'main.py'),library_training_id=meta['dataset_id'],library_train_sha256=meta['training_sample_sha256'],library_manifest_sha256=h(lib/'library_manifest.json'),library_role_sha256={r:h(lib/'role_libraries'/f'{r}.jsonl') for r in ('agent_k1','agent_k2','agent_k3','agent_g')},ir_history_manifest_sha256=(h(root/a.ir_history/'ir_manifest.json') if ish else None),ir_history_data_present=ish,model=model,examples=(3 if model.startswith('qwen') else (1 if a.method=='knn' else 0)),temperature=(float(a.temperature) if model.startswith('qwen') else None),max_tokens=(int(a.max_tokens) if model.startswith('qwen') else None),stream=(True if model.startswith('qwen') else None),p2a_rerank_config=(rr if a.method=='ir_rerank' else None),policy=('qwen_ir_risk_reranked_coop_icl_local_p2a_v1' if a.method=='ir_rerank' else 'other'),checkpoint_sha256=(h(root/'marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth') if a.method=='mappo' else None))
(out/'manifest.json').write_text(json.dumps(m))
rows=[]
with (out/'records.jsonl').open('w') as rec:
 for i in range(int(a.start),int(a.start)+int(a.count)):
  g=(i%2==0); c=(i%3==0)
  row=dict(original_scene_id=i,status='ok',g_done=g,cooperative_valid=c,individual_rationality=c,V=340.+i/10,V_g=20.,V_k1=100.,V_k2=110.,V_k3=110.+i/10,cost_sum=35.+i/10,policy_latency_s=1.0,env_step_s=0.02,failed_role='')
  rows.append(row)
  rec.write(json.dumps(dict(original_scene_id=i,status='ok',fallback_used=False))+'\n')
with (out/'metrics.csv').open('w',newline='') as fp:
 w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
s=dict(n_attempted=len(rows),n_scored=len(rows),n_failed_policy=0,fallback_rate=0.0,eval_dataset_id=d['dataset_id'],split='test',method=a.method,mock_not_real_llm=False,acceptance_rate_scored_only=sum(r['g_done'] for r in rows)/len(rows),cooperative_valid_rate_scored_only=sum(r['cooperative_valid'] for r in rows)/len(rows),V_scored_only=sum(r['V'] for r in rows)/len(rows),cost_sum_scored_only=sum(r['cost_sum'] for r in rows)/len(rows))
(out/'summary.json').write_text(json.dumps(s))
'''

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for d in ('p1b_data','p1a','p1b','p1c','p1d','p2a'):
            (self.root/d).mkdir(parents=True,exist_ok=True)
        for f in ('main.py','configs_main.yaml','p1a_runner.py'):
            (self.root/f).write_text('fixture '+f)
        (self.root/'p1b_generate.py').write_text(GENERATOR)
        (self.root/'p1c_runner.py').write_text(RUNNER)
        # The queue now freezes the audited evaluation adapter and provenance.
        # This mock runner never imports a real simulator or model service.
        (self.root/'p2c_eval_runner.py').write_text(RUNNER)
        provenance = self.root / queue.HOTFIX_PROVENANCE
        provenance.mkdir(parents=True)
        (provenance/'main_generation.py').write_text('fixture generation source')
        (provenance/'eval_hotfix_attestation.json').write_text('{}')
        cp=self.root/queue.CHECKPOINT
        cp.parent.mkdir(parents=True,exist_ok=True)
        cp.write_text('fixture_checkpoint')
        for job in queue.JOBS:
            d=self.root/job.library
            (d/'role_libraries').mkdir(parents=True, exist_ok=True)
            (d/'library_manifest.json').write_text(json.dumps(dict(dataset_id='train',training_sample_sha256='hash_training_not_test',development_only=False)))
            for role in queue.ROLES:
                (d/'role_libraries'/f'{role}.jsonl').write_text('{}\n')
            if job.ir:
                d=self.root/job.ir;d.mkdir(parents=True,exist_ok=True)
                (d/'ir_manifest.json').write_text('{}')
                (d/'agent_k1.jsonl').write_text('{}\n')
        self.constants=[patch.object(queue,'N_SCENES',2), patch.object(queue,'CHUNK_SIZE',2), patch.object(queue,'SEEDS',(6601,))]
        for p in self.constants:p.start()
    def tearDown(self):
        for p in self.constants:p.stop()
        self.tmp.cleanup()

    def test_complete_resume_and_corruption_rejection(self):
        queue.plan(self.root)
        queue.prepare(self.root)
        queue.prepare(self.root)
        self.assertEqual(len(list((self.root/'p1b_data').glob('*.json'))),1)
        with patch.object(queue,'paired_bootstrap',return_value={'coop_pp_ci95':(0,0),'cost_ci95':(0,0),'V_ci95':(0,0)}):
            queue.execute(self.root)
        out=self.root/queue.OUTPUT_REL
        paths=list(out.glob('p2c_test_seed*/**/chunk_*'))
        self.assertEqual(len(paths),1*8*1)
        with (out/'p2c_all_scene_metrics.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),1*8*2)
        with (out/'p2c_paired_comparisons.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))),2*len(queue.PAIRS))
        with patch.object(queue,'paired_bootstrap',return_value={'coop_pp_ci95':(0,0),'cost_ci95':(0,0),'V_ci95':(0,0)}):
            queue.execute(self.root)
        bad=paths[0]
        (bad/'records.jsonl').unlink()
        with self.assertRaisesRegex(RuntimeError,'records.jsonl'):
            queue.execute(self.root)

    def test_prepare_does_not_overwrite_orphan(self):
        (self.root/'p1b_data'/'p2c_test_seed6601.pkl').write_text('orphan')
        with self.assertRaisesRegex(RuntimeError,'INCOMPLETE generation artifacts'):
            queue.prepare(self.root)

    def test_generation_orphan_in_actual_geant_default_path(self):
        sample = self.root/'data'/'GEANT'/'p2c_test_seed6601.pkl'
        sample.parent.mkdir(parents=True)
        sample.write_text('incomplete_pickle')
        with self.assertRaisesRegex(RuntimeError, 'INCOMPLETE generation artifacts'):
            queue.prepare(self.root)

    def test_all_mode_prepares_before_executing(self):
        calls=[]
        with patch.object(queue, 'prepare', side_effect=lambda root: calls.append('prepare')):
            with patch.object(queue, 'execute', side_effect=lambda root: calls.append('execute')):
                queue.run_all(self.root)
        self.assertEqual(calls, ['prepare', 'execute'])

    def test_run_all_end_to_end_and_reuse_seed(self):
        with patch.object(queue, 'paired_bootstrap', return_value={
            'coop_pp_ci95': (0,0), 'cost_ci95': (0,0), 'V_ci95': (0,0)
        }):
            queue.run_all(self.root)
            self.assertTrue((self.root/queue.OUTPUT_REL/'p2c_all_scene_metrics.csv').exists())
            first_log=self.root/queue.OUTPUT_REL/'logs'/'generation'/'p2c_test_seed6601.log'
            self.assertTrue(first_log.is_file())
            self.assertEqual(first_log.read_text().count('### GENERATE START'), 1)
            queue.run_all(self.root)
            self.assertEqual(first_log.read_text().count('### GENERATE START'), 1)

    def test_command_icl_without_ir_and_rerank_with_ir(self):
        a=queue.JOBS[2];b=queue.JOBS[4]
        c=queue.command_for(a,6601,32,32,Path('foo'))
        self.assertIn('icl',c)
        self.assertNotIn('--ir-history',c)
        d=queue.command_for(b,6601,32,32,Path('foo'))
        self.assertIn('--ir-history',d)
        self.assertIn('--risk-weight',d)
        self.assertEqual(len(list(queue.all_blocks(self.root))),8)

if __name__=='__main__':unittest.main()
