"""End-to-end simulated smoke test: no real model, no GEANT, no credentials."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / 'gpt61_sol_dev.py'
spec = importlib.util.spec_from_file_location('gpt61_for_integration', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAKE_RUNNER = '''\
import argparse,csv,json
from pathlib import Path
from urllib.request import Request,urlopen
p=argparse.ArgumentParser()
for x in ['output-dir','base-url','start','count','dataset-manifest','method']:
 p.add_argument('--'+x,required=True)
a,_=p.parse_known_args()
output=Path(vars(a)['output_dir']);output.mkdir(parents=True)
start,count=int(a.start),int(a.count)
rows=[]
for scene in range(start,start+count):
 for agent in range(4):
  payload={'messages':[{'role':'system','content':'Rules'},{'role':'user','content':'scene '+str(scene)}],
           'stream':False,'temperature':0.1,'max_tokens':8192}
  req=Request(a.base_url+'/chat/completions', data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  with urlopen(req,timeout=10) as resp:
   doc=json.loads(resp.read())
  assert json.loads(doc['choices'][0]['message']['content'])['ratios'][0]==0.2
 rows.append({'original_scene_id':scene,'status':'ok','fallback_used':False,
              'policy_latency_s':0.1,'metrics':{'g_done':1,'cooperative_valid':1,
                                               'V':340.0,'V_g':26.0,'cost_sum':32.0}})
(output/'manifest.json').write_text(json.dumps({'eval_dataset_id':'gpt61_dev_seed7601','method':a.method,
        'scene_ids':list(range(start,start+count))}),encoding='utf-8')
(output/'summary.json').write_text(json.dumps({'n_attempted':count,'n_scored':count,'fallback_rate':0}),encoding='utf-8')
with (output/'records.jsonl').open('w',encoding='utf-8') as f:
 for x in rows:f.write(json.dumps(x)+'\\n')
with (output/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['original_scene_id','status']);w.writeheader()
 for x in rows:w.writerow({'original_scene_id':x['original_scene_id'],'status':'ok'})
'''

class WholeFlow(unittest.TestCase):
    def test_fake_project_and_resume_10_to_20(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'p1b_data').mkdir()
            (root/'data'/'GEANT').mkdir(parents=True)
            (root/m.DEFAULT_LIBRARY).mkdir(parents=True)
            (root/m.DEFAULT_IR).mkdir(parents=True)
            for name,value in [('main.py','safe-main'),('configs_main.yaml','safe-config'),
                               ('p1c_runner.py',FAKE_RUNNER)]:
                (root/name).write_text(value,encoding='utf-8')
            (root/m.DEFAULT_LIBRARY/'library_manifest.json').write_text('{}')
            (root/m.DEFAULT_IR/'ir_manifest.json').write_text('{}')
            sample=root/'data'/'GEANT'/'gpt61_dev_seed7601.pkl'
            sample.write_bytes(b'simulated-pickle-no-real-env')
            m.save_json(root/'p1b_data'/'gpt61_dev_seed7601.json',{
                'dataset_id':'gpt61_dev_seed7601','split':'validation','seed':7601,
                'n_scenes':20,'sample_file':str(sample),'sample_sha256':m.digest(sample),
                'main_py_sha256':m.digest(root/'main.py'),
                'config_sha256':m.digest(root/'configs_main.yaml')})
            def fake_response(body,*,url,api_key,timeout):
                return {'id':'mock-only','choices':[{'message':{'role':'assistant','content':'{"ratios":[0.2,0.3,0.4,0.5]}'}}],
                        'usage':{'prompt_tokens':12000,'completion_tokens':1000,
                                 'completion_tokens_details':{'reasoning_tokens':950}}}
            def args(n):
                return m.parser().parse_args(['--mode','run','--scenes',str(n),'--budget-usd','8'])
            with patch.dict(os.environ,{'GPT61_BASE_URL':'https://mock.test/v1','GPT61_API_KEY':'secretfake'}),\
                 patch.object(m,'invoke_upstream',side_effect=fake_response),\
                 contextlib.redirect_stdout(io.StringIO()):
                m.run_evaluation(args(10),root)
                m.run_evaluation(args(20),root)
                m.run_evaluation(args(20),root)   # completed -> skip all
            out=root/m.DEFAULT_RUN_FOLDER
            ledger=m.read_events(out/'usage.jsonl')
            self.assertEqual(m.totals(ledger)['requests_ok'],80)
            self.assertEqual(m.load_json(out/'dev20_summary.json')['scenes_completed'],20)
            self.assertEqual(len(list((out/'chunks').iterdir())),4)
            self.assertEqual(m.load_json(out/'protocol_lock.json')['scenes_capacity'],20)

if __name__ == '__main__':unittest.main()
