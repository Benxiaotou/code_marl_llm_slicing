# coding: utf-8
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import p2e_prompt_3way as p
import gpt61_sol_dev as adapter

SYS = p.SYSTEM_START + ' ' + p.ROLE_TAG + ' Produce exactly 4 values: [node_control_0,node_control_1,link_control_0,link_control_1].'
SYS_G = p.SYSTEM_START + ' You represent NEW slice g. Produce exactly 3 values: [compute_compression,cache_compression,bandwidth_compression].'
USER = 'One current observation for your role (all node/link IDs are LOCAL):\n{"role":"agent_k1","nodes":[]}'
USER_G = 'One current observation for your role (all node/link IDs are LOCAL):\n{"role":"agent_g","nodes":[]}'


def msg(existing=True):
    return {'model': 'qwen3-next', 'temperature': 0.1, 'max_tokens': 4096,
            'messages': [{'role':'system','content':SYS if existing else SYS_G},
                         {'role':'user','content':USER if existing else USER_G}],
            'stream':False}


class PromptTests(unittest.TestCase):
    def test_variants_are_distinct_and_nonempty(self):
        ts=[p.read_prompt(v) for v in p.VARIANT_FILES]
        self.assertEqual(len(set(ts)),3)
        self.assertTrue(all(len(x)>150 for x in ts))

    def test_existing_prompt_only_applies_to_system(self):
        for v in p.VARIANT_FILES:
            base=msg()
            new, aud=p.inject_existing_prompt(base,variant=v,appendix=p.read_prompt(v))
            self.assertEqual(base['messages'][0]['content'],SYS)
            self.assertEqual(new['messages'][1],base['messages'][1])
            self.assertEqual(new['messages'][0]['content'],SYS+'\n\n'+p.read_prompt(v))
            self.assertTrue(aud['prompt_injected'])
            self.assertEqual(aud['role_type'],'existing')
            self.assertEqual(new['temperature'],0.1)

    def test_new_slice_prompt_unchanged(self):
        for v in p.VARIANT_FILES:
            original=msg(False)
            new,aud=p.inject_existing_prompt(original,variant=v,appendix=p.read_prompt(v))
            self.assertEqual(new,original)
            self.assertFalse(aud['prompt_injected'])
            self.assertEqual(aud['role_type'],'new_slice_g')

    def test_mismatched_role_rejected(self):
        bad=msg();bad['messages'][1]['content']=USER_G
        with self.assertRaisesRegex(ValueError,'Role mismatch'):
            p.inject_existing_prompt(bad,variant='semantic_balance',appendix='hello')

    def test_missing_original_context_rejected(self):
        bad=msg();bad['messages'][0]['content']='Generic role you are a bot'
        with self.assertRaises(ValueError):
            p.inject_existing_prompt(bad,variant='semantic_balance',appendix='hello')

    def test_unexpected_message_count_rejected(self):
        bad=msg();bad['messages'].append({'role':'assistant','content':'hi'})
        with self.assertRaises(ValueError):
            p.inject_existing_prompt(bad,variant='teacher_scale',appendix='hello')

    def test_prompt_effective_hash_changes(self):
        b,a=p.inject_existing_prompt(msg(),variant='teacher_scale',appendix=p.read_prompt('teacher_scale'))
        self.assertNotEqual(a['original_system_sha256'],a['effective_system_sha256'])
        self.assertEqual(a['effective_system_sha256'],p.hash_str(b['messages'][0]['content']))

    def test_original_g_prompt_effective_hash_same(self):
        _,a=p.inject_existing_prompt(msg(False),variant='semantic_balance',appendix=p.read_prompt('semantic_balance'))
        self.assertEqual(a['original_system_sha256'],a['effective_system_sha256'])

    def test_proxy_http_roundtrip_both_roles_no_real_upstream(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);seen=[]
            def fake(body,*,url,api_key,timeout):
                seen.append(body)
                return {'choices':[{'message':{'role':'assistant','content':'{"ratios":[0.3,0.4,0.7,0.8]}'},'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':5100,'completion_tokens':200,'completion_tokens_details':{'reasoning_tokens':180}}}
            state=p.VariantState(variant='semantic_balance',appendix=p.read_prompt('semantic_balance'),
                audit=d/'audit.jsonl',upstream='https://dummy/v1/chat/completions',key='fake_hidden',
                api='chat',model='gpt-6.1-sol',effort='medium',max_output_tokens=8192,
                ledger=d/'usage.jsonl',prices={'input':2.0,'cached_input':0.1,'output':10.0},
                budget=2.0,max_calls=2,timeout=30)
            class Handler(adapter.LocalProxyHandler):pass
            Handler.state=state
            srv=adapter.ThreadingHTTPServer(('127.0.0.1',0),Handler)
            t=threading.Thread(target=srv.serve_forever,daemon=True);t.start()
            try:
                with patch.object(adapter,'invoke_upstream',side_effect=fake):
                    for incoming in (msg(True),msg(False)):
                        req=Request(f'http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions',
                            data=json.dumps(incoming).encode(),headers={'Content-Type':'application/json'})
                        with urlopen(req,timeout=5) as response:
                            self.assertEqual(response.status,200)
                    self.assertEqual(len(seen),2)
            finally:
                srv.shutdown();srv.server_close();t.join(3)
            self.assertIn('semantic_balance',state.variant)
            self.assertIn('GPT-6.1 Sol existing-slice',seen[0]['messages'][0]['content'])
            self.assertNotIn('GPT-6.1 Sol existing-slice',seen[1]['messages'][0]['content'])
            self.assertEqual(seen[0]['messages'][1]['content'],USER)
            audit=adapter.read_events(d/'audit.jsonl')
            self.assertEqual([r['prompt_injected'] for r in audit],[True,False])
            self.assertEqual(adapter.totals(adapter.read_events(d/'usage.jsonl'))['requests_ok'],2)
            self.assertNotIn('fake_hidden',(d/'usage.jsonl').read_text())

    def test_aggregate(self):
        recs=[]
        for i in range(10):
            recs.append({'status':'ok','original_scene_id':i,'policy_latency_s':10,
                         'metrics':{'g_done': i < 6,'cooperative_valid':i < 5,
                                    'V':100+i,'V_g':10 if i<6 else 0,'cost_sum':20+i}})
        s=p.aggregate(recs,'test')
        self.assertEqual((s['accepted'],s['cooperative'],s['accept_rate'],s['coop_rate']),(6,5,0.6,0.5))
        self.assertEqual(s['V_mean'],104.5)

    def test_verification_excludes_partial_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(p.collect_new(Path(d),Path('not_there')))
            with self.assertRaises(RuntimeError):
                p.collect_new(Path(d),Path('not_there'),require_complete=True)

if __name__=='__main__':
    unittest.main()
