import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

SCRIPT = Path(__file__).parents[1] / 'gpt61_sol_dev.py'
spec = importlib.util.spec_from_file_location('gpt61_sol_dev', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class AdapterTests(unittest.TestCase):
    def test_chat_request_transforms_qwen_fields(self):
        inb={'model':'qwen','messages':[{'role':'system','content':'rules'},{'role':'user','content':'input'}],
             'temperature':0.1,'max_tokens':4096,'stream':False,'top_p':0.7}
        o=m.transform_request(inb,api='chat',model='gpt-6.1-sol',effort='medium',max_output_tokens=8192)
        self.assertEqual(o['model'],'gpt-6.1-sol')
        self.assertEqual(o['max_completion_tokens'],8192)
        self.assertEqual(o['reasoning_effort'],'medium')
        self.assertEqual(o['messages'][0]['role'],'developer')
        self.assertNotIn('temperature',o)
        self.assertNotIn('top_p',o)
        self.assertNotIn('max_tokens',o)

    def test_responses_request(self):
        o=m.transform_request({'messages':[{'role':'system','content':'x'},{'role':'user','content':'y'}], 'stream':False},api='responses',model='gpt-6.1-sol',effort='low',max_output_tokens=6000)
        self.assertEqual(o['reasoning'],{'effort':'low'})
        self.assertEqual(o['max_output_tokens'],6000)
        self.assertEqual(o['store'],False)

    def test_stream_and_tool_denied(self):
        with self.assertRaises(ValueError):
            m.transform_request({'messages':[{'role':'user','content':'x'}],'stream':True},api='chat',model='gpt-6.1-sol',effort='low',max_output_tokens=256)
        with self.assertRaises(ValueError):
            m.transform_request({'messages':[{'role':'tool','content':'x'}]},api='chat',model='gpt-6.1-sol',effort='low',max_output_tokens=256)

    def test_usage_and_pricing(self):
        resp={'usage':{'prompt_tokens':10000,'completion_tokens':3000,
                       'prompt_tokens_details':{'cached_tokens':2000},
                       'completion_tokens_details':{'reasoning_tokens':2900}}}
        usage=m.parse_usage(resp,'chat')
        self.assertEqual(usage['reasoning_tokens'],2900)
        self.assertAlmostEqual(m.price(usage,m.PRICES),0.0462)

    def test_responses_output_translation(self):
        resp={'id':'r1','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"ratios":[0.1,0.2,0.3,0.4]}'}]}],
              'usage':{'input_tokens':95,'output_tokens':500,'output_tokens_details':{'reasoning_tokens':490}}}
        usage=m.parse_usage(resp,'responses')
        obj=m.normalise_response(resp,'responses',usage)
        self.assertEqual(obj['choices'][0]['message']['content'],'{"ratios":[0.1,0.2,0.3,0.4]}')
        self.assertEqual(obj['usage']['completion_tokens'],500)

    def test_http_url_tls_guard(self):
        self.assertEqual(m.upstream_url('https://example.net/v1','chat'),'https://example.net/v1/chat/completions')
        self.assertEqual(m.upstream_url('https://example.net/v1/chat/completions','chat'),'https://example.net/v1/chat/completions')
        with self.assertRaises(ValueError):
            m.upstream_url('http://example.net/v1','chat')
        with self.assertRaises(ValueError):
            m.upstream_url('https://user:secret@example.net/v1','chat')

    def test_mock_http_proxy_full_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=Path(d)/'usage.jsonl'
            seen=[]
            def fake_invoke(body,*,url,api_key,timeout):
                seen.append(body)
                return {'id':'chat-test','choices':[{'message':{'role':'assistant','content':'{"ratios":[0.2,0.3,0.4,0.5]}'},'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':12000,'completion_tokens':400,
                                 'completion_tokens_details':{'reasoning_tokens':370}}}
            state=m.ProxyState(upstream='https://localhost/v1/chat/completions',key='SECRET_NO_LOG',api='chat',model='gpt-6.1-sol',effort='medium',max_output_tokens=8192,ledger=ledger,prices=m.PRICES,budget=3,max_calls=1,timeout=5)
            class Handler(m.LocalProxyHandler):pass
            Handler.state=state
            server=m.ThreadingHTTPServer(('127.0.0.1',0),Handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_address[1]}/v1/chat/completions'
            payload={'messages':[{'role':'system','content':'demo'},{'role':'user','content':'demo'}],'temperature':0.1,'max_tokens':4096,'stream':False}
            try:
                with patch.object(m,'invoke_upstream',side_effect=fake_invoke):
                    req=Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    with urlopen(req,timeout=5) as r:
                        obj=json.loads(r.read())
                    self.assertEqual(obj['choices'][0]['message']['content'],'{"ratios":[0.2,0.3,0.4,0.5]}')
                    with self.assertRaises(Exception):
                        with urlopen(req,timeout=5) as r:r.read()
            finally:
                server.shutdown();server.server_close();thread.join(timeout=3)
            self.assertEqual(len(seen),1)
            self.assertEqual(m.totals(m.read_events(ledger))['requests_ok'],1)
            self.assertNotIn('SECRET_NO_LOG', ledger.read_text())

    def test_manifest_hash_verification(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'p1b_data').mkdir();(root/'data'/'GEANT').mkdir(parents=True)
            (root/'main.py').write_text('environment')
            (root/'configs_main.yaml').write_text('settings')
            sample=root/'data'/'GEANT'/'gpt61_dev_seed7601.pkl'
            sample.write_bytes(b'fake-offline-only')
            mf={'dataset_id':'gpt61_dev_seed7601','split':'validation','n_scenes':20,
                'seed':7601, 'sample_file':str(sample),'sample_sha256':m.digest(sample),
                'main_py_sha256':m.digest(root/'main.py'),
                'config_sha256':m.digest(root/'configs_main.yaml')}
            m.save_json(root/'p1b_data'/'gpt61_dev_seed7601.json',mf)
            m.verify_dataset(root,'gpt61_dev_seed7601',20)
            sample.write_bytes(b'corrupted')
            with self.assertRaises(RuntimeError):
                m.verify_dataset(root,'gpt61_dev_seed7601',20)

if __name__ == '__main__':unittest.main()
