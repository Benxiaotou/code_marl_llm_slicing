"""Run: python p0_selftest.py  (no API token, no model training required)."""
import json
import math
import unittest

import numpy as np

from p0_policy import (AGENTS, InvalidDecision, FixedPolicy, GreedyPolicy,
                       RandomPolicy, LLMPolicy, ratios_to_actions, build_prompt,
                       describe_observations)


def sample_obs():
    return {k:np.zeros(240,dtype=np.float32) for k in AGENTS}


def valid_decision():
    return {k:([0., .5, 1., .25] if k!='agent_g' else [0., .5, 1.]) for k in AGENTS}


class FakeResponse:
    def __init__(self, content=None, finish_reason='stop'):
        self.content = content if content is not None else json.dumps(valid_decision())
        self.finish_reason = finish_reason
    def raise_for_status(self): pass
    def json(self):
        return {'choices':[{'message':{'content':self.content},'finish_reason':self.finish_reason}],
                'usage':{'total_tokens':42}}


class FakeSession:
    def __init__(self, content=None, error=None, finish_reason='stop'):
        self.content=content
        self.error=error
        self.finish_reason=finish_reason
        self.calls=[]
    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({'url':url,'json':json,'headers':headers,'timeout':timeout})
        if self.error:
            raise self.error
        return FakeResponse(self.content,self.finish_reason)


class P0ContractTests(unittest.TestCase):
    def test_boundaries_and_unused_dimension(self):
        a=ratios_to_actions(json.dumps(valid_decision()))
        np.testing.assert_allclose(a['agent_k1'],[-1.,0.,1.,-.5])
        np.testing.assert_allclose(a['agent_g'],[-1.,0.,1.,0.])
        self.assertTrue(all(a[k].shape==(4,) for k in AGENTS))

    def test_rejects_bad_json_and_extra_keys(self):
        for bad in ('not-json','{"agent_k1":[]}', '{}', '', '[]'):
            with self.subTest(raw=bad):
                with self.assertRaises(InvalidDecision): ratios_to_actions(bad)
        x=valid_decision(); x['extra']=[.2]
        with self.assertRaises(InvalidDecision): ratios_to_actions(json.dumps(x))

    def test_rejects_bad_ranges_nonfinite_and_wrong_dimensions(self):
        tests = [([-.01,.5,1.,.1],'negative'), ([1.01,.5,.5,.1],'too large'),
                 ([float('nan'),.5,.5,.1],'nan'),([float('inf'),.5,.5,.1],'inf'),
                 (['0.2',.5,.5,.1],'string'),([True,.5,.5,.1],'bool'),
                 ([.5,.5,.5],'missing dimension')]
        for v,case in tests:
            x=valid_decision();x['agent_k1']=v
            with self.subTest(case=case):
                with self.assertRaises(InvalidDecision): ratios_to_actions(json.dumps(x))
        x=valid_decision();x['agent_g']=[.5,.5,.5,.5]
        with self.assertRaises(InvalidDecision): ratios_to_actions(json.dumps(x))

    def test_code_fences_and_think_tags(self):
        x=json.dumps(valid_decision())
        for raw in ('```json\n'+x+'\n```','<think>analysis</think>\n'+x):
            y=ratios_to_actions(raw)
            self.assertEqual(y['agent_k1'].shape,(4,))

    def test_obs_shape_and_prompt(self):
        obs=sample_obs()
        s=build_prompt(obs)
        self.assertIn('agent_k1',s)
        self.assertIn('physical_cpu_allocated_ratio', s)
        self.assertEqual(len(describe_observations(obs)),4)
        obs['agent_k1']=np.zeros(239)
        with self.assertRaises(ValueError): build_prompt(obs)

    def test_heuristic_baselines_do_not_depend_on_api(self):
        for policy in (FixedPolicy(),GreedyPolicy(),RandomPolicy(1)):
            out=policy.act(sample_obs())
            self.assertFalse(out.used_fallback)
            self.assertTrue(all(np.isfinite(out.action[k]).all() for k in AGENTS))
            self.assertTrue(all((out.action[k] >= -1.).all() and (out.action[k] <= 1.).all() for k in AGENTS))

    def test_llm_mock_request_success(self):
        ses=FakeSession()
        policy=LLMPolicy(api_key='dummy-key',session=ses)
        out=policy.act(sample_obs())
        self.assertEqual(out.origin,'llm')
        self.assertFalse(out.used_fallback)
        self.assertEqual(out.usage['total_tokens'],42)
        self.assertTrue(out.prompt_hash)
        self.assertEqual(len(ses.calls),1)
        self.assertTrue(ses.calls[0]['url'].endswith('/v1/chat/completions'))
        self.assertFalse(ses.calls[0]['json']['stream'])
        self.assertEqual(ses.calls[0]['json']['model'],'deepseek-v4-flash-0731')
        self.assertEqual(ses.calls[0]['headers']['Authorization'],'Bearer dummy-key')

    def test_llm_failure_aborts_by_default(self):
        for ses in (FakeSession(content='bad'), FakeSession(error=TimeoutError('server timeout')),
                    FakeSession(finish_reason='length')):
            with self.subTest(error=ses.error, raw=ses.content):
                policy=LLMPolicy(api_key='dummy',session=ses)
                with self.assertRaises(RuntimeError): policy.act(sample_obs())

    def test_llm_failure_fallback_is_labeled(self):
        policy=LLMPolicy(api_key='dummy',session=FakeSession(content='bad'),fail_mode='rule')
        r=policy.act(sample_obs())
        self.assertTrue(r.used_fallback)
        self.assertIsNotNone(r.error)
        self.assertEqual(r.origin,'llm')
        self.assertTrue(all(k in r.action for k in AGENTS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
