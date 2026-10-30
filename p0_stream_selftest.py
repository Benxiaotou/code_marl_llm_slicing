"""Offline regression tests for P0 optional SSE decoding; no API or training."""
import json
import unittest
import numpy as np

from p0_policy import AGENTS, LLMPolicy, InvalidDecision


def demo_obs():
    return {k: np.zeros(240, dtype=np.float32) for k in AGENTS}


def demo_answer():
    return json.dumps({k: [0.5] * (3 if k == 'agent_g' else 4) for k in AGENTS})


class StreamingResponse:
    def __init__(self, lines):
        self.lines = list(lines)
        self.closed = False
    def raise_for_status(self):
        pass
    def iter_lines(self, chunk_size=64, decode_unicode=False):
        for line in self.lines:
            yield line.encode('utf-8')
    def close(self):
        self.closed = True


class StreamingSession:
    def __init__(self, lines):
        self.calls = []
        self.response = StreamingResponse(lines)
    def post(self, url, json=None, headers=None, timeout=None, stream=None):
        self.calls.append((url, json, timeout, stream))
        return self.response


def line(delta=None, finish=None, usage=None):
    chunk = {'choices': []}
    if delta is not None or finish is not None:
        chunk['choices'] = [{'delta': delta or {}, 'finish_reason': finish}]
    if usage is not None:
        chunk['usage'] = usage
    return 'data: ' + json.dumps(chunk)


class StreamTests(unittest.TestCase):
    def run_once(self, lines, fail_mode='abort'):
        ses = StreamingSession(lines)
        policy = LLMPolicy(api_key='test-secret', session=ses, stream=True,
                           fail_mode=fail_mode, timeout=180)
        return ses, policy.act(demo_obs())

    def test_normal_stream_with_reasoning_and_usage(self):
        text = demo_answer()
        lines = [line({'role': 'assistant', 'reasoning_content': 'thinking'}),
                 line({'content': text[:20]}), line({'content': text[20:]}),
                 line(finish='stop'), line(usage={'total_tokens': 123}),
                 'data: [DONE]']
        ses, answer = self.run_once(lines)
        self.assertEqual(answer.origin, 'llm')
        self.assertFalse(answer.used_fallback)
        self.assertEqual(answer.usage['total_tokens'], 123)
        self.assertTrue(all(x.shape == (4,) for x in answer.action.values()))
        self.assertTrue(ses.response.closed)
        self.assertTrue(ses.calls[0][-1])
        self.assertTrue(ses.calls[0][1]['stream'])
        self.assertEqual(ses.calls[0][2], (10, 180.0))

    def test_stop_without_done_is_accepted(self):
        _, answer = self.run_once([line({'content': demo_answer()}), line(finish='stop')])
        self.assertFalse(answer.used_fallback)

    def test_truncated_or_reasoning_only_streams_rejected(self):
        tests = [
            [line({'content': demo_answer()[:20]})],
            [line({'reasoning_content': 'thinking'}), line(finish='stop'), 'data: [DONE]'],
            [line({'content': demo_answer()}), line(finish='length'), 'data: [DONE]'],
            [line({'content': 'invalid'}), 'data: [DONE]'],
            ['data: {bad json', 'data: [DONE]'],
            ['event: message', 'data: [DONE]'],
        ]
        for chunks in tests:
            with self.subTest(chunks=chunks):
                with self.assertRaises(RuntimeError):
                    self.run_once(chunks)

    def test_fallback_is_labeled(self):
        _, a = self.run_once([line({'reasoning_content': 'thinking'}), 'data: [DONE]'],
                             fail_mode='rule')
        self.assertTrue(a.used_fallback)
        self.assertIsNotNone(a.error)

    def test_nonstream_compatibility_with_previous_tests(self):
        self.assertFalse(LLMPolicy(api_key='dummy').stream)


if __name__ == '__main__':
    unittest.main(verbosity=2)
