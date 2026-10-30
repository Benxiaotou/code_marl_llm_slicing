import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from p1a.local_obs import AGENTS, OBS_SIZE, describe_local_observation, make_messages
from p1a.policy import (ChatCompletionsClient, Completion, DecentralizedPolicy,
                        InvalidDecision, MockClient, PolicyFailure, parse_ratios, to_env_action)
from p1a_runner import get_args, make_mini_env_type, run, summarize


def valid_obs(role, mark=0.1, n_links=2):
    node = np.zeros((12, 5), dtype=np.float32)
    node[:, 1:] = mark
    if role != "agent_g":
        node[[1, 4], 0] = 1
    link = np.zeros((36, 3), dtype=np.float32)
    if role == "agent_g":
        link[:n_links, 2] = 0.3
        # physical allocated bandwidth can be zero: still a real link
    else:
        link[:n_links, 1] = 0.25
        link[:n_links, 2] = 0.2
        link[0, 0] = 1
    topology = np.zeros((36, 2), dtype=np.float32)
    topology[:n_links] = np.array([[0, 1], [0.1, 1]])[:n_links]
    return np.concatenate((node.ravel(), link.ravel(), topology.ravel()))


class CaptureClient:
    model = "fake_for_test"

    def __init__(self):
        self.calls = []

    def complete(self, agent, messages):
        self.calls.append((agent, messages))
        return Completion(json.dumps({"ratios": [0.75] * (3 if agent == "agent_g" else 4)}),
                          0.001, "stop", {"test": True})


class FakeResponse:
    def __init__(self, events, status=200):
        self.events = events
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self, decode_unicode=False):
        return iter(self.events)


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *a, **kw):
        self.calls.append((a, kw))
        return self.response


class FakeEnv:
    def __init__(self):
        self.sample_dict = {
            0: (SimpleNamespace(tag=x) for x in ("agent_k1", "agent_k2", "agent_k3", "agent_g", "dg")),
            1: (SimpleNamespace(tag=x) for x in ("agent_k1", "agent_k2", "agent_k3", "agent_g", "dg")),
        }
        self.sample_dict = {i: tuple(v) for i, v in self.sample_dict.items()}
        self.sample_idx = 0
        self.g_done_list = []
        self.dict_eval_index = {k: [] for k in (
            "V", "V_k1", "V_k2", "V_k3", "V_g", "V_k1_0", "V_k2_0",
            "V_k3_0", "X_k1", "X_k2", "X_k3", "cost_k1", "cost_k2",
            "cost_k3", "revenue_k1", "revenue_k2", "revenue_k3", "revenue_g")}
        self.observations = None
        self.infos = None

    def _get_overlap_node_link(self, k_net, g_net):
        return [1, 4], [0], [(1, 4)]

    def _get_v_net_observations(self, k_net, nodes, links, pad):
        assert nodes == [1, 4] and links == [0] and pad
        return valid_obs(k_net.tag)

    def _get_g_net_observations(self, pad):
        return valid_obs("agent_g")

    def reset(self):
        return self.observations, self.infos

    def step(self, actions):
        assert set(actions) == set(AGENTS)
        assert all(len(v) == 4 for v in actions.values())
        self.g_done_list.append(True)
        d = self.dict_eval_index
        values = {"V": 340, "V_k1": 105, "V_k2": 107, "V_k3": 107,
                  "V_g": 21, "V_k1_0": 100, "V_k2_0": 100, "V_k3_0": 100,
                  "cost_k1": 3, "cost_k2": 4, "cost_k3": 5}
        for k in d:
            d[k].append(values.get(k, 0))
        return {}, {a: 1.0 for a in AGENTS}, {}, False, {}


class ObservationTests(unittest.TestCase):
    def test_shapes_and_padding(self):
        self.assertEqual(OBS_SIZE, 240)
        for role in AGENTS:
            payload = describe_local_observation(role, valid_obs(role))
            self.assertEqual(payload["node_count"], 12)
            self.assertEqual(payload["link_count"], 2)
            self.assertEqual(len(payload["links"]), 2)
        g = describe_local_observation("agent_g", valid_obs("agent_g"))
        self.assertEqual(g["links"][0]["physical_allocated_bandwidth_fraction"], 0.0)

    def test_detects_invalid_and_no_global_enrichment(self):
        with self.assertRaises(ValueError):
            describe_local_observation("agent_k1", np.zeros(239))
        broken = valid_obs("agent_g")
        broken[0] = float("nan")
        with self.assertRaises(ValueError):
            describe_local_observation("agent_g", broken)
        self.assertNotIn("physical_network_graph", json.dumps(describe_local_observation("agent_g", valid_obs("agent_g"))))

    def test_independent_prompt(self):
        one = valid_obs("agent_k1", 0.12)
        one2 = valid_obs("agent_k1", 0.41)
        msgs = make_messages("agent_k1", one)
        self.assertNotEqual(msgs, make_messages("agent_k1", one2))
        self.assertEqual(msgs, make_messages("agent_k1", one))
        self.assertNotIn("other_agents_observations", str(msgs))
        self.assertNotIn("allocated_comp", str(msgs))


class ActionTests(unittest.TestCase):
    def test_action_g_fourth_unused(self):
        self.assertEqual(to_env_action("agent_g", [0.0, 0.5, 1.0]).tolist(),
                         [-1.0, 0.0, 1.0, 0.0])
        self.assertEqual(len(to_env_action("agent_k1", [0.0, 1.0, 0.5, 0.5])), 4)

    def test_bad_ratios_rejected(self):
        for text in ('{"ratios": [0.1, 0.2, 0.3]}',
                     '{"ratios":[0.1, 0.2, 0.3, 1.3]}',
                     '{"ratios": [0.1, 0.2, "0.3", 0.4]}',
                     '{"ratios": [0.1, 0.2, NaN, 0.4]}',
                     '{"ratios": [0.1, 0.2, 0.3, 0.4],"other":42}',
                     'junk {"ratios":[0,0,0,0]}'):
            with self.subTest(text=text), self.assertRaises(InvalidDecision):
                parse_ratios(text, "agent_k1")
        self.assertEqual(parse_ratios('```json\n{"ratios":[0,0,1]}\n```', "agent_g"), [0, 0, 1])

    def test_four_calls_and_isolation(self):
        client = CaptureClient()
        source = {a: valid_obs(a, (i + 1) / 10.) for i, a in enumerate(AGENTS)}
        policy = DecentralizedPolicy(client)
        acts, log = policy.act(source)
        self.assertEqual([x[0] for x in client.calls], list(AGENTS))
        self.assertEqual(len(acts), 4)
        for a in AGENTS:
            self.assertEqual(set(log[a]["messages"][1]), {"role", "content"})
            self.assertEqual(log[a]["env_action"], acts[a].tolist())
        # If k2 changes, k1's prompt (and g's prompt) is bit-for-bit unchanged.
        before = [x[1] for x in client.calls]
        client.calls = []
        source["agent_k2"] = valid_obs("agent_k2", 0.9)
        policy.act(source)
        after = [x[1] for x in client.calls]
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[3], after[3])
        self.assertNotEqual(before[1], after[1])

    def test_failure_no_joint_action(self):
        class Broken(CaptureClient):
            def complete(self, agent, messages):
                if agent == "agent_k3":
                    return Completion('{"ratios": []}', 0.1)
                return super().complete(agent, messages)
        client = Broken()
        with self.assertRaises(PolicyFailure) as cm:
            DecentralizedPolicy(client).act({a: valid_obs(a) for a in AGENTS})
        self.assertEqual(cm.exception.role, "agent_k3")
        self.assertEqual(set(cm.exception.partial), {"agent_k1", "agent_k2"})


class TransportTests(unittest.TestCase):
    def test_stream_reasoning_is_excluded(self):
        events = [
            b'data: {"choices":[{"delta":{"reasoning_content":"think","content":null},"finish_reason":null}]}',
            b'data: {"choices":[{"delta":{"content":"{\\"ratios\\":[0,0,0]}"},"finish_reason":"stop"}]}',
            b'data: [DONE]',
        ]
        fake = FakeSession(FakeResponse(events))
        client = ChatCompletionsClient(base_url="http://test/v1", api_key="test", model="qwen",
                                       stream=True, session=fake)
        result = client.complete("agent_g", [])
        self.assertEqual(parse_ratios(result.text, "agent_g"), [0, 0, 0])
        self.assertEqual(fake.calls[0][1]["json"]["stream"], True)
        self.assertEqual(fake.calls[0][0][0], "http://test/v1/chat/completions")

    def test_truncated_stream_rejected(self):
        events = [b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"length"}]}',
                  b'data: [DONE]']
        with self.assertRaises(InvalidDecision):
            ChatCompletionsClient._read_sse(FakeResponse(events))


class RunnerTests(unittest.TestCase):
    def test_minienv_override_without_code_edits(self):
        class Base:
            def __init__(self, conf):
                self.sample_cnt = 600
                self.sample_dict_save_file = "old_full.pkl"
        cls = make_mini_env_type(Base, "mini.pkl", 5)
        env = cls(None)
        self.assertEqual(env.sample_cnt, 5)
        self.assertEqual(env.sample_dict_save_file, "mini.pkl")

    def test_full_mock_pipeline_no_real_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = get_args(["--policy", "mock", "--dataset", "mini", "--count", "2",
                             "--output-dir", tmp + "/run"])
            res = run(args, env_override=FakeEnv(), mapping_override=[0, 599])
            self.assertEqual(res["n_scored"], 2)
            self.assertEqual(res["acceptance_rate_all_attempts"], 1.0)
            self.assertEqual(res["cooperative_valid_rate_all_attempts"], 1.0)
            self.assertEqual(res["policy"], "mock_not_an_llm_result")
            path = Path(tmp) / "run"
            rows = [json.loads(x) for x in (path / "records.jsonl").read_text().splitlines()]
            self.assertEqual([r["original_scene_id"] for r in rows], [0, 599])
            self.assertEqual(len(rows[0]["per_role"]), 4)
            self.assertFalse(rows[0]["fallback_used"])

    def test_no_claim_on_partial_failures(self):
        records = [{"status": "policy_error"}, {"status": "ok", "metrics": {
            "g_done": True, "cooperative_valid": True, "V": 1.0, "V_g": 1.0,
            "V_k1": 1.0, "V_k2": 1.0, "V_k3": 1.0, "cost_sum": 1.0, "reward_mean": 1.0},
            "policy_latency_s": 1.0,
            "per_role": {a: {"model_latency_s": 0.2} for a in AGENTS}}]
        s = summarize(records, "qwen_zeroshot_local")
        self.assertIsNone(s["acceptance_rate_all_attempts"])
        self.assertEqual(s["n_failed_policy"], 1)


if __name__ == "__main__":
    unittest.main()

# Separate comparison tests are discovered by unittest as well.
