"""P0: no-RL joint resource decisions from four existing MARL observations.

All policies receive only the original four 240-dimensional observations.
The real LLM policy outputs ratios [0,1], converted to environment actions [-1,1].
The unused fourth action of agent_g is always fixed to the neutral value 0.
"""
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

AGENTS = ("agent_k1", "agent_k2", "agent_k3", "agent_g")
NODE_COUNT = 12
LINK_SLOTS = 36
OBS_SIZE = 5 * NODE_COUNT + 5 * LINK_SLOTS


class InvalidDecision(ValueError):
    pass


@dataclass
class Decision:
    action: Dict[str, np.ndarray]
    origin: str
    latency_s: float = 0.0
    used_fallback: bool = False
    error: Optional[str] = None
    raw_response: Optional[str] = None
    prompt_hash: Optional[str] = None
    usage: Optional[dict] = None


def check_obs(observations):
    if not isinstance(observations, dict) or set(observations) != set(AGENTS):
        raise ValueError("Observation must contain exactly four agent keys")
    result = {}
    for agent in AGENTS:
        vec = np.asarray(observations[agent], dtype=float).reshape(-1)
        if vec.size != OBS_SIZE or not np.isfinite(vec).all():
            raise ValueError("Bad observation for {}: expected {} finite values".format(agent, OBS_SIZE))
        result[agent] = vec
    return result


def _float(x):
    return round(float(x), 4)


def _two_control_groups(indices):
    """Matches main.py:_get_scale_repeat_action (np.repeat with ceil(n/2))."""
    scale = int(math.ceil(len(indices) / 2.0)) if indices else 0
    return [indices[:scale], indices[scale:]]


def describe_observations(observations):
    """Interpret the ORIGINAL, fully normalized obs; no hidden env/global state.

    First 60 numbers: 12 node slots * 5.
    Next 108: 36 link slots * 3.
    Last 72: 36 link endpoints * 2. Zero-padded links are not reliably
    distinguishable from legitimate all-zero records; keep a zero-pad flag.
    """
    obs = check_obs(observations)
    desc = {}
    for agent in AGENTS:
        x = obs[agent]
        nodes = x[:60].reshape(12, 5)
        links = x[60:168].reshape(36, 3)
        edges = x[168:240].reshape(36, 2)
        if agent != "agent_g":
            active_nodes = [i for i, row in enumerate(nodes) if row[0] >= .5]
            active_links = [i for i, row in enumerate(links) if row[0] >= .5]
            ng, lg = _two_control_groups(active_nodes), _two_control_groups(active_links)
            node_rows = [{"i": i, "cpu_capacity_ratio": _float(row[1]),
                          "cpu_used_ratio": _float(row[2]),
                          "cache_capacity_ratio": _float(row[3]),
                          "cache_used_ratio": _float(row[4])}
                         for i, row in enumerate(nodes) if i in active_nodes]
            link_rows = [{"i": i, "bandwidth_capacity_ratio": _float(links[i,1]),
                          "bandwidth_used_ratio": _float(links[i,2])}
                         for i in active_links]
            desc[agent] = {
                "role": "existing_slice", "overlap_node_count": len(active_nodes),
                "overlap_link_count": len(active_links),
                "node_control_groups": ng, "link_control_groups": lg,
                "overlap_nodes": node_rows, "overlap_links": link_rows,
                "overlap_mean_cpu_used_ratio": _float(np.mean(nodes[active_nodes, 2])) if active_nodes else None,
                "overlap_mean_cache_used_ratio": _float(np.mean(nodes[active_nodes, 4])) if active_nodes else None,
                "overlap_mean_link_used_ratio": _float(np.mean(links[active_links, 2])) if active_links else None,
            }
        else:
            desc[agent] = {
                "role": "new_slice", "node_count": NODE_COUNT,
                "nodes": [{"i": i, "physical_cpu_allocated_ratio": _float(row[1]),
                           "requested_cpu_capacity_ratio": _float(row[2]),
                           "physical_cache_allocated_ratio": _float(row[3]),
                           "requested_cache_capacity_ratio": _float(row[4])}
                          for i, row in enumerate(nodes)],
                "links": [{
                    "i": i,
                    "physical_bandwidth_allocated_ratio": _float(row[1]),
                    "requested_bandwidth_ratio": _float(row[2]),
                    "free_bandwidth_ratio_before_k_release": _float(1.0 - row[1]),
                    "bandwidth_shortfall_ratio_before_k_release":
                        _float(max(0.0, row[1] + row[2] - 1.0)),
                    "min_g_bw_reduction_if_no_k_link_release":
                        _float(max(0.0, (row[1] + row[2] - 1.0)
                                   / (0.7 * row[2]))) if row[2] > 1e-12 else 0.0,
                    "shortfall_after_max_g_bw_reduction_if_no_k_link_release":
                        _float(max(0.0, row[1] + 0.3 * row[2] - 1.0)),
                }
                for i, row in enumerate(links)
                if row[1] != 0 or row[2] != 0],
                "max_cpu_pressure": _float(np.max(nodes[:, 1] + nodes[:, 2])),
                "max_cache_pressure": _float(np.max(nodes[:, 3] + nodes[:, 4])),
                "max_bandwidth_pressure_observed": _float(np.max(links[:, 1] + links[:, 2])),
                "note": "physical_*_allocated_ratio is ALREADY allocated, not free; 1 - value is unused physical capacity",
            }
    return desc


SYSTEM_PROMPT = """You control cooperative admission of a new slice g on a shared GEANT substrate.
Return a single JSON object only, with keys agent_k1, agent_k2, agent_k3, agent_g.
Use ratios in [0,1]: k1/k2/k3: FOUR numbers [node_group_0, node_group_1, link_group_0, link_group_1]; g: THREE numbers [cpu_reduction, cache_reduction, bandwidth_reduction].
Higher existing-slice ratios release more of their allocated resources but can incur migrations and rerouting costs.
Higher g ratios reduce the new slice's resource demand, which can itself reduce service and incur costs.
The new slice must be admitted if feasible, while controlling disruption to the existing slices.
Do not confuse capacity and utilization. Already-allocated physical resources are not free resources.
Existing nodes/links are grouped in ascending overlap index order, matching the original environment's 2-way repeated actions.
Do not invent missing network attributes. Do not use tool calls, Markdown fences, explanations, or extra fields.
Choose the fifteen numeric ratios based on THIS scenario, not a canned vector.
Feasibility rules of the ORIGINAL environment (important, not generic assumptions):
1. For an existing slice k, a node-group action ratio r releases 0.8*r of
   that slice's previously allocated CPU AND cache capacity at its controlled
   overlapping nodes. A link-group action ratio r likewise releases 0.8*r
   of the controlled bandwidth capacities. Larger releases can cause migration
   or rerouting costs. The two groups use the indices supplied below.
2. For a new-slice g node, the ORIGINAL environment checks available physical
   CPU/cache against the ORIGINAL g CPU/cache requests before adding the
   reduced g allocation. Thus increasing g's CPU/cache reduction ratios does
   NOT relax this implementation's node-admission checks. To make g feasible,
   the existing slices must free enough physical CPU and cache at the relevant
   overlapping nodes. Do not assume all g node indices match k node indices.
3. In the new-slice observation, physical_*_allocated_ratio means occupied
   capacity, NOT free capacity. Before existing slices release resources,
   occupied_ratio + requested_ratio > 1 indicates a normalized CPU/cache
   shortfall at that g node. Existing-slice node releases can close it.
4. For g bandwidth, effective requested bandwidth is original*(1-0.7*r_g_bw).
   More reduction may help fit physical link capacity, BUT resulting new-slice
   link utilization must be at most 0.7 or admission still fails.
Decision priorities: (a) find a plausible joint action that enables g admission,
especially by addressing positive CPU/cache pressure; (b) avoid unnecessary
migration, rerouting, and bandwidth reduction; (c) preserve system utility.
The g-link bandwidth-capacity pressure fields are DERIVED FROM THE SAME ORIGINAL OBSERVATIONS and refer to the state BEFORE any existing-slice link release. The min_g_bw_reduction_if_no_k_link_release field is a per-link lower bound, NOT a guaranteed feasible solution. If its value exceeds 1 for any g link, reducing g bandwidth alone cannot fit that link: some relevant existing-slice link-group release is necessary. If it is at most 1, increasing g's bandwidth reduction MAY help without k-link releases; also respect the <=0.7 link-utilization check. Do not give all existing-slice link groups zero by default while bandwidth shortages remain. Avoid excessive node releases once CPU/cache are plausible. Prefer the smallest joint releases that plausibly satisfy both NODE and LINK admission checks. Use the overlap group and utilization/capacity observations to choose ratios.
Do not make all values equal by default; do not invent exact physical-node
mappings or claim to have proved feasibility from the limited observations.
Return ONLY the required JSON object, with finite numeric values in [0,1],
exactly four numbers for each k agent and three for g. No example values,
no explanations, no Markdown.
"""


def build_prompt(observations):
    desc = describe_observations(observations)
    return "Current four-agent observations (normalized fractions, original MARL information only):\n" + \
           json.dumps(desc, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + \
           "\nChoose one coordinated action with the required exact JSON schema."


def _parse_json_object(text):
    if not isinstance(text, str) or not text.strip():
        raise InvalidDecision("Empty model response")
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    if s.startswith("<think>") and "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError) as exc:
        raise InvalidDecision("Expected a single valid JSON object: " + str(exc))
    if not isinstance(obj, dict) or set(obj) != set(AGENTS):
        raise InvalidDecision("Missing or extra agent keys")
    return obj


def ratios_to_actions(raw_json):
    """Reject malformed/out-of-bounds/nonfinite values; never silently clip."""
    obj = _parse_json_object(raw_json)
    actions = {}
    for agent in AGENTS:
        vals = obj[agent]
        length = 3 if agent == "agent_g" else 4
        if not isinstance(vals, list) or len(vals) != length:
            raise InvalidDecision("{} requires {} numbers".format(agent, length))
        if any(isinstance(v, (bool, str, dict, list)) or not isinstance(v, (int, float)) for v in vals):
            raise InvalidDecision("{} contains a nonnumeric field".format(agent))
        x = np.asarray(vals, dtype=float)
        if not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise InvalidDecision("{} ratio must be finite and within [0,1]".format(agent))
        if agent == "agent_g":
            x = np.append(x, 0.5)  # fourth agent_g component unused; 0 neutral environment action
        a = (2.0 * x - 1.0).astype(np.float32)
        actions[agent] = a
    return actions


def ratio_actions(ratios):
    return ratios_to_actions(json.dumps(ratios, ensure_ascii=False, allow_nan=False))


class FixedPolicy:
    name = "fixed_half"
    def act(self, observations):
        check_obs(observations)
        return Decision(ratio_actions({k: ([.5] * (3 if k == "agent_g" else 4)) for k in AGENTS}), self.name)


class GreedyPolicy:
    name = "rule"
    def act(self, observations):
        d = describe_observations(observations)
        out = {}
        for k in AGENTS[:3]:
            def group_ratio(rows, indices, use_keys):
                sub = [rows[i] for i in indices]
                if not sub: return 0.0
                vals = [min(1.0, max(0.0, row[key])) for row in sub for key in use_keys]
                # Keep used resources; release more from the lowest utilized groups.
                return round(max(0.0, min(.55, .7 * (1.0 - max(vals)))), 4)
            ns = {x["i"]:x for x in d[k]["overlap_nodes"]}
            ls = {x["i"]:x for x in d[k]["overlap_links"]}
            out[k] = [group_ratio(ns, idx, ("cpu_used_ratio", "cache_used_ratio"))
                      for idx in d[k]["node_control_groups"]] + [group_ratio(ls,idx,("bandwidth_used_ratio",))
                      for idx in d[k]["link_control_groups"]]
        g = d["agent_g"]
        cpu = max(0., g["max_cpu_pressure"] - 1.)
        cache = max(0., g["max_cache_pressure"] - 1.)
        bw = max(0., g["max_bandwidth_pressure_observed"] - 1.)
        out["agent_g"] = [min(1., cpu), min(1., cache), min(1., bw)]
        return Decision(ratio_actions(out), self.name)


class RandomPolicy:
    name = "random"
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
    def act(self, observations):
        check_obs(observations)
        ratios = {k: self.rng.uniform(0, 1, size=(3 if k == "agent_g" else 4)).tolist() for k in AGENTS}
        return Decision(ratio_actions(ratios), self.name)


class LLMPolicy:
    name = "llm"
    def __init__(self, api_key=None, base_url=None, model=None, timeout=90,
                 temperature=0.0, max_tokens=4096, fail_mode="abort", session=None, stream=False):
        self.key = api_key or os.environ.get("LLM_API_KEY")
        if not self.key:
            raise ValueError("LLM_API_KEY is required for LLM evaluation; no silent heuristic substitution")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "http://192.168.131.35:30001/v1").rstrip("/")
        if not (self.base_url.endswith("/v1")):
            raise ValueError("LLM_BASE_URL should end with /v1, not /chat/completions")
        self.model = model or os.environ.get("LLM_MODEL") or "deepseek-v4-flash-0731"
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.stream = bool(stream)
        if self.timeout <= 0 or self.max_tokens <= 0:
            raise ValueError("timeout and max_tokens must be positive")
        if fail_mode not in ("abort", "rule"):
            raise ValueError("fail_mode must be abort or rule")
        self.fail_mode = fail_mode
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    @staticmethod
    def _decode_stream(reply):
        """Consume OpenAI-compatible SSE; do not mistake reasoning_content for final JSON.

        The server may send thinking chunks before final content. Streamed HTTP
        only helps if the gateway actually sends chunks before read_timeout.
        """
        fragments = []
        usage = None
        finish_reason = None
        saw_done = False
        saw_event = False
        try:
            # A small chunk size avoids substantial client-side buffering of
            # small token events by requests/urllib3.
            for line in reply.iter_lines(chunk_size=64, decode_unicode=False):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk = json.loads(payload)
                except (ValueError, TypeError) as exc:
                    raise InvalidDecision("Malformed SSE data: " + str(exc)) from exc
                saw_event = True
                if not isinstance(chunk, dict):
                    raise InvalidDecision("SSE chunk must be an object")
                if "error" in chunk:
                    raise InvalidDecision("SSE server returned error: " + str(chunk["error"])[:300])
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        raise InvalidDecision("Nontext content fragment in SSE")
                    fragments.append(content)
                # Ignore reasoning_content: not part of the answer JSON.
        finally:
            if hasattr(reply, "close"):
                reply.close()
        if not saw_event:
            raise InvalidDecision("No SSE data chunks received (streaming unsupported or empty)")
        if finish_reason == "length":
            raise InvalidDecision("Model response reached max_tokens")
        if finish_reason not in (None, "stop"):
            raise InvalidDecision("Unexpected stream finish_reason: " + str(finish_reason))
        if not saw_done and finish_reason != "stop":
            raise InvalidDecision("SSE stream terminated without [DONE] or finish_reason=stop")
        raw = "".join(fragments)
        if not raw.strip():
            raise InvalidDecision("SSE ended without final assistant content (reasoning-only response)")
        return raw, usage

    def act(self, observations):
        prompt = build_prompt(observations)
        prompt_hash = hashlib.sha256((SYSTEM_PROMPT + prompt).encode("utf-8")).hexdigest()
        before = time.perf_counter()
        raw = None
        usage = None
        try:
            body = {"model": self.model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": prompt}],
                    "temperature": self.temperature, "max_tokens": self.max_tokens,
                    "stream": self.stream}
            kwargs = {"json": body, "headers": {"Authorization": "Bearer " + self.key},
                      "timeout": (10, self.timeout)}
            if self.stream:
                kwargs["stream"] = True
            reply = self.session.post(self.base_url + "/chat/completions", **kwargs)
            reply.raise_for_status()
            if self.stream:
                raw, usage = self._decode_stream(reply)
            else:
                response = reply.json()
                choice = response["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise InvalidDecision("Model response reached max_tokens")
                if choice.get("finish_reason") not in (None, "stop"):
                    raise InvalidDecision("Unexpected finish_reason: " + str(choice.get("finish_reason")))
                raw = choice["message"]["content"]
                usage = response.get("usage")
            action = ratios_to_actions(raw)
            return Decision(action, "llm", time.perf_counter() - before,
                            raw_response=raw, prompt_hash=prompt_hash, usage=usage)
        except Exception as exc:
            spent = time.perf_counter() - before
            if self.fail_mode == "abort":
                raise RuntimeError("LLM decision failed after {:.2f}s: {}".format(spent, exc)) from exc
            rule = GreedyPolicy().act(observations)
            return Decision(rule.action, "llm", spent, used_fallback=True,
                            error=str(exc), raw_response=raw, prompt_hash=prompt_hash, usage=usage)
