"""Independent per-role LLM calls and strict 4+4+4+3 action contract."""
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import requests

from .local_obs import AGENTS, make_messages


class InvalidDecision(ValueError):
    pass


class PolicyFailure(RuntimeError):
    def __init__(self, role, cause, partial=None):
        super().__init__("{} failed: {}".format(role, cause))
        self.role = role
        self.partial = partial or {}


@dataclass
class Completion:
    text: str
    latency_s: float
    finish_reason: str = "stop"
    usage: object = None


def parse_ratios(text, agent):
    if not isinstance(text, str):
        raise InvalidDecision("Response content is not a string")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if cleaned.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if not match:
            raise InvalidDecision("Malformed JSON code fence")
        cleaned = match.group(1).strip()
    try:
        parsed = json.loads(cleaned, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError) as e:
        raise InvalidDecision("Not a valid JSON object: {}".format(e))
    if not isinstance(parsed, dict) or set(parsed) != {"ratios"}:
        raise InvalidDecision("Response must contain only 'ratios'")
    vals = parsed["ratios"]
    expected = 3 if agent == "agent_g" else 4
    if not isinstance(vals, list) or len(vals) != expected:
        raise InvalidDecision("{}: expected {} ratios".format(agent, expected))
    out = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (float, int)):
            raise InvalidDecision("Ratios must be JSON numbers, not strings/bools")
        if not math.isfinite(v) or v < 0.0 or v > 1.0:
            raise InvalidDecision("Ratios must be finite in [0,1]")
        out.append(float(v))
    return out


def to_env_action(agent, ratios):
    """Map physical [0,1] ratios to XuanCe [-1,1] actions; pad unused g slot."""
    expected = 3 if agent == "agent_g" else 4
    if len(ratios) != expected:
        raise InvalidDecision("Wrong action length for {}".format(agent))
    arr = np.asarray(ratios, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or np.any(arr > 1):
        raise InvalidDecision("Invalid action ratios")
    env_action = 2.0 * arr - 1.0
    if agent == "agent_g":
        env_action = np.concatenate((env_action, [0.0]))  # fourth value is unused
    return env_action.astype(np.float32)


class ChatCompletionsClient:
    """OpenAI-compatible Chat Completions API. No external framework required."""

    def __init__(self, base_url=None, api_key=None, model=None, temperature=0.1,
                 max_tokens=4096, read_timeout=120.0, stream=True, session=None):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or
                         "http://192.168.131.35:30001/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
        if not self.api_key or self.api_key.strip().lower() in ("sk-xxx", "your-key", "your_api_key"):
            raise ValueError("LLM_API_KEY must be configured with a real token")
        self.model = model or os.getenv("P1A_MODEL", "qwen3-next-80b-a3b-instruct")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.read_timeout = float(read_timeout)
        self.stream = bool(stream)
        self.session = session or requests.Session()

    @staticmethod
    def _read_sse(response):
        pieces = []
        usage = None
        finish_reason = None
        done = False
        saw_event = False
        # Each delta.content belongs to the final answer. reasoning_content is
        # intentionally NOT included in JSON parsing.
        for line in response.iter_lines(decode_unicode=False):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", "replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            try:
                event = json.loads(data)
            except ValueError as exc:
                raise InvalidDecision("Malformed SSE chunk") from exc
            saw_event = True
            if event.get("error"):
                raise InvalidDecision("Model stream returned an error")
            if event.get("usage") is not None:
                usage = event["usage"]
            choices = event.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    pieces.append(content)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
        if not saw_event:
            raise InvalidDecision("No valid SSE events")
        if finish_reason == "length":
            raise InvalidDecision("Model reached max_tokens")
        if finish_reason not in ("stop", None):
            raise InvalidDecision("Unexpected finish_reason: {}".format(finish_reason))
        if not done and finish_reason != "stop":
            raise InvalidDecision("Stream ended without an explicit stop marker")
        if not "".join(pieces).strip():
            raise InvalidDecision("Empty final response (possibly reasoning-only)")
        return "".join(pieces), finish_reason or "stop", usage

    def complete(self, agent, messages):
        headers = {"Authorization": "Bearer " + self.api_key,
                   "Content-Type": "application/json"}
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "stream": self.stream}
        start = time.perf_counter()
        with self.session.post(
            self.base_url + "/chat/completions", json=body, headers=headers,
            timeout=(10.0, self.read_timeout), stream=self.stream
        ) as resp:
            # Don't print response body (may contain sensitive generated content).
            if resp.status_code != 200:
                raise RuntimeError("LLM HTTP {}".format(resp.status_code))
            if self.stream:
                content, reason, usage = self._read_sse(resp)
            else:
                try:
                    data = resp.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"] or ""
                    reason = choice.get("finish_reason")
                    usage = data.get("usage")
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    raise InvalidDecision("Malformed non-stream response") from exc
                if reason == "length":
                    raise InvalidDecision("Model reached max_tokens")
                if reason != "stop" or not content:
                    raise InvalidDecision("No completed final answer")
        return Completion(content, time.perf_counter() - start, reason, usage)


class MockClient:
    """EXPLICITLY not a model result; use only for pipeline testing."""
    model = "mock_not_a_llm_result"

    def complete(self, agent, messages):
        vals = [0.2, 0.2, 0.2] if agent == "agent_g" else [0.2] * 4
        return Completion(json.dumps({"ratios": vals}), 0.0, "stop", None)


class DecentralizedPolicy:
    def __init__(self, client):
        self.client = client

    def act(self, observations):
        if not isinstance(observations, dict) or set(observations) != set(AGENTS):
            raise ValueError("Expected exactly four raw local observations")
        result = {}
        env_actions = {}
        for agent in AGENTS:
            try:
                # Only obs[agent] is ever passed to prompt construction.
                messages = make_messages(agent, observations[agent])
                prompt_bytes = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
                prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
                completion = self.client.complete(agent, messages)
                if completion.finish_reason != "stop":
                    raise InvalidDecision("Model did not finish normally")
                ratios = parse_ratios(completion.text, agent)
                env_actions[agent] = to_env_action(agent, ratios)
            except Exception as exc:
                raise PolicyFailure(agent, str(exc), partial=result) from exc
            result[agent] = {
                "ratios": ratios,
                "raw_model_output": completion.text,
                "model_latency_s": completion.latency_s,
                "usage": completion.usage,
                "prompt_sha256": prompt_hash,
                "messages": messages,
                "env_action": env_actions[agent].tolist(),
            }
        return env_actions, result
