"""P1-A+ policy: no change to original P1-A policy or API client."""
import hashlib
import json

from p1a.local_obs import AGENTS
from p1a.policy import InvalidDecision, PolicyFailure, parse_ratios, to_env_action
from .semantics import make_semantic_messages


class SemanticDecentralizedPolicy:
    def __init__(self, client):
        self.client = client

    def act(self, observations):
        if not isinstance(observations, dict) or set(observations) != set(AGENTS):
            raise ValueError('Expected exactly four raw local observations')
        result = {}
        env_actions = {}
        for agent in AGENTS:
            try:
                messages = make_semantic_messages(agent, observations[agent])
                prompt_bytes = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')
                prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
                completion = self.client.complete(agent, messages)
                if completion.finish_reason != 'stop':
                    raise InvalidDecision('Model did not finish normally')
                ratios = parse_ratios(completion.text, agent)
                env_actions[agent] = to_env_action(agent, ratios)
            except Exception as exc:
                raise PolicyFailure(agent, str(exc), partial=result) from exc
            result[agent] = {
                'ratios': ratios,
                'raw_model_output': completion.text,
                'model_latency_s': completion.latency_s,
                'usage': completion.usage,
                'prompt_sha256': prompt_hash,
                'messages': messages,
                'env_action': env_actions[agent].tolist(),
            }
        return env_actions, result
