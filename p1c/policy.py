"""Frozen Qwen/nearest-neighbour policies with role-only MAPPO history."""
import hashlib
import json
import time

import numpy as np

from p1a.local_obs import AGENTS, make_messages
from p1a.policy import DecentralizedPolicy
from .retrieval import local_numeric_summary
from p1a.policy import PolicyFailure, InvalidDecision, parse_ratios, to_env_action


LABELS = {
    'knn': 'knn_role_local_teacher_actions',
    'icl': 'qwen_coop_icl_role_local_v1',
    'ir_icl': 'qwen_ir_aware_coop_icl_local_v1',
    'ir_prompt_only': 'qwen_ir_prompt_only_coop_icl_local_v1',
    'ir_data_only': 'qwen_ir_data_only_coop_icl_local_v1',
    'ir_rerank': 'qwen_ir_risk_reranked_coop_icl_local_p2a_v1',
    'random_icl': 'qwen_random_icl_role_local_v1',
    'mappo': 'frozen_mappo_teacher_local_v1',
    'zero': 'qwen_zeroshot_local_on_new_validation',
}


def _prompt_features(summary):
    # Do not repeat the static safety-limit numeric anchors that correlated
    # with P1-A+ over-reduction; the retrieval algorithm may still use them.
    return {k: round(float(v), 5) for k, v in summary.items()
            if '_static_no_' not in k}


def _public_history_view(examples):
    """Whitelist only same-role summaries + own teacher actions into Prompt."""
    return [{
        'historical_local_features': _prompt_features(ex['summary']),
        'historical_own_action_ratios': ex['ratios'],
    } for ex in examples]


def make_icl_messages(role, observation, examples, *, is_random=False):
    """Current role local observation + own role local historical demonstrations.

    Historical source IDs, other-role actions and joint rewards are deliberately
    not inserted. History is untrusted demonstration data, not instruction text.
    """
    base = make_messages(role, observation)
    source = ('randomly sampled training history (control experiment)' if is_random
              else 'numerically nearest locally similar training history')
    system = (base[0]['content'] + '\nAdditional historical demonstrations follow. '
              'All historical examples are ONLY your role\'s local summary and '
              'your own frozen MAPPO teacher action, retained from jointly '
              'cooperative training cases. Joint success does not prove that '
              'any one action was individually optimal. Do not blindly copy a '
              'single ratio, do not infer other agents\' observations or actions, '
              'and do not regard static reference limits as precise migration '
              'costs. Select a new action from your own CURRENT local state. '
              'Respond with exactly {"ratios":[...]} and no explanation.')
    current_text = base[1]['content'].split('\nReply as ')[0]
    current_summary = _prompt_features(local_numeric_summary(role, observation))
    user = (current_text + '\n\nYour CURRENT role-local summary, calculated only from the 240D observation:\n'
            + json.dumps(current_summary, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            + '\n\nOwn-role historical demonstrations from ' + source
            + ' (local-only numeric features; these may be imperfect matches):\n'
            + json.dumps(_public_history_view(examples), separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False)
            + '\nOutput exactly one JSON object {"ratios":[...]} using the correct role dimensions.')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


class RetrievalPolicy:
    def __init__(self, library, method, *, client=None, examples=3, seed=1):
        if method not in ('knn', 'icl', 'random_icl'):
            raise ValueError('RetrievalPolicy method must be knn, icl or random_icl')
        if method != 'knn' and client is None:
            raise ValueError('ICL requires a real or explicitly mocked client')
        self.library = library
        self.method = method
        self.client = client
        self.examples = int(examples)
        self.seed = int(seed)
        self.scene_id = None
        if not 1 <= self.examples <= min(len(library.cases[r]) for r in AGENTS):
            raise ValueError('Example count out of range for P1-B library')

    def set_scene(self, scene_id):
        self.scene_id = int(scene_id)

    def _make_messages(self, role, observation, matches):
        return make_icl_messages(role, observation, matches,
                                 is_random=self.method == 'random_icl')

    def act(self, observations):
        if not isinstance(observations, dict) or set(observations) != set(AGENTS):
            raise ValueError('Expected exactly four original local observations')
        if self.scene_id is None:
            raise ValueError('Set current scenario ID before evaluating policy')
        per_role = {}
        env_actions = {}
        for role in AGENTS:
            try:
                start = time.perf_counter()
                if self.method == 'random_icl':
                    matches = self.library.random_cases(role, self.examples,
                                                         self.seed, self.scene_id)
                else:
                    matches = self.library.query(role, observations[role],
                                                 top_k=1 if self.method == 'knn' else self.examples)
                retrieval_time = time.perf_counter() - start
                if self.method == 'knn':
                    ratios = list(matches[0]['ratios'])
                    model_output = None
                    model_time = 0.0
                    usage = None
                    messages = None
                    prompt_hash = None
                else:
                    messages = self._make_messages(role, observations[role], matches)
                    prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True,
                        ensure_ascii=False).encode('utf-8')).hexdigest()
                    completion = self.client.complete(role, messages)
                    if completion.finish_reason != 'stop':
                        raise InvalidDecision('Model did not finish normally')
                    ratios = parse_ratios(completion.text, role)
                    model_output = completion.text
                    model_time = float(completion.latency_s)
                    usage = completion.usage
                env_actions[role] = to_env_action(role, ratios)
                per_role[role] = {
                    'ratios': ratios,
                    'env_action': env_actions[role].tolist(),
                    'retrieval_latency_s': retrieval_time,
                    'model_latency_s': model_time,
                    'usage': usage,
                    'raw_model_output': model_output,
                    'messages': messages,
                    'prompt_sha256': prompt_hash,
                    'retrieved_ids_for_audit_only': [m['experience_id'] for m in matches],
                    'distances_for_audit_only': [m['distance'] for m in matches],
                    'history_shown_to_model': self.method != 'knn',
                }
            except Exception as exc:
                raise PolicyFailure(role, str(exc), partial=per_role) from exc
        return env_actions, per_role


class FrozenMAPPOEvalPolicy:
    """Optional independent benchmark, NOT a source of ICL experiences."""
    def __init__(self, teacher, seed=1):
        self.teacher = teacher
        self.seed = seed
        self.scene_id = None

    def set_scene(self, scene_id):
        self.scene_id = int(scene_id)

    def act(self, observations):
        if self.scene_id is None:
            raise ValueError('Set current scenario ID before evaluating MAPPO')
        start = time.perf_counter()
        try:
            acts = self.teacher.act(observations, original_scene_id=self.scene_id,
                                    seed=self.seed)
            elapsed = time.perf_counter() - start
            from p1b.core import normalize_teacher_actions
            acts = normalize_teacher_actions(acts, assumed_space='env')
            entries = {}
            for role in AGENTS:
                dims = 3 if role == 'agent_g' else 4
                ratios = (np.asarray(acts[role][:dims], dtype=np.float64) + 1) / 2
                entries[role] = {'ratios': ratios.tolist(),
                                 'env_action': acts[role].tolist(),
                                 'retrieval_latency_s': 0.0,
                                 # Total MAPPO inference is measured at policy level.
                                 'model_latency_s': 0.0,
                                 'teacher_joint_inference_s_for_audit': elapsed,
                                 'messages': None, 'prompt_sha256': None,
                                 'raw_model_output': None, 'usage': None,
                                 'retrieved_ids_for_audit_only': []}
            return acts, entries
        except Exception as exc:
            raise PolicyFailure('frozen_mappo', str(exc)) from exc


class ZeroShotEvalPolicy:
    """Same zero-shot P1-A prompt/model on the new held-out P1-C scenes."""
    def __init__(self, client):
        self.base = DecentralizedPolicy(client)
    def set_scene(self, scene_id):
        del scene_id
    def act(self, observations):
        return self.base.act(observations)
