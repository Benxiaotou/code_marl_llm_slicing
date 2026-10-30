"""Validated data contracts shared by P1-B collection, filtering and audit."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS, OBS_SIZE

METRIC_NAMES = (
    'V', 'V_k1', 'V_k2', 'V_k3', 'V_g', 'V_k1_0', 'V_k2_0', 'V_k3_0',
    'cost_k1', 'cost_k2', 'cost_k3', 'cost_sum', 'g_done',
    'individual_rationality', 'cooperative_valid',
)


class ContractError(ValueError):
    pass


def canonical_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ContractError('{} line {}: bad JSON'.format(path, line_no)) from exc


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_obs(obs):
    if not isinstance(obs, dict) or set(obs) != set(AGENTS):
        raise ContractError('Exactly four local observations required')
    result = {}
    for agent in AGENTS:
        arr = np.asarray(obs[agent], dtype=np.float64)
        if arr.shape != (OBS_SIZE,) or not np.all(np.isfinite(arr)):
            raise ContractError('{}: invalid observation shape / nonfinite'.format(agent))
        if np.any(arr < -1e-5) or np.any(arr > 1 + 1e-5):
            raise ContractError('{}: observation outside [0,1]'.format(agent))
        result[agent] = [float(v) for v in arr]
    return result


def _extract_action_container(item):
    """Extract only explicit action containers; never guess from arbitrary arrays."""
    if isinstance(item, dict):
        if set(AGENTS).issubset(item):
            return {a: item[a] for a in AGENTS}, None
        for field, space in (('env_actions', 'env'), ('actions_env', 'env'),
                             ('action_dict', 'env'), ('action_env', 'env'),
                             ('actions_01', 'ratio'), ('ratios', 'ratio'),
                             ('action_ratios', 'ratio'), ('actions', None),
                             ('action', None)):
            if field in item:
                return item[field], space
        if 'per_role' in item and isinstance(item['per_role'], dict):
            entries = item['per_role']
            if set(entries) == set(AGENTS):
                if all(isinstance(entries[a], dict) and 'env_action' in entries[a]
                       for a in AGENTS):
                    return {a: entries[a]['env_action'] for a in AGENTS}, 'env'
                if all(isinstance(entries[a], dict) and 'ratios' in entries[a]
                       for a in AGENTS):
                    return {a: entries[a]['ratios'] for a in AGENTS}, 'ratio'
        raise ContractError('Cannot find per-agent actions in keys: {}'.format(sorted(item)))
    if hasattr(item, 'env_actions'):
        return getattr(item, 'env_actions'), 'env'
    if hasattr(item, 'action_dict'):
        return getattr(item, 'action_dict'), 'env'
    if hasattr(item, 'actions_dict'):
        return getattr(item, 'actions_dict'), 'env'
    if hasattr(item, 'action_env'):
        return getattr(item, 'action_env'), 'env'
    if hasattr(item, 'env_action'):
        return getattr(item, 'env_action'), 'env'
    if hasattr(item, 'actions'):
        return getattr(item, 'actions'), None
    if hasattr(item, 'action'):
        return getattr(item, 'action'), None
    return item, None


def normalize_teacher_actions(teacher_output, *, assumed_space='env'):
    """Return canonical four [−1,1] env actions. Explicitly validate space.

    MAPPO generally emits env actions [-1,1]. Logs may contain [0,1] ratios;
    those MUST be marked as ratio inputs. We never infer action space from
    numeric ranges alone, since positive-only env actions are ambiguous.
    """
    container, declared = _extract_action_container(teacher_output)
    space = declared or assumed_space
    if space not in ('env', 'ratio'):
        raise ContractError('Action space must be env or ratio')
    if hasattr(container, 'detach'):
        container = container.detach().cpu().numpy()
    if isinstance(container, dict):
        if set(container) != set(AGENTS):
            raise ContractError('Action dict must have exactly 4 agents')
        by_role = {a: container[a] for a in AGENTS}
    else:
        if hasattr(container, 'tolist'):
            container = container.tolist()
        if isinstance(container, (list, tuple)) and len(container) == 4 and all(
            isinstance(v, (list, tuple, np.ndarray)) for v in container
        ):
            by_role = dict(zip(AGENTS, container))
        else:
            try:
                arr = np.asarray(container, dtype=np.float64).reshape(-1)
            except (ValueError, TypeError) as exc:
                raise ContractError('Unsupported teacher action container') from exc
            if arr.size not in (15, 16):
                raise ContractError('Flat actions must have 15 or 16 entries')
            by_role = {a: arr[4*i:4*i+4] for i, a in enumerate(AGENTS[:3])}
            by_role['agent_g'] = arr[12:]
    result = {}
    for role in AGENTS:
        v = by_role[role]
        if hasattr(v, 'detach'):
            v = v.detach().cpu().numpy()
        # String/bool action values are not teacher numeric outputs.
        if np.asarray(v).dtype.kind in ('U', 'S', 'O', 'b'):
            raise ContractError('{}: action numbers cannot be strings/bools'.format(role))
        try:
            a = np.asarray(v, dtype=np.float64).reshape(-1)
        except (ValueError, TypeError) as exc:
            raise ContractError('Invalid action vector for {}'.format(role)) from exc
        if len(a) not in ((3, 4) if role == 'agent_g' else (4,)):
            raise ContractError('{} expects {} actions, received {}'.format(
                role, '3 or 4' if role == 'agent_g' else '4', len(a)))
        if not np.all(np.isfinite(a)):
            raise ContractError('{} has nonfinite action'.format(role))
        lo, hi = (-1.0, 1.0) if space == 'env' else (0.0, 1.0)
        if np.any(a < lo - 1e-5) or np.any(a > hi + 1e-5):
            raise ContractError('{} is out of {} action range'.format(role, space))
        a = np.clip(a, lo, hi)
        if space == 'ratio':
            a = 2 * a - 1
        if role == 'agent_g' and len(a) == 3:
            a = np.r_[a, 0.0]
        result[role] = a.astype(np.float32)
    return result


def make_experience(dataset_id, scene_id, obs, env_actions, metrics, teacher_provenance):
    obs = validate_obs(obs)
    env_actions = normalize_teacher_actions(env_actions, assumed_space='env')
    for key in METRIC_NAMES:
        if key not in metrics:
            raise ContractError('Missing original metric: ' + key)
    ms = {}
    for key in METRIC_NAMES:
        if key in ('g_done', 'individual_rationality', 'cooperative_valid'):
            ms[key] = bool(metrics[key])
        else:
            ms[key] = float(metrics[key])
            if not math.isfinite(ms[key]):
                raise ContractError('Nonfinite metric: '+key)
    ir = all(ms['V_'+k] >= ms['V_'+k+'_0'] - 1e-8 for k in ('k1','k2','k3')) and ms['V_g'] >= -1e-8
    if ir != ms['individual_rationality'] or (ms['g_done'] and ir) != ms['cooperative_valid']:
        raise ContractError('Original individual rationality / cooperative validity inconsistent')
    did = str(dataset_id).strip()
    if not did or '::' in did or not did.replace('-', '').replace('_', '').isalnum():
        raise ContractError('Invalid dataset ID')
    sid = int(scene_id)
    joint_id = '{}::{:06d}'.format(did, sid)
    actions_env = {k: [float(v) for v in env_actions[k]] for k in AGENTS}
    ratios = {k: [float(v) for v in (env_actions[k][:3] if k == 'agent_g' else env_actions[k]) / 2 + 0.5] for k in AGENTS}
    return {
        'record_version': 'p1b_joint_v1', 'joint_id': joint_id,
        'dataset_id': did, 'scene_id': sid,
        'observations': obs, 'teacher_actions_env': actions_env,
        'teacher_actions_ratio': ratios, 'metrics': ms,
        'teacher_provenance': teacher_provenance,
    }
