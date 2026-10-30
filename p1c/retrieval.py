"""Strict role-local numeric retrieval from P1-B public role libraries ONLY.

Never opens private_offline, team metrics, other-role observations or environment
objects. Similarity scaling is fit once on the frozen TRAIN role library.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS, N_NODES, MAX_LINKS, OBS_SIZE
from p1b.core import ContractError, file_sha256, read_json, read_jsonl

ROLE_FIELDS = {'schema', 'role', 'experience_id', 'observation_240',
               'own_teacher_ratios', 'own_teacher_env_action', 'source'}


def _stat(a, fun='mean'):
    if len(a) == 0:
        return 0.0
    arr = np.asarray(a, dtype=np.float64)
    if fun == 'max':
        return float(arr.max())
    if fun == 'min':
        return float(arr.min())
    if fun == 'std':
        return float(arr.std())
    return float(arr.mean())


def _fields_with_prefix(target, prefix, values):
    target[prefix + '_mean'] = _stat(values)
    target[prefix + '_max'] = _stat(values, 'max')


def _split_groups(rows):
    count = len(rows)
    boundary = (count + 1) // 2
    return (rows[:boundary], rows[boundary:])


def local_numeric_summary(role, observation):
    """Fixed-dimensional local features aligned with resource control groups.

    Existing slices: controls 0/1 apply to contiguous halves of sorted
    overlapping local indices; g uses compressed compute/cache/bandwidth.
    Historical and current summaries use exactly the same formula.
    """
    if role not in AGENTS:
        raise ContractError('Unknown role')
    arr = np.asarray(observation, dtype=np.float64)
    if arr.shape != (OBS_SIZE,) or not np.isfinite(arr).all():
        raise ContractError('Expected one finite 240-dimensional local observation')
    if np.any(arr < -1e-5) or np.any(arr > 1 + 1e-5):
        raise ContractError('Local observation outside [0,1]')
    nodes = arr[:N_NODES * 5].reshape(N_NODES, 5)
    links_all = arr[N_NODES * 5:N_NODES * 5 + MAX_LINKS * 3].reshape(MAX_LINKS, 3)
    edge_all = arr[N_NODES * 5 + MAX_LINKS * 3:].reshape(MAX_LINKS, 2)
    n_real = int(np.count_nonzero(links_all[:, 2] > 0 if role == 'agent_g'
                                  else links_all[:, 1] > 0))
    links = links_all[:n_real]
    if np.any(np.abs(links_all[n_real:]) > 1e-7) or np.any(np.abs(edge_all[n_real:]) > 1e-7):
        raise ContractError('Padded link or topology fields are not zero')
    f = {'link_count_frac': float(n_real / MAX_LINKS)}
    # Local-only topology descriptors. Endpoints are normalised local IDs,
    # not recoverable global physical node identifiers.
    if n_real:
        edges = edge_all[:n_real]
        f['topo_endpoint_mean'] = float(edges.mean())
        f['topo_endpoint_std'] = float(edges.std())
    else:
        f['topo_endpoint_mean'] = f['topo_endpoint_std'] = 0.0
    if role != 'agent_g':
        if np.any((np.abs(nodes[:, 0]) > 1e-5) & (np.abs(nodes[:, 0] - 1) > 1e-5)):
            raise ContractError('Invalid k node overlap flags')
        if np.any((np.abs(links[:, 0]) > 1e-5) & (np.abs(links[:, 0] - 1) > 1e-5)):
            raise ContractError('Invalid k link overlap flags')
        sel_node = nodes[nodes[:, 0] > .5]
        sel_link = links[links[:, 0] > .5]
        f['overlap_nodes_frac'] = float(len(sel_node) / N_NODES)
        f['overlap_links_frac'] = float(len(sel_link) / max(1, n_real))
        _fields_with_prefix(f, 'all_compute_util', nodes[:, 2])
        _fields_with_prefix(f, 'all_cache_util', nodes[:, 4])
        _fields_with_prefix(f, 'all_link_util', links[:, 2])
        for gidx, group in enumerate(_split_groups(sel_node)):
            prefix = 'node_control_' + str(gidx)
            f[prefix + '_count_frac'] = float(len(group) / N_NODES)
            for name, index in (('compute_capacity', 1), ('compute_util', 2),
                                ('cache_capacity', 3), ('cache_util', 4)):
                _fields_with_prefix(f, prefix + '_' + name, group[:, index])
            if len(group):
                max_used = np.maximum(group[:, 2], group[:, 4])
                f[prefix + '_static_no_migration_limit'] = float(np.clip(
                    np.min((1 - max_used) / .8), 0, 1))
            else:
                f[prefix + '_static_no_migration_limit'] = 0.0
        for gidx, group in enumerate(_split_groups(sel_link)):
            prefix = 'link_control_' + str(gidx)
            f[prefix + '_count_frac'] = float(len(group) / MAX_LINKS)
            for name, index in (('bandwidth_capacity', 1), ('bandwidth_util', 2)):
                _fields_with_prefix(f, prefix + '_' + name, group[:, index])
            if len(group):
                # Reference only, not a physical feasibility guarantee.
                f[prefix + '_static_no_reroute_limit'] = float(np.clip(
                    np.min((1 - group[:, 2]) / .8), 0, 1))
            else:
                f[prefix + '_static_no_reroute_limit'] = 0.0
    else:
        if np.any(np.abs(nodes[:, 0]) > 1e-6) or np.any(np.abs(links[:, 0]) > 1e-6):
            raise ContractError('Invalid g padding feature')
        for name, index in (('allocated_compute', 1), ('requested_compute', 2),
                            ('allocated_cache', 3), ('requested_cache', 4)):
            _fields_with_prefix(f, 'g_' + name, nodes[:, index])
        _fields_with_prefix(f, 'g_compute_shortage',
                            np.maximum(0, nodes[:, 1] + nodes[:, 2] - 1))
        _fields_with_prefix(f, 'g_cache_shortage',
                            np.maximum(0, nodes[:, 3] + nodes[:, 4] - 1))
        for name, values in (
            ('g_allocated_bandwidth', links[:, 1]),
            ('g_requested_bandwidth', links[:, 2]),
            ('g_bandwidth_shortage', np.maximum(0, links[:, 1] + links[:, 2] - 1))):
            _fields_with_prefix(f, name, values)
    if any(not math.isfinite(value) for value in f.values()):
        raise ContractError('Nonfinite derived local feature')
    return f


class LocalHistory:
    """Immutable role-specific train-only experience database."""
    def __init__(self, directory, *, require_audit=True):
        base = Path(directory).resolve()
        self.directory = base
        meta = read_json(base / 'library_manifest.json')
        if meta.get('schema') != 'p1b_library_manifest_v1':
            raise ContractError('Unsupported P1-B library manifest')
        if meta.get('development_only') or not meta.get('no_current_global_information'):
            raise ContractError('Only audited non-development role-local libraries permitted')
        if not meta.get('teacher_provenance', {}).get('verified_mappo'):
            raise ContractError('Experience source is not verified frozen MAPPO')
        if meta.get('policy_used_during_collection') != 'frozen_MAPPO_only':
            raise ContractError('Unexpected teacher source')
        if require_audit:
            audit_path = base / 'audit_report.json'
            if not audit_path.is_file():
                raise ContractError('Run p1b_audit.py on the library first')
            audit = read_json(audit_path)
            if not (audit.get('audit_passed') and audit.get('no_global_fields_in_role_libraries')
                    and audit.get('matched_joint_case_ids') and audit.get('dataset_id') == meta.get('dataset_id')):
                raise ContractError('P1-B library audit did not pass')
        self.meta = meta
        self.library_hashes = {}
        self.cases = {}
        self.features = {}
        self.scales = {}
        self.keys = {}
        idsets = []
        for role in AGENTS:
            path = base / 'role_libraries' / (role + '.jsonl')
            self.library_hashes[role] = file_sha256(path)
            rows = list(read_jsonl(path))
            if not rows or len(rows) != meta.get('n_experiences_per_role'):
                raise ContractError('Incorrect library size for ' + role)
            seen = set()
            raw_features = []
            stored = []
            for rec in rows:
                if set(rec) != ROLE_FIELDS or rec['schema'] != 'p1b_local_case_v1' or rec['role'] != role:
                    raise ContractError('Non-local/unknown fields in ' + role)
                case_id = str(rec['experience_id'])
                if not case_id.startswith(str(meta['dataset_id']) + '::') or case_id in seen:
                    raise ContractError('Unexpected / duplicate training case id')
                seen.add(case_id)
                vec = np.asarray(rec['own_teacher_ratios'], dtype=np.float64)
                env_act = np.asarray(rec['own_teacher_env_action'], dtype=np.float64)
                dims = 3 if role == 'agent_g' else 4
                if vec.shape != (dims,) or env_act.shape != (dims,) or not np.isfinite(vec).all():
                    raise ContractError('Invalid historical action')
                if np.any(vec < 0) or np.any(vec > 1) or not np.allclose(env_act, 2 * vec - 1, atol=2e-6):
                    raise ContractError('Historical action mismatch')
                desc = local_numeric_summary(role, rec['observation_240'])
                raw_features.append(desc)
                stored.append({'experience_id': case_id, 'ratios': [float(x) for x in vec],
                               'summary': desc})
            idsets.append(seen)
            keys = sorted(raw_features[0])
            if any(sorted(row) != keys for row in raw_features):
                raise ContractError('Local feature schema changed within ' + role)
            matrix = np.array([[row[k] for k in keys] for row in raw_features], dtype=np.float64)
            # Scaling is TRAIN-only. A 0.12 floor prevents tiny training
            # variances from causing pathological nearest-neighbour weights.
            scales = np.maximum(matrix.std(axis=0), .12)
            self.cases[role] = stored
            self.features[role] = matrix
            self.scales[role] = scales
            self.keys[role] = keys
        if any(ids != idsets[0] for ids in idsets[1:]):
            raise ContractError('Role libraries do not share selected joint case IDs')
        self.manifest_sha256 = file_sha256(base / 'library_manifest.json')

    def assert_independent(self, eval_manifest):
        if eval_manifest.get('schema') != 'p1b_dataset_v1':
            raise ContractError('P1-C requires P1-B generated dataset manifest')
        if eval_manifest.get('split') not in ('validation', 'test'):
            raise ContractError('P1-C evaluation data must be validation or test, NEVER train')
        if eval_manifest.get('dataset_id') == self.meta['dataset_id']:
            raise ContractError('TRAIN/EVAL dataset IDs collide')
        if eval_manifest.get('sample_sha256') == self.meta['training_sample_sha256']:
            raise ContractError('TRAIN/EVAL sample hashes collide')
        # A file hash only detects identical file bytes; it cannot guarantee
        # independent generation seeds. Keep split seeds in manifests.

    def query(self, role, observation, top_k=3):
        if role not in AGENTS or top_k < 1 or top_k > len(self.cases[role]):
            raise ContractError('Invalid role or top_k')
        local = local_numeric_summary(role, observation)
        curr = np.array([local[k] for k in self.keys[role]], dtype=np.float64)
        hist = self.features[role]
        dist = np.sqrt(np.mean(((hist - curr) / self.scales[role]) ** 2, axis=1))
        selected = sorted(range(len(dist)), key=lambda j: (float(dist[j]),
                           self.cases[role][j]['experience_id']))[:top_k]
        return [{**self.cases[role][idx], 'distance': float(dist[idx])}
                for idx in selected]

    def random_cases(self, role, n, seed, scene_id):
        if role not in AGENTS or n < 1 or n > len(self.cases[role]):
            raise ContractError('Invalid role/count')
        raw = '{}:{}:{}:p1c_random_v1'.format(int(seed), int(scene_id), role)
        key = int.from_bytes(hashlib.sha256(raw.encode('utf-8')).digest()[:8], 'big')
        rng = np.random.default_rng(key)
        indices = rng.choice(len(self.cases[role]), size=n, replace=False)
        return [{**self.cases[role][int(idx)], 'distance': None} for idx in indices]
