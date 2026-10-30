"""Build/verify self-only historical IR annotations from frozen TRAIN joint outcomes.

The OFFLINE builder is the only component permitted to read P1-B private
selected_joint_records. At execution, IRHistory opens only role-specific
annotations and the audited original role-local library.
"""
import json
import math
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import ContractError, file_sha256, read_json, read_jsonl, save_json
from p1c.retrieval import LocalHistory

K_FIELDS = frozenset(('schema', 'role', 'experience_id',
                      'own_historical_ir_margin',
                      'own_historical_relative_ir_margin',
                      'own_historical_adjustment_cost'))
G_FIELDS = frozenset(('schema', 'role', 'experience_id',
                      'own_historical_new_slice_utility'))
ROW_SCHEMA = 'p1d_role_ir_v1'
MANIFEST_SCHEMA = 'p1d_ir_overlay_manifest_v1'


def _nonnegative(value, key, *, eps=1e-8):
    if isinstance(value, bool):
        raise ContractError('Boolean is not a numeric metric: ' + key)
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError('Invalid metric: ' + key) from exc
    if not math.isfinite(f) or f < -eps:
        raise ContractError('Invalid or negative metric: ' + key)
    return max(0., f)


def _record_from_joint(role, joint):
    m = joint['metrics']
    rec = {'schema': ROW_SCHEMA, 'role': role, 'experience_id': joint['joint_id']}
    if role == 'agent_g':
        rec['own_historical_new_slice_utility'] = _nonnegative(m['V_g'], 'V_g')
    else:
        short = role.replace('agent_', '')
        before = float(m['V_' + short + '_0'])
        after = float(m['V_' + short])
        if not (math.isfinite(before) and math.isfinite(after)):
            raise ContractError('Nonfinite original utility')
        rec['own_historical_ir_margin'] = _nonnegative(after - before, 'IR margin ' + role)
        rec['own_historical_relative_ir_margin'] = rec['own_historical_ir_margin'] / max(1.0, abs(before))
        rec['own_historical_adjustment_cost'] = _nonnegative(m['cost_' + short], 'cost ' + role)
    return rec


def build_ir_overlay(library_path, output_path):
    """Offline-only build; never invoked by the runtime policy."""
    lib = LocalHistory(library_path)
    lib_base = Path(library_path).resolve()
    target = Path(output_path).resolve()
    if target == lib_base or lib_base in target.parents:
        raise ContractError('IR overlay must not modify/overwrite original P1-B library')
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('Refusing to overwrite existing IR overlay: ' + str(target))
    source = lib_base / 'private_offline' / 'selected_joint_records.jsonl'
    if not source.is_file():
        raise ContractError('Missing P1-B offline selected joint records: ' + str(source))
    joints = list(read_jsonl(source))
    expected = {r: set(c['experience_id'] for c in lib.cases[r]) for r in AGENTS}
    seen = set()
    by_role = {r: {} for r in AGENTS}
    original_local = {r: {rec['experience_id']: rec for rec in read_jsonl(
        lib_base / 'role_libraries' / (r + '.jsonl'))} for r in AGENTS}
    for joint in joints:
        cid = str(joint.get('joint_id', ''))
        if cid in seen or cid not in expected['agent_k1']:
            raise ContractError('Duplicate, out-of-library or foreign training case ID: ' + cid)
        seen.add(cid)
        if joint.get('status') != 'ok' or joint.get('dataset_id') != lib.meta['dataset_id']:
            raise ContractError('Not a successful frozen-training joint record: ' + cid)
        m = joint.get('metrics')
        if not isinstance(m, dict) or not all(bool(m.get(k)) for k in ('g_done', 'individual_rationality', 'cooperative_valid')):
            raise ContractError('Selected joint case does not satisfy cooperation conditions: ' + cid)
        for role in AGENTS:
            old = original_local[role][cid]
            obs = np.asarray(joint['observations'][role], dtype=np.float64)
            prev_obs = np.asarray(old['observation_240'], dtype=np.float64)
            ratio = np.asarray(joint['teacher_actions_ratio'][role], dtype=np.float64)
            prev_ratio = np.asarray(old['own_teacher_ratios'], dtype=np.float64)
            if (obs.shape != (240,) or not np.allclose(obs, prev_obs, rtol=0, atol=3e-6)
                    or ratio.shape != prev_ratio.shape
                    or not np.allclose(ratio, prev_ratio, rtol=0, atol=3e-6)):
                raise ContractError('Historical local observation / teacher action mismatch: ' + role + '/' + cid)
            by_role[role][cid] = _record_from_joint(role, joint)
    if seen != expected['agent_k1'] or any(ids != seen for ids in expected.values()):
        raise ContractError('IR source does not match all selected P1-B case IDs')
    target.mkdir(parents=True, exist_ok=True)
    role_dir = target / 'role_ir_history'
    role_dir.mkdir()
    hashes = {}
    for role in AGENTS:
        path = role_dir / (role + '.jsonl')
        with path.open('w', encoding='utf-8') as f:
            for case in lib.cases[role]:
                f.write(json.dumps(by_role[role][case['experience_id']],
                                   ensure_ascii=False, allow_nan=False) + '\n')
        hashes[role] = file_sha256(path)
    manifest = {
        'schema': MANIFEST_SCHEMA, 'training_dataset_id': lib.meta['dataset_id'],
        'training_sample_sha256': lib.meta['training_sample_sha256'],
        'baseline_library_manifest_sha256': lib.manifest_sha256,
        'baseline_role_libraries_sha256': lib.library_hashes,
        'source_selected_joint_sha256': file_sha256(source),
        'role_ir_history_sha256': hashes,
        'n_selected_training_cases': len(seen),
        'roles': list(AGENTS), 'development_only': False,
        'scope': 'offline historical OWN-role realized utility/cost only',
        'runtime_does_not_read_joint_records': True,
        'no_current_global_state': True,
        'caution': ('Historical IR margin/cost is a result of the whole historical joint action. '
                    'It is not an action-specific causal prediction or a current-scene observation.'),
    }
    save_json(target / 'ir_manifest.json', manifest)
    # The constructor audits both manifest provenance and all role-only data.
    IRHistory(target, lib)
    save_json(target / 'ir_audit.json', {
        'audit_passed': True, 'training_dataset_id': lib.meta['dataset_id'],
        'n_per_role': len(seen), 'role_only_fields': True,
        'source_joint_used_offline_only': True,
    })
    return manifest


class IRHistory:
    """Runtime IR annotations: no private_offline access or team result fields."""
    def __init__(self, directory, library):
        base = Path(directory).resolve()
        self.directory = base
        meta_path = base / 'ir_manifest.json'
        meta = read_json(meta_path)
        if (meta.get('schema') != MANIFEST_SCHEMA or meta.get('development_only')
                or not meta.get('no_current_global_state')
                or not meta.get('runtime_does_not_read_joint_records')):
            raise ContractError('Invalid IR overlay provenance')
        if (meta.get('training_dataset_id') != library.meta['dataset_id']
                or meta.get('training_sample_sha256') != library.meta['training_sample_sha256']
                or meta.get('baseline_library_manifest_sha256') != library.manifest_sha256
                or meta.get('baseline_role_libraries_sha256') != library.library_hashes):
            raise ContractError('IR overlay does not match the audited frozen P1-B library')
        self.meta = meta
        self.manifest_sha256 = file_sha256(meta_path)
        self.role_hashes = {}
        self.rows = {}
        for role in AGENTS:
            path = base / 'role_ir_history' / (role + '.jsonl')
            digest = file_sha256(path)
            if digest != meta.get('role_ir_history_sha256', {}).get(role):
                raise ContractError('Tampered IR history file for ' + role)
            self.role_hashes[role] = digest
            rows = list(read_jsonl(path))
            ref_ids = {rec['experience_id'] for rec in library.cases[role]}
            if len(rows) != len(ref_ids) or len(rows) != meta['n_selected_training_cases']:
                raise ContractError('Wrong IR history count for ' + role)
            lookup = {}
            for rec in rows:
                expected = G_FIELDS if role == 'agent_g' else K_FIELDS
                if (set(rec) != expected or rec['schema'] != ROW_SCHEMA
                        or rec['role'] != role or rec['experience_id'] in lookup
                        or rec['experience_id'] not in ref_ids):
                    raise ContractError('Unexpected / private IR role fields for ' + role)
                for key in expected - {'schema','role','experience_id'}:
                    _nonnegative(rec[key], key)
                lookup[rec['experience_id']] = rec
            if set(lookup) != ref_ids:
                raise ContractError('IR case IDs differ from original local history')
            self.rows[role] = lookup

    def for_examples(self, role, matches):
        if role not in AGENTS:
            raise ContractError('Unknown role')
        out = []
        for case in matches:
            item = self.rows[role][case['experience_id']]
            out.append({key: round(float(value), 5) for key, value in item.items()
                        if key.startswith('own_historical_')})
        return out
