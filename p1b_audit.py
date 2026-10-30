"""Independent offline audit of four P1-B role-local experience libraries."""
import argparse
import json
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import ContractError, read_json, read_jsonl, save_json

REQUIRED = {'schema', 'role', 'experience_id', 'observation_240',
            'own_teacher_ratios', 'own_teacher_env_action', 'source'}


def audit(folder, holdout_manifests=()):
    folder = Path(folder)
    mf = read_json(folder / 'library_manifest.json')
    if mf.get('schema') != 'p1b_library_manifest_v1':
        raise ContractError('Bad library manifest')
    for path in holdout_manifests:
        h = read_json(path)
        if h.get('dataset_id') == mf['dataset_id'] or h.get('sample_sha256') == mf['training_sample_sha256']:
            raise ContractError('Training / holdout leakage detected')
    by_role = {}
    for role in AGENTS:
        rows = list(read_jsonl(folder / 'role_libraries' / (role+'.jsonl')))
        ids = []
        for item in rows:
            if set(item) != REQUIRED or item['schema'] != 'p1b_local_case_v1' or item['role'] != role:
                raise ContractError('Role DB has unexpected/global fields: '+role)
            obs = np.asarray(item['observation_240'], dtype=np.float64)
            if obs.shape != (240,) or not np.isfinite(obs).all():
                raise ContractError('Bad local observation for '+role)
            n = 3 if role == 'agent_g' else 4
            ratio = np.asarray(item['own_teacher_ratios'], dtype=np.float64)
            act = np.asarray(item['own_teacher_env_action'], dtype=np.float64)
            if ratio.shape != (n,) or act.shape != (n,) or not np.all(np.isfinite(ratio)):
                raise ContractError('Bad action dimensions for '+role)
            if not np.allclose(act, 2*ratio - 1, atol=2e-6):
                raise ContractError('Action ratio / env mismatch for '+role)
            if np.any(ratio < -1e-6) or np.any(ratio > 1+1e-6):
                raise ContractError('Action outside [0,1]')
            ids.append(item['experience_id'])
        if len(ids) != len(set(ids)):
            raise ContractError('Duplicate experiences within '+role)
        by_role[role] = ids
    # Cross-check role exports against the full PRIVATE offline source.
    joint_path = folder / 'private_offline' / 'selected_joint_records.jsonl'
    if not joint_path.is_file():
        raise ContractError('Missing private offline joint source for audit')
    joint_rows = list(read_jsonl(joint_path))
    joint_map = {r['joint_id']:r for r in joint_rows}
    if len(joint_map) != len(joint_rows):
        raise ContractError('Duplicate private joint IDs')
    for role in AGENTS:
        for item in read_jsonl(folder / 'role_libraries' / (role+'.jsonl')):
            source = joint_map.get(item['experience_id'])
            if source is None:
                raise ContractError('Role item not selected in private joint source')
            if not np.allclose(item['observation_240'], source['observations'][role], atol=1e-7):
                raise ContractError('Role observation differs from selected source')
            if not np.allclose(item['own_teacher_ratios'], source['teacher_actions_ratio'][role], atol=1e-7):
                raise ContractError('Role action differs from teacher source')
    first = by_role[AGENTS[0]]
    if any(by_role[a] != first for a in AGENTS):
        raise ContractError('Four roles must use selected matched joint cases')
    if len(first) != mf['n_experiences_per_role']:
        raise ContractError('Library size != manifest')
    if not first:
        raise ContractError('Empty library')
    report = {
        'audit_passed': True, 'dataset_id': mf['dataset_id'],
        'development_only': mf['development_only'],
        'n_per_role': len(first), 'roles': list(AGENTS),
        'no_global_fields_in_role_libraries': True,
        'matched_joint_case_ids': True,
        'holdout_check': len(holdout_manifests),
    }
    save_json(folder / 'audit_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def get_args(argv=None):
    p = argparse.ArgumentParser(description='Audit P1-B role-local history libraries')
    p.add_argument('--library', required=True)
    p.add_argument('--holdout-manifest', action='append', default=[])
    return p.parse_args(argv)


if __name__ == '__main__':
    args = get_args()
    audit(args.library, args.holdout_manifest)
