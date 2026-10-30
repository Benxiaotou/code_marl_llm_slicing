"""Filter P1-B high-quality joint teacher data, export local role-only DBs."""
import argparse
import csv
import datetime
import json
from pathlib import Path

from p1a.local_obs import AGENTS
from p1b.core import ContractError, read_json, read_jsonl, save_json, file_sha256
from p1b.select import analyze_and_select


def get_args(argv=None):
    p = argparse.ArgumentParser(description='P1-B cooperative high-quality teacher selection')
    p.add_argument('--collection', required=True, help='Folder created by p1b_collect.py')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-keep', type=int, default=400)
    p.add_argument('--cost-quantile', type=float, default=.95)
    p.add_argument('--quality-weight', type=float, default=.65)
    p.add_argument('--diversity-weight', type=float, default=.20)
    p.add_argument('--dev-analysis', action='store_true',
                   help='Allow incomplete/used samples for engineering, never final test')
    p.add_argument('--holdout-manifest', action='append', default=[],
                   help='Optional validation/test dataset manifest: require distinct hash and ID')
    return p.parse_args(argv)


def select(args):
    root = Path(args.collection).resolve()
    dest = Path(args.output_dir).resolve()
    manifest = read_json(root / 'manifest.json')
    if manifest.get('schema') != 'p1b_collection_v1':
        raise ContractError('Unknown collection manifest')
    if manifest.get('dataset_split') != 'train' or manifest.get('development_only'):
        if not args.dev_analysis:
            raise ContractError('Only independent TRAIN collection can enter P1-C. '
                                'For used 600 sample diagnostics, pass --dev-analysis')
    if not manifest.get('teacher', {}).get('verified_mappo'):
        if not args.dev_analysis:
            raise ContractError('Teacher source was not verified as MAPPO')
    if manifest.get('guard_used') or not manifest.get('no_training'):
        raise ContractError('Unexpected modified/learned teacher run')
    for hp in args.holdout_manifest:
        holdout = read_json(hp)
        if holdout.get('dataset_id') == manifest['dataset_id']:
            raise ContractError('Data leakage: holdout dataset ID overlaps training')
        if holdout.get('sample_sha256') == manifest['dataset_sha256']:
            raise ContractError('Data leakage: holdout sample hash overlaps training')
        if holdout.get('split') not in ('validation','test'):
            raise ContractError('Holdout manifest must be validation or test')
    data = list(read_jsonl(root / 'joint_records.jsonl'))
    if len(data) != len(manifest['scene_ids']):
        if not args.dev_analysis:
            raise ContractError('Incomplete teacher run; select only after finishing all scenes')
    ids = [r.get('scene_id') for r in data]
    if len(ids) != len(set(ids)) or any(r.get('dataset_id') != manifest['dataset_id'] for r in data):
        raise ContractError('Duplicate IDs or mixed datasets')
    if any(r.get('status') == 'ok' and (r.get('teacher_provenance') or {}).get('kind')
           != manifest.get('teacher', {}).get('kind') for r in data):
        raise ContractError('Teacher provenance inconsistent with manifest')
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError('Output already contains files; choose a new directory')
    dest.mkdir(parents=True, exist_ok=True)
    chosen, stats = analyze_and_select(
        data, args.max_keep, args.cost_quantile, args.quality_weight, args.diversity_weight)
    if not chosen:
        raise ContractError('No high-quality cooperative experience found. '
                            'Cannot fabricate local positive examples.')
    private_dir = dest / 'private_offline'
    role_dir = dest / 'role_libraries'
    private_dir.mkdir()
    role_dir.mkdir()
    with (private_dir / 'selected_joint_records.jsonl').open('w', encoding='utf-8') as f:
        for c in chosen:
            r = c['row']
            private = dict(r)
            private['selection_stratum'] = c['stratum']
            private['selection_quality_score'] = float(c['quality_score'])
            private['offline_pressure'] = c['features']['pressure']
            private['offline_overlap'] = c['features']['overlap']
            f.write(json.dumps(private, ensure_ascii=False, allow_nan=False)+'\n')
    for role in AGENTS:
        target = role_dir / (role+'.jsonl')
        with target.open('w', encoding='utf-8') as f:
            for c in chosen:
                r = c['row']
                # Whitelist ONLY. No reward, outcomes, other agents' observations,
                # other agents' actions, physical graph or current global state.
                item = {
                    'schema': 'p1b_local_case_v1', 'role': role,
                    'experience_id': r['joint_id'],
                    'observation_240': r['observations'][role],
                    'own_teacher_ratios': r['teacher_actions_ratio'][role],
                    'own_teacher_env_action': r['teacher_actions_env'][role][:3]
                                               if role == 'agent_g' else r['teacher_actions_env'][role],
                    'source': 'frozen_MAPPO_selected_joint_cooperation',
                }
                f.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
    selected_ids = [{'joint_id': c['row']['joint_id'], 'stratum': c['stratum'],
                     'quality_score': float(c['quality_score'])} for c in chosen]
    save_json(private_dir / 'selection_index.json', selected_ids)
    output_meta = {
        'schema': 'p1b_library_manifest_v1',
        'dataset_id': manifest['dataset_id'],
        'training_sample_sha256': manifest['dataset_sha256'],
        'teacher_provenance': manifest['teacher'],
        'development_only': manifest['development_only'] or bool(args.dev_analysis),
        'n_experiences_per_role': len(chosen),
        'output_role_fields': ['schema','role','experience_id','observation_240',
                               'own_teacher_ratios','own_teacher_env_action','source'],
        'roles': list(AGENTS), 'no_current_global_information': True,
        'policy_used_during_collection': 'frozen_MAPPO_only',
        'selection': {'max_keep': args.max_keep, 'cost_quantile': args.cost_quantile,
                      'quality_weight':args.quality_weight, 'diversity_weight':args.diversity_weight},
        'warnings': ['Joint cooperative success is not individual action optimality.',
                     'Do not use joint records or offline scores as Qwen prompt contents.',
                     'No final independent test has been performed by this script.'],
    }
    save_json(dest / 'library_manifest.json', output_meta)
    save_json(dest / 'selection_report.json', stats)
    with (dest / 'selection_report.txt').open('w', encoding='utf-8') as f:
        f.write('P1-B Frozen MAPPO / cooperative experience selection\n')
        f.write('Dataset: {} | development_only: {}\n\n'.format(
            manifest['dataset_id'], output_meta['development_only']))
        f.write('Input scenes: {}\nCooperative valid: {}\n'.format(
            stats['n_input'], stats['n_cooperative_candidates']))
        f.write('After stratified cost filter: {}\nSelected: {}\n\n'.format(
            stats['n_after_relative_cost_filter'], stats['n_selected']))
        f.write('Rejections: {}\n'.format(stats['hard_filter_rejections']))
        f.write('Selected strata: {}\n'.format(stats['strata_selected']))
        f.write('\nWARNING: Joint success is NOT per-agent causal optimality.\n')
    print('Selected {}/{} candidates, {} per role. Output: {}'.format(
        len(chosen), len(data), len(chosen), dest), flush=True)
    return output_meta, stats


if __name__ == '__main__':
    select(get_args())
