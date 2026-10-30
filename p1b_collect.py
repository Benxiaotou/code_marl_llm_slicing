"""Collect P1-B frozen-MAPPO joint experience on an ORIGINAL GEANT environment.

Supports (1) live inference through the previously validated P0 MAPPOPolicy,
and (2) strict replay of saved MAPPO actions. Neither trains a model.
"""
import argparse
import csv
import datetime
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from p1b.core import (ContractError, file_sha256, make_experience,
                      read_json, read_jsonl, save_json)
from p1b.teacher import P0FrozenTeacher, RecordedMAPPO


def get_args(argv=None):
    p = argparse.ArgumentParser(description='P1-B frozen MAPPO teacher collection')
    p.add_argument('--dataset', choices=['custom', 'legacy'], default='custom')
    p.add_argument('--dataset-manifest', default='', help='p1b_data/<new_train_name>.json (custom)')
    p.add_argument('--allow-used-dev-dataset', action='store_true',
                   help='Explicitly mark original 600 GEANT as development only')
    p.add_argument('--teacher', choices=['live', 'recorded'], default='live')
    p.add_argument('--teacher-run', default='', help='P0 run directory for recorded teacher')
    p.add_argument('--checkpoint', default='marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth')
    p.add_argument('--teacher-space', choices=['env', 'ratio'], default='env',
                   help='Default [-1,1] action; use ratio only for saved 0..1 logs')
    p.add_argument('--allow-unverified-source', action='store_true',
                   help='Development diagnosis only; selection rejects this source')
    p.add_argument('--config', default='configs_main.yaml')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--on-error', choices=['abort', 'record'], default='abort')
    return p.parse_args(argv)


def construct_env(args, root):
    from p1a_runner import (_config, _compat_pickle_main_classes, make_mini_env_type,
                            create_env)
    if args.dataset == 'legacy':
        if not args.allow_used_dev_dataset:
            raise ContractError('Original 600 GEANT scenes are already used in P0/P1-A. '
                                'Use --allow-used-dev-dataset for engineering ONLY.')
        dummy = argparse.Namespace(dataset='full', config=args.config)
        env, original_ids = create_env(dummy, root)
        return env, {'schema': 'p1b_dataset_v1', 'dataset_id': 'legacy600_devonly',
                     'split': 'development', 'n_scenes': len(original_ids),
                     'sample_file': str(Path(env.sample_dict_save_file).resolve()),
                     'sample_sha256': file_sha256(env.sample_dict_save_file),
                     'source_note': 'Previously used original 600 samples; never untouched test'}
    if not args.dataset_manifest:
        raise ValueError('--dataset-manifest required for new custom GEANT samples')
    dm = read_json(root / args.dataset_manifest)
    if dm.get('schema') != 'p1b_dataset_v1':
        raise ContractError('Unexpected dataset manifest version')
    if dm.get('split') != 'train':
        raise ContractError('Teacher experiences must come from TRAIN datasets only')
    sample = Path(dm['sample_file']).resolve()
    if not sample.is_file():
        raise FileNotFoundError('New sample file unavailable: '+str(sample))
    if file_sha256(sample) != dm['sample_sha256']:
        raise ContractError('Dataset hash mismatch, refusing collection')
    if dm.get('main_py_sha256') and file_sha256(root / 'main.py') != dm['main_py_sha256']:
        raise ContractError('GEANT main.py changed since sample generation; audit environment before use')
    if dm.get('config_sha256') and file_sha256(root / args.config) != dm['config_sha256']:
        raise ContractError('GEANT configuration changed since sample generation')
    if int(dm['n_scenes']) < 1:
        raise ContractError('Bad scene count in manifest')
    main, conf = _config(root, root / args.config)
    _compat_pickle_main_classes(main)
    cls = make_mini_env_type(main.CustomMultiAgentEnv, sample, int(dm['n_scenes']))
    env = cls(conf)
    env.is_eval = True
    if env.sample_cnt != len(env.sample_dict):
        raise ContractError('Custom sample count mismatch')
    if set(env.sample_dict) != set(range(int(dm['n_scenes']))):
        raise ContractError('Samples must use sequential integer keys')
    return env, dm


def _prepare_teacher_conf(root, config_file):
    from p1a_runner import _config
    main, conf = _config(root, root / config_file)
    # XuanCe make_envs(config) needs this registry entry; the P0 policy loads
    # its own evaluation vector env solely to build/check the frozen actor.
    main.REGISTRY_MULTI_AGENT_ENV[conf.env_name] = main.CustomMultiAgentEnv
    for attr, folder in (('path_model_save', 'model_save'),
                         ('path_model_train_log', 'model_train_log'),
                         ('path_F_save', 'F_save')):
        setattr(conf, attr, str(root / 'result' / folder))
    return conf


def collect(args, *, root=None, env_override=None, dataset_override=None, teacher_override=None):
    root = Path(root or Path.cwd()).resolve()
    if args.count < 1 or args.start < 0 or args.seed < 0:
        raise ValueError('start >=0, count >=1 and seed>=0 required')
    output = (root / args.output_dir).resolve()
    if args.allow_unverified_source and args.teacher == 'live':
        raise ValueError('--allow-unverified-source is for recorded runs only')
    if env_override is None:
        env, dm = construct_env(args, root)
    else:
        if dataset_override is None:
            raise ValueError('Test env_override requires dataset_override')
        env, dm = env_override, dataset_override
    if args.start + args.count > int(dm['n_scenes']):
        raise ValueError('start/count outside dataset')
    requested = list(range(args.start, args.start + args.count))
    checkpoint = (root / args.checkpoint).resolve()
    if teacher_override is not None:
        teacher = teacher_override
    elif args.teacher == 'live':
        teacher = P0FrozenTeacher(_prepare_teacher_conf(root, args.config), checkpoint,
                                  action_space=args.teacher_space)
    else:
        if not args.teacher_run:
            raise ValueError('--teacher-run needed when --teacher recorded')
        teacher = RecordedMAPPO(root / args.teacher_run, action_space=args.teacher_space,
                                allow_unverified=args.allow_unverified_source)
    src = dict(teacher.source_info)
    if not src.get('verified_mappo') and not args.allow_unverified_source:
        raise ContractError('Unverified teacher cannot be used')
    if src.get('kind') == 'recorded_MAPPO_replay' and args.dataset == 'custom':
        # Most older P0 logs have only scene 0..599 without dataset fingerprint.
        # They may not be relabeled as actions for an unrelated new dataset.
        raise ContractError('Recorded P0 actions cannot be paired with an unrelated '
                            'NEW sample dataset. Use --teacher live for new train data.')
    out_manifest = {
        'schema': 'p1b_collection_v1', 'dataset_id': dm['dataset_id'],
        'dataset_sha256': dm['sample_sha256'], 'dataset_split': dm['split'],
        'dataset_path': dm['sample_file'],
        'development_only': dm['split'] != 'train' or not src.get('verified_mappo'),
        'scene_ids': requested, 'teacher': src, 'seed': args.seed,
        'main_py_sha256': file_sha256(root / 'main.py') if (root / 'main.py').is_file() else None,
        'config_sha256': file_sha256(root / args.config) if (root / args.config).is_file() else None,
        'environment_is_original': True, 'guard_used': False,
        'no_training': True, 'no_test_time_global_state': True,
        'joint_records_private': True, 'policy_fallback': False,
        'notes': 'Environment is evaluated AFTER teacher action is frozen; no LP action repair',
    }
    mf = output / 'manifest.json'
    old_records = []
    if args.resume:
        if not mf.is_file():
            raise FileNotFoundError('Cannot resume without prior manifest')
        prior = read_json(mf)
        if prior != out_manifest:
            raise ContractError('Resume manifest differs: data / policy / scenes / seed changed')
        records_path = output / 'joint_records.jsonl'
        if records_path.is_file():
            old_records = list(read_jsonl(records_path))
        done = [r['scene_id'] for r in old_records]
        if len(done) != len(set(done)) or any(s not in requested for s in done):
            raise ContractError('Duplicate/out-of-range records prevent safe resume')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Output directory already contains files; use a new directory')
        output.mkdir(parents=True, exist_ok=True)
        save_json(mf, out_manifest)
    from p1a_runner import load_scene_local_observations, _get_step_metrics
    done_set = {r['scene_id'] for r in old_records}
    records = list(old_records)
    try:
        with (output / 'joint_records.jsonl').open('a' if args.resume else 'w', encoding='utf-8') as sink:
            for scene_id in requested:
                if scene_id in done_set:
                    continue
                try:
                    # All four original local observations are captured BEFORE step.
                    obs = load_scene_local_observations(env, scene_id)
                    t0 = time.perf_counter()
                    acts = teacher.act(obs, original_scene_id=scene_id, seed=args.seed)
                    teacher_elapsed = time.perf_counter() - t0
                    # Strict validation before touching the environment.
                    from p1b.core import normalize_teacher_actions
                    acts = normalize_teacher_actions(acts, assumed_space='env')
                    random.seed(args.seed + scene_id)
                    np.random.seed(args.seed + scene_id)
                    before_len = len(env.g_done_list)
                    _, rewards, _, _, _ = env.step(acts)
                    metrics = _get_step_metrics(env, before_len, rewards)
                    rec = make_experience(dm['dataset_id'], scene_id, obs, acts,
                                          metrics, src)
                    rec['teacher_latency_s'] = float(teacher_elapsed)
                    rec['status'] = 'ok'
                    print('scene={} accept={} IR={} V={:.3f} cost={:.3f}'.format(
                        scene_id, int(metrics['g_done']), int(metrics['individual_rationality']),
                        metrics['V'], metrics['cost_sum']), flush=True)
                except Exception as exc:
                    rec = {'record_version': 'p1b_joint_v1', 'dataset_id': dm['dataset_id'],
                           'scene_id': scene_id, 'joint_id': '{}::{:06d}'.format(dm['dataset_id'],scene_id),
                           'status': 'failed', 'failure_type': type(exc).__name__,
                           'failure_message': str(exc)}
                    print('scene={} FAILED {}'.format(scene_id, str(exc)), flush=True)
                sink.write(json.dumps(rec, ensure_ascii=False, allow_nan=False)+'\n')
                sink.flush()
                records.append(rec)
                if rec['status'] != 'ok' and args.on_error == 'abort':
                    raise RuntimeError('Stopped on scene {}. Can resume with --resume'.format(scene_id))
    finally:
        if hasattr(teacher, 'close'):
            teacher.close()
        good = [r for r in records if r['status'] == 'ok']
        failures = [r for r in records if r['status'] != 'ok']
        outcome = {
            'schema': 'p1b_collection_summary_v1', 'dataset_id': dm['dataset_id'],
            'n_requested': len(requested), 'n_processed': len(records),
            'n_ok': len(good), 'n_failed': len(failures),
            'complete': len(records) == len(requested) and not failures,
            'n_admitted': sum(r['metrics']['g_done'] for r in good),
            'n_cooperative_valid': sum(r['metrics']['cooperative_valid'] for r in good),
            'n_individual_rational': sum(r['metrics']['individual_rationality'] for r in good),
            'policy': 'frozen_mapppo_teacher', 'teacher_source_kind': src.get('kind'),
            'not_final_test_results': True,
        }
        save_json(output / 'summary.json', outcome)
        with (output / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
            keys = ['scene_id','joint_id','status','g_done','individual_rationality',
                    'cooperative_valid','V','V_g','V_k1','V_k2','V_k3','cost_sum',
                    'delta_V_k1','delta_V_k2','delta_V_k3']
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in records:
                row = {k: r.get(k,'') for k in ('scene_id','joint_id','status')}
                if r['status'] == 'ok':
                    m = r['metrics']
                    row.update({k:m[k] for k in keys if k in m})
                    for k in ('k1','k2','k3'):
                        row['delta_V_'+k] = m['V_'+k] - m['V_'+k+'_0']
                writer.writerow(row)
        print('Collection output: {}'.format(output), flush=True)
        print(json.dumps(outcome, ensure_ascii=False, indent=2), flush=True)
    return outcome


if __name__ == '__main__':
    collect(get_args())
