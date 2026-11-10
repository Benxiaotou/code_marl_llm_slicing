"""Zero-API Qwen/MAPPO counterfactual replay on val5601 scenes 0..199.

This script scores only cached joint actions in the unchanged GEANT evaluation
environment. It never constructs an LLM client and never selects a deployed
policy. MM and QQ controls must match their archived metrics before mixed
actions are interpreted. Outputs are always written to a new timestamp folder.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('agent_k1', 'agent_k2', 'agent_k3', 'agent_g')
K_ROLES = ROLES[:3]
NUMERIC = ('V', 'V_g', 'V_k1', 'V_k2', 'V_k3', 'V_k1_0', 'V_k2_0',
           'V_k3_0', 'X_k1', 'X_k2', 'X_k3', 'cost_k1', 'cost_k2',
           'cost_k3', 'revenue_k1', 'revenue_k2', 'revenue_k3',
           'revenue_g', 'cost_sum', 'reward_mean')
BOOLEANS = ('g_done', 'individual_rationality', 'cooperative_valid')
SAMPLE_SHA = '7d92e97808f3269dcf11b3275c1cf85f6daae50d1678f412b7c29e347ad79c86'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    rows = {}
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            scene = row.get('original_scene_id')
            if not isinstance(scene, int) or scene in rows:
                raise ValueError(f'{path}: invalid/duplicate scene at line {line_no}')
            if row.get('status') != 'ok' or row.get('fallback_used'):
                raise ValueError(f'{path}: incomplete/fallback scene {scene}')
            actions = row.get('env_actions')
            metrics = row.get('metrics')
            if not isinstance(actions, dict) or not isinstance(metrics, dict):
                raise ValueError(f'{path}: missing actions/metrics at scene {scene}')
            for role in ROLES:
                vec = actions.get(role)
                if (not isinstance(vec, list) or len(vec) != 4 or
                        any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                            not math.isfinite(x) or x < -1.00001 or x > 1.00001 for x in vec)):
                    raise ValueError(f'{path}: malformed {role} action at scene {scene}')
            rows[scene] = row
    if sorted(rows) != list(range(200)):
        raise ValueError(f'{path}: expected exactly scenes 0..199')
    return rows


def chosen_actions(mappo, qwen, case, scene):
    m = mappo[scene]['env_actions']
    q = qwen[scene]['env_actions']
    if case == 'MM':
        return {r: list(m[r]) for r in ROLES}
    if case == 'QQ':
        return {r: list(q[r]) for r in ROLES}
    if case == 'MQ':  # MAPPO existing slices, Qwen new slice
        return {r: list(m[r] if r in K_ROLES else q[r]) for r in ROLES}
    if case == 'QM':  # Qwen existing slices, MAPPO new slice
        return {r: list(q[r] if r in K_ROLES else m[r]) for r in ROLES}
    if case == 'Q_NODE_M':  # Qwen g; MAPPO node controls, Qwen link controls
        out = {r: list(q[r]) for r in K_ROLES}
        for r in K_ROLES:
            out[r][:2] = m[r][:2]
        out['agent_g'] = list(q['agent_g'])
        return out
    if case == 'Q_LINK_M':  # Qwen g; Qwen node controls, MAPPO link controls
        out = {r: list(q[r]) for r in K_ROLES}
        for r in K_ROLES:
            out[r][2:] = m[r][2:]
        out['agent_g'] = list(q['agent_g'])
        return out
    raise ValueError(case)


def metrics_from_env(env, before, rewards):
    from p1a_runner import _get_step_metrics
    return _get_step_metrics(env, before, rewards)


def replay_scene(env, cached, case, scene, seed):
    from p1a_runner import load_scene_local_observations
    load_scene_local_observations(env, scene)
    actions = {r: np.asarray(v, dtype=np.float32) for r, v in
               chosen_actions(*cached, case, scene).items()}
    random.seed(seed + scene)
    np.random.seed(seed + scene)
    before = len(env.g_done_list)
    _, rewards, _, _, _ = env.step(actions)
    metrics = metrics_from_env(env, before, rewards)
    return {'status': 'ok', 'original_scene_id': scene, 'case': case,
            'metrics': metrics, 'env_actions': {r: actions[r].tolist() for r in ROLES},
            'policy_latency_s': 0.0, 'no_model_call': True}


def control_diff(expected, got, atol=1e-4, rtol=1e-6):
    diffs = []
    for k in BOOLEANS:
        if bool(expected[k]) != bool(got[k]):
            diffs.append(f'{k}: expected={expected[k]} got={got[k]}')
    for k in NUMERIC:
        if k in expected and k in got and not math.isclose(float(expected[k]), float(got[k]),
                                                           abs_tol=atol, rel_tol=rtol):
            diffs.append(f'{k}: expected={expected[k]} got={got[k]}')
    return diffs


def aggregate(rows):
    n = len(rows)
    def mean(key):
        return float(np.mean([float(r['metrics'][key]) for r in rows]))
    return {'n': n, 'acceptance_rate': mean('g_done'),
            'cooperative_rate': mean('cooperative_valid'),
            'ir_valid_rate': mean('individual_rationality'),
            'V_mean': mean('V'), 'V_g_mean': mean('V_g'),
            'cost_mean': mean('cost_sum'),
            'V_k1_mean': mean('V_k1'), 'V_k2_mean': mean('V_k2'),
            'V_k3_mean': mean('V_k3'),
            'accepted_ir_invalid_n': sum(int(bool(r['metrics']['g_done']) and
                                             not bool(r['metrics']['individual_rationality']))
                                         for r in rows)}


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', default='p2f_analysis/val5601_counterfactual')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--control-atol', type=float, default=1e-4)
    args = ap.parse_args()
    if args.seed != 1:
        raise ValueError('Cached validation actions are locked to seed=1')
    mpath = ROOT / 'p2b_runs/val5601_mappo200/records.jsonl'
    qpath = ROOT / 'p2b_runs/val5601_standard1000_p2a200/records.jsonl'
    manifest_path = ROOT / 'p1b_data/p2b_val_seed5601.json'
    for p in (mpath, qpath, manifest_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('dataset_id') != 'p2b_val_seed5601' or
            manifest.get('split') != 'validation' or manifest.get('n_scenes') != 200 or
            manifest.get('sample_sha256') != SAMPLE_SHA):
        raise ValueError('Unexpected val5601 manifest')
    mappo, qwen = read_jsonl(mpath), read_jsonl(qpath)
    for scene in range(200):
        for key in ('V_k1_0', 'V_k2_0', 'V_k3_0'):
            if not math.isclose(mappo[scene]['metrics'][key], qwen[scene]['metrics'][key], abs_tol=1e-8):
                raise ValueError(f'Unpaired source baselines scene {scene} {key}')
    # Validate source CSVs against cached records before loading the environment.
    for tag, directory, records in (
            ('mappo', ROOT / 'p2b_runs/val5601_mappo200', mappo),
            ('qwen', ROOT / 'p2b_runs/val5601_standard1000_p2a200', qwen)):
        csv_path = directory / 'metrics.csv'
        run_meta = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if (run_meta['eval_dataset_id'] != manifest['dataset_id'] or
                run_meta['eval_dataset_split'] != 'validation' or
                run_meta['eval_dataset_sha256'] != SAMPLE_SHA or
                run_meta['scene_ids'] != list(range(200)) or run_meta['seed'] != args.seed or
                run_meta['config_sha256'] != manifest['config_sha256'] or
                run_meta.get('guard_used') or run_meta.get('fallback_used') or
                run_meta.get('mock_not_real_llm') or run_meta.get('test_time_global_state_access')):
            raise ValueError(f'{tag}: source protocol mismatch')
        with csv_path.open(encoding='utf-8-sig', newline='') as f:
            csv_rows = {int(r['original_scene_id']): r for r in csv.DictReader(f)}
        if set(csv_rows) != set(range(200)):
            raise ValueError(f'{tag}: metrics.csv scene mismatch')
        for scene, row in records.items():
            met = row['metrics']
            for key, csv_key in (('V', 'V'), ('V_g', 'V_g'), ('cost_sum', 'cost_sum')):
                if not math.isclose(float(met[key]), float(csv_rows[scene][csv_key]),
                                    abs_tol=1e-8):
                    raise ValueError(f'{tag}: records/CSV mismatch scene {scene} {key}')
    # Install legacy pickle aliases, then use the val5601 compatibility adapter's
    # exact audited environment path. This reads the existing 644 MiB sample once.
    import sys
    sys.path.insert(0, str(ROOT))
    import p1a_runner
    import p2e_prompt.val5601_eval as adapter
    main_module, _ = p1a_runner._config(ROOT, ROOT / 'configs_main.yaml')
    p1a_runner._compat_pickle_main_classes(main_module)
    environment_check = adapter.verify_environment(ROOT, manifest)
    import p1c_runner
    cases = ('MM', 'QQ', 'MQ', 'QM', 'Q_NODE_M', 'Q_LINK_M')
    out_root = ROOT / args.output_root
    out = out_root / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'env_runtime'
    runtime.mkdir(exist_ok=False)
    # Original evaluation periodically saves diagnostics. Redirect those saves
    # to this NEW output folder before any step, preserving official results.
    log = (out / 'env_stdout.log').open('x', encoding='utf-8')
    with contextlib.redirect_stdout(log):
        env = p1c_runner.create_eval_env(ROOT, manifest, 'configs_main.yaml')
    env.path_reward_save = str(runtime)
    env.path_env_save = str(runtime)
    protocol = {'schema': 'val5601_qwen_mappo_counterfactual_v1',
                'dataset_id': 'p2b_val_seed5601', 'dataset_split': 'validation',
                'dataset_sha256': SAMPLE_SHA, 'scene_ids': list(range(200)),
                'seed': args.seed, 'cases': list(cases), 'no_model_calls': True,
                'environment_check': environment_check,
                'source_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in
                                  (Path(__file__).resolve(), ROOT / 'main.py', ROOT / 'configs_main.yaml',
                                   ROOT / 'p1a_runner.py', ROOT / 'p2e_prompt/val5601_eval.py')},
                'no_api_cost_usd': 0.0, 'control_cases': ['MM', 'QQ'],
                'action_sources': {
                    'MM': 'all MAPPO', 'QQ': 'all Qwen P2-A Standard-1000',
                    'MQ': 'MAPPO k1/k2/k3 + Qwen g',
                    'QM': 'Qwen k1/k2/k3 + MAPPO g',
                    'Q_NODE_M': 'Qwen g; MAPPO k node dims 0:2; Qwen k link dims 2:4',
                    'Q_LINK_M': 'Qwen g; Qwen k node dims 0:2; MAPPO k link dims 2:4'},
                'input_files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
                                  'sha256': sha256(p)} for p in (mpath, qpath, manifest_path)}}
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    all_rows = {}
    try:
        for case in cases:
            rows = []
            for scene in range(200):
                with contextlib.redirect_stdout(log):
                    row = replay_scene(env, (mappo, qwen), case, scene, args.seed)
                rows.append(row)
                if case in ('MM', 'QQ'):
                    expected = (mappo if case == 'MM' else qwen)[scene]['metrics']
                    diffs = control_diff(expected, row['metrics'], args.control_atol)
                    if diffs:
                        (out / 'CONTROL_FAILED.json').write_text(
                            json.dumps({'case': case, 'scene': scene, 'diff': diffs}, indent=2) + '\n',
                            encoding='utf-8')
                        raise RuntimeError(f'{case} control mismatch scene {scene}: {diffs[:3]}')
            all_rows[case] = rows
            (out / (case + '_records.jsonl')).write_text(
                ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows),
                encoding='utf-8')
            print('COMPLETE', case, json.dumps(aggregate(rows), ensure_ascii=False), flush=True)
        summary = {case: aggregate(rows) for case, rows in all_rows.items()}
        comparison = []
        by_case = {case: {r['original_scene_id']: r for r in rows}
                   for case, rows in all_rows.items()}
        for scene in range(200):
            row = {'scene': scene}
            for case in cases:
                met = by_case[case][scene]['metrics']
                row.update({case + '_accept': int(met['g_done']),
                            case + '_cooperative': int(met['cooperative_valid']),
                            case + '_IR': int(met['individual_rationality']),
                            case + '_V': float(met['V']),
                            case + '_V_g': float(met['V_g']),
                            case + '_cost': float(met['cost_sum'])})
            comparison.append(row)
        write_csv(out / 'paired_counterfactual.csv', comparison)
        report = {'status': 'VALIDATED', 'protocol': protocol, 'summary': summary,
                  'control_match_MM': True, 'control_match_QQ': True,
                  'diagnostic_only': True,
                  'interpretation': 'Mixed cached actions are descriptive counterfactuals, not a deployable policy.'}
        (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('OUTPUT', out)
    except Exception:
        # Preserve partial records and protocol for audit; never overwrite.
        raise
    finally:
        log.close()


if __name__ == '__main__':
    main()
