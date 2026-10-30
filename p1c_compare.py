"""Strict matched-scene comparison for P1-C runs, with paired bootstrap intervals."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import ContractError, read_json, read_jsonl, save_json

FIELDS = ('admission', 'cooperative', 'IR', 'V', 'V_g', 'cost_sum',
          'V_k1', 'V_k2', 'V_k3', 'policy_latency_s')
COMMON = ('eval_dataset_id','eval_dataset_split','eval_dataset_sha256',
          'eval_generation_seed','scene_ids','seed','config_sha256','main_py_sha256',
          'library_training_id','library_train_sha256',
          'library_manifest_sha256','library_role_sha256')


def read_run(directory, expected_method):
    folder = Path(directory)
    meta = read_json(folder / 'manifest.json')
    if meta.get('schema') != 'p1c_run_v1' or meta.get('method') != expected_method:
        raise ContractError('Not the requested P1-C method run: ' + str(folder))
    if meta.get('mock_not_real_llm') or meta.get('guard_used') or meta.get('fallback_used'):
        raise ContractError('Mock/Guard/fallback run cannot enter performance comparison')
    if meta.get('task_finetuning') or meta.get('cross_agent_communication') or meta.get('test_time_global_state_access'):
        raise ContractError('Information boundary mismatch')
    cases = {}
    for item in read_jsonl(folder / 'records.jsonl'):
        scene = int(item['original_scene_id'])
        if scene in cases or item.get('status') != 'ok' or item.get('fallback_used'):
            raise ContractError('Duplicate / incomplete / fallback scene in ' + str(folder))
        for role in AGENTS:
            d = item.get('per_role', {}).get(role, {})
            ratios = np.asarray(d.get('ratios', []), dtype=np.float64)
            if ratios.shape != ((3,) if role == 'agent_g' else (4,)) or not np.isfinite(ratios).all():
                raise ContractError('Invalid per-role action in ' + str(folder))
        m = item['metrics']
        for k in ('V','V_g','V_k1','V_k2','V_k3','cost_sum','g_done',
                  'cooperative_valid','individual_rationality'):
            if k not in m:
                raise ContractError('Missing GEANT metric: ' + k)
        if any(not math.isfinite(float(m[k])) for k in ('V','V_g','V_k1','V_k2','V_k3','cost_sum')):
            raise ContractError('Nonfinite metrics')
        cases[scene] = item
    if sorted(cases) != sorted(int(s) for s in meta['scene_ids']):
        raise ContractError('Incomplete/mismatched scene coverage in ' + str(folder))
    return meta, cases


def values(row):
    m = row['metrics']
    return {'admission': int(m['g_done']),
            'cooperative': int(m['cooperative_valid']),
            'IR': int(m['individual_rationality']),
            'V': float(m['V']), 'V_g': float(m['V_g']),
            'V_k1': float(m['V_k1']), 'V_k2': float(m['V_k2']),
            'V_k3': float(m['V_k3']), 'cost_sum': float(m['cost_sum']),
            'policy_latency_s': float(row['policy_latency_s'])}


def paired_ci(x, rng, n_boot):
    if len(x) < 2:
        return [None, None]
    samples = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[samples].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def compare(run_paths, *, bootstrap=3000, seed=2026):
    if 'knn' not in run_paths or 'icl' not in run_paths:
        raise ValueError('At least knn and icl runs are required')
    if not 1 <= bootstrap <= 10000:
        raise ValueError('bootstrap must be between 1 and 10000')
    loaded = {name: read_run(path, name) for name, path in run_paths.items()}
    primary_meta = loaded['knn'][0]
    for name, (meta, records) in loaded.items():
        for key in COMMON:
            if meta.get(key) != primary_meta.get(key):
                raise ContractError('Runs differ in experimental condition {}: {}'.format(key, name))
        if meta.get('on_error') not in ('abort','record'):
            raise ContractError('Unknown error policy')
        if meta['eval_dataset_split'] != 'validation' and meta['eval_dataset_split'] != 'test':
            raise ContractError('Invalid P1-C data split')
    if 'zero' in loaded:
        # Zero-shot and Coop-ICL must share identical model decoding parameters.
        x, y = loaded['icl'][0], loaded['zero'][0]
        for key in ('model','temperature','max_tokens','stream'):
            if x.get(key) != y.get(key):
                raise ContractError('Zero-shot/ICL mismatch in model setting: ' + key)
    if 'ir_icl' in loaded:
        x, y = loaded['icl'][0], loaded['ir_icl'][0]
        for key in ('model', 'temperature', 'max_tokens', 'stream', 'examples', 'retrieval'):
            if x.get(key) != y.get(key):
                raise ContractError('ICL/IR-aware ICL must match setting: ' + key)
        if not y.get('ir_history_manifest_sha256') or not y.get('ir_history_role_sha256'):
            raise ContractError('IR-aware method is missing verified training IR provenance')
        for scene in loaded['icl'][1]:
            baseline_row = loaded['icl'][1][scene]
            ir_row = loaded['ir_icl'][1][scene]
            for role in AGENTS:
                a = baseline_row['per_role'][role]
                b = ir_row['per_role'][role]
                if a.get('retrieved_ids_for_audit_only') != b.get('retrieved_ids_for_audit_only'):
                    raise ContractError('IR-aware retrieval examples differ from baseline at scene {} role {}'.format(scene,role))
                da = a.get('distances_for_audit_only')
                db = b.get('distances_for_audit_only')
                if da is None or db is None or not np.allclose(da, db, atol=1e-9, rtol=0):
                    raise ContractError('IR-aware retrieval distances differ from baseline')
    # P1-E 2x2 factorial: all four conditions must use identical current
    # observations, frozen training history and nearest-neighbour retrieval.
    for method in ('ir_prompt_only', 'ir_data_only'):
        if method not in loaded:
            continue
        original, ablated = loaded['icl'][0], loaded[method][0]
        for key in ('model', 'temperature', 'max_tokens', 'stream', 'examples', 'retrieval'):
            if original.get(key) != ablated.get(key):
                raise ContractError('Ablation differs in model/retrieval setting {}: {}'.format(key, method))
        expect_risk = method == 'ir_prompt_only'
        expect_data = method == 'ir_data_only'
        if (ablated.get('ir_risk_prompt_present') is not expect_risk or
                ablated.get('ir_history_data_present') is not expect_data or
                ablated.get('ir_ablation_factorial') != method):
            raise ContractError('Invalid factor flags in ' + method)
        if expect_data:
            if not ablated.get('ir_history_manifest_sha256') or not ablated.get('ir_history_role_sha256'):
                raise ContractError('IR data-only missing verified training IR source')
            if 'ir_icl' in loaded:
                full = loaded['ir_icl'][0]
                for key in ('ir_history_manifest_sha256', 'ir_history_role_sha256', 'ir_feedback_basis'):
                    if ablated.get(key) != full.get(key):
                        raise ContractError('Data-only / full IR history differs: ' + key)
        elif (ablated.get('ir_history_manifest_sha256') is not None or
              ablated.get('ir_history_role_sha256') is not None or
              ablated.get('ir_feedback_basis') is not None):
            raise ContractError('Prompt-only must not load historical IR numbers')
        for scene, baseline_row in loaded['icl'][1].items():
            other_row = loaded[method][1][scene]
            for role in AGENTS:
                baseline = baseline_row['per_role'][role]
                ablation = other_row['per_role'][role]
                if baseline.get('retrieved_ids_for_audit_only') != ablation.get('retrieved_ids_for_audit_only'):
                    raise ContractError('Ablation retrieved different examples: {} scene {} {}'.format(method, scene, role))
                if not np.allclose(baseline.get('distances_for_audit_only', []),
                                   ablation.get('distances_for_audit_only', []), rtol=0, atol=1e-9):
                    raise ContractError('Ablation retrieval distances differ: {} scene {} {}'.format(method, scene, role))
                base_messages, ab_messages = baseline.get('messages'), ablation.get('messages')
                if (not isinstance(base_messages, list) or len(base_messages) != 2 or
                        not isinstance(ab_messages, list) or len(ab_messages) != 2):
                    raise ContractError('Missing ablation prompts for audit')
                if expect_risk and ab_messages[1] != base_messages[1]:
                    raise ContractError('Prompt-only must retain the exact same user content as ICL')
                if expect_data and ab_messages[0] != base_messages[0]:
                    raise ContractError('Data-only must retain the exact same system content as ICL')
                if 'ir_icl' in loaded:
                    full_message = loaded['ir_icl'][1][scene]['per_role'][role]['messages']
                    if expect_risk and ab_messages[0] != full_message[0]:
                        raise ContractError('Prompt-only and full IR must share exactly the same risk instruction')
                    if expect_data and ab_messages[1] != full_message[1]:
                        raise ContractError('Data-only and full IR must share exactly the same historical feedback')
    if 'random_icl' in loaded:
        x, y = loaded['icl'][0], loaded['random_icl'][0]
        for key in ('model', 'temperature','max_tokens', 'stream', 'examples'):
            if x.get(key) != y.get(key):
                raise ContractError('ICL/Random-ICL must match model setting: ' + key)
    scene_ids = sorted(loaded['knn'][1])
    rng = np.random.default_rng(seed)
    result = {
        'schema': 'p1c_comparison_v1', 'dataset_id': primary_meta['eval_dataset_id'],
        'dataset_split': primary_meta['eval_dataset_split'], 'n_paired': len(scene_ids),
        'scene_ids': scene_ids, 'methods': list(loaded),
        'warning': 'Paired bootstrap assumes independent scenarios; validation results are development estimates. '
                   'No model-sampling uncertainty is captured by one inference per scene.',
        'method_means': {}, 'pairwise_comparisons': {}, 'action_diversity': {},
    }
    by_method = {}
    for name, (manifest, rows) in loaded.items():
        by_method[name] = {scene: values(rows[scene]) for scene in scene_ids}
        result['method_means'][name] = {
            field: float(np.mean([by_method[name][s][field] for s in scene_ids])) for field in FIELDS}
        result['method_means'][name]['n_admitted'] = sum(by_method[name][s]['admission'] for s in scene_ids)
        result['method_means'][name]['n_cooperative'] = sum(by_method[name][s]['cooperative'] for s in scene_ids)
        diversity = {}
        for role in AGENTS:
            actions = np.array([rows[s]['per_role'][role]['ratios'] for s in scene_ids], dtype=np.float64)
            diversity[role] = {
                'unique_actions_rounded_3dp': len({tuple(np.round(v,3)) for v in actions}),
                'per_dimension_std': [float(v) for v in np.std(actions, axis=0)],
            }
        result['action_diversity'][name] = diversity
    pairs = [('knn','icl')]
    if 'zero' in loaded:
        pairs.extend([('zero','icl'),('knn','zero')])
    if 'random_icl' in loaded:
        pairs.extend([('knn','random_icl'),('random_icl','icl')])
    if 'mappo' in loaded:
        pairs.extend([('mappo','knn'),('mappo','icl')])
        if 'zero' in loaded:
            pairs.append(('mappo','zero'))
        if 'random_icl' in loaded:
            pairs.append(('mappo','random_icl'))
    if 'ir_icl' in loaded:
        pairs.append(('icl','ir_icl'))
        pairs.append(('knn','ir_icl'))
        if 'mappo' in loaded:
            pairs.append(('mappo','ir_icl'))
        if 'random_icl' in loaded:
            pairs.append(('random_icl','ir_icl'))
        if 'zero' in loaded:
            pairs.append(('zero','ir_icl'))
    if 'ir_prompt_only' in loaded:
        pairs.append(('icl', 'ir_prompt_only'))
        if 'ir_icl' in loaded:
            pairs.append(('ir_prompt_only', 'ir_icl'))
        if 'mappo' in loaded:
            pairs.append(('mappo', 'ir_prompt_only'))
    if 'ir_data_only' in loaded:
        pairs.append(('icl', 'ir_data_only'))
        if 'ir_icl' in loaded:
            pairs.append(('ir_data_only', 'ir_icl'))
        if 'mappo' in loaded:
            pairs.append(('mappo', 'ir_data_only'))
    if 'ir_prompt_only' in loaded and 'ir_data_only' in loaded:
        pairs.append(('ir_prompt_only', 'ir_data_only'))
    for a, b in pairs:
        pr = {}
        for field in FIELDS:
            deltas = np.array([by_method[b][s][field] - by_method[a][s][field]
                               for s in scene_ids], dtype=np.float64)
            pr[field] = {'delta_{}_minus_{}'.format(b,a): float(deltas.mean()),
                         'paired_bootstrap_95pct_interval': paired_ci(deltas, rng, bootstrap)}
        if a == 'knn' and b == 'icl':
            table = {(x,y):0 for x in (0,1) for y in (0,1)}
            for s in scene_ids:
                key = (by_method[a][s]['admission'], by_method[b][s]['admission'])
                table[key] += 1
            pr['admission_paired_contingency'] = {
                'both_fail': table[(0,0)], 'knn_only': table[(1,0)],
                'icl_only': table[(0,1)], 'both_admit': table[(1,1)]}
        result['pairwise_comparisons'][b + '_vs_' + a] = pr
    # Interaction = full - data-only - prompt-only + ordinary ICL.
    # Calculating it on the SAME scenes gives a legitimate paired bootstrap
    # rather than combining unrelated marginal confidence intervals.
    if all(m in loaded for m in ('icl', 'ir_prompt_only', 'ir_data_only', 'ir_icl')):
        effects = {
            'factor_definition': {
                'baseline': 'icl: no extra IR risk instruction, no historical IR numbers',
                'instruction_only': 'ir_prompt_only: same full P1-D instruction, no historical IR numbers',
                'data_only': 'ir_data_only: same full P1-D historical IR numbers and factual explanation, no extra IR instruction',
                'both': 'ir_icl: original full P1-D instruction and historical data',
                'interaction': 'ir_icl - ir_data_only - ir_prompt_only + icl',
            },
            'per_metric': {},
        }
        for field in FIELDS:
            effect_series = {}
            for label, (a, b) in {
                'instruction_only_minus_baseline': ('icl', 'ir_prompt_only'),
                'data_only_minus_baseline': ('icl', 'ir_data_only'),
                'full_minus_baseline': ('icl', 'ir_icl'),
            }.items():
                effect_series[label] = np.array(
                    [by_method[b][scene][field] - by_method[a][scene][field] for scene in scene_ids],
                    dtype=np.float64)
            effect_series['factor_interaction'] = np.array([
                by_method['ir_icl'][scene][field] - by_method['ir_data_only'][scene][field]
                - by_method['ir_prompt_only'][scene][field] + by_method['icl'][scene][field]
                for scene in scene_ids], dtype=np.float64)
            effects['per_metric'][field] = {
                label: {'mean_delta': float(delta.mean()),
                        'paired_bootstrap_95pct_interval': paired_ci(delta, rng, bootstrap)}
                for label, delta in effect_series.items()
            }
        result['ir_ablation_factorial'] = effects
    per_scene = []
    for s in scene_ids:
        row = {'scene_id':s}
        for name in loaded:
            for field in FIELDS:
                row[name + '_' + field] = by_method[name][s][field]
        for a, b in pairs:
            for field in ('admission','cooperative','V','V_g','cost_sum'):
                row['delta_' + b + '_vs_' + a + '_' + field] = (by_method[b][s][field] -
                                                               by_method[a][s][field])
        per_scene.append(row)
    return result, per_scene


def main(argv=None):
    p = argparse.ArgumentParser(description='P1-C strict matched-scene comparison')
    p.add_argument('--knn', required=True)
    p.add_argument('--icl', required=True)
    p.add_argument('--ir-icl', default='', help='New IR-aware Qwen-ICL P1-D run')
    p.add_argument('--ir-prompt-only', default='', help='P1-E IR instruction only, no historical IR numbers')
    p.add_argument('--ir-data-only', default='', help='P1-E IR numbers only, no extra IR risk instruction')
    p.add_argument('--random-icl', default='')
    p.add_argument('--zero', default='')
    p.add_argument('--mappo', default='')
    p.add_argument('--bootstrap', type=int, default=3000)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)
    paths = {'knn': args.knn, 'icl':args.icl}
    if args.ir_icl:
        paths['ir_icl'] = args.ir_icl
    if args.ir_prompt_only:
        paths['ir_prompt_only'] = args.ir_prompt_only
    if args.ir_data_only:
        paths['ir_data_only'] = args.ir_data_only
    if args.random_icl:
        paths['random_icl'] = args.random_icl
    if args.zero:
        paths['zero'] = args.zero
    if args.mappo:
        paths['mappo'] = args.mappo
    summary, rows = compare(paths, bootstrap=args.bootstrap)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory already contains data')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'comparison.json', summary)
    with (output / 'paired_scenes.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
