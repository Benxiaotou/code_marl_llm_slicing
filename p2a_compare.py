"""P2-A paired 200-scene comparison, preserving all P1-E audit checks.

The earlier eight-method comparator is invoked UNMODIFIED; only the new P2-A
run is added afterwards, with strict cross-run and nearest-anchor checks.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import ContractError, save_json
from p1c_compare import COMMON, FIELDS, compare as compare_prior, paired_ci, read_run, values
from p2a.rerank import RerankConfig

_SOURCE_DELIMITER = 'Own-role historical demonstrations from '


def _same_current_prompt_prefix(messages):
    if not isinstance(messages, list) or len(messages) != 2:
        raise ContractError('Expected 2-message role-local IR-aware prompt')
    text = messages[1]['content']
    if text.count(_SOURCE_DELIMITER) != 1:
        raise ContractError('Unknown current observation prompt structure')
    return text.split(_SOURCE_DELIMITER, 1)[0]


def compare_with_rerank(existing_paths, risk_path, *, bootstrap=3000, seed=2026):
    if 'ir_icl' not in existing_paths:
        raise ContractError('P2-A comparison requires existing full IR-aware baseline')
    summary, scene_rows = compare_prior(existing_paths, bootstrap=bootstrap, seed=seed)
    base_meta, base_records = read_run(existing_paths['ir_icl'], 'ir_icl')
    new_meta, new_records = read_run(risk_path, 'ir_rerank')
    for key in COMMON:
        if base_meta.get(key) != new_meta.get(key):
            raise ContractError('P2-A mismatched run condition: ' + key)
    # Original P1-D runs predate the P1-E manifest factor flags. Their
    # manifest intentionally has no ir_risk_prompt_present and
    # ir_history_data_present keys. Compare all original provenance keys and
    # verify the *actual prompts* below, rather than interpreting absence as
    # False. Reject explicit False values if a newer baseline supplies flags.
    for key in ('model','temperature','max_tokens','stream','examples','seed',
                'ir_history_manifest_sha256','ir_history_role_sha256',
                'ir_feedback_basis'):
        if base_meta.get(key) != new_meta.get(key):
            raise ContractError('P2-A must match full IR-aware setting: ' + key)
    for key in ('ir_risk_prompt_present', 'ir_history_data_present'):
        if key in base_meta and base_meta[key] is not True:
            raise ContractError('P2-A baseline explicitly disables: ' + key)
    if (new_meta.get('ir_risk_prompt_present') is not True
            or new_meta.get('ir_history_data_present') is not True
            or not new_meta.get('ir_history_manifest_sha256')
            or new_meta.get('retrieval') != 'train_fitted_local_numeric_group_features_v1_ir_rerank_p2a'
            or new_meta.get('p2a_nearest_anchor') is not True):
        raise ContractError('Invalid P2-A reranker provenance/information-boundary flags')
    cfg = new_meta.get('p2a_rerank_config')
    if not isinstance(cfg, dict):
        raise ContractError('Missing P2-A config')
    expected_cfg = RerankConfig(pool_size=cfg['pool_size'], risk_weight=cfg['risk_weight'],
                                max_abs_distance_gap=cfg['max_abs_distance_gap'],
                                max_distance_ratio=cfg['max_distance_ratio']).as_dict()
    if cfg != expected_cfg:
        raise ContractError('Rerank algorithm settings are inconsistent')
    scenes = summary['scene_ids']
    if sorted(new_records) != scenes:
        raise ContractError('P2-A rerank scenes differ from original validation runs')
    if base_meta['eval_dataset_split'] not in ('validation', 'test'):
        raise ContractError('Not valid evaluation split')
    altered = {role: 0 for role in AGENTS}
    all_eligible = {role: [] for role in AGENTS[:3]}
    for scene in scenes:
        for role in AGENTS:
            a = base_records[scene]['per_role'][role]
            b = new_records[scene]['per_role'][role]
            ia = a.get('retrieved_ids_for_audit_only')
            ib = b.get('retrieved_ids_for_audit_only')
            db = b.get('distances_for_audit_only')
            da = a.get('distances_for_audit_only')
            tr = b.get('p2a_rerank_trace_for_audit_only')
            if (not isinstance(ib, list) or len(ib) != new_meta['examples']
                    or len(set(ib)) != len(ib) or not isinstance(ia, list)
                    or not isinstance(db, list) or len(db) != len(ib)
                    or not isinstance(tr, dict) or tr.get('schema') != 'p2a_rerank_trace_v1'
                    or tr.get('role') != role or tr.get('selected_ids') != ib
                    or tr.get('baseline_top_ids') != ia):
                raise ContractError('Invalid P2-A role-level retrieval evidence at {} {}'.format(scene, role))
            if a.get('messages', [None])[0] != b.get('messages', [None])[0]:
                raise ContractError('P2-A changed risk instruction instead of only selecting cases')
            if _same_current_prompt_prefix(a['messages']) != _same_current_prompt_prefix(b['messages']):
                raise ContractError('P2-A changed current-state prompt')
            got_sha = hashlib.sha256(json.dumps(b['messages'], ensure_ascii=False,
                                                sort_keys=True).encode('utf-8')).hexdigest()
            if b.get('prompt_sha256') != got_sha:
                raise ContractError('Tampered P2-A prompt SHA256')
            if role == 'agent_g':
                if (ia != ib or a['messages'] != b['messages']
                        or not np.allclose(da, db, atol=1e-10, rtol=0)
                        or tr.get('strategy') != 'unchanged_original_nearest'):
                    raise ContractError('P2-A must not change g retrieval or prompt')
            else:
                if (ib[0] != ia[0] or abs(float(db[0]) - float(da[0])) > 1e-9
                        or tr.get('strategy') != 'nearest_anchor_plus_local_risk_rerank'
                        or tr.get('changed_vs_nearest') is not (ib != ia)
                        or not set(ib).issubset(set(tr.get('candidate_top_ids', [])))
                        or tr['candidate_top_ids'][0] != ia[0]
                        or len(tr['candidate_top_ids']) > cfg['pool_size']
                        or not 3 <= int(tr.get('eligible_count', 0)) <= len(tr['candidate_top_ids'])):
                    raise ContractError('P2-A anchor/candidate rule violation at {} {}'.format(scene, role))
                selected_scores = tr.get('selected_scores')
                if not isinstance(selected_scores, list) or len(selected_scores) != len(ib):
                    raise ContractError('Missing P2-A risk score audit')
                for c, score in zip(ib, selected_scores):
                    if score['experience_id'] != c or not all(math.isfinite(float(score[k])) for k in (
                            'local_distance','historical_train_risk','rerank_score')):
                        raise ContractError('Invalid P2-A risk scores')
                    if not 0 <= float(score['historical_train_risk']) <= 1:
                        raise ContractError('P2-A risk outside training quantiles')
                eligible_rows = tr.get('eligible_scores')
                if not isinstance(eligible_rows, list) or len(eligible_rows) != tr['eligible_count']:
                    raise ContractError('Missing reranking eligible candidate audit')
                candidate_ids = tr['candidate_top_ids']
                if [item.get('experience_id') for item in eligible_rows] != candidate_ids[:len(eligible_rows)]:
                    raise ContractError('Risk candidate list must be nearest-local prefix')
                first_distance = float(db[0])
                allowed = min(first_distance + cfg['max_abs_distance_gap'],
                              first_distance * cfg['max_distance_ratio'])
                if len(eligible_rows) > new_meta['examples']:
                    if any(float(item['local_distance']) > allowed + 1e-10
                           for item in eligible_rows):
                        raise ContractError('Reranking used a distant training example')
                for pos, score_item in enumerate(eligible_rows):
                    d = float(score_item['local_distance'])
                    historical = float(score_item['historical_train_risk'])
                    if not math.isfinite(d) or not 0 <= historical <= 1:
                        raise ContractError('Nonfinite rerank candidate audit')
                    if pos and d < float(eligible_rows[pos - 1]['local_distance']) - 1e-10:
                        raise ContractError('Candidates not ordered by local distance')
                    expected_score = ((1 - cfg['risk_weight']) *
                                      max(0., (d - first_distance) / max(.15, first_distance))
                                      + cfg['risk_weight'] * historical)
                    if abs(float(score_item['rerank_score']) - expected_score) > 1e-9:
                        raise ContractError('Risk ranking score cannot be reproduced')
                expected_other = sorted(eligible_rows[1:], key=lambda item: (
                    float(item['rerank_score']), float(item['local_distance']),
                    item['experience_id']))[:new_meta['examples'] - 1]
                predicted_ids = [ia[0]] + [item['experience_id'] for item in expected_other]
                if ib != predicted_ids:
                    raise ContractError('Risk-aware examples do not match recorded selection scores')
                if selected_scores != [next(item for item in eligible_rows
                                            if item['experience_id'] == cid) for cid in ib]:
                    raise ContractError('Selected risk case scores were altered')
                all_eligible[role].append(int(tr['eligible_count']))
            if ia != ib:
                altered[role] += 1
    # Add exactly one new method to P1-E's already audited comparison.
    summary['methods'].append('ir_rerank')
    raw = {scene: values(new_records[scene]) for scene in scenes}
    summary['method_means']['ir_rerank'] = {
        field: float(np.mean([raw[s][field] for s in scenes])) for field in FIELDS
    }
    summary['method_means']['ir_rerank']['n_admitted'] = sum(raw[s]['admission'] for s in scenes)
    summary['method_means']['ir_rerank']['n_cooperative'] = sum(raw[s]['cooperative'] for s in scenes)
    summary['action_diversity']['ir_rerank'] = {}
    for role in AGENTS:
        actions = np.asarray([new_records[s]['per_role'][role]['ratios'] for s in scenes], dtype=float)
        summary['action_diversity']['ir_rerank'][role] = {
            'unique_actions_rounded_3dp': len({tuple(np.round(vec, 3)) for vec in actions}),
            'per_dimension_std': [float(v) for v in np.std(actions, axis=0)]}
    existing = {key: read_run(path, key)[1] for key, path in existing_paths.items()}
    rng = np.random.default_rng(seed + 208)
    for other_name in ('ir_icl','mappo','ir_prompt_only','ir_data_only','knn','icl'):
        if other_name not in existing:
            continue
        other_values = {s: values(existing[other_name][s]) for s in scenes}
        entry = {}
        for field in FIELDS:
            deltas = np.array([raw[s][field] - other_values[s][field] for s in scenes])
            entry[field] = {
                'delta_ir_rerank_minus_' + other_name: float(deltas.mean()),
                'paired_bootstrap_95pct_interval': paired_ci(deltas, rng, bootstrap)}
        for field in ('admission', 'cooperative'):
            table = {(x,y):0 for x in (0,1) for y in (0,1)}
            for s in scenes:
                table[(other_values[s][field], raw[s][field])] += 1
            entry[field + '_paired_contingency'] = {
                'both_fail': table[(0,0)], 'other_only': table[(1,0)],
                'rerank_only': table[(0,1)], 'both_succeed': table[(1,1)]}
        summary['pairwise_comparisons']['ir_rerank_vs_' + other_name] = entry
    summary['p2a_retrieval_audit'] = {
        'schema': 'p2a_rerank_audit_v1',
        'config': cfg, 'new_method': 'ir_rerank',
        'reference': 'ir_icl', 'n_scenes': len(scenes),
        'changed_scene_count_per_role': altered,
        'mean_eligible_candidates_per_existing_role': {
            role: float(np.mean(values)) for role, values in all_eligible.items()},
        'g_unchanged_all_scenes': altered['agent_g'] == 0,
        'same_current_local_prompt_and_same_ir_overlay': True,
        'development_only_warning': 'These are repeatedly used validation scenarios, NOT an untouched test set',
    }
    for row in scene_rows:
        scene = row['scene_id']
        for field in FIELDS:
            row['ir_rerank_' + field] = raw[scene][field]
            for ref in ('ir_icl','mappo','ir_prompt_only'):
                if ref in existing:
                    row['delta_ir_rerank_vs_' + ref + '_' + field] = raw[scene][field] - values(existing[ref][scene])[field]
        for role in AGENTS:
            row['p2a_changed_' + role] = int(
                new_records[scene]['per_role'][role]['p2a_rerank_trace_for_audit_only']['changed_vs_nearest'])
    return summary, scene_rows


def main(argv=None):
    p = argparse.ArgumentParser(description='P2-A nine-way matched-scene comparison')
    p.add_argument('--knn', required=True)
    p.add_argument('--icl', required=True)
    p.add_argument('--ir-icl', required=True)
    p.add_argument('--ir-rerank', required=True)
    for method in ('ir-prompt-only','ir-data-only','mappo','random-icl','zero'):
        p.add_argument('--' + method, default='')
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)
    paths = {'knn': args.knn, 'icl': args.icl, 'ir_icl': args.ir_icl}
    for name in ('ir_prompt_only','ir_data_only','mappo','random_icl','zero'):
        path = getattr(args, name)
        if path:
            paths[name] = path
    report, rows = compare_with_rerank(paths, args.ir_rerank, bootstrap=args.bootstrap)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite comparison output')
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'comparison.json', report)
    with (out / 'paired_scenes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'n_paired': report['n_paired'],
                      'method_means': report['method_means'],
                      'p2a_retrieval_audit': report['p2a_retrieval_audit'],
                      'p2a_vs_ir_icl': report['pairwise_comparisons']['ir_rerank_vs_ir_icl']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
