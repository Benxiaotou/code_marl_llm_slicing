"""P2-A no-LLM, no-step, role-local retrieval preview on known validation scenes.

Prints how often risk reranking changes a training demonstration, before paying
for 800 real Qwen requests. Never claims performance improvement from preview.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import save_json
from p1c.retrieval import LocalHistory
from p1d.ir import IRHistory
from p1c_runner import checked_eval_manifest, create_eval_env
from p1a_runner import load_scene_local_observations
from p2a.rerank import IRRiskReranker, RerankConfig


def preview(library, ir_history, env, scenes, *, examples=3, config=None):
    ranker = IRRiskReranker(library, ir_history, config=config or RerankConfig())
    rows = []
    for scene in scenes:
        local = load_scene_local_observations(env, scene)
        for role in AGENTS:
            chosen, trace = ranker.query(role, local[role], top_k=examples)
            row = {
                'scene_id': scene, 'role': role,
                'changed_from_nearest': int(trace['changed_vs_nearest']),
                'closest_distance': float(chosen[0]['distance']),
                'eligible_count': int(trace['eligible_count']),
                'baseline_ids': '|'.join(trace['baseline_top_ids']),
                'risk_selected_ids': '|'.join(trace['selected_ids']),
                'mean_selected_distance': float(np.mean([c['distance'] for c in chosen])),
            }
            rows.append(row)
    summary = {
        'schema': 'p2a_preview_v1', 'n_scenes': len(scenes),
        'n_roles': len(AGENTS), 'config': ranker.config.as_dict(),
        'changes_by_role': {role: int(sum(x['changed_from_nearest'] for x in rows
                                          if x['role'] == role)) for role in AGENTS},
        'mean_eligible_by_role': {role: float(np.mean([x['eligible_count'] for x in rows
                                                     if x['role'] == role])) for role in AGENTS},
        'nearest_distance_quantiles_by_role': {
            role: {str(q): float(np.quantile([x['closest_distance'] for x in rows
                                           if x['role'] == role], q)) for q in (.5, .9, .95)}
            for role in AGENTS},
        'warning': ('This is a retrospective validation-set preview of retrieval structure, '
                    'NOT a test-time action simulation or estimate of future utility.'),
    }
    return summary, rows


def main(argv=None):
    p = argparse.ArgumentParser(description='P2-A retrieval-only preview without Qwen/GEANT step')
    p.add_argument('--library', required=True)
    p.add_argument('--ir-history', required=True)
    p.add_argument('--dataset-manifest', required=True)
    p.add_argument('--config', default='configs_main.yaml')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--examples', type=int, default=3)
    p.add_argument('--risk-pool', type=int, default=20)
    p.add_argument('--risk-weight', type=float, default=.35)
    p.add_argument('--risk-max-gap', type=float, default=.10)
    p.add_argument('--risk-max-ratio', type=float, default=1.35)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args(argv)
    root = Path.cwd().resolve()
    library = LocalHistory(root / args.library)
    history = IRHistory(root / args.ir_history, library)
    manifest = checked_eval_manifest(root, args.dataset_manifest, library, args.config)
    if args.start < 0 or args.count < 1 or args.start + args.count > manifest['n_scenes']:
        raise ValueError('Invalid preview scene range')
    cfg = RerankConfig(args.risk_pool, args.risk_weight, args.risk_max_gap,
                       args.risk_max_ratio)
    if args.examples != 3 or args.examples > cfg.pool_size:
        raise ValueError('Example count exceeds candidate pool')
    env = create_eval_env(root, manifest, args.config)
    summary, rows = preview(library, history, env,
                            range(args.start, args.start + args.count),
                            examples=args.examples, config=cfg)
    summary['dataset_id'] = manifest['dataset_id']
    summary['training_library_sha256'] = library.manifest_sha256
    summary['ir_history_sha256'] = history.manifest_sha256
    out = (root / args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite preview output')
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'preview.json', summary)
    with (out / 'retrieval_preview.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
