"""Offline paired analysis of an already completed val5601 100-scene run."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

METHODS = ('mappo', 'semantic_balance', 'joint_feasibility')
PAIRS = (('semantic_balance', 'mappo'), ('joint_feasibility', 'mappo'),
         ('joint_feasibility', 'semantic_balance'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', required=True)
    args = parser.parse_args()
    folder = Path(args.comparison_dir).resolve()
    with (folder / 'paired_100scene_metrics.csv').open(encoding='utf-8-sig', newline='') as fp:
        rows = list(csv.DictReader(fp))
    comparison = json.loads((folder / 'comparison.json').read_text(encoding='utf-8'))
    protocol = comparison['protocol']
    assert protocol['dataset_id'] == 'p2b_val_seed5601'
    assert protocol['dataset_split'] == 'validation'
    assert protocol['scene_ids'] == list(range(100))
    assert len(rows) == 300 and {r['method'] for r in rows} == set(METHODS)
    by = {m: sorted([r for r in rows if r['method'] == m], key=lambda r: int(r['scene_id']))
          for m in METHODS}
    assert all([int(r['scene_id']) for r in values] == list(range(100)) for values in by.values())
    rng = np.random.default_rng(20261104)
    indices = rng.integers(0, 100, size=(10000, 100))
    results = []
    for a, b in PAIRS:
        result = {'method_A': a, 'method_B': b, 'differences_are_A_minus_B': True, 'metrics': {}}
        for metric in ('accept', 'cooperative', 'V', 'V_g', 'cost', 'node_ratio_mean', 'link_ratio_mean'):
            first = np.array([float(r[metric]) for r in by[a]])
            second = np.array([float(r[metric]) for r in by[b]])
            difference = first - second
            lo, hi = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
            item = {'mean_difference': float(difference.mean()),
                    'paired_bootstrap_95pct_CI': [float(lo), float(hi)],
                    'A_higher_scenes': int((difference > 1e-8).sum()),
                    'A_lower_scenes': int((difference < -1e-8).sum()),
                    'ties': int((np.abs(difference) <= 1e-8).sum())}
            if metric in ('accept', 'cooperative'):
                wins = int(((first == 1) & (second == 0)).sum())
                losses = int(((first == 0) & (second == 1)).sum())
                discordant = wins + losses
                p_value = min(1.0, 2 * sum(math.comb(discordant, k)
                                          for k in range(min(wins, losses) + 1)) / 2**discordant)
                item.update(A_only_success=wins, B_only_success=losses,
                            exact_McNemar_two_sided_p=p_value)
            result['metrics'][metric] = item
        results.append(result)
    out = folder / 'paired_analysis.json'
    with out.open('x', encoding='utf-8') as fp:
        json.dump({'dataset': protocol['dataset_id'], 'n_scenes': 100,
                   'bootstrap_seed': 20261104, 'bootstrap_replicates': 10000,
                   'limitations': 'Existing development data; exploratory, unadjusted comparisons; one LLM draw per scene.',
                   'comparisons': results}, fp, ensure_ascii=False, indent=2)
        fp.write('\n')
    print('OFFLINE_PAIRED_ANALYSIS_SAVED', out)


if __name__ == '__main__':
    main()
