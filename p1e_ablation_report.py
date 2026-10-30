"""Readable 2x2 IR-ablation comparison from verified P1-C/P1-D/P1-E logs.

Does not recompute environment results, select models or access any Qwen API.
"""
import argparse
from pathlib import Path

from p1b.core import ContractError, read_json

NAMES = {
    'mappo': 'MAPPO',
    'knn': 'kNN-Action',
    'icl': 'Ordinary Qwen-ICL (no IR extras)',
    'ir_prompt_only': 'IR-Prompt-only (instruction)',
    'ir_data_only': 'IR-Data-only (historical numbers)',
    'ir_icl': 'IR-aware Qwen-ICL (both)',
    'random_icl': 'Random-ICL',
    'zero': 'Qwen-ZeroShot',
}


def make_report(data):
    keys = ('icl', 'ir_prompt_only', 'ir_data_only', 'ir_icl')
    if data.get('schema') != 'p1c_comparison_v1' or not all(
            k in data.get('method_means', {}) for k in keys):
        raise ContractError('Need all four factorial variants in one matched-scene comparison')
    effects = data.get('ir_ablation_factorial', {}).get('per_metric', {})
    if not effects:
        raise ContractError('Missing audited 2x2 factorial paired comparisons')
    n = int(data['n_paired'])
    out = [
        '# P1-E: Role-local IR data vs IR risk instruction (2x2 ablation)',
        '',
        'Dataset: `{}` | split: `{}` | paired scenes: **{}**'.format(
            data['dataset_id'], data['dataset_split'], n),
        '',
        '| Method | Admission | Cooperative valid | V | Adjustment cost | Latency (s) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for key in ('mappo', 'knn', 'icl', 'ir_prompt_only', 'ir_data_only',
                'ir_icl', 'random_icl', 'zero'):
        if key not in data['method_means']:
            continue
        m = data['method_means'][key]
        out.append('| {} | {:.1%} ({}/{}) | {:.1%} ({}/{}) | {:.3f} | {:.3f} | {:.3f} |'.format(
            NAMES[key], m['admission'], m['n_admitted'], n,
            m['cooperative'], m['n_cooperative'], n,
            m['V'], m['cost_sum'], m['policy_latency_s']))
    out += ['', '## IR ablation contrasts and 95% paired bootstrap intervals', '',
            'All effects are mean differences over the SAME scenarios; pp means percentage points.', '',
            '| Metric | Instruction only - ICL | Data only - ICL | Full - ICL | Interaction |',
            '|---|---:|---:|---:|---:|']
    labels = ('instruction_only_minus_baseline', 'data_only_minus_baseline',
              'full_minus_baseline', 'factor_interaction')
    for metric in ('admission', 'cooperative', 'V', 'cost_sum', 'V_g', 'policy_latency_s'):
        def f(label):
            e = effects[metric][label]
            m, (low, high) = e['mean_delta'], e['paired_bootstrap_95pct_interval']
            if metric in ('admission', 'cooperative'):
                return '{:+.2f} pp [{:+.2f}, {:+.2f}]'.format(100*m, 100*low, 100*high)
            return '{:+.3f} [{:+.3f}, {:+.3f}]'.format(m, low, high)
        out.append('| {} | {} | {} | {} | {} |'.format(metric, *(f(x) for x in labels)))
    out += [
        '',
        'The interaction is **Full - Data-only - Prompt-only + ordinary ICL**.',
        'It is a descriptive factorial contrast, not proof that the IR numbers causally predict a current action.',
        '', '## Research limitations', '',
        '- These are validation/development results, not a new untouched final test set.',
        '- Each scene has only one Qwen realization; intervals do not cover inference randomness.',
        '- Data-only includes factual field definitions but no additional IR *instruction*; full uses the exact same data block.',
        '- Prompt-only receives the full IR risk instruction but no historical IR numbers or private joint records.',
        '- All variants retain identical current role-only 240D observations, original teacher actions, and three retrieved cases.',
        '- Training-history IR margins are outcomes of historical JOINT actions, not independent causal effects.',
        '',
    ]
    return '\n'.join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description='Summarize P1-E IR 2x2 ablation from verified comparison')
    p.add_argument('--comparison', required=True)
    p.add_argument('--output', default='')
    args = p.parse_args(argv)
    report = make_report(read_json(args.comparison))
    print(report)
    if args.output:
        target = Path(args.output)
        if target.exists():
            raise FileExistsError('Refuse to replace existing ablation report: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding='utf-8')
        print('\nSaved ' + str(target))


if __name__ == '__main__':
    main()
