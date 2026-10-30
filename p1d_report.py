"""Summarize P1-D six-way paired comparison without editing original results."""
import argparse
from pathlib import Path

from p1b.core import ContractError, read_json

NAMES = {
    'mappo': 'MAPPO',
    'knn': 'kNN-Action',
    'icl': 'Qwen-Coop-ICL',
    'ir_icl': 'IR-aware Qwen-ICL',
    'random_icl': 'Random-ICL',
    'zero': 'Qwen-ZeroShot',
}


def make_report(data):
    if (data.get('schema') != 'p1c_comparison_v1' or
            'ir_icl' not in data.get('methods', ())):
        raise ContractError('Report requires a complete P1-D (ir_icl) comparison file')
    n = int(data['n_paired'])
    lines = [
        '# P1-D: IR-aware Qwen-ICL matched-scene experiment',
        '',
        'Dataset: `{}` | split: `{}` | paired scenes: **{}**'.format(
            data['dataset_id'], data['dataset_split'], n),
        '',
        '| Method | Admission | Cooperative valid | V | Cost | Latency (s) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for key in ('mappo','knn','icl','ir_icl','random_icl','zero'):
        if key not in data['method_means']:
            continue
        m=data['method_means'][key]
        lines.append('| {} | {:.1%} ({}/{}) | {:.1%} ({}/{}) | {:.3f} | {:.3f} | {:.3f} |'.format(
            NAMES[key],m['admission'],m['n_admitted'],n,
            m['cooperative'],m['n_cooperative'],n,m['V'],m['cost_sum'],m['policy_latency_s']))
    lines.extend(['','## IR-aware Qwen-ICL paired changes', '',
                  '| Comparison | Metric | Mean delta | Paired bootstrap 95% interval |',
                  '|---|---|---:|---|'])
    pairwise=data['pairwise_comparisons']
    for other in ('icl','mappo','knn','random_icl','zero'):
        k='ir_icl_vs_'+other
        if k not in pairwise:
            continue
        for metric in ('admission','cooperative','V','V_g','cost_sum','policy_latency_s'):
            entry=pairwise[k][metric]
            mean=entry['delta_ir_icl_minus_'+other]
            ci=entry['paired_bootstrap_95pct_interval']
            if metric in ('admission','cooperative'):
                fmt=lambda x:'{:+.2f} pp'.format(100*x)
            else:
                fmt=lambda x:'{:+.3f}'.format(x)
            lines.append('| IR-aware minus {} | {} | {} | [{}, {}] |'.format(
                NAMES[other],metric,fmt(mean),fmt(ci[0]),fmt(ci[1])))
    lines.extend(['','## Notes', '',
                  '- These are **development/validation** estimates, NOT untouched final test results.',
                  '- Bootstrap intervals treat scenes as independent and do not capture repeated LLM sampling uncertainty.',
                  '- IR annotations come exclusively from the **same role\'s training-case** returns/costs.',
                  '- Historical own IR margin is a **joint historical outcome**, not the causal effect of one agent\'s action.',
                  '- Check `paired_scenes.csv` for per-scene changes and verify `manifest.json` in each run.',
                  ''])
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description='Human-readable P1-D metrics and paired differences')
    p.add_argument('--comparison', required=True)
    p.add_argument('--output', default='',help='Optional path for Markdown summary')
    args=p.parse_args(argv)
    report=make_report(read_json(args.comparison))
    print(report)
    if args.output:
        path=Path(args.output)
        if path.exists():
            raise FileExistsError('Refusing to overwrite existing report: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(report,encoding='utf-8')
        print('\nSummary saved to '+str(path))


if __name__=='__main__':
    main()
