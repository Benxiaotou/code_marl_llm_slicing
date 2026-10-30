"""Aggregate independently produced P0 runs; never compare unmatched scene sets."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from p0_runner import summarize


def compare(paths, output):
    summaries = []
    per_policy = {}
    for p in paths:
        folder = Path(p)
        files = list(folder.glob('*_records.jsonl'))
        if len(files) != 1:
            raise ValueError('Expected exactly one records.jsonl in ' + str(folder))
        rows = [json.loads(line) for line in files[0].read_text(encoding='utf-8').splitlines() if line.strip()]
        summ = summarize(rows)
        if summ['policy'] in per_policy:
            raise ValueError('Duplicate policy ' + summ['policy'])
        per_policy[summ['policy']] = {r['original_scene_index']:r for r in rows}
        summaries.append(summ)
    sets = [set(d) for d in per_policy.values()]
    if not all(s == sets[0] for s in sets[1:]):
        raise ValueError('Policies evaluated different original scene sets; cannot compare fairly')
    with Path(output).open('w', newline='', encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print('Wrote',output)
    for item in summaries:
        print('{policy}: n={n}, acceptance={acceptance_rate:.3f}, V={V:.3f}, fallback={fallback_rate:.3f}'.format(**item))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run_folders',nargs='+')
    parser.add_argument('--output',default='p0_comparison.csv')
    args=parser.parse_args()
    compare(args.run_folders,args.output)
