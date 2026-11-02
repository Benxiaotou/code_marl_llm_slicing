#!/usr/bin/env python3
"""Safe sequential runner for P2-B val5601: checkpoint-by-validation, never overwrite.

Run in the original Windows experiment project with the working 'marl' environment.
By default only preflights; --execute starts the serial queue.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

DATASET_REL = Path('p1b_data/p2b_val_seed5601.json')
DATASET_ID = 'p2b_val_seed5601'
DATASET_SHA = '7d92e97808f3269dcf11b3275c1cf85f6daae50d1678f412b7c29e347ad79c86'
MODEL = 'qwen3-next-80b-a3b-instruct'
COUNT = 200
SEED = 1
TEMPERATURE = 0.1
TOKENS = 4096
EXPECTED_P2A_POLICY = 'qwen_ir_risk_reranked_coop_icl_local_p2a_v1'
RERANK = {'pool_size': 20, 'risk_weight': 0.35,
          'max_abs_distance_gap': 0.10, 'max_distance_ratio': 1.35,
          'anchor_nearest': True, 'g_retrieval_unchanged': True}
ROLES = ('agent_k1', 'agent_k2', 'agent_k3', 'agent_g')

@dataclass(frozen=True)
class Job:
    tag: str
    method: str
    lib: str
    ir: str | None = None

    @property
    def run_dir(self) -> Path:
        return Path('p2b_runs') / f'val5601_{self.tag}200'


def standard_jobs(extended_ir: bool = False) -> list[Job]:
    base = 'p1b_libraries/train_seed2601_v1'
    base_ir = 'p1d_libraries/train_seed2601_ir_v1'
    cases = [('base152', base, base_ir)]
    for mode in ('standard', 'balanced'):
        for size in (300, 500, 1000):
            cases.append((f'{mode}{size}', f'p2b_libraries/p2b_{mode}_{size}_v2',
                          f'p2b_ir_libraries/p2b_{mode}_{size}_ir_v2'))
    jobs: list[Job] = []
    # Non-Qwen checks first; fastest way to catch environment issues before API calls.
    for tag, lib, _ in cases:
        jobs.append(Job(f'{tag}_knn', 'knn', lib))
    jobs.append(Job('mappo', 'mappo', base))
    # Original two completed jobs reuse precisely these output locations.
    for tag, lib, ir in cases:
        jobs.append(Job(f'{tag}_p2a', 'ir_rerank', lib, ir))
    # Main IR-aware ablation first at base, standard 500, balanced 500.
    wanted = {'base152', 'standard500', 'balanced500'}
    if extended_ir:
        wanted.update(tag for tag, _, _ in cases)
    for tag, lib, ir in cases:
        if tag in wanted:
            jobs.append(Job(f'{tag}_p1d', 'ir_icl', lib, ir))
    return jobs


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def get_overlay_manifest(root: Path, rel: str) -> Path:
    p = root / rel
    require(p.is_dir(), f'Missing IR overlay: {p}')
    candidates = [p / 'ir_manifest.json', p / 'manifest.json']
    found = [x for x in candidates if x.is_file()]
    require(len(found) == 1, f'Expected exactly one IR manifest in {p}: {found}')
    return found[0]


def check_dataset(root: Path) -> dict:
    dataset = root / DATASET_REL
    require(dataset.is_file(), f'Missing dataset manifest: {dataset}')
    d = load_json(dataset)
    for k, v in [('dataset_id', DATASET_ID), ('split', 'validation'), ('n_scenes', COUNT),
                 ('sample_sha256', DATASET_SHA)]:
        require(d.get(k) == v, f'{dataset}: {k} must equal {v!r}, got {d.get(k)!r}')
    sample = Path(d['sample_file'])
    if not sample.is_absolute():
        sample = root / sample
    require(sample.is_file(), f'Missing validation sample file: {sample}')
    require(sha256(sample) == DATASET_SHA, 'Validation sample SHA256 mismatch; stop')
    for filename, field in [('main.py', 'main_py_sha256'),
                            ('configs_main.yaml', 'config_sha256')]:
        expected = d.get(field)
        if expected:
            actual = sha256(root / filename)
            require(actual == expected, f'{filename} changed compared to dataset generation: stop')
    return d


def get_library_meta(root: Path, rel: str) -> dict:
    p = root / rel / 'library_manifest.json'
    require(p.is_file(), f'Missing library: {p}')
    m = load_json(p)
    require(not m.get('development_only', False), f'Development-only source: {rel}')
    for role in ROLES:
        require((root / rel / 'role_libraries' / f'{role}.jsonl').is_file(),
                f'Missing role library: {rel}, {role}')
    return {'meta': m, 'sha': sha256(p)}


def check_existing(job: Job, root: Path, cache: dict) -> dict:
    """Refuse to trust partial/stale runs. Any existing directory must fully validate."""
    out = root / job.run_dir
    require(out.is_dir(), f'Not a directory: {out}')
    for name in ('summary.json', 'manifest.json', 'metrics.csv', 'records.jsonl'):
        require((out / name).is_file(), f'INCOMPLETE {job.tag}: missing {name} in {out}')
    m = load_json(out / 'manifest.json')
    summary = load_json(out / 'summary.json')
    expected = {
        'method': job.method, 'mock_not_real_llm': False,
        'eval_dataset_id': DATASET_ID, 'eval_dataset_split': 'validation',
        'eval_dataset_sha256': DATASET_SHA, 'eval_generation_seed': 5601,
        'scene_ids': list(range(COUNT)), 'seed': SEED, 'on_error': 'abort',
        'fallback_used': False, 'guard_used': False, 'task_finetuning': False,
        'test_time_global_state_access': False, 'cross_agent_communication': False,
    }
    for field, value in expected.items():
        require(m.get(field) == value, f'INVALID existing {job.tag}: {field} differs')
    if job.method == 'ir_rerank':
        require(m.get('policy') == EXPECTED_P2A_POLICY,
                f'{job.tag}: unexpected P2-A policy')
    # Config and implementation fingerprints should match the current checkout.
    require(m.get('config_sha256') == cache['config_sha'], f'{job.tag}: config changed')
    require(m.get('main_py_sha256') == cache['main_sha'], f'{job.tag}: main.py changed')
    lib = cache['libs'][job.lib]
    require(m.get('library_training_id') == lib['meta']['dataset_id'],
            f'{job.tag}: wrong library training ID')
    require(m.get('library_train_sha256') == lib['meta']['training_sample_sha256'],
            f'{job.tag}: wrong training-source SHA')
    require(m.get('library_manifest_sha256') == lib['sha'],
            f'{job.tag}: wrong library manifest SHA')
    for role in ROLES:
        actual = sha256(root / job.lib / 'role_libraries' / f'{role}.jsonl')
        require(m.get('library_role_sha256', {}).get(role) == actual,
                f'{job.tag}: wrong role library SHA for {role}')
    if job.ir:
        ir_sha = cache['irs'][job.ir]
        require(m.get('ir_history_manifest_sha256') == ir_sha,
                f'{job.tag}: wrong IR manifest SHA')
        require(m.get('ir_history_data_present') is True,
                f'{job.tag}: history IR data missing')
        if job.method == 'ir_rerank':
            for k, value in RERANK.items():
                require((m.get('p2a_rerank_config') or {}).get(k) == value,
                        f'{job.tag}: rerank parameter mismatch: {k}')
    else:
        require(m.get('ir_history_manifest_sha256') is None,
                f'{job.tag}: unexpected IR overlay')
    if job.method in ('ir_rerank', 'ir_icl'):
        for field, value in [('model', MODEL), ('examples', 3),
                             ('temperature', TEMPERATURE), ('max_tokens', TOKENS),
                             ('stream', True)]:
            require(m.get(field) == value, f'{job.tag}: {field} mismatch')
    if job.method == 'knn':
        require(m.get('model') == 'no_llm_knn', f'{job.tag}: wrong knn model')
        require(m.get('examples') == 1, f'{job.tag}: expected one kNN case')
    if job.method == 'mappo':
        require(m.get('model') == 'frozen_mappo', f'{job.tag}: wrong MAPPO model')
        require(bool(m.get('checkpoint_sha256')), f'{job.tag}: no MAPPO checkpoint hash')
    for field, value in [('n_attempted', COUNT), ('n_scored', COUNT),
                         ('n_failed_policy', 0), ('fallback_rate', 0.0),
                         ('eval_dataset_id', DATASET_ID), ('method', job.method),
                         ('mock_not_real_llm', False)]:
        require(summary.get(field) == value,
                f'INCOMPLETE existing {job.tag}: summary {field} mismatch')
    with (out / 'metrics.csv').open('r', encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    require(len(records) == COUNT, f'{job.tag}: metrics rows {len(records)} != 200')
    seen = set()
    for i, row in enumerate(records):
        scene_id = int(row['original_scene_id'])
        require(scene_id not in seen and scene_id == i, f'{job.tag}: metrics scenes not ordered 0..199')
        seen.add(scene_id)
        require(row['status'] == 'ok' and not row.get('failed_role'),
                f'{job.tag}: failed scene {scene_id}')
    with (out / 'records.jsonl').open('r', encoding='utf-8-sig') as f:
        record_lines = [line for line in f if line.strip()]
    require(len(record_lines) == COUNT, f'{job.tag}: records.jsonl != 200 rows')
    for i, line in enumerate(record_lines):
        rec = json.loads(line)
        require(rec.get('original_scene_id') == i and rec.get('status') == 'ok'
                and not rec.get('fallback_used', False),
                f'{job.tag}: invalid record at row {i}')
    for field, col, kind in [('acceptance_rate_scored_only', 'g_done', 'bool'),
                             ('cooperative_valid_rate_scored_only', 'cooperative_valid', 'bool'),
                             ('V_scored_only', 'V', 'float'),
                             ('cost_sum_scored_only', 'cost_sum', 'float')]:
        if kind == 'bool':
            require(all(row[col] in ('True', 'False') for row in records),
                    f'{job.tag}: invalid bool field {col}')
            vals = [float(row[col] == 'True') for row in records]
        else:
            vals = [float(row[col]) for row in records]
        require(all(math.isfinite(v) for v in vals), f'{job.tag}: nonfinite {col}')
        mean = sum(vals) / COUNT
        require(math.isclose(mean, float(summary[field]), abs_tol=1e-8, rel_tol=1e-9),
                f'{job.tag}: summary/metrics mismatch {field}: {mean} / {summary[field]}')
    return summary


def job_command(job: Job) -> list[str]:
    cmd = [sys.executable, '-u', 'p1c_runner.py',
           '--library', job.lib, '--dataset-manifest', str(DATASET_REL),
           '--method', job.method, '--start', '0', '--count', str(COUNT),
           '--seed', str(SEED), '--on-error', 'abort',
           '--output-dir', str(job.run_dir)]
    if job.ir:
        cmd += ['--ir-history', job.ir]
    if job.method in ('ir_icl', 'ir_rerank'):
        cmd += ['--examples', '3', '--model', MODEL, '--temperature', str(TEMPERATURE),
                '--max-tokens', str(TOKENS), '--api-timeout', '120']
    if job.method == 'ir_rerank':
        cmd += ['--risk-pool', '20', '--risk-weight', '0.35',
                '--risk-max-gap', '0.10', '--risk-max-ratio', '1.35']
    return cmd


def append_queue_event(path: Path, action: str, job: Job, extra: dict | None = None) -> None:
    row = {'time_utc': datetime.now(timezone.utc).isoformat(), 'action': action,
           'job': job.tag, 'method': job.method,
           'output_dir': str(job.run_dir)}
    if extra:
        row.update(extra)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def build_cache(root: Path, jobs: list[Job]) -> dict:
    require((root / 'p1c_runner.py').is_file(), 'Run from original project root: missing p1c_runner.py')
    check_dataset(root)
    cache = {'libs': {}, 'irs': {}, 'config_sha': sha256(root / 'configs_main.yaml'),
             'main_sha': sha256(root / 'main.py')}
    for job in jobs:
        if job.lib not in cache['libs']:
            cache['libs'][job.lib] = get_library_meta(root, job.lib)
        if job.ir and job.ir not in cache['irs']:
            cache['irs'][job.ir] = sha256(get_overlay_manifest(root, job.ir))
    if any(job.method == 'mappo' for job in jobs):
        checkpoint = root / 'marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth'
        require(checkpoint.is_file(), f'Missing frozen MAPPO checkpoint: {checkpoint}')
    return cache


def run_new(job: Job, root: Path, event_file: Path, cache: dict) -> None:
    out = root / job.run_dir
    require(not out.exists(), f'REFUSING to overwrite existing run directory: {out}')
    logs = root / 'p2b_runs' / 'p2b_val5601_queue_logs'
    logs.mkdir(parents=True, exist_ok=True)
    log_file = logs / (job.tag + '.log')
    cmd = job_command(job)
    append_queue_event(event_file, 'start', job, {'command': cmd, 'log': str(log_file)})
    print('\nRUN', job.tag, '|', job.method, '|', ' '.join(cmd), flush=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with log_file.open('a', encoding='utf-8') as log:
        log.write('\n### START ' + datetime.now(timezone.utc).isoformat() + ' ###\n')
        log.write('COMMAND: ' + repr(cmd) + '\n')
        log.flush()
        with subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace',
                              bufsize=1, env=env) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if (not line.startswith('method=') or 'FAILED' in line or
                    (' scene=' in line and any(' scene=' + str(i) + ' ' in line
                                                for i in range(0, COUNT, 25)))):
                    print(line.rstrip(), flush=True)
            ret = proc.wait()
    if ret != 0:
        append_queue_event(event_file, 'failed_process', job, {'return_code': ret})
        raise RuntimeError(f'{job.tag} returned exit={ret}; partial output kept at {out}; log={log_file}')
    try:
        summary = check_existing(job, root, cache)
    except Exception as exc:
        append_queue_event(event_file, 'failed_validation', job, {'error': str(exc)})
        raise
    append_queue_event(event_file, 'completed', job, {
        'cooperative_valid_rate': summary['cooperative_valid_rate_scored_only'],
        'acceptance_rate': summary['acceptance_rate_scored_only'],
        'V': summary['V_scored_only'], 'cost': summary['cost_sum_scored_only']})
    print('COMPLETE', job.tag, 'coop=', summary['cooperative_valid_rate_scored_only'],
          'V=', round(summary['V_scored_only'], 3),
          'cost=', round(summary['cost_sum_scored_only'], 3), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Actually run pending jobs sequentially; default read-only preflight')
    parser.add_argument('--scope', choices=('all', 'knn', 'mappo', 'rerank', 'ir_icl'), default='all')
    parser.add_argument('--extended-ir', action='store_true', help='Add 4 extra P1-D evaluations (22 jobs total)')
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    jobs = standard_jobs(args.extended_ir)
    if args.scope != 'all':
        jobs = [job for job in jobs if job.method == {'rerank': 'ir_rerank'}.get(args.scope, args.scope)]
    cache = build_cache(root, jobs)
    print('P2-B validation dataset:', DATASET_ID, 'SHA256:', DATASET_SHA, 'scenes:', COUNT)
    print('Queue:', len(jobs), '| scope:', args.scope, '| mode:', 'EXECUTE' if args.execute else 'READ-ONLY PREFLIGHT')
    completed: list[Job] = []
    pending: list[Job] = []
    for job in jobs:
        path = root / job.run_dir
        if path.exists():
            check_existing(job, root, cache)
            completed.append(job)
            print('SKIP VERIFIED', job.tag, str(job.run_dir))
        else:
            pending.append(job)
            print('PENDING', job.tag, str(job.run_dir))
    print('Validated completed:', len(completed), 'Pending:', len(pending))
    if not args.execute:
        print('PRECHECK OK. Add --execute to launch serial tests. This preflight wrote no files.')
        return 0
    if not pending:
        print('ALL FINISHED. No new experiment launched.')
        return 0
    print('Starting sequential queue. Any error stops and retains all outputs without overwriting.')
    logdir = root / 'p2b_runs' / 'p2b_val5601_queue_logs'
    logdir.mkdir(parents=True, exist_ok=True)
    event_file = logdir / 'queue_events.jsonl'
    for i, job in enumerate(pending, start=1):
        print(f'[{i}/{len(pending)}] {job.tag}', flush=True)
        run_new(job, root, event_file, cache)
    print('SUCCESS: all selected tasks have completed and passed strict validation.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print('STOPPED:', type(exc).__name__ + ':', str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
