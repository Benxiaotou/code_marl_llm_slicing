"""Frozen MAPPO/V1/V3 on predeclared val5601 scenes 0..99. No generation."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer

import p2e_prompt_3way as prompts
import gpt61_sol_dev as adapter
import val5601_eval as compat

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('gpt61_runs/val5601_prompt100_v1')
METHODS = ('mappo', 'semantic_balance', 'joint_feasibility')
SCENE_IDS = list(range(100))
CHUNK_SIZE = 5
MODEL = 'gpt-6.1-sol'
CHECKPOINT = 'marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth'


def save_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.write('\n')


def blocks(method):
    return [(0, 100)] if method == 'mappo' else [(i, CHUNK_SIZE) for i in range(0, 100, CHUNK_SIZE)]


def preflight(root: Path):
    manifest_path = root / 'p1b_data/p2b_val_seed5601.json'
    manifest = adapter.load_json(manifest_path)
    data = compat.verify_environment(root, manifest)
    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    library = LocalHistory(root / adapter.DEFAULT_LIBRARY)
    library.assert_independent(manifest)
    ir = IRHistory(root / adapter.DEFAULT_IR, library)
    sources = ['main.py', 'configs_main.yaml', 'p1c_runner.py', 'p1a_runner.py',
               'p1a/local_obs.py', 'p1a/policy.py', 'p1c/policy.py', 'p1c/retrieval.py',
               'p1d/policy.py', 'p1d/ir.py', 'p1b/teacher.py', 'p0_runner.py',
               'p2e_prompt/val5601_100.py', 'p2e_prompt/val5601_eval.py',
               'p2e_prompt/p2e_prompt_3way.py', 'p2e_prompt/gpt61_sol_dev.py']
    return {'dataset_id': compat.DATASET_ID, 'dataset_split': 'validation',
            'dataset_generation_seed': 5601, 'source_n_scenes': 200,
            'dataset_manifest_sha256': adapter.digest(manifest_path), **data,
            'scene_ids': SCENE_IDS, 'selection': 'first 100 scene IDs fixed before evaluation; no outcome filtering',
            'methods': list(METHODS), 'model': MODEL, 'api': 'chat', 'effort': 'medium',
            'max_output_tokens': 8192, 'timeout_seconds': 360, 'policy_seed': 1,
            'examples': 3, 'method': 'ir_icl', 'library': adapter.DEFAULT_LIBRARY,
            'ir_history': adapter.DEFAULT_IR, 'library_manifest_sha256': library.manifest_sha256,
            'library_role_sha256': library.library_hashes,
            'ir_manifest_sha256': ir.manifest_sha256, 'ir_role_sha256': ir.role_hashes,
            'checkpoint': CHECKPOINT, 'checkpoint_sha256': adapter.digest(root / CHECKPOINT),
            'prompt_sha256': {v: prompts.hash_str(prompts.read_prompt(v)) for v in METHODS[1:]},
            'source_sha256': {f: adapter.digest(root / f) for f in sources},
            'chunk_size_GPT': CHUNK_SIZE, 'max_calls_per_GPT_variant': 400,
            'budget_usd_per_GPT_variant': 6.0, 'budget_usd_total_GPT': 12.0,
            'prices_usd_per_million': adapter.PRICES,
            'gateway_prices_verified': False, 'python_version': sys.version,
            'training_or_final_test_data_used': False}


def read_chunk(folder: Path, method: str, start: int, count: int, protocol: dict):
    if not all((folder / f).is_file() for f in ('manifest.json', 'summary.json', 'metrics.csv', 'records.jsonl')):
        raise RuntimeError('Incomplete chunk preserved: ' + str(folder))
    manifest = adapter.load_json(folder / 'manifest.json')
    summary = adapter.load_json(folder / 'summary.json')
    expected = {'eval_dataset_id': compat.DATASET_ID, 'eval_dataset_split': 'validation',
                'eval_dataset_sha256': compat.SAMPLE_SHA, 'scene_ids': list(range(start, start + count)),
                'seed': 1, 'method': 'mappo' if method == 'mappo' else 'ir_icl',
                'model': 'frozen_mappo' if method == 'mappo' else MODEL,
                'main_py_sha256': protocol['evaluation_main_sha256'],
                'library_manifest_sha256': protocol['library_manifest_sha256'],
                'library_role_sha256': protocol['library_role_sha256'],
                'mock_not_real_llm': False}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Chunk protocol mismatch: ' + str(folder))
    if method == 'mappo':
        if manifest.get('checkpoint_sha256') != protocol['checkpoint_sha256']:
            raise RuntimeError('Frozen MAPPO checkpoint differs')
    elif manifest.get('ir_history_role_sha256') != protocol['ir_role_sha256']:
        raise RuntimeError('IR overlay differs')
    if (summary.get('n_attempted') != count or summary.get('n_scored') != count
            or summary.get('fallback_rate') != 0):
        raise RuntimeError('Chunk did not score all scenes without fallback')
    rows = adapter.read_events(folder / 'records.jsonl')
    if (len(rows) != count or [r['original_scene_id'] for r in rows] != expected['scene_ids']
            or any(r['status'] != 'ok' or r.get('fallback_used') for r in rows)):
        raise RuntimeError('Invalid scene records')
    with (folder / 'metrics.csv').open(encoding='utf-8-sig', newline='') as fp:
        metrics = list(csv.DictReader(fp))
    if len(metrics) != count or any(r['status'] != 'ok' for r in metrics):
        raise RuntimeError('Invalid metrics CSV')
    return rows


def collect(root, method, protocol):
    rows = []
    for start, count in blocks(method):
        rows.extend(read_chunk(root / OUT / method / 'chunks' / f'{start:04d}_{count:03d}',
                               method, start, count, protocol))
    return rows


def verify_audit(root, method, rows):
    folder = root / OUT / method
    audits = adapter.read_events(folder / 'prompt_injection_audit.jsonl')
    events = adapter.read_events(folder / 'usage.jsonl')
    journal = adapter.read_events(folder / 'usage_attempts.jsonl')
    if (len(audits) != 400 or len(events) != 400
            or any(e['status'] != 'ok' for e in events) or len(journal) != 800):
        raise RuntimeError('Expected exactly 400 successful calls and 800 journal entries')
    started = {e['request_id'] for e in journal if e['status'] == 'started'}
    finished = {e['request_id'] for e in journal if e['status'] == 'finished'}
    if len(started) != 400 or started != finished:
        raise RuntimeError('Unresolved or duplicate request journal entries')
    for i, row in enumerate(rows):
        for j, role in enumerate(('agent_k1', 'agent_k2', 'agent_k3', 'agent_g')):
            _, expected = prompts.inject_existing_prompt(
                {'messages': row['per_role'][role]['messages']}, variant=method,
                appendix=prompts.read_prompt(method))
            if any(expected[k] != audits[4*i+j].get(k) for k in expected):
                raise RuntimeError('Prompt injection audit mismatch')
    return adapter.totals(events)


def run_method(root, method, protocol):
    out = root / OUT / method
    state = None
    server = None
    thread = None
    key = os.environ.get('GPT61_API_KEY', '').strip()
    if method != 'mappo':
        base_url = os.environ.get('GPT61_BASE_URL', '').strip()
        if not key or not base_url:
            raise RuntimeError('GPT credentials missing; no API call made')
        state = prompts.VariantState(variant=method, appendix=prompts.read_prompt(method),
                    audit=out / 'prompt_injection_audit.jsonl',
                    upstream=adapter.upstream_url(base_url, 'chat'), key=key, api='chat',
                    model=MODEL, effort='medium', max_output_tokens=8192,
                    ledger=out / 'usage.jsonl', prices=adapter.PRICES, budget=6, max_calls=400, timeout=360)
        class Handler(adapter.LocalProxyHandler):
            pass
        Handler.state = state
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    try:
        for start, count in blocks(method):
            chunk = out / 'chunks' / f'{start:04d}_{count:03d}'
            if chunk.exists() and any(chunk.iterdir()):
                read_chunk(chunk, method, start, count, protocol)
                print('SKIP_VERIFIED', method, start, start+count-1, flush=True)
                continue
            print('RUN_CHUNK', method, start, start+count-1, flush=True)
            cmd = [sys.executable, '-B', '-u', str(ROOT / 'p2e_prompt/val5601_eval.py'),
                   '--library', adapter.DEFAULT_LIBRARY,
                   '--dataset-manifest', 'p1b_data/p2b_val_seed5601.json',
                   '--method', 'mappo' if method == 'mappo' else 'ir_icl',
                   '--start', str(start), '--count', str(count), '--seed', '1',
                   '--checkpoint', CHECKPOINT, '--on-error', 'abort', '--output-dir', str(chunk)]
            if state:
                cmd += ['--ir-history', adapter.DEFAULT_IR, '--examples', '3', '--model', MODEL,
                        '--base-url', f'http://127.0.0.1:{server.server_address[1]}/v1',
                        '--temperature', '0.1', '--max-tokens', '8192', '--api-timeout', '360', '--no-stream']
            log_path = out / 'logs' / f'{start:04d}_{count:03d}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tail = []
            with log_path.open('x', encoding='utf-8') as log:
                proc = subprocess.Popen(cmd, cwd=root, env=prompts.runner_environment(),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                        encoding='utf-8', errors='replace')
                for line in proc.stdout:
                    safe = line.replace(key, '[REDACTED_KEY]') if key else line
                    log.write(safe)
                    log.flush()
                    tail = (tail + [safe.rstrip()])[-12:]
                    if safe.startswith('method='):
                        print(method, safe.rstrip(), flush=True)
                code = proc.wait()
            if code:
                raise RuntimeError('Runner failed; no retry; records preserved. ' + '\n'.join(tail))
            read_chunk(chunk, method, start, count, protocol)
            total = adapter.totals(adapter.read_events(out / 'usage.jsonl')) if state else {}
            print('CHUNK_VERIFIED', method, start+count, 'of 100', json.dumps(total), flush=True)
    finally:
        if server:
            server.shutdown()
            server.server_close()
            thread.join(4)
    rows = collect(root, method, protocol)
    summary = prompts.aggregate(rows, method)
    if state:
        summary['usage'] = verify_audit(root, method, rows)
    target = out / 'summary.json'
    if target.exists():
        if adapter.load_json(target) != summary:
            raise RuntimeError('Existing summary differs; preserved without overwrite')
    else:
        save_new(target, summary)
    print('METHOD_COMPLETE', json.dumps(summary), flush=True)


def compare(root, protocol):
    runs = {m: collect(root, m, protocol) for m in METHODS}
    for method in METHODS[1:]:
        verify_audit(root, method, runs[method])
    for scene in SCENE_IDS:
        for role in ('agent_k1', 'agent_k2', 'agent_k3', 'agent_g'):
            if (runs[METHODS[1]][scene]['per_role'][role]['messages']
                    != runs[METHODS[2]][scene]['per_role'][role]['messages']):
                raise RuntimeError('Base messages differ between V1/V3')
    out = root / OUT / ('comparison_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    summaries = []
    for method, rows in runs.items():
        summary = prompts.aggregate(rows, method)
        if method != 'mappo':
            summary.update(verify_audit(root, method, rows))
        summaries.append(summary)
    save_new(out / 'comparison.json', {'protocol': protocol, 'results': summaries,
              'notes': 'Previously used development validation set; not independent final testing.'})
    cols = list(dict.fromkeys(k for r in summaries for k in r))
    with (out / 'comparison.csv').open('x', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=cols)
        writer.writeheader()
        writer.writerows(summaries)
    fields = ['method', 'scene_id', 'accept', 'cooperative', 'individual_rationality',
              'V', 'V_g', 'cost', 'node_ratio_mean', 'link_ratio_mean',
              'k1_ir_margin', 'k2_ir_margin', 'k3_ir_margin']
    with (out / 'paired_100scene_metrics.csv').open('x', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for method, rows in runs.items():
            for row in rows:
                metric = row['metrics']
                writer.writerow({'method': method, 'scene_id': row['original_scene_id'],
                        'accept': int(metric['g_done']), 'cooperative': int(metric['cooperative_valid']),
                        'individual_rationality': int(metric['individual_rationality']),
                        'V': metric['V'], 'V_g': metric['V_g'], 'cost': metric['cost_sum'],
                        'node_ratio_mean': prompts.node_ratio_mean(row),
                        'link_ratio_mean': prompts.link_ratio_mean(row),
                        **{role+'_ir_margin': value for role, value in prompts.ir_margins(row).items()}})
    print('COMPARISON', out, json.dumps(summaries), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'run', 'compare'), default='check')
    parser.add_argument('--method', choices=('all',) + METHODS, default='all')
    args = parser.parse_args()
    protocol = preflight(ROOT)
    if args.mode == 'check':
        print('PREFLIGHT_PASS', json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    folder = ROOT / OUT
    lock = folder / 'protocol_lock.json'
    if lock.exists():
        if adapter.load_json(lock) != protocol:
            raise RuntimeError('Protocol changed; existing results preserved')
    else:
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError('Nonempty output directory without lock; preserved')
        save_new(lock, protocol)
    if args.mode == 'compare':
        compare(ROOT, protocol)
        return
    for method in METHODS if args.method == 'all' else (args.method,):
        run_method(ROOT, method, protocol)
    if args.method == 'all':
        compare(ROOT, protocol)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('STOPPED_NO_RETRY:', str(exc), file=sys.stderr)
        sys.exit(2)
