#!/usr/bin/env python3
# coding: utf-8
"""P2-E: GPT-6.1 Sol existing-slice prompt variants on fixed dev7601 scenes 0..9.

Does not modify original project code or call APIs except in explicit --mode run.
Fresh local-only proxy modifies ONLY system prompts for agent_k1/k2/k3.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
import statistics
import subprocess
import sys
import threading

import gpt61_sol_dev as adapter

BASE = Path(__file__).resolve().parent
VERSION = 'p2e_gpt61_prompt_dev7601_v1_runtime_fix1'
VARIANT_FILES = {
    'semantic_balance': BASE / 'prompts' / '01_semantic_balance.txt',
    'teacher_scale': BASE / 'prompts' / '02_teacher_scale.txt',
    'joint_feasibility': BASE / 'prompts' / '03_joint_feasibility.txt',
}
DEFAULT_OUT = 'gpt61_runs/dev7601_prompt_ablation_v1'
ROLE_TAG = 'You represent an EXISTING slice.'
SYSTEM_START = 'You are a frozen language-model policy in a partially observable GEANT multi-agent network slicing experiment.'


def hash_str(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_prompt(variant: str) -> str:
    p = VARIANT_FILES[variant]
    t = p.read_text(encoding='utf-8').strip()
    if not (50 < len(t) < 5000) or '{"ratios"' in t:
        raise ValueError('Unexpected prompt variant content or format: ' + variant)
    return t


def inject_existing_prompt(inbound: dict, *, variant: str, appendix: str) -> tuple[dict, dict]:
    request = copy.deepcopy(inbound)
    messages = request.get('messages')
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError('Unexpected messages structure. No API call made.')
    if messages[0].get('role') != 'system' or messages[1].get('role') != 'user':
        raise ValueError('Expected original system+user role prompts only. No API call made.')
    base = messages[0].get('content')
    user_text = messages[1].get('content')
    if not isinstance(base, str) or not isinstance(user_text, str) or not base.startswith(SYSTEM_START):
        raise ValueError('Original GEANT system prompt changed; abort before API call.')
    is_existing = ROLE_TAG in base
    is_new = 'You represent NEW slice g.' in base
    if is_existing == is_new:
        raise ValueError('Could not unambiguously determine current role. No API call made.')
    if is_existing and '"role":"agent_g"' in user_text[:200]:
        raise ValueError('Role mismatch in prompt. No API call made.')
    if is_new and '"role":"agent_g"' not in user_text[:500]:
        raise ValueError('Unexpected new-slice observation. No API call made.')
    role = 'existing' if is_existing else 'new_slice_g'
    effective = base + ('\n\n' + appendix if is_existing else '')
    messages[0]['content'] = effective
    return request, {
        'variant': variant, 'role_type': role,
        'original_system_sha256': hash_str(base),
        'effective_system_sha256': hash_str(effective),
        'user_message_sha256': hash_str(user_text),
        'variant_sha256': hash_str(appendix),
        'prompt_injected': is_existing,
    }


class VariantState(adapter.ProxyState):
    def __init__(self, *, variant: str, appendix: str, audit: Path, **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.appendix = appendix
        self.audit = audit

    def handle(self, request: dict):
        modified, audit_record = inject_existing_prompt(request, variant=self.variant, appendix=self.appendix)
        # Important: budget reserve uses modified request lengths; do not bill for an unmodified body.
        result, event = super().handle(modified)
        audit_record['time_utc'] = adapter.utcnow()
        audit_record['input_tokens'] = event['input_tokens']
        audit_record['output_tokens'] = event['output_tokens']
        adapter.append_jsonl(self.audit, audit_record)
        return result, event


def results_rows(chunk: Path, start: int, n: int, dataset: str) -> list[dict] | None:
    if not adapter.verify_results(chunk, start, n, dataset):
        return None
    records = [json.loads(x) for x in (chunk / 'records.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    return records


def collect_new(root: Path, folder: Path, *, require_complete=False) -> list[dict] | None:
    out = root / folder
    rows = []
    for start in (0, 5):
        recs = results_rows(out / 'chunks' / f'{start:04d}_005', start, 5, adapter.DEFAULT_DATASET)
        if recs is None:
            if require_complete:
                raise RuntimeError('Not all ten scenes verified: ' + str(out))
            return None
        rows.extend(recs)
    return rows


def collect_baseline(root: Path) -> list[dict] | None:
    base = root / adapter.DEFAULT_RUN_FOLDER
    records = []
    for start in (0, 5):
        recs = results_rows(base / 'chunks' / f'{start:04d}_005', start, 5, adapter.DEFAULT_DATASET)
        if recs is None:
            return None
        records.extend(recs)
    return records


def collect_mappo(root: Path) -> list[dict] | None:
    folder = root / 'gpt61_runs' / 'dev7601_mappo_frozen_v1'
    if not (folder / 'records.jsonl').is_file():
        return None
    records = [json.loads(x) for x in (folder / 'records.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(records) != 10 or [r.get('original_scene_id') for r in records] != list(range(10)):
        return None
    if any(r.get('status') != 'ok' for r in records):
        return None
    return records


def data_lock(root: Path, args, variant: str, dataset: dict) -> dict:
    required = {
        'p1c_runner.py': root / 'p1c_runner.py',
        'main.py': root / 'main.py',
        'configs_main.yaml': root / 'configs_main.yaml',
        'library_manifest.json': root / adapter.DEFAULT_LIBRARY / 'library_manifest.json',
    }
    ir_root = root / adapter.DEFAULT_IR
    ir_mf = [p for p in (ir_root / 'ir_manifest.json', ir_root / 'manifest.json') if p.is_file()]
    if len(ir_mf) != 1:
        raise RuntimeError('IR overlay manifest not found; do not continue')
    required['ir_overlay_manifest'] = ir_mf[0]
    for filename in ('p1a_runner.py', 'p1a/local_obs.py', 'p1a/policy.py',
                     'p1c/policy.py', 'p1c/retrieval.py', 'p1d/policy.py', 'p1d/ir.py'):
        required[filename] = root / filename
    if not all(p.is_file() for p in required.values()):
        raise RuntimeError('Original project file/library missing; do not run')
    return {
        'version': VERSION, 'variant': variant, 'prompt_file': VARIANT_FILES[variant].name,
        'variant_sha256': hash_str(read_prompt(variant)),
        'injection_target': 'existing agent_k1/agent_k2/agent_k3 ONLY (system prompt appended)',
        'new_slice_g_prompt': 'unchanged; fresh calls still made to g',
        'dataset': dataset, 'scene_ids': list(range(10)),
        'project_hashes': {k: adapter.digest(v) for k, v in required.items()},
        'adapter_sha256': adapter.digest(BASE / 'gpt61_sol_dev.py'),
        'driver_sha256': adapter.digest(BASE / 'p2e_prompt_3way.py'),
        'python_version': sys.version, 'runtime_fixes': 'local_proxy_token_and_durable_attempt_accounting_v1',
        'model': args.model, 'upstream_api': args.api, 'effort': args.effort,
        'max_output_tokens': args.max_output_tokens,
        'timeout': args.timeout, 'chunk_size': 5,
        'policy': 'ir_icl', 'library': adapter.DEFAULT_LIBRARY, 'ir_history': adapter.DEFAULT_IR,
        'role_policy_seed': 1, 'examples': 3,
        'max_calls': 40, 'cost_budget_usd': args.budget_usd,
        'rates_usd_per_million': {'input': args.input_usd_per_m,
                                  'cached': args.cached_usd_per_m,
                                  'output': args.output_usd_per_m},
    }


def runner_environment() -> dict:
    env = os.environ.copy()
    # This token is used only for the 127.0.0.1 shim, never for the upstream API.
    env['LLM_API_KEY'] = 'p2e-local-proxy-token'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def run_variant(root: Path, args):
    if not args.variant:
        raise ValueError('--variant is required for run')
    rates = (args.input_usd_per_m, args.cached_usd_per_m, args.output_usd_per_m)
    if (not math.isfinite(args.budget_usd) or args.budget_usd <= 0
            or any(not math.isfinite(x) or x < 0 for x in rates)
            or args.max_output_tokens < 128 or args.timeout < 20):
        raise ValueError('Invalid budget, token ceiling or timeout')
    key = os.environ.get('GPT61_API_KEY', '').strip()
    base_url = os.environ.get('GPT61_BASE_URL', '').strip()
    if not key or not base_url:
        raise RuntimeError('Set GPT61_API_KEY and GPT61_BASE_URL; no output directory created')
    upstream = adapter.upstream_url(base_url, args.api, allow_http=args.allow_http_upstream)
    dataset = adapter.verify_dataset(root, adapter.DEFAULT_DATASET, 20)
    out = root / DEFAULT_OUT / args.variant
    expected = data_lock(root, args, args.variant, dataset)
    lockpath = out / 'protocol_lock.json'
    if lockpath.exists():
        if adapter.load_json(lockpath) != expected:
            raise RuntimeError('Protocol lock changed; do not overwrite old run or silently change settings')
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError('Existing non-empty directory without protocol lock; do not overwrite: ' + str(out))
        adapter.save_json(lockpath, expected)
    ledger = out / 'usage.jsonl'
    audit = out / 'prompt_injection_audit.jsonl'
    state = VariantState(variant=args.variant, appendix=read_prompt(args.variant), audit=audit,
                         upstream=upstream, key=key, api=args.api,
                         model=args.model, effort=args.effort, max_output_tokens=args.max_output_tokens,
                         ledger=ledger, prices={'input': args.input_usd_per_m,
                                                'cached_input': args.cached_usd_per_m,
                                                'output': args.output_usd_per_m},
                         budget=args.budget_usd, max_calls=40, timeout=args.timeout)
    class Handler(adapter.LocalProxyHandler):
        pass
    Handler.state = state
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        local_base = f'http://127.0.0.1:{server.server_address[1]}/v1'
        print('VARIANT', args.variant, 'sha256=', expected['variant_sha256'])
        print('BUDGET', args.budget_usd, 'USD estimate; HARD MAX 40 SUCCESS/LOGGED REQUESTS; 10 scenes only')
        for start in (0, 5):
            chunk = out / 'chunks' / f'{start:04d}_005'
            if chunk.exists() and any(chunk.iterdir()):
                if results_rows(chunk, start, 5, adapter.DEFAULT_DATASET) is not None:
                    print('SKIP verified existing chunk', start, 'to', start + 4)
                    continue
                raise RuntimeError('Incomplete chunk preserved: ' + str(chunk) + '; do not overwrite; inspect first')
            cmd = [sys.executable, '-u', 'p1c_runner.py',
                   '--library', adapter.DEFAULT_LIBRARY, '--ir-history', adapter.DEFAULT_IR,
                   '--dataset-manifest', f'p1b_data/{adapter.DEFAULT_DATASET}.json',
                   '--method', 'ir_icl', '--examples', '3',
                   '--start', str(start), '--count', '5', '--seed', '1',
                   '--model', args.model, '--base-url', local_base,
                   '--temperature', '0.1', '--max-tokens', str(args.max_output_tokens),
                   '--api-timeout', str(args.timeout), '--no-stream',
                   '--on-error', 'abort', '--output-dir', str(chunk)]
            print('RUN', args.variant, 'scenes', start, '-', start + 4, flush=True)
            exitcode = subprocess.call(cmd, cwd=root, env=runner_environment())
            if exitcode or results_rows(chunk, start, 5, adapter.DEFAULT_DATASET) is None:
                raise RuntimeError(f'Chunk {chunk} incomplete or runner error {exitcode}; preserve results and usage')
            usage = adapter.totals(adapter.read_events(ledger))
            print(f'CHUNK PASS; recorded estimated cost ${usage["cost_usd_estimate"]:.4f}')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=4)
    rows = collect_new(root, Path(DEFAULT_OUT) / args.variant, require_complete=True)
    audit_rows = adapter.read_events(audit)
    usage = adapter.totals(adapter.read_events(ledger))
    if len(audit_rows) != 40 or sum(bool(r['prompt_injected']) for r in audit_rows) != 30:
        raise RuntimeError('Prompt injection audit mismatch: expected 30 existing + 10 new-slice requests')
    if usage['requests_ok'] != 40:
        raise RuntimeError('API count inconsistent: expected 40 successful role calls')
    obj = aggregate(rows, args.variant)
    obj['usage'] = usage
    obj['prompt_injection_audit'] = {'total_role_calls': 40, 'existing_prompt_injections': 30,
                                      'g_unmodified': 10, 'sha256': expected['variant_sha256']}
    obj['dataset_id'] = adapter.DEFAULT_DATASET
    obj['prompt_variant'] = args.variant
    obj['diagnostic_only_not_independent_test'] = True
    adapter.save_json(out / 'summary.json', obj)
    print('COMPLETE', args.variant, json.dumps(obj, ensure_ascii=False, indent=2))


def aggregate(records: list[dict], label: str) -> dict:
    if not records:
        raise ValueError('No records')
    m = [r['metrics'] for r in records if r['status'] == 'ok']
    if len(m) != len(records):
        raise ValueError('Incomplete policy records: ' + label)
    summary = {'name': label, 'n': len(m),
            'accepted': sum(bool(x['g_done']) for x in m),
            'cooperative': sum(bool(x['cooperative_valid']) for x in m),
            'accept_rate': statistics.mean(float(x['g_done']) for x in m),
            'coop_rate': statistics.mean(float(x['cooperative_valid']) for x in m),
            'V_mean': statistics.mean(float(x['V']) for x in m),
            'V_g_mean': statistics.mean(float(x['V_g']) for x in m),
            'cost_mean': statistics.mean(float(x['cost_sum']) for x in m),
            'latency_mean_s': statistics.mean(float(r['policy_latency_s']) for r in records)}
    if all('individual_rationality' in x for x in m):
        summary['IR_violation_scenes'] = sum(not x['individual_rationality'] for x in m)
        summary['accepted_IR_violation_scenes'] = sum(x['g_done'] and not x['individual_rationality'] for x in m)
    if all('per_role' in r for r in records):
        summary['node_ratio_mean'] = statistics.mean(node_ratio_mean(r) for r in records)
        summary['link_ratio_mean'] = statistics.mean(link_ratio_mean(r) for r in records)
    return summary


def ir_margins(record: dict) -> dict:
    metrics = record['metrics']
    return {role: float(metrics['V_' + role]) - float(metrics['V_' + role + '_0'])
            for role in ('k1', 'k2', 'k3')}


def node_ratio_mean(record: dict) -> float:
    return statistics.mean(float(v) for role in ('agent_k1', 'agent_k2', 'agent_k3')
                           for v in record['per_role'][role]['ratios'][:2])


def link_ratio_mean(record: dict) -> float:
    return statistics.mean(float(v) for role in ('agent_k1', 'agent_k2', 'agent_k3')
                           for v in record['per_role'][role]['ratios'][2:4])


def compare(root: Path, report_dir: str | None = None):
    baselines = {'GPT_original': collect_baseline(root), 'MAPPO': collect_mappo(root)}
    baselines.update({v: collect_new(root, Path(DEFAULT_OUT) / v) for v in VARIANT_FILES})
    included = {k: v for k, v in baselines.items() if v is not None}
    if not included:
        raise RuntimeError('No complete baseline or variants found')
    run_root = root / DEFAULT_OUT
    out = root / (report_dir or (DEFAULT_OUT + '/comparison_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    out.mkdir(parents=True, exist_ok=False)
    summaries = [aggregate(r, k) for k, r in included.items()]
    for row in summaries:
        if row['name'] in VARIANT_FILES:
            ledger = run_root / row['name'] / 'usage.jsonl'
            u = adapter.totals(adapter.read_events(ledger))
            row['estimated_api_cost_usd'] = u['cost_usd_estimate']
            row['input_tokens'] = u['input_tokens']
            row['output_tokens'] = u['output_tokens']
            row['reasoning_tokens'] = u['reasoning_tokens']
            row['cached_input_tokens'] = u['cached_input_tokens']
        elif row['name'] == 'GPT_original':
            ledger = root / adapter.DEFAULT_RUN_FOLDER / 'usage.jsonl'
            u = adapter.totals(adapter.read_events(ledger))
            row['estimated_api_cost_usd'] = u['cost_usd_estimate']
            row['input_tokens'] = u['input_tokens']
            row['output_tokens'] = u['output_tokens']
            row['reasoning_tokens'] = u['reasoning_tokens']
            row['cached_input_tokens'] = u['cached_input_tokens']
    if baselines['GPT_original'] is not None:
        originals = baselines['GPT_original']
        for row in summaries:
            cur = included[row['name']]
            row['delta_cooperative_cases_vs_GPT_original'] = row['cooperative'] - sum(bool(x['metrics']['cooperative_valid']) for x in originals)
            row['delta_V_vs_GPT_original'] = row['V_mean'] - aggregate(originals, 'base')['V_mean']
            row['wins_in_accept_vs_GPT_original'] = [i for i,(a,b) in enumerate(zip(cur,originals)) if a['metrics']['g_done'] and not b['metrics']['g_done']]
            row['losses_in_accept_vs_GPT_original'] = [i for i,(a,b) in enumerate(zip(cur,originals)) if not a['metrics']['g_done'] and b['metrics']['g_done']]
    adapter.save_json(out / 'comparison.json', {'dataset_id': adapter.DEFAULT_DATASET,
                                                 'n_scenes_per_method': 10,
                                                 'complete_methods': list(included),
                                                 'results': summaries,
                                                 'notes': '10-scene exploratory development; not independent final testing; no statistical significance claim.'})
    cols = ['name','n','accepted','cooperative','accept_rate','coop_rate','V_mean','V_g_mean','cost_mean','latency_mean_s',
            'node_ratio_mean','link_ratio_mean','IR_violation_scenes','accepted_IR_violation_scenes',
            'estimated_api_cost_usd','input_tokens','output_tokens','cached_input_tokens','reasoning_tokens',
            'delta_cooperative_cases_vs_GPT_original','delta_V_vs_GPT_original']
    with (out / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in summaries:
            w.writerow(r)
    per_cols=['scene_id','method','accept','cooperative','V','V_g','cost','k1_ir_margin','k2_ir_margin','k3_ir_margin',
              'individual_rationality','node_ratio_mean','link_ratio_mean']
    with (out / 'paired_10scene_metrics.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=per_cols);w.writeheader()
        for name, recs in included.items():
            for i,r in enumerate(recs):
                if r['original_scene_id'] != i:
                    raise RuntimeError('Scene ID misalignment for ' + name)
                m=r['metrics']
                ir=ir_margins(r)
                w.writerow({'scene_id':i,'method':name,'accept':int(m['g_done']),
                            'cooperative':int(m['cooperative_valid']),'V':m['V'],
                            'V_g':m['V_g'],'cost':m['cost_sum'],
                            'k1_ir_margin':ir['k1'], 'k2_ir_margin':ir['k2'],
                            'k3_ir_margin':ir['k3'], 'individual_rationality':int(m['individual_rationality']),
                            'node_ratio_mean':node_ratio_mean(r), 'link_ratio_mean':link_ratio_mean(r)})
    print('COMPARE', json.dumps(summaries, ensure_ascii=False, indent=2))
    print('FILES', out / 'comparison.csv', out / 'paired_10scene_metrics.csv')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['plan','check','run','compare'],default='plan')
    p.add_argument('--project',default='.')
    p.add_argument('--variant',choices=list(VARIANT_FILES))
    p.add_argument('--report-dir',help='New comparison directory; refuses to overwrite an existing path')
    p.add_argument('--model',default='gpt-6.1-sol')
    p.add_argument('--api',choices=['chat','responses'],default='chat')
    p.add_argument('--effort',choices=['low','medium','high','xhigh','max'],default='medium')
    p.add_argument('--max-output-tokens',type=int,default=8192)
    p.add_argument('--timeout',type=int,default=360)
    p.add_argument('--budget-usd',type=float,default=2.0)
    p.add_argument('--input-usd-per-m',type=float,default=2.0)
    p.add_argument('--cached-usd-per-m',type=float,default=0.1)
    p.add_argument('--output-usd-per-m',type=float,default=10.0)
    p.add_argument('--allow-http-upstream',action='store_true')
    args=p.parse_args()
    root=Path(args.project).resolve()
    for v in VARIANT_FILES:
        text=read_prompt(v)
        print('PROMPT',v,'SHA256',hash_str(text),'chars',len(text))
    if args.mode=='plan':
        print('Model:',args.model,'effort:',args.effort,'API:',args.api)
        print('Fixed scenes seed7601 0..9; original baseline GPT 10 scenes reused (zero repeat API calls)')
        print('Three variants x 10 scenes x 4 roles = 120 new calls; approximate total $1.43 at observed $0.477/10 scenes')
        return
    if args.mode in ('check','run'):
        ds=adapter.verify_dataset(root,adapter.DEFAULT_DATASET,20)
        if args.mode=='check':
            print('DATASET VERIFIED',ds['dataset_id'],ds['sample_sha256'])
            for v in VARIANT_FILES:
                expected=data_lock(root,args,v,ds)
                print('LOCK PREVIEW',v,expected['variant_sha256'], 'project hashes checked')
            return
        run_variant(root,args)
    if args.mode=='compare':
        compare(root,args.report_dir)


if __name__=='__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        print('STOPPED:',str(exc),file=sys.stderr)
        sys.exit(2)
