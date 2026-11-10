#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT-6.1 Sol compatibility adapter for the existing P1-C role-local GEANT runner.

No original project code, MAPPO weights, experience bank, P2-C data or API key is edited.
Upstream model is selected via environment/CLI, and requests are forwarded from a
127.0.0.1-only OpenAI-Chat-Completions shim to Chat Completions or Responses API.

Modes: plan, prepare, probe, run, costs.
Python standard library only. Use the project's `marl` Python environment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

VERSION = 'gpt61_sol_dev20_adapter_v1'
DEFAULT_DATASET = 'gpt61_dev_seed7601'
DEFAULT_SEED = 7601
DEFAULT_MAX_SCENES = 20
DEFAULT_MODEL = 'gpt-6.1-sol'
DEFAULT_LIBRARY = 'p2b_libraries/p2b_standard_1000_v2'
DEFAULT_IR = 'p2b_ir_libraries/p2b_standard_1000_ir_v2'
DEFAULT_RUN_FOLDER = 'gpt61_runs/dev7601_iraware_std1000_v1'
PRICES = {'input': 2.0, 'cached_input': 0.1, 'output': 10.0}  # USD/1M, official standard, unverified for third-party gateway


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def append_jsonl(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fp:
        fp.write(json.dumps(obj, ensure_ascii=False) + '\n')
        fp.flush()
        os.fsync(fp.fileno())


def read_events(path: Path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as fp:
        return [json.loads(line) for line in fp if line.strip()]


def totals(events: list[dict]) -> dict:
    ok = [e for e in events if e.get('status') == 'ok']
    billed = [e for e in events if 'cost_usd_estimate' in e]
    return {
        'requests_ok': len(ok),
        'requests_logged': len(events),
        'cost_unknown_requests': sum(bool(e.get('cost_unknown')) for e in events),
        'input_tokens': sum(e.get('input_tokens', 0) for e in billed),
        'cached_input_tokens': sum(e.get('cached_input_tokens', 0) for e in billed),
        'output_tokens': sum(e.get('output_tokens', 0) for e in billed),
        'reasoning_tokens': sum(e.get('reasoning_tokens', 0) for e in billed),
        'cost_usd_estimate': round(sum(e['cost_usd_estimate'] for e in billed), 8),
    }


def parse_usage(response: dict, api: str) -> dict:
    usage = response.get('usage')
    if not isinstance(usage, dict):
        raise ValueError('Upstream response has no usage object; cannot estimate bill reliably')
    if api == 'chat':
        inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0) or 0
    else:
        inp, out = usage.get('input_tokens'), usage.get('output_tokens')
        cached = (usage.get('input_tokens_details') or {}).get('cached_tokens', 0) or 0
        reasoning = (usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0) or 0
    if not isinstance(inp, int) or not isinstance(out, int) or inp < 0 or out < 0:
        raise ValueError('Usage missing valid input/output token counts')
    if not isinstance(cached, int) or cached < 0 or cached > inp:
        raise ValueError('Cached input tokens invalid')
    if not isinstance(reasoning, int) or reasoning < 0:
        reasoning = 0
    return {'input_tokens': inp, 'output_tokens': out, 'cached_input_tokens': cached,
            'reasoning_tokens': reasoning}


def price(usage: dict, prices: dict) -> float:
    a, b, c = usage['input_tokens'], usage['cached_input_tokens'], usage['output_tokens']
    return round(((a - b) * prices['input'] + b * prices['cached_input'] +
                  c * prices['output']) / 1_000_000, 8)


def response_text(response: dict) -> str:
    if isinstance(response.get('output_text'), str):
        return response['output_text']
    texts = []
    for item in response.get('output', []):
        if item.get('type') != 'message':
            continue
        for part in item.get('content', []):
            if part.get('type') in ('output_text', 'text') and isinstance(part.get('text'), str):
                texts.append(part['text'])
    return ''.join(texts)


def normalise_response(response: dict, api: str, usage: dict) -> dict:
    if api == 'chat':
        if not response.get('choices') or not isinstance(response['choices'][0].get('message'), dict):
            raise ValueError('Chat response missing choices[0].message')
        return response
    if response.get('status') == 'incomplete':
        raise ValueError('Responses incomplete (often max_output_tokens); no usable action returned')
    txt = response_text(response)
    if not txt.strip():
        raise ValueError('Responses returned no visible action text')
    return {
        'id': response.get('id', 'gpt61-proxy'), 'object': 'chat.completion',
        'model': response.get('model', DEFAULT_MODEL),
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': txt},
                     'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': usage['input_tokens'],
                  'completion_tokens': usage['output_tokens'],
                  'total_tokens': usage['input_tokens'] + usage['output_tokens'],
                  'prompt_tokens_details': {'cached_tokens': usage['cached_input_tokens']},
                  'completion_tokens_details': {'reasoning_tokens': usage['reasoning_tokens']}},
    }


def transform_request(inbound: dict, *, api: str, model: str,
                      effort: str, max_output_tokens: int) -> dict:
    messages = inbound.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('Incoming request must contain non-empty messages[]')
    if inbound.get('stream') is True:
        raise ValueError('Streaming unsupported by this billing-safe adapter; use --no-stream')
    if any('tool_calls' in m or m.get('role') == 'tool' for m in messages):
        raise ValueError('Tools unsupported; role-local single-shot decisions only')
    safe_messages = []
    for m in messages:
        role = m.get('role')
        if role not in ('system', 'developer', 'user', 'assistant') or not isinstance(m.get('content'), str):
            raise ValueError('Unsupported message role/content; expected plain text')
        safe_messages.append({'role': 'developer' if role == 'system' else role,
                              'content': m['content']})
    if api == 'chat':
        # Explicitly do NOT forward Qwen temperature, max_tokens, top_p, or unsupported fields.
        return {'model': model, 'messages': safe_messages, 'stream': False,
                'reasoning_effort': effort, 'max_completion_tokens': max_output_tokens}
    if api == 'responses':
        return {'model': model, 'input': safe_messages,
                'reasoning': {'effort': effort},
                'max_output_tokens': max_output_tokens,
                'store': False}
    raise ValueError('api must be chat or responses')


class UpstreamError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f'Upstream HTTP {status}: {body[:700]}')
        self.status = status


def invoke_upstream(body: dict, *, url: str, api_key: str, timeout: int = 360) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode('utf-8')
    req = Request(url, data=data, method='POST',
                  headers={'Authorization': 'Bearer ' + api_key,
                           'Content-Type': 'application/json', 'Accept': 'application/json'})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read(10_000_001)
    except HTTPError as exc:
        raw = exc.read(1500).decode('utf-8', errors='replace')
        raise UpstreamError(exc.code, raw) from None
    except URLError as exc:
        raise RuntimeError('Upstream connection failed: ' + str(exc.reason)) from None
    if len(raw) > 10_000_000:
        raise ValueError('Upstream response too large')
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError('Upstream response was not a JSON object')
    if parsed.get('error'):
        raise RuntimeError('Upstream reported error: ' + str(parsed['error'])[:700])
    return parsed


def upstream_url(base_url: str, api: str, *, allow_http=False) -> str:
    p = urlparse(base_url.rstrip('/'))
    if p.scheme not in ('https', 'http') or not p.netloc or p.username or p.password or p.query or p.fragment:
        raise ValueError('GPT61_BASE_URL must be a normal http(s) base URL, without embedded credentials')
    if p.scheme == 'http' and not allow_http:
        raise ValueError('HTTP upstream sends your API key without TLS. Pass --allow-http-upstream only for a trusted internal network')
    path = (p.path or '').rstrip('/')
    if path.endswith('/chat/completions') or path.endswith('/responses'):
        path = path.rsplit('/', 2)[0] if path.endswith('/chat/completions') else path.rsplit('/', 1)[0]
    if not path.endswith('/v1'):
        path += '/v1'
    target = 'chat/completions' if api == 'chat' else 'responses'
    return f'{p.scheme}://{p.netloc}{path}/{target}'


class ProxyState:
    def __init__(self, *, upstream: str, key: str, api: str, model: str, effort: str,
                 max_output_tokens: int, ledger: Path, prices: dict, budget: float,
                 max_calls: int, timeout: int):
        self.url = upstream
        self.key = key
        self.api = api
        self.model = model
        self.effort = effort
        self.max_output_tokens = max_output_tokens
        self.ledger = ledger
        self.attempt_ledger = ledger.with_name(ledger.stem + '_attempts.jsonl')
        self.prices = prices
        self.budget = budget
        self.max_calls = max_calls
        self.timeout = timeout
        self.lock = threading.Lock()
        self.inflight = False

    def handle(self, request: dict) -> tuple[dict, dict]:
        with self.lock:
            if self.inflight:
                raise RuntimeError('Concurrent model calls disabled for repeatable billing')
            before = totals(read_events(self.ledger))
            attempts = read_events(self.attempt_ledger)
            started_ids = {e['request_id'] for e in attempts if e['status'] == 'started'}
            finished_ids = {e['request_id'] for e in attempts if e['status'] == 'finished'}
            if started_ids - finished_ids or before['cost_unknown_requests']:
                raise RuntimeError('Unresolved API attempt or unknown cost; inspect preserved ledgers before continuing')
            if max(before['requests_logged'], len(started_ids)) >= self.max_calls:
                raise RuntimeError(f'Maximum {self.max_calls} API requests reached; stop')
            if before['cost_usd_estimate'] >= self.budget:
                raise RuntimeError(f'Budget estimate ${self.budget:.2f} reached; stop')
            # Conservative preflight reserve: at most one token per UTF-8 byte of input,
            # plus the maximum output-token allocation. Cache discounts are ignored.
            # This guards the configured token-rate budget, not third-party markups.
            input_bytes = sum(len(str(message.get('content', '')).encode('utf-8'))
                              for message in request.get('messages', []) if isinstance(message, dict))
            maximum_one_request = ((input_bytes + 1000) * self.prices['input'] +
                                   self.max_output_tokens * self.prices['output']) / 1_000_000
            if before['cost_usd_estimate'] + maximum_one_request > self.budget:
                raise RuntimeError(f'Budget safety reserve would exceed ${self.budget:.2f}; stop before upstream call')
            self.inflight = True
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        usage = None
        try:
            outgoing = transform_request(request, api=self.api, model=self.model,
                                         effort=self.effort,
                                         max_output_tokens=self.max_output_tokens)
            # Persist before network I/O; a crash leaves a blocking unresolved attempt.
            append_jsonl(self.attempt_ledger, {'time_utc': utcnow(), 'status': 'started',
                                              'request_id': request_id,
                                              'maximum_cost_reserve_usd': maximum_one_request})
            reply = invoke_upstream(outgoing, url=self.url, api_key=self.key, timeout=self.timeout)
            usage = parse_usage(reply, self.api)
            norm = normalise_response(reply, self.api, usage)
            event = {'time_utc': utcnow(), 'status': 'ok', 'model': self.model,
                     'request_id': request_id, 'response_model': reply.get('model'),
                     'api': self.api, 'effort': self.effort,
                     'request_seconds': round(time.monotonic() - started, 3),
                     **usage, 'cost_usd_estimate': price(usage, self.prices)}
            append_jsonl(self.ledger, event)
            append_jsonl(self.attempt_ledger, {'time_utc': utcnow(), 'status': 'finished',
                                              'request_id': request_id, 'outcome': 'ok'})
            return norm, event
        except Exception as exc:
            safe_error = str(exc).replace(self.key, '[REDACTED_KEY]')[:600]
            event = {'time_utc': utcnow(), 'status': 'error', 'request_id': request_id,
                                       'model': self.model, 'api': self.api,
                                       'request_seconds': round(time.monotonic() - started, 3),
                                       'error': safe_error, 'cost_unknown': usage is None}
            if usage is not None:
                event.update(usage)
                event['cost_usd_estimate'] = price(usage, self.prices)
            append_jsonl(self.ledger, event)
            append_jsonl(self.attempt_ledger, {'time_utc': utcnow(), 'status': 'finished',
                                              'request_id': request_id, 'outcome': 'error'})
            raise RuntimeError(safe_error) from None
        finally:
            with self.lock:
                self.inflight = False


class LocalProxyHandler(BaseHTTPRequestHandler):
    state: ProxyState
    server_version = 'GPT61LocalShim/1.0'

    def log_message(self, fmt, *args):
        # No request bodies or bearer tokens in logs.
        pass

    def send(self, status: int, obj: dict):
        raw = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path.rstrip('/') != '/v1/chat/completions':
            self.send(404, {'error': {'message': 'Only /v1/chat/completions accepted'}})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 2_000_000:
                raise ValueError('Invalid or excessively large JSON request')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Request JSON must be an object')
            response, event = self.state.handle(payload)
            self.send(200, response)
            print('[API] input={input_tokens} cached={cached_input_tokens} '
                  'output={output_tokens} reasoning={reasoning_tokens} '
                  'estimated=${cost_usd_estimate:.4f} {request_seconds}s'.format(**event),
                  flush=True)
        except Exception as exc:
            msg = str(exc).replace(self.state.key, '[REDACTED_KEY]')[:400]
            print('[API STOP] ' + msg, flush=True)
            self.send(502, {'error': {'message': msg, 'type': 'gpt61_adapter_error'}})


def source_sample(project: Path, manifest: dict) -> Path:
    path = Path(manifest['sample_file'])
    return path if path.is_absolute() else project / path


def verify_dataset(project: Path, dataset_name: str, count: int, split='validation') -> dict:
    manifest_path = project / 'p1b_data' / f'{dataset_name}.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'Missing {manifest_path}. Run --mode prepare first')
    m = load_json(manifest_path)
    for k, expected in [('dataset_id', dataset_name), ('split', split), ('seed', DEFAULT_SEED), ('n_scenes', count)]:
        if m.get(k) != expected:
            raise RuntimeError(f'Dataset {k} mismatch: expected {expected!r}, got {m.get(k)!r}')
    sample = source_sample(project, m)
    if not sample.is_file() or digest(sample) != m.get('sample_sha256'):
        raise RuntimeError('Dataset sample missing or SHA256 mismatch; STOP')
    if m.get('config_sha256') and digest(project / 'configs_main.yaml') != m['config_sha256']:
        raise RuntimeError('Dataset config SHA256 does not match current environment')
    if m.get('main_py_sha256') and digest(project / 'main.py') != m['main_py_sha256']:
        raise RuntimeError('Dataset main.py SHA256 does not match current environment')
    return {'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path),
            'sample': str(sample), 'sample_sha256': digest(sample), 'count': count,
            'dataset_id': dataset_name}


def prepare_dataset(project: Path, dataset_name: str, count: int, seed: int):
    manifest_path = project / 'p1b_data' / f'{dataset_name}.json'
    if manifest_path.is_file():
        info = verify_dataset(project, dataset_name, count)
        print('REUSE VERIFIED:', dataset_name, info['sample_sha256'])
        return
    orphans = list((project / 'data' / 'GEANT').glob(dataset_name + '.pkl*'))
    if orphans:
        raise RuntimeError('Partial dataset exists without manifest; preserve and inspect: ' + repr(orphans))
    cmd = [sys.executable, '-u', 'p1b_generate.py', '--name', dataset_name,
           '--split', 'validation', '--seed', str(seed), '--count', str(count)]
    print('GENERATION (one time):', ' '.join(cmd), flush=True)
    exit_code = subprocess.call(cmd, cwd=project)
    if exit_code:
        raise RuntimeError(f'p1b_generate.py failed with exit code {exit_code}; inspect partial output')
    info = verify_dataset(project, dataset_name, count)
    print('GENERATED VERIFIED:', info['dataset_id'], info['sample_sha256'])


def verify_results(chunk: Path, start: int, count: int, dataset_name: str) -> bool:
    needed = [chunk / x for x in ('manifest.json', 'summary.json', 'metrics.csv', 'records.jsonl')]
    if not all(x.exists() for x in needed):
        return False
    manifest = load_json(chunk / 'manifest.json')
    summ = load_json(chunk / 'summary.json')
    if (manifest.get('eval_dataset_id') != dataset_name or manifest.get('method') != 'ir_icl'
            or manifest.get('scene_ids') != list(range(start, start + count))
            or summ.get('n_attempted') != count or summ.get('n_scored') != count
            or summ.get('fallback_rate', 1) != 0):
        return False
    with (chunk / 'records.jsonl').open(encoding='utf-8') as fp:
        records = [json.loads(line) for line in fp if line.strip()]
    if len(records) != count or [r.get('original_scene_id') for r in records] != list(range(start, start + count)):
        return False
    if any(r.get('status') != 'ok' or r.get('fallback_used') is True for r in records):
        return False
    with (chunk / 'metrics.csv').open(encoding='utf-8-sig', newline='') as fp:
        metrics = list(csv.DictReader(fp))
    return len(metrics) == count and all(r['status'] == 'ok' for r in metrics)


def summarize_results(out: Path, count: int, chunk_size: int, ledger: Path) -> dict:
    all_records = []
    for start in range(0, count, chunk_size):
        length = min(chunk_size, count - start)
        chunk = out / 'chunks' / f'{start:04d}_{length:03d}'
        if not verify_results(chunk, start, length, DEFAULT_DATASET):
            return {'complete': False, 'scenes_completed': len(all_records),
                    'api_usage': totals(read_events(ledger))}
        all_records += [json.loads(line) for line in (chunk / 'records.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    import statistics
    metrics = [row['metrics'] for row in all_records]
    stats = {'complete': True, 'scenes_completed': len(all_records),
             'acceptance_rate': statistics.mean(float(m['g_done']) for m in metrics),
             'cooperative_valid_rate': statistics.mean(float(m['cooperative_valid']) for m in metrics),
             'V_mean': statistics.mean(float(m['V']) for m in metrics),
             'V_g_mean': statistics.mean(float(m['V_g']) for m in metrics),
             'cost_sum_mean': statistics.mean(float(m['cost_sum']) for m in metrics),
             'policy_latency_s_mean': statistics.mean(float(r['policy_latency_s']) for r in all_records),
             'api_usage': totals(read_events(ledger))}
    save_json(out / 'dev20_summary.json', stats)
    return stats


def configs(args):
    key = os.environ.get('GPT61_API_KEY', '').strip()
    base = os.environ.get('GPT61_BASE_URL', '').strip()
    if not key or not base:
        raise RuntimeError('Set GPT61_API_KEY and GPT61_BASE_URL in the current PowerShell session (do NOT upload your key)')
    url = upstream_url(base, args.upstream_api, allow_http=args.allow_http_upstream)
    return key, url


def make_state(args, ledger: Path) -> ProxyState:
    key, url = configs(args)
    prices = {'input': args.input_usd_per_m, 'cached_input': args.cached_input_usd_per_m,
              'output': args.output_usd_per_m}
    return ProxyState(upstream=url, key=key, api=args.upstream_api,
                      model=args.upstream_model, effort=args.effort,
                      max_output_tokens=args.max_output_tokens,
                      ledger=ledger, prices=prices, budget=args.budget_usd,
                      max_calls=args.max_calls, timeout=args.timeout)


def run_probe(args, project: Path):
    out = project / 'gpt61_runs' / 'connection_probe'
    ledger = out / 'usage.jsonl'
    state = make_state(args, ledger)
    request = {'messages': [
        {'role': 'system', 'content': 'Act as agent_k1. Output only JSON: {"ratios":[0.2,0.4,0.6,0.8]}. No prose.'},
        {'role': 'user', 'content': 'Return exactly four finite control ratios between 0 and 1 as a JSON object.'}],
        'temperature': 0.1, 'max_tokens': args.max_output_tokens, 'stream': False}
    result, event = state.handle(request)
    txt = result['choices'][0]['message']['content']
    try:
        obj = json.loads(txt.strip())
        vals = obj['ratios']
        valid = isinstance(vals, list) and len(vals) == 4 and all(type(x) in (int, float) and 0 <= x <= 1 for x in vals)
    except (ValueError, TypeError, KeyError):
        valid = False
    print('PROBE RESULT:', 'VALID_JSON_ACTION' if valid else 'INVALID_JSON_ACTION')
    print('MODEL USAGE:', {k: event[k] for k in ('input_tokens', 'output_tokens', 'reasoning_tokens', 'cost_usd_estimate')})
    if not valid:
        print('Action preview (truncated):', txt[:150])
        raise RuntimeError('API returned an invalid action; cannot start model evaluation')
    print('API PROBE PASSED. No GEANT experiment was run.')


def lock_expected(args, project: Path, ds: dict):
    lib = project / DEFAULT_LIBRARY / 'library_manifest.json'
    ir = project / DEFAULT_IR
    ir_candidates = [ir / 'ir_manifest.json', ir / 'manifest.json']
    found = [p for p in ir_candidates if p.is_file()]
    if not lib.is_file() or len(found) != 1:
        raise RuntimeError('Standard-1000 library or IR overlay missing')
    for p in ('p1c_runner.py', 'main.py', 'configs_main.yaml'):
        if not (project / p).is_file():
            raise RuntimeError('Missing original project file: ' + p)
    return {'version': VERSION, 'dataset': ds, 'method': 'ir_icl',
            'library': DEFAULT_LIBRARY, 'library_manifest_sha256': digest(lib),
            'ir': DEFAULT_IR, 'ir_manifest_sha256': digest(found[0]),
            'runner_sha256': digest(project / 'p1c_runner.py'),
            'environment_sha256': digest(project / 'main.py'),
            'config_sha256': digest(project / 'configs_main.yaml'),
            'model': args.upstream_model, 'api': args.upstream_api,
            'effort': args.effort, 'max_output_tokens': args.max_output_tokens,
            'scenes_capacity': DEFAULT_MAX_SCENES, 'chunk_size': args.chunk_size,
            'seed_for_role_policy': 1, 'examples': 3,
            'temperature_sent_to_upstream': None,
            'original_runner_temperature_metadata': 0.1,
            'max_calls': args.max_calls,
            'unit_rates_usd_per_million_tokens': {
                'input': args.input_usd_per_m, 'cached_input': args.cached_input_usd_per_m,
                'output': args.output_usd_per_m}}


def run_evaluation(args, project: Path):
    if args.scenes < 1 or args.scenes > DEFAULT_MAX_SCENES:
        raise RuntimeError('This package is deliberately limited to at most 20 development scenes')
    if args.chunk_size < 1 or args.chunk_size > args.scenes:
        raise RuntimeError('chunk-size must be between 1 and scenes')
    ds = verify_dataset(project, DEFAULT_DATASET, DEFAULT_MAX_SCENES)
    out = project / DEFAULT_RUN_FOLDER
    lock = out / 'protocol_lock.json'
    intended = lock_expected(args, project, ds)
    if lock.exists() and load_json(lock) != intended:
        raise RuntimeError('Existing protocol lock mismatch. Keep old results; do NOT overwrite or bypass')
    if not lock.exists():
        out.mkdir(parents=True, exist_ok=True)
        save_json(lock, intended)
        print('PROTOCOL LOCKED:', lock)
    ledger = out / 'usage.jsonl'
    state = make_state(args, ledger)
    class Handler(LocalProxyHandler):
        pass
    Handler.state = state
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name='gpt61-local-proxy', daemon=True)
    thread.start()
    local_base = f'http://127.0.0.1:{port}/v1'
    print('LOCAL PROXY READY:', local_base, '(binds localhost only)')
    print(f'MAX SPEND ESTIMATE ${args.budget_usd:.2f}; MAX API ATTEMPTS {args.max_calls}; model={args.upstream_model}')
    try:
        for start in range(0, args.scenes, args.chunk_size):
            length = min(args.chunk_size, args.scenes-start)
            chunk = out / 'chunks' / f'{start:04d}_{length:03d}'
            if chunk.exists() and any(chunk.iterdir()):
                if verify_results(chunk, start, length, DEFAULT_DATASET):
                    print(f'SKIP VERIFIED scene {start}-{start+length-1}')
                    continue
                raise RuntimeError(f'Partial or invalid chunk {chunk}; preserve it and inspect; do not overwrite')
            cmd = [sys.executable, '-u', 'p1c_runner.py',
                   '--library', DEFAULT_LIBRARY,
                   '--ir-history', DEFAULT_IR,
                   '--dataset-manifest', 'p1b_data/' + DEFAULT_DATASET + '.json',
                   '--method', 'ir_icl', '--examples', '3',
                   '--start', str(start), '--count', str(length), '--seed', '1',
                   '--model', args.upstream_model, '--base-url', local_base,
                   '--temperature', '0.1', '--max-tokens', str(args.max_output_tokens),
                   '--api-timeout', str(args.timeout), '--no-stream',
                   '--on-error', 'abort', '--output-dir', str(chunk)]
            print(f'RUN scenes {start}-{start+length-1}, output {chunk}')
            code = subprocess.call(cmd, cwd=project)
            if code or not verify_results(chunk, start, length, DEFAULT_DATASET):
                raise RuntimeError(f'Runner failed or incomplete in scenes {start}-{start+length-1}; exit={code}; preserved {chunk}')
            print(f'PASSED scenes {start}-{start+length-1}; running cost approx ${totals(read_events(ledger))["cost_usd_estimate"]:.4f}')
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        result = summarize_results(out, args.scenes, args.chunk_size, ledger)
        print('USAGE / SUMMARY:', json.dumps(result, ensure_ascii=False, indent=2))
    print('ALL DEVELOPMENT SCENES VERIFIED. This is NOT an independent final test set.')


def print_costs(args, project):
    ledger = project / DEFAULT_RUN_FOLDER / 'usage.jsonl'
    usage = totals(read_events(ledger))
    print('Recorded usage:', json.dumps(usage, ensure_ascii=False, indent=2))
    if usage['requests_ok']:
        scenes = usage['requests_ok'] / 4
        usd = usage['cost_usd_estimate'] / scenes
        print(f'ESTIMATED ${usd:.5f}/scene at recorded usage (four calls/scene assumed).')
        for n in (20, 200, 1536):
            print(f'   {n} scenes at observed usage: ~${usd*n:.2f} USD')
        print('These are projections, NOT provider invoices or performance predictions.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['plan', 'prepare', 'probe', 'run', 'costs'], default='plan')
    p.add_argument('--project', default='.', help='Original project root (default current directory)')
    p.add_argument('--scenes', type=int, default=20, help='Actual run scenes (1..20); generator always makes 20')
    p.add_argument('--chunk-size', type=int, default=5)
    p.add_argument('--upstream-api', choices=['chat', 'responses'], default='chat')
    p.add_argument('--upstream-model', default=DEFAULT_MODEL)
    p.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max'], default='medium')
    p.add_argument('--max-output-tokens', type=int, default=8192)
    p.add_argument('--budget-usd', type=float, default=8.0)
    p.add_argument('--max-calls', type=int, default=85, help='Includes API probe only if same ledger; 20 scenes use 80 calls')
    p.add_argument('--timeout', type=int, default=360)
    p.add_argument('--input-usd-per-m', type=float, default=PRICES['input'])
    p.add_argument('--cached-input-usd-per-m', type=float, default=PRICES['cached_input'])
    p.add_argument('--output-usd-per-m', type=float, default=PRICES['output'])
    p.add_argument('--allow-http-upstream', action='store_true', help='Only for trusted internal networks; sends key via HTTP')
    return p


def main():
    args = parser().parse_args()
    root = Path(args.project).resolve()
    if not root.is_dir():
        raise RuntimeError(f'Project directory not found: {root}')
    if any(x < 0 for x in (args.input_usd_per_m, args.cached_input_usd_per_m, args.output_usd_per_m)):
        raise RuntimeError('Prices must be nonnegative')
    if args.budget_usd <= 0 or args.max_calls <= 0 or args.max_output_tokens < 64 or args.timeout < 20:
        raise RuntimeError('Invalid cost ceiling, request ceiling, output ceiling, or timeout')
    if args.mode == 'plan':
        print('MODEL', args.upstream_model, '| upstream API', args.upstream_api, '| reasoning effort', args.effort)
        print('PROJECT', root)
        print('ISOLATED DEV DATASET', DEFAULT_DATASET, '| 20 new validation scenes')
        print('EXPERIMENT IR-aware + Standard-1000 (no P2-C final test reuse)')
        print('PRICE assumption USD / 1M tokens', args.input_usd_per_m, '/', args.cached_input_usd_per_m, '/', args.output_usd_per_m)
        print('Commands: --mode probe -> --mode prepare -> --mode run -> --mode costs')
        print('The adapter will not modify P2-C results, Qwen code, checkpoint or experience bank.')
    elif args.mode == 'prepare':
        prepare_dataset(root, DEFAULT_DATASET, DEFAULT_MAX_SCENES, DEFAULT_SEED)
    elif args.mode == 'probe':
        run_probe(args, root)
    elif args.mode == 'run':
        run_evaluation(args, root)
    elif args.mode == 'costs':
        print_costs(args, root)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('INTERRUPTED. Existing results and usage ledger are preserved.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print('STOPPED:', str(exc), file=sys.stderr)
        sys.exit(2)
