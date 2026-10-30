"""Minimal intranet LLM connectivity probe (never prints the API key).

Run inside the original marl environment:
  python p0_api_probe.py
  python p0_api_probe.py --stream --timeout 180
  python p0_api_probe.py --stream --no-proxy --timeout 180
Only use on a network you trust: the documented internal endpoint uses plain HTTP.
"""
import argparse
import os
import sys
import time

import requests


def run_probe(url, key, model, timeout=90.0, stream=False, no_proxy=False, out=print):
    if not key or key.strip().lower() in ('sk-xxx', 'your-key', 'your_token', 'xxx'):
        raise ValueError('LLM_API_KEY is missing or still the sample placeholder')
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    url = url.rstrip('/')
    if not url.endswith('/v1'):
        raise ValueError('API URL must end with /v1')
    sess = requests.Session()
    if no_proxy:
        sess.trust_env = False
    body = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Reply with just OK.'}],
        'temperature': 0,
        'max_tokens': 512,
        'stream': bool(stream),
    }
    start = time.monotonic()
    out('Probe mode={}, model={}, proxy_env={}, read_timeout={}s'.format(
        'SSE' if stream else 'nonstream', model, 'ignored' if no_proxy else 'enabled', timeout))
    try:
        with sess.post(url + '/chat/completions',
                       headers={'Authorization': 'Bearer ' + key},
                       json=body, timeout=(10.0, timeout), stream=stream) as resp:
            t_headers = time.monotonic() - start
            out('HTTP status={}, response_headers_after={:.2f}s'.format(resp.status_code, t_headers))
            if resp.status_code >= 400:
                # The response may explain authentication, model or gateway issues.
                out('HTTP error excerpt: ' + resp.text[:250].replace('\n', ' '))
                return False
            if not stream:
                payload = resp.json()
                choices = payload.get('choices') or []
                choice = choices[0] if choices else {}
                message = choice.get('message') or {}
                content = message.get('content') or ''
                out('finish_reason={}, content_chars={}, usage={}'.format(
                    choice.get('finish_reason'), len(content), payload.get('usage')))
                out('Total time: {:.2f}s'.format(time.monotonic() - start))
                return bool(content.strip())
            first_event = None
            count = 0
            content_chars = 0
            reasoning_chars = 0
            finish_reason = None
            saw_done = False
            for line in resp.iter_lines(chunk_size=64, decode_unicode=False):
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode('utf-8', errors='replace')
                line = line.strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    saw_done = True
                    break
                import json
                chunk = json.loads(data)
                if 'error' in chunk:
                    out('SSE error: ' + str(chunk['error'])[:200])
                    return False
                if first_event is None:
                    first_event = time.monotonic() - start
                    out('First SSE event after {:.2f}s'.format(first_event))
                count += 1
                for choice in chunk.get('choices') or []:
                    delta = choice.get('delta') or {}
                    content_chars += len(delta.get('content') or '')
                    reasoning_chars += len(delta.get('reasoning_content') or '')
                    if choice.get('finish_reason') is not None:
                        finish_reason = choice['finish_reason']
            out('SSE chunks={}, final_content_chars={}, reasoning_chars={}, finish_reason={}, done_marker={}'.format(
                count, content_chars, reasoning_chars, finish_reason, saw_done))
            out('Total time: {:.2f}s'.format(time.monotonic() - start))
            return count > 0 and content_chars > 0 and (saw_done or finish_reason == 'stop')
    except requests.exceptions.ConnectTimeout as exc:
        out('CONNECT TIMEOUT: ' + str(exc))
        return False
    except requests.exceptions.ReadTimeout as exc:
        out('READ TIMEOUT: ' + str(exc))
        return False
    except requests.exceptions.RequestException as exc:
        out('NETWORK ERROR: ' + str(exc))
        return False
    except (ValueError, KeyError, IndexError) as exc:
        out('RESPONSE PARSE ERROR: ' + str(exc))
        return False
    finally:
        sess.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default=os.getenv('LLM_BASE_URL', 'http://192.168.131.35:30001/v1'))
    parser.add_argument('--model', default=os.getenv('LLM_MODEL', 'deepseek-v4-flash-0731'))
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--no-proxy', action='store_true')
    args = parser.parse_args(argv)
    try:
        ok = run_probe(args.url, os.getenv('LLM_API_KEY'), args.model, args.timeout,
                       args.stream, args.no_proxy)
    except (ValueError, TypeError) as exc:
        print('CONFIG ERROR: ' + str(exc))
        return 2
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
