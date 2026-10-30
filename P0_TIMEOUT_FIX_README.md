# P0 DeepSeek ReadTimeout: diagnostic & optional SSE patch

Current state: 9 original offline tests passed; GEANT mini mock + frozen MAPPO ran. Real DeepSeek first decision failed after 90s with `requests.exceptions.ReadTimeout` in *nonstreaming* `/v1/chat/completions`. This cannot by itself identify whether it is slow reasoning, server queuing, gateway buffering, or local proxy behaviour.

## Files

- `p0_api_probe.py`: tiny API-only diagnostic, without loading GEANT and without printing the token.
- `p0_policy.py`: original controller plus OPTIONAL `stream=True` SSE support and strict streamed JSON assembly. No changes to prompt/environment/reward/action contract.
- `p0_runner.py`: adds `--api-stream` flag and records mode in manifest.
- `p0_stream_selftest.py`: standalone offline tests for stream assembly and failure paths.

Do not copy the patched `p0_policy.py` / `p0_runner.py` until you have done the short diagnostic. If copying, **back up your current two files first**. Do not replace `main.py`, model checkpoints, GEANT data, or modify the existing P0 scene results.

## 1. Probe quick API response (first priority)

In project root, Conda `marl`, use an actual credential, not the literal string `sk-XXX`.

```powershell
conda activate marl
$env:LLM_API_KEY="<your actual token>"
$env:LLM_BASE_URL="http://192.168.131.35:30001/v1"
$env:LLM_MODEL="deepseek-v4-flash-0731"
python p0_api_probe.py --timeout 90
python p0_api_probe.py --stream --timeout 180
```

Interpretation:
- HTTP 401/403: token/permissions.
- HTTP 404: wrong URL or route.
- HTTP 429/503: quota/concurrency/back-end service availability.
- Tiny prompt nonstream also ReadTimeout: service/queue/network/gateway problem **not caused by the large P0 prompt**.
- Tiny prompt succeeds but full P0 times out: likely prompt size / generation latency / token budget; compare streaming behaviour.
- SSE yields events early: switch P0 to `--api-stream` for evaluation.
- SSE also waits for headers or first event until timeout: increasing read timeout alone may not solve the underlying problem. Check server logs and gateway buffering.

If Python is abnormal but Cherry Studio works with the same model: compare the **exact API route, key, machine network and proxy setting**. Optional `--no-proxy` tests whether your shell proxy environment is the difference; use only when accessing a trusted internal network.

## 2. Install optional streaming changes

Copy the patched files alongside `main.py`, first backing up your existing ones:

```powershell
Copy-Item .\p0_policy.py .\p0_policy.py.before-stream-backup
Copy-Item .\p0_runner.py .\p0_runner.py.before-stream-backup
Copy-Item .\patch\p0_policy.py .\p0_policy.py -Force
Copy-Item .\patch\p0_runner.py .\p0_runner.py -Force
python p0_selftest.py
python p0_stream_selftest.py
python p0_runner.py --dataset mini --policy llm --count 1 --api-stream --api-timeout 180 --max-tokens 4096 --fail-mode abort
```

`--api-stream` is optional; without it, existing behaviour is unchanged. SSE support consumes chunks and **only** combines `delta.content`, never `delta.reasoning_content`. It requires a completed assistant JSON answer and rejects missing, malformed, out-of-range, or truncated decisions. The API may not actually stream early tokens: a server-side queue or a gateway that buffers responses can still cause ReadTimeout.

If streaming fails but a tiny nonstream probe succeeds, use the older nonstream request with `--api-timeout 300` rather than blindly increasing output tokens. If the API shows `finish_reason=length`, discuss raising `max_tokens` and reducing thinking/Prompt size rather than treating a partial JSON as a valid action.

Do not enable `--fail-mode rule` when establishing pure LLM performance; fallback results are not independent LLM decisions. Keep `--fail-mode abort` for first P0 test.

**Important:** the current internal base URL uses `http://` and sends the token over an unencrypted connection. Use it only within your organization's trusted network according to lab guidance. Do not share real tokens or raw private network-state transcripts.
