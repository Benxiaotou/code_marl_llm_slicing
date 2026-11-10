# P2-E runtime fixes (2026-11-04)

The original V1 `semantic_balance` run is preserved in
`gpt61_runs/dev7601_prompt_ablation_v1/semantic_balance/`. Its original driver and
adapter are archived in `p2e_prompt/source_archive_v1/`; both archive hashes were
verified against the saved V1 protocol lock before running V2/V3.

Only P2-E runtime and reporting code changed. The prompts, `main.py`, existing
runner/policy code, datasets, experience libraries, MAPPO checkpoints and saved
V1/baseline results were not changed.

## Fixes

- Pass a placeholder token to the child process connecting to the localhost
  proxy. Do not replace the saved Qwen credentials.
- Flush and fsync request-attempt records before upstream I/O. Refuse another
  call if an attempt is unresolved or an error has unknown cost. Keep valid
  usage even if the response cannot be converted into a usable action.
- Count known error-response usage in estimated costs. No automatic retries.
- Export IR margins from the original `V_k - V_k_0` fields, original IR flags,
  and existing-slice node/link action-ratio means. These are reporting additions;
  the environment's evaluation definitions are unchanged.
- Write comparisons to a new directory and refuse to overwrite an existing
  report. Old V1 protocol locks are not rewritten to match the changed sources.

`usage.jsonl` records call outcomes; `usage_attempts.jsonl` records the durable
start/finish journal. A pending start after a crash blocks further calls until
the records and gateway bill are reconciled. The configured budget uses the
recorded token rates; gateway markups and unknown charges are not a guaranteed
billing cap.

## Configuration and authorized protocol

GPT uses User environment variables `GPT61_API_KEY` and `GPT61_BASE_URL` through
`run_configured.ps1`. Qwen uses `LLM_API_KEY`, `LLM_BASE_URL` and `P1A_MODEL`; the
user reports that Qwen is an internal deployment with unlimited quota, so its
incremental API cost is treated as zero. No Qwen calls are needed for this run.

V2/V3: `gpt-6.1-sol`, Chat Completions, `medium`, max output 8192 tokens,
timeout 360 seconds, seed7601 scenes 0..9, policy seed 1, IR-aware Standard-1000,
three own-role examples, two chunks of five scenes, at most 40 attempts per
variant and a $2 estimated budget per variant. Before execution the user
authorized these two runs, estimated at 80 calls and about $0.96 total. The rates
are assumptions inherited from the prior run: input $2, cached input $0.10,
output $10 per million tokens. Actual gateway rates remain unverified.

These ten scenes are exploratory development data, not an independent final
test. The P2-C final test data is not used.

## Commands (from the project root)

```powershell
conda run -n marl python -B -m unittest discover -s .\p2e_prompt\tests -p 'test_*.py' -v
.\p2e_prompt\run_configured.ps1 -Mode check
.\p2e_prompt\run_configured.ps1 -Mode run -Variant teacher_scale -BudgetUsd 2
.\p2e_prompt\run_configured.ps1 -Mode run -Variant joint_feasibility -BudgetUsd 2
.\p2e_prompt\run_configured.ps1 -Mode compare
```

Do not rerun the preserved V1 with the modified driver: its saved protocol lock
intentionally refers to the archived original implementation. Complete verified
chunks of V2/V3 can be skipped with identical settings; partial chunks are
preserved and block automatic reruns.
