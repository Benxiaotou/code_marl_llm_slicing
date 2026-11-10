# val5601 100-scene paired experiment

The user authorized MAPPO, GPT V1 and GPT V3 on the same 100 predeclared scenes
from the existing 200-scene `p2b_val_seed5601` validation dataset, followed by
comparison. Scene IDs are 0..99, fixed before any new inference. No dataset
generation, training, P2-C final-test access, or Qwen inference is required.

`val5601_eval.py` verifies the source sample hash, config hash, archived source,
and the exact previously audited NumPy path-equality compatibility fix. It
never rewrites the old manifest or the environment. MAPPO's actor-construction
environment reads the same existing 200-scene file, avoiding the unrelated
600-scene default. Checkpoint loading, test-mode actions and seed convention
come from the existing frozen teacher implementation.

`val5601_100.py` reuses the frozen P2-E prompt injection and billing adapter.
V1/V3 use unchanged prompt text, g's unchanged prompt, Standard-1000 v2,
IR-aware, three own-role examples, GPT model `gpt-6.1-sol`, Chat Completions,
`medium`, max output 8192, timeout 360 seconds, and policy/environment seed 1.
All original scene IDs are retained. The actor is not exposed to the current
scene's MAPPO action or other roles' observations.

The independent output directory is `gpt61_runs/val5601_prompt100_v1/`.
It contains the immutable protocol lock, one MAPPO 100-scene chunk, and twenty
5-scene chunks for each GPT prompt variant. Each GPT variant has its own
400-request maximum, $6 configured token-rate estimate ceiling, usage ledger,
durable request journal, and prompt hash audit. Expected total is 800 calls,
about 4.50 million tokens and $8.91 at historical usage; the total configured
estimate ceiling is $12. Gateway billing rates remain unverified. No automatic
retries or overwrites of partial chunks are allowed.

Before starting, 22 offline tests passed, along with full dataset/library/
IR-overlay/checkpoint hashes and source-compatibility checks. Source data size
is about 615 MiB. The baseline temporarily loads scoring and actor-construction
environments, so several GiB of RAM are reserved; available RAM was about 21 GiB.
The source code, data and prompts are locked before execution. Previously
validated ten-scene results remain in their original directories.

```powershell
.\p2e_prompt\run_val5601_100.ps1 -Mode check
.\p2e_prompt\run_val5601_100.ps1 -Mode run -Method all
```

The run automatically creates a fresh comparison directory only after all
three methods and both prompt audits pass. It exports summary JSON/CSV and
300 paired rows with IR margins, original IR flags and node/link action means.
To export another comparison without model calls or overwriting reports:

```powershell
.\p2e_prompt\run_val5601_100.ps1 -Mode compare
```

`analyze_val5601_100.py --comparison-dir <completed comparison folder>` adds
offline paired bootstrap intervals and exact McNemar discordance tests. These
are exploratory unadjusted analyses of existing development data, not evidence
from an untouched final-test set. The analysis never calls a model.
