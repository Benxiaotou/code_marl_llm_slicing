# Fixed-candidate validation on new scenes

User-authorized comparison of the frozen `buffer01_d025` node postprocessor
against MAPPO and the preceding frozen link-only candidate. No tuning occurs
during this run. The original Qwen P2-A actions are an additional reference.

- Two new generation seeds: `2026110501`, `2026110502`; 100 scenes each.
- Split: validation, using the unchanged original GEANT generator and config.
- Node parameters: max ratio increment 0.025, extra predicted overload 0.01.
- Link parameters: three nearest TRAIN link-local cases averaged, alpha 0.5.
- Frozen MAPPO checkpoint: `512000_GEANT_mappo_model_save.pth`, SHA256
  `fa36695c78feb28b3506fbd2858e6efbc2c8f44e1f98fd7fe0282acf191ebf03`.
- Qwen: `qwen3-next-80b-a3b-instruct`, temperature 0.1, max tokens4096,
  streaming, timeout120s, 800 requests, up to8 concurrent requests.
- One shared fresh Qwen action draw per scene; neither postprocessor requests
  another model response. No GPT inference, fallback, or automatic retries.
- Primary endpoint: new node candidate mean V minus MAPPO on all200 scenes.
  Secondary comparisons include node minus link and node minus raw Qwen.
- Paired bootstrap:10000 samples stratified by generation seed. Scene identity
  is `(generation_seed, scene_id)`. Report each seed separately as well.

Only one role's current observation and own public TRAIN role history enter its
prompt or postprocessor. Offline scoring uses the original global environment.
MAPPO actions/outcomes never enter Qwen prompts or postprocessing. All original
datasets, checkpoints, environment mathematics, rewards and metric definitions
are preserved. Old independent test results are not accessed. This is new
validation under the same generator, not a new final test or a distribution shift.

## Sources and outputs

Inspected: root `AGENTS.md` and `PROJECT_HANDOFF.md`, frozen `candidate.json`,
`main.py`, `configs_main.yaml`, `p1b_generate.py`, `p1a_runner.py`, `p1c_runner.py`,
`p0_runner.py`, `p1b/teacher.py`, `p1b/core.py`, local observation/API/policy/
retrieval/IR/rerank implementations, prior link teacher and evaluation runner
source. Public training library files are read only by their exact runtime
paths. Exact source hashes and public input paths/sizes are in each protocol.

New files: `run_validation.py`, `run_validation.ps1`, `audit_validation.py`,
`test_validation.py` and this README. Launcher loads User-scope Qwen credentials
without printing them, clears GPT credentials from its child process, and puts
temporary files in a new workspace directory. All experimental outputs and
new datasets go to an exclusive timestamped run directory.

Four contract tests actually passed before launch. The live preflight also
requires12 old development prompts,3 frozen MAPPO action controls and6 old
metric controls to reproduce before any API call. Final audit reconstructs all
800 prompts/responses,400 postprocessed actions and800 score rows, checks200
new MAPPO metric controls, and verifies source/dataset hashes.

```powershell
& 'python' -B -m unittest discover -s .\p2j_validation -p test_validation.py -v
& .\p2j_validation\run_validation.ps1
```

The second command generates new files and makes800 internal Qwen requests.
According to the user's unlimited internal quota the extra fee is estimated$0;
token counts are recorded when returned and reported unknown when absent.
Never run another copy while the current one is active. A complete run requires
`summary.json` statusCOMPLETE, a passing `audit.json`, and `RESULTS.md`.

Initial attempt `run_20261105T014617_460924Z` stopped before generation/API calls
because the existing generator's override takes a callable rather than a tuple.
That compatibility issue was fixed without editing the original generator.
The subsequent run is `run_20261105T014642_470733Z`.

## Completed result

That run completed with exit0 and a passing full audit. No process is running.
Node candidate meanV346.5482 vs MAPPO344.7959 (+1.7523,about0.508%);
paired95%CI[-1.8731,5.5156], so a stable advantage is not established.
Node minus link meanV+2.7966,CI[.3250,5.5853]. Node admits191/200 and is
cooperative170/200, compared with MAPPO167/200 and164/200. Node average modeled
cost36.5396 exceeds MAPPO35.6751. Both seeds have positive mean gains against
MAPPO, while seed2 has a negative gain against link-only. No tuning followed.
All800 requests succeeded; no usage was returned, so tokens are unknown.
The API fee estimate remains$0 under the user's internal quota. See the run's
`RESULTS.md`, `summary.json` and `audit.json` for full data and limitations.
