# Frozen node policy on existing P2-C 1536 scenes

User-requested additional evaluation on the existing test datasets, with
horizontal comparison against all eight original formal methods and the
previous frozen link method. These tests were previously inspected; this
run is not an untouched independent final test. Parameters are fixed and
must not be tuned using this run.

## Inputs and policy

- Existing dataset manifests: `p1b_data/p2c_test_seed6601.json`,
  `p1b_data/p2c_test_seed6701.json`, `p1b_data/p2c_test_seed6801.json`.
  Each names one original 512-scene pickle and its SHA256.
- Exact original chunks from `p2c_qwen8_queue.JOBS`: each of the eight
  methods, each seed, starts 0,32,...,480, under
  `p2c_runs/qwen8_test_v1/p2c_test_seed<seed>/<tag>/chunk_<start>_032/`.
  Read only each chunk's manifest, summary and streamed records. Retain
  metrics, plus MAPPO and P2-A Standard1000 actions; omit prompt logs.
- Prior link controls:
  `p2g_autonomous/results/formal_test_replay_20261104T165841_092219Z/records_<seed>.jsonl`.
- Frozen candidate:
  `p2i_node_boundary/run_20261105T012338_935948Z/candidate.json`.
  Link `link_mean3` alpha 0.5; node ratio increment <=0.025 and extra
  projected overload <=0.01 of own capacity.
- Audited public TRAIN Standard1000 library and unchanged validated helpers
  from `p2j_validation/run_validation.py`. Each role's postprocessor uses
  its own local observation/action and local TRAIN retrieval only.

Original environment, metrics, rewards, datasets, checkpoints and existing
results are preserved. A single 512-scene environment is loaded at a time;
preflight requires at least 8 GiB available memory and 2 GiB free disk.
New outputs are placed in an exclusive timestamped directory.

## Implementation and validation

- `run_cached.ps1`: removes API credentials from the child process and runs
  the offline evaluation with a workspace-local temporary directory.
- `run_cached.py`: checks metadata/hashes; scores four arms on all 1536
  scenes; reproduces 4608 historical control scores and 12 post-controls;
  summarizes all ten methods and nine paired candidate comparisons.
- `audit_cached.py`: independently reconstructs saved deterministic
  actions, checks archived/replayed metrics and hashes, and recomputes
  summary values and paired point estimates without constructing an
  environment or API client.
- `test_cached.py`: three contract tests for scene coverage, rejection of
  mismatched dataset metadata, and action preservation/bounded adjustment.
  All three tests passed before execution.

Bootstrap intervals use 10000 paired scene resamples stratified by the
three generation seeds, RNG 20261105. The primary contrast is candidate
minus MAPPO mean V; the other eight contrasts are descriptive, without
familywise multiplicity adjustment. No new LLM draw is tested: derived
methods share archived P2-A actions.

## Commands

```powershell
& 'python' -B -m unittest discover -s .\p2l_cached_test -p test_cached.py -v
& .\p2l_cached_test\run_cached.ps1
```

The second command starts a NEW offline run; never overwrite a previous
run. API calls: 0; additional API cost: $0. Source/sample hashes and
controls are required to pass before the run can be marked COMPLETE.

Outputs include `protocol.json`, `historical_metrics.jsonl`,
`records.jsonl`, `post_controls.jsonl`, `scored_summary.json`,
`audit.json`, `summary.json` and `RESULTS.md`. On failure, preserve
`STOPPED.json` and partial records for diagnosis.

## Completed execution

`run_20261105T023605_900720Z`: exit 0, COMPLETE, audit passed. Available RAM
19.48 GiB at preflight. All 6144 evaluations, 4608 original controls, 12
post-controls and 1536 reconstructed action checks passed. API calls 0,
additional API cost $0. Initial sandbox attempt was denied access to the
public TRAIN manifest before scoring; the scoped approved launcher completed.

Candidate mean V 343.579617 versus MAPPO 341.994095, difference +1.585522
(+0.4636%), stratified paired 95% interval [0.118391,3.139406]. Candidate
ranks first among ten on mean V, admission and cooperation in these scenes.
Mean cost increases by 0.446076 versus MAPPO. See the run's `RESULTS.md`
for the full ten-method table and `ANALYSIS.md` for interpretation/limits.

Relevant implementation inspected: `run_cached.py`, `audit_cached.py`,
`run_cached.ps1`, `test_cached.py`, `p2j_validation/run_validation.py`,
`p2i_node_boundary/bounded_release.py`,
`p2g_autonomous/jobs/teacher_link_multicohort.py`,
`p2f_analysis/val5601_counterfactual_replay.py`, `p2c_qwen8_queue.py` and
`PROJECT_HANDOFF.md`. Exact input metadata/hashes are preserved in protocol.
Only new files in `p2l_cached_test` and the project handoff were changed;
pre-existing edits to `main.py` and `.gitignore` were left intact.

To inspect the completed report without starting another experiment:

```powershell
Get-Content -LiteralPath .\p2l_cached_test\run_20261105T023605_900720Z\RESULTS.md
Get-Content -LiteralPath .\p2l_cached_test\run_20261105T023605_900720Z\ANALYSIS.md
```
