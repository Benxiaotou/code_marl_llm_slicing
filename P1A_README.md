# P1-A: decentralized zero-shot Qwen on the original GEANT slicing environment

## What is included

`p1a_runner.py` is a **new**, stand-alone experiment entry point. It does not
rewrite `main.py`, the existing `p0_*.py` scripts, the MAPPO checkpoint, the
reward function, or the GEANT datasets. `p1a/` implements the observation,
API and action policies. `tests/` supplies offline unit and synthetic pipeline
tests. The real internal API and real GEANT environment still need local acceptance testing.

The procedure is one API call per role, in the deterministic order `agent_k1`,
`agent_k2`, `agent_k3`, `agent_g`. All four calls target the same model but have
separate messages, with **only that role's original 240-dimensional observation**.
The policy does not see `env.state()`, `env.global_state`, `env.DG`, other roles'
observations/actions, global feasibility calculations, or environment feedback.
Only the experiment runner handles environment state to select the scenario and
compute metrics **after** the action is finalized.

This is a frozen-model zero-shot baseline: no examples, historical cases,
training, MAPPO behavior cloning, LP, Guard, extra negotiation or test-time
candidate search are used.

## Install

Unzip the package into the directory holding the existing `main.py` and
`configs_main.yaml` (do not overwrite other files). Use your already working
Windows `(marl)` Conda environment.

```powershell
conda activate marl
cd F:\Projects\code_marl_v0_exper_1
python -m unittest discover -s tests -p test_p1a.py -v
```

Only P1-A tests are run by that command. API transport uses `requests`, already
present in the P0 environment. The code uses Python 3.8-compatible syntax.

## Local mock smoke test (not a Qwen result)

The mini dataset is the earlier `P0_small_samples.pkl` with its accompanying
`P0_small_samples.indices.json`. Original scene IDs are kept.

```powershell
python p1a_runner.py --dataset mini --policy mock --count 5
```

The `mock` policy always returns fixed ratios to exercise observation parsing,
the environment and metrics. **Do not report mock results as Qwen performance**.

## Live Qwen on one real mini scenario

```powershell
$env:LLM_API_KEY = "<your real internal API key>"
$env:LLM_BASE_URL = "http://192.168.131.35:30001/v1"
python p1a_runner.py --dataset mini --policy qwen --count 1 --max-tokens 4096 --api-timeout 120 --on-error abort
```

Use `--no-stream` only when needed to diagnose server streaming support.
`--api-timeout` is the **network read timeout**, not a strict wall-clock limit.

After smoke testing, a broader P1-A development run can use the full local
600-scene file (not uploaded or changed) and original IDs:

```powershell
python p1a_runner.py --dataset full --policy qwen --scenes 100,101,102,103,104,105,106,107,108,109 --on-error abort
```

The prior P0 512 scenes have been examined; these P1-A runs are for development
and do **not** count as an untouched final test. Make a separate held-out set
for P1-B/C/E after freezing the method. To compare with MAPPO, rerun the
frozen MAPPO policy on the **exact same scene IDs and environment mode** using
matched random seeds and verify record-level metric definitions. An older
aggregate 512 result is a reference only, not an automatically matched baseline.

## Outputs

A unique directory under `p1a_runs/` contains:

- `manifest.json`: model, scene IDs, experiment parameters, `main.py` hash.
- `records.jsonl`: each role's prompt, raw answer, parsed ratios, final action,
  separate latency, token usage if provided, and the original environment metrics.
- `metrics.csv`: machine-readable per-scene outcomes and errors.
- `summary.json`: admission, individual rationality, cooperative validity,
  utility and adjustment cost with explicit denominators.

Failed LLM decisions are never secretly substituted with a deterministic policy.
By default, `--on-error abort` stops the run and preserves prior records.
`--on-error record` records missing outcomes; full-denominator acceptance and
cooperative-validity metrics are then null, not silently computed from successful
responses alone. An unsuccessful admission is different from an API failure.

`g` returns three active ratios, but the original XuanCe environment declares
four action dimensions for all agents. The unused fourth `g` action is set
to `0` in [-1,1] and the environment only reads the first three. The model's
ratio `p` is mapped to the original action `2*p-1`; no LP/Guard repair is used.

## Important scope / known limitations

1. We checked upstream public GEANT `main.py` observation/action implementations,
   but do not have your latest locally patched P0 `main.py`, small sample data,
   model checkpoint, or intranet connection in this session. Real integration
   and API performance are **not yet verified** by us.
2. The original `main.py` hard-codes 600 scenes. For `--dataset mini`, a small
   subclass intercepts only initialization assignments for the sample path and
   count; no global source rewrite is done. This mechanism has a synthetic test
   but must be checked against your locally patched main.py.
3. The default GEANT observation schema assumes 12 local nodes, 36 maximum
   padded links and a 240-value vector, per the original code. Unexpected shapes
   fail loudly rather than leaking full network state or inventing features.
4. In the original code, g observes **physical allocated resource fractions**
   at g's own nodes/links, not the entire global network state. These values
   remain in the g local prompt as permitted by its source observation.
5. The original environment's new-slice compute/cache admission check appears
   to compare against the pre-compression demand. This package does **not**
   change or silently repair that legacy rule; align all baseline evaluations
   to the same unmodified environment.
6. The result of four **independently chosen** actions may be uncoordinated.
   P1-A measures that weakness; P1-B/P1-C will test historical-cooperation ICL.
7. No asynchronous model execution: calls are sequential, giving a clear
   four-call latency and isolated contexts. A parallel version needs separate
   experimental reporting if later introduced.

## Rollback

Delete `p1a_runner.py`, `p1a/`, `tests/test_p1a.py` and the new `p1a_runs/`
results. No original files are patched, and deleting them does not change
MAPPO or P0.

## Optional matched-scene comparison to existing MAPPO logs

The package also includes `p1a_compare.py`. The P0 runner previously wrote
`mappo_records.jsonl` with `original_scene_index`, `g_done`, `V`, `cost_sum`,
`V_k1_0` etc. For example, once both methods have been evaluated on the
identical five original mini scenes, run:

```powershell
python p1a_compare.py --p1a .\p1a_runs\<YOUR_QWEN_RUN>\records.jsonl --mappo .\p0_runs\mappo_mini_20261012_163149\mappo_records.jsonl --output-dir .\p1a_runs\paired_example
```

The comparison script rejects missing/extra scene IDs, failed model calls,
fallback-contaminated results and duplicate records. It reports paired differences
but cannot infer whether the two runs used identical environment code versions or
stochastic seeds: check those experimentally before interpreting small differences.
