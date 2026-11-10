# Bounded local node-release development replay

This experiment applies five fixed deterministic node adjustments to each of
three cached Qwen draws on the same 200 `p2b_val_seed5601` development scenes.
The base is frozen `link_mean3_a0.5`. All original g and link actions stay fixed.
There are no API requests, model regeneration, checkpoint changes, independent
test reads, or changes to environment math or metric definitions.

`bounded_release.adjust_local` receives only one existing role's original
240-vector and its own cached four-dimensional action. It increases each node
ratio by at most 0.025 or 0.05, with static compute/cache overload increments
bounded by 0, 0.01 or 0.02 of that slice's own capacity. It uses the worst resource
within the original environment's contiguous overlap control group. These
bounds do not guarantee migration feasibility, admission, or non-increasing
cost. No current scene ID, peer observation/action, g request, global deficit,
or replay result is a policy input.

The fixed screening rule requires positive mean V change in all three draws,
no baseline admission losses, and no net cooperation decline in any draw.
The repeated development scenes and five-arm selection prevent interpreting
exploratory intervals as independent confirmatory evidence. Averages over
draws are formed within each scene before bootstrapping (n=200, not n=600).

## Files and execution

- `bounded_release.py`: pure role-local policy function and fixed grid.
- `test_bounded_release.py`: six boundary/semantics tests.
- `run_replay.py`: preflight, 800 baseline controls, 3000 interventions,
  12 post-controls, original accounting/IR validation, paired summaries.
- `audit_replay.py`: saved-action bounds/non-node preservation, all 812
  controls, all 3000 rows, summaries/transitions, source and dataset hashes.

```powershell
& 'python' -B -m unittest discover -s .\p2i_node_boundary -p test_bounded_release.py -v
& 'python' -B .\p2i_node_boundary\run_replay.py
```

The runner prints its new exclusive output directory. Audit that exact directory:

```powershell
& 'python' -B .\p2i_node_boundary\audit_replay.py --run-dir .\p2i_node_boundary\run_20261105T012338_935948Z
```

The auditor creates `audit.json` exclusively; do not rerun into an already
audited directory. No existing experiment outputs are overwritten.

Inspected implementations: `main.py` node release and action expansion,
`p1a/local_obs.py`, `p1a_runner.py`, `p1c/retrieval.py`, `p1c_runner.py`,
`p2e_prompt/val5601_eval.py`, prior node/link replay source, and the passive
service/gate trace. Exact experimental inputs and their sizes/hashes are in
each run's `protocol.json`. Source sample is 644,537,810 bytes; replay loads
one original 200-scene environment and checks available RAM and output space.

The first startup attempt stopped before environment scoring because a relative
`__file__` path was used in hashing. This was corrected to an absolute path;
the subsequent run creates a different directory. Its completion must be judged
from `summary.json` and `audit.json`, not from startup or partial records.
