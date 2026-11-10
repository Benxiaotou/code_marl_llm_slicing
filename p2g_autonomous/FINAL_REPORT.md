# Autonomous Qwen research session — final frozen candidate

## Frozen method

Use the original Qwen P2-A policy for all four roles. For each existing slice,
retrieve the three nearest training cases using only that role's link-local
features. Average the three historical link ratios and blend that target 50%
with the Qwen link controls. Keep Qwen node controls and g controls unchanged.

This was selected on the val5601 development pool and then evaluated without
further tuning on the independent P2-C replay. The postprocessor makes no
additional model calls.

## Independent P2-C replay

Three seeds, 512 scenes each; 1536 paired scenes. All 3072 MAPPO/Qwen control
replays reproduced their saved metrics. New API calls: 0; estimated incremental
API cost: $0.

| Method | Accepted | Cooperative | V | V_g | Cost | Accepted IR invalid |
|---|---:|---:|---:|---:|---:|---:|
| MAPPO | 1283/1536 | 1252/1536 | 341.9941 | 24.8084 | 33.9102 | 31 |
| Original Qwen | 1363/1536 | 1192/1536 | 339.1035 | 26.4618 | 34.0928 | 171 |
| Frozen link calibration | 1363/1536 | 1261/1536 | 341.4151 | 26.4304 | 32.7057 | 102 |

Frozen candidate minus original Qwen: cooperative +4.49 percentage points
(96 wins, 27 losses pooled), V +2.3115, cost −1.3872, with unchanged
admission. Per-seed cooperation gains were +4.49, +5.86 and +3.13 points.

Frozen candidate minus MAPPO: cooperative +0.59 points (136 wins, 127 losses),
V −0.5790, V_g +1.6220, cost −1.2045. This does not establish superiority on
every MAPPO endpoint.

Pooled diagnostics over the same 1536 scenes:

- effective V_g: candidate 24.8953, MAPPO 24.3313, original Qwen 23.5031;
- V lower 10% mean: candidate 261.7607, MAPPO 261.0802, original Qwen 260.5432;
- cost upper 10% mean: candidate 46.7368, MAPPO 51.2911, original Qwen 54.4278;
- mean IR shortfall: candidate 2.8280, MAPPO 2.7606, original Qwen 3.3445.

## Development diagnostics

The full link-calibration prompt failed on 200 fresh scenes. Fresh node
calibration plus 25% blend also failed to transfer. Offline crossover isolated
the useful block to link actions. Three cached Qwen cohorts supported the
equal three-neighbor link mean at 50%:

- historical Qwen: cooperative 158→164, V 343.7608→346.2280;
- fresh cohort A: 159→167, V 344.1437→346.5745;
- fresh cohort B: 157→164, V 343.8840→346.1681.

Single-role and pair-role ablations were weaker than all three roles together.
Distance weighting, inverse-cost weighting, risk gating, nearest-only and
node/link prompt variants were not selected. Two early offline scripts that
failed to reload scenes between counterfactual cases are marked INVALID.md and
excluded from all conclusions.

## Scope and files

All P2-C inputs were used only after candidate freezing and were not used for
tuning. The original environment, rewards, metrics, migration/rerouting logic,
datasets, checkpoints and prior results were not modified. Full records and
audits are under p2g_autonomous/results/; the formal replay summary is
p2g_autonomous/results/formal_test_replay_20261104T165841_092219Z/pooled_summary.json.
No GPT calls occurred.
