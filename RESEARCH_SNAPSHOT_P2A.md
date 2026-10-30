# Research snapshot: P0 to P2-A

This is a **development-stage** snapshot of decentralized network-slice
resource allocation experiments with a frozen Qwen model and a frozen MAPPO
teacher. Evaluation-time agents act on their own original local observations.
The current pipeline does **not** use test-time global state, Guard/LP action
repair, or LLM fine-tuning.

## Development milestones

- P0: baseline/diagnostics and MAPPO reproduction.
- P1-A/P1-A+: local-role zero-shot Qwen baselines and semantic observation prompts.
- P1-B: MAPPO teacher history generation, selection, and role-local histories.
- P1-C: role-local experience retrieval, kNN and Qwen-ICL matched comparisons.
- P1-D: self-role historical IR/cost feedback added to frozen Qwen ICL.
- P1-E: prompt-only vs historical-data-only ablation.
- P2-A: local IR-risk-aware historical example reranking.

## Existing matched 200-scene *development validation* observations

| Method | Admission | Cooperative-valid | Mean V | Mean adjustment cost |
|---|---:|---:|---:|---:|
| Frozen MAPPO | 85.5% | 82.5% | 346.05 | 34.59 |
| Full IR-aware Qwen-ICL | 95.0% | 74.5% | 342.41 | 38.63 |
| P2-A IR-risk reranking | 93.0% | 77.0% | 341.67 | 37.99 |

These 200 scenarios were used for method development; **they are not an
independent final test set**. The P2-A vs IR-aware +2.5 percentage-point
cooperative-valid difference was not conclusive under paired resampling.

## Data and reproducibility

This public code snapshot excludes generated GEANT datasets, selected teacher
histories, model checkpoints, raw prompts/responses, API configuration/secrets,
and detailed run logs. Regenerate datasets and IR role overlays using the
P1-B and P1-D instructions. Record the data split hashes before comparison.
Use separate withheld scenarios for final performance reporting.
