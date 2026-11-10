# Reproducibility and publication provenance

## What this snapshot verifies

The release contains method source, offline contracts, static prompt templates and selected small summary/protocol artifacts. Source syntax, mock-based tests, source/artifact hashes and publication hygiene are checked separately from historical environment audits. Saved audit.json files report the completed historical runs; publication tests do not rerun those experiments.

## Historical inputs that are not bundled

- GEANT topology and traffic inputs, generated scenario pickles and train/validation/test manifests.
- Frozen MAPPO checkpoint `marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth`.
- TRAIN role libraries `p2b_libraries/p2b_standard_1000_v2` and `p2b_ir_libraries/p2b_standard_1000_ir_v2`.
- Raw per-scene actions, model prompts/replies, request journals and replay caches.
- The historical `tmgen==0.1.2` source installation. The original environment export only recorded a local editable path, not a portable repository revision. Obtain a compatible source checkout independently and record its revision. Do not assume a similarly named PyPI package is identical.

`requirements.txt` retains historical version pins where available, removes machine-specific pip URLs, and writes UTF-8. The original typing_extensions file URL did not preserve a version pin; the portable minimum is explicitly not an exact historical lock. The minimal offline requirements support contract tests and do not install the full simulator/training stack. Neither a clean full-environment install nor a new numerical reproduction is claimed.

## Publication-only changes

Internal private-IP defaults are replaced by reserved `.invalid` example endpoints. Set actual endpoints and tokens through environment variables. PowerShell paths use the script location and the active Python interpreter; legacy plotting/data helper paths are relative or configurable. Environment equations, reward/metric definitions, migration/rerouting calculations and frozen policy parameters are unchanged. The existing NumPy array equality compatibility fix in `main.py` is included with its original attestation.

The publication also repairs the P2C test fixture to include the evaluation adapter/provenance files now required by its source checks, and makes two replay-test path assertions portable across Windows/POSIX. Test scenarios/assertions are retained; production evaluation gates are not weakened. Ordinary public source uses LF newlines. `.gitattributes` preserves exact bytes for curated artifacts, so historical main-source snapshots and artifact hashes survive a checkout with a different Git newline setting. Archived source whitespace is preserved deliberately.

`research_artifacts/SOURCE_PROVENANCE.json` records reviewed research-source hashes and public hashes. Public transport/path cleanup changes source fingerprints. Existing experiment locks are preserved as historical evidence, not regenerated to match public edits. Original protocol assertions may therefore refuse a rerun from this checkout. Resolve inputs and establish a new explicitly documented protocol for a new experiment; do not disable checks or rewrite an old lock to imply historical identity.

## Curated artifacts and original locations

`research_artifacts/MANIFEST.json` maps each public artifact to its original relative path, original SHA256 and public SHA256. Summaries with unchanged bytes retain original fingerprints. The public P2C protocol has local project paths replaced with a neutral placeholder, and its different hash is recorded. For historical analysis in an authorized private checkout, restore copies using that path map together with the matching original data and source versions. The public placeholder path is not a runnable data location.

The P2C `main_generation.py` and `main_evaluation.py` snapshots are byte-identical to the recorded generation/evaluation sources. Their hashes match `eval_hotfix_attestation.json`; only the approved path-array equality line differs between these snapshots. The public root `main.py` has the same code as the evaluation snapshot with LF newline normalization, so its byte hash differs. This does not supply the missing datasets or reconstruct omitted runtime files.

Fresh mainline confirmation uses seeds 2026110511/2026110512/2026110513, 512 scenes each. Cached comparison uses 6601/6701/6801 and is previously inspected data. The service-demand pilot covers only the first 16 scenes of each fresh seed (48 total). All twelve mainline arms remain visible. Supplementary metrics are descriptive and preserve their historical definitions.

## Deferred P3 material

The autonomous role-allocation/repetition workspace is omitted because its historical source/retained-arm statistics were being consolidated and one summary changed during publication review. Removed g-node-gate research is not restored. The retained k-role bounded-node postprocessor is a separate, frozen part of the published mainline. No incomplete three-seed or repeated-development-scene result is presented as a completed independent confirmation.
