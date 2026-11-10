# P2-B v2: Fix composite case IDs for LocalHistory

## Diagnosis based on the reported output and v1 code

All six `p2b_*_v1` libraries passed the original `p1b_audit.py` checks. The error then occurred while the validation program initialized `LocalHistory`, before IR generation started.

The v1 builder assigned a composite `dataset_id` in the manifest while retaining source-dataset `joint_id` and `experience_id` strings. This is the likely cause of the `Unexpected / duplicate training case id` exception. The exact implementation of `p1c/retrieval.py` was not included in the uploaded sources; local validation is therefore still required.

## What the fix does

- Builds **new** `p2b_standard_{300,500,1000}_v2` and `p2b_balanced_{300,500,1000}_v2` directories from the same two original, audited P1-B source libraries and the same selection rules.
- Uses a stable composite case namespace: `<composite_dataset_id>::000000`, `<composite_dataset_id>::000001`, etc.
- Global indices depend on the **original source order** (152 old cases first, then 1,209 new-source cases); the same case has the same ID across all six v2 libraries even when the selected subset/order differs.
- Rewrites the four local-role `experience_id`s, private joint `dataset_id`/`joint_id`, and `private_offline/selection_index.json` consistently.
- Leaves the **observation vectors, MAPPO teacher action vectors, local-role schema, selection score and selection order** unchanged.
- Records the source dataset ID, training sample SHA, original joint ID, and original scene ID in the **private-only** `private_offline/source_case_id_map.jsonl` file. Never inject this file into the Qwen prompt.
- `p2b_validate_ir_v2.py` verifies v1/v2 case-for-case equivalence before calling the original `p1b_audit.py`, `LocalHistory`, and `p1d_build_ir.py` / `IRHistory`.

No existing v1 library, old P1-B library, P2-A baseline, IR file, or Qwen experimental results are overwritten. Existing output names cause the script to abort.

## Install

Extract the ZIP. Copy the two `.py` files into the **existing** experiment project root:

`path/to/code_marl_llm_slicing`

Do not replace `p2b_merge.py` or `p2b_validate_ir.py`; the new script names include `_v2`.

## Step 1. Read-only preflight

```powershell
conda activate marl
cd path/to/code_marl_llm_slicing

python .\p2b_merge_v2.py `
  --holdout-manifest .\p1b_data\p1c_val_seed4601.json
```

Expected final line: `STATUS: read-only preflight passed; add --execute to write new libraries.`

## Step 2. Write new v2 libraries

Only after the preflight succeeds:

```powershell
python .\p2b_merge_v2.py `
  --holdout-manifest .\p1b_data\p1c_val_seed4601.json `
  --execute
```

This produces six new `p2b_*_v2` libraries and `p2b_libraries/p2b_composite_build_summary_v2.json`. Existing `p2b_*_v1` libraries remain untouched.

## Step 3. Verify content parity, audit, retrieval and IR

```powershell
python .\p2b_validate_ir_v2.py `
  --holdout-manifest .\p1b_data\p1c_val_seed4601.json
```

The verifier performs, in this order:

1. `V1/V2 CONTENT PARITY OK` for all six libraries (only case IDs may differ; source ID mapping is checked).
2. Original P1-B audit for each v2 library.
3. `LocalHistory` construction and train/eval independence assertion for all six.
4. Fresh P1-D IR construction at `p2b_ir_libraries/p2b_*_ir_v2` and `IRHistory` compatibility verification for all six.
5. Final `"ok": true`, `"audited_libraries": 6`, `"verified_ir_overlays": 6`.

If one of these steps fails, **stop** and share the first complete traceback/output. Do not change the original `p1c/retrieval.py` ID-prefix constraint and do not remove the leakage checks.

### Important uncertainty

We tested v2 on synthetic source libraries, including the original user-uploaded `p1b_audit.py` with test stubs for its dependencies, and an ID-prefix compatibility assumption inferred from the reported error and source-data structure. We **have not executed against your true 152 + 1,209 cases**, nor against the original unshared `p1d/ir.py` and full `p1c/retrieval.py` implementations. Actual IR build success must be verified on your machine.

The original validation dataset is only used to verify separation, never to collect experience. The v2 patch neither trains MAPPO nor calls Qwen.

## Optional offline test

```powershell
python -m unittest discover -s tests -p "test_p2b_merge_v2.py" -v
```

The 10 tests use synthetic inputs and do not touch the experiment directories. Two tests explicitly check that the v1-to-v2 equivalence verifier accepts case-ID-only changes and rejects action changes.

## Optional v1 ID inspection (read-only)

```powershell
$lib = ".\p2b_libraries\p2b_standard_300_v1"
$meta = Get-Content "$lib\library_manifest.json" -Raw | ConvertFrom-Json
$item = Get-Content "$lib\role_libraries\agent_k1.jsonl" -TotalCount 1 | ConvertFrom-Json
"Library dataset ID: $($meta.dataset_id)"
"First case ID:      $($item.experience_id)"
```

Different prefixes are evidence of the v1 namespace mismatch. The v2 patch never edits these existing v1 files.
