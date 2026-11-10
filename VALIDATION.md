# Publication validation — 2026-11-10

Python 3.8.20; offline tests in twelve isolated suite processes, with external network blocked and actual model credentials removed.

- 201 tests discovered/run, 198 passed, 3 skipped, zero failures or errors on the final run.
- The three skips require optional private/user-supplied ZIP archives omitted from the public snapshot. All synthetic replay and action-component checks ran.
- 13 PowerShell files parsed without syntax errors; no launcher was executed.
- Python syntax, publication payload hygiene, original/public artifact hashes and the P2C source attestation were checked. Exact source count is in the local publication receipt.
- No new model inference, original environment scoring, historical numerical auditor, full training run or clean-environment package install was performed. Additional model API cost: $0.

The first restricted run encountered Windows temporary-directory permissions. The offline harness now places temporary fixtures under the checkout. Further testing exposed an outdated P2C mock fixture and two POSIX-only test assertions; these were corrected in the public copy and the final complete suite passed. Production protocol checks and evaluation formulas were not relaxed.

Run again from the repository root:

```powershell
python -B tools/run_offline_checks.py
```

The minimal offline requirements do not establish a fully reproducible simulator installation. See REPRODUCIBILITY.md and VALIDATION.json for limits and per-suite counts.
