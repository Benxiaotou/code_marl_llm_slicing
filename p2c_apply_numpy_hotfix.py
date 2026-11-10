#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-time, audited migration after P2-C failed with NumPy path truth ambiguity.

Run in the ORIGINAL project root:
  python p2c_apply_numpy_hotfix.py --check
  python p2c_apply_numpy_hotfix.py --apply

STRICT GATES:
 - Exactly one known source line is edited in main.py.
 - Three complete test manifests already exist with identical original main hash.
 - The original protocol lock exists and references the original main hash.
 - No P2-C evaluation scene has been scored; the single expected failed chunk is 0/0.
 - All original manifests and .pkl files remain completely untouched.
 - The failed attempt and old lock are archived, not deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path.cwd().resolve()
SEEDS = (6601, 6701, 6801)
RUN_ROOT = ROOT / 'p2c_runs/qwen8_test_v1'
PROV = RUN_ROOT / 'provenance'
MAIN = ROOT / 'main.py'
QUEUE = ROOT / 'p2c_qwen8_queue.py'
WRAPPER = ROOT / 'p2c_eval_runner.py'
OLD_LOCK = RUN_ROOT / 'protocol_lock.json'
OLD_LINE = b'if k_net_new_shortest_paths[idx_flow] != k_net.shortest_paths_node[idx_flow]:'
NEW_LINE = b'if not np.array_equal(k_net_new_shortest_paths[idx_flow], k_net.shortest_paths_node[idx_flow]):'
FAILED = RUN_ROOT / 'p2c_test_seed6601/mappo/chunk_0000_032'
PROV_MAIN = PROV / 'main_generation.py'
PROV_LOCK = PROV / 'protocol_lock_before_eval_hotfix.json'
PROV_FAILED = PROV / 'failed_attempt_mappo_6601_chunk_0000_032'
ATTESTATION = PROV / 'eval_hotfix_attestation.json'


def require(ok: bool, why: str):
    if not ok:
        raise RuntimeError(why)


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for piece in iter(lambda: f.read(1024*1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def read(path: Path):
    with path.open('r', encoding='utf-8-sig') as f:
        return json.load(f)


def preflight():
    require(MAIN.is_file(), 'main.py not found; run in experiment root')
    require(QUEUE.is_file() and WRAPPER.is_file(),
            'Copy the v1.2 queue and p2c_eval_runner.py into the project root first')
    require('eval_hotfix_approved' in QUEUE.read_text(encoding='utf-8'),
            'Queue is not v1.2: do not apply the hotfix using the old queue')
    require('approved_test_manifest' in WRAPPER.read_text(encoding='utf-8'),
            'P2-C evaluation adapter is missing or wrong')
    require((ROOT / 'p1c_runner.py').is_file(), 'Missing original p1c_runner.py')
    require(OLD_LOCK.is_file(), 'Original protocol_lock.json missing; STOP')
    require(not ATTESTATION.exists(), 'Hotfix already applied; STOP')
    require(not any(x.exists() for x in (PROV_MAIN, PROV_LOCK, PROV_FAILED)),
            'Existing provenance archive detected; STOP to avoid overwrite')

    source = MAIN.read_bytes()
    require(b'import numpy as np' in source, 'main.py does not import numpy as np; inspect manually')
    require(source.count(OLD_LINE) == 1, 'Expected original failing line exactly once in main.py')
    patched = source.replace(OLD_LINE, NEW_LINE, 1)
    try:
        compile(patched, str(MAIN), 'exec')
    except Exception as exc:
        raise RuntimeError(f'Patched main.py would not compile: {exc}') from exc
    old_main_hash = sha(MAIN)
    manifests = {}
    for seed in SEEDS:
        path = ROOT / 'p1b_data' / f'p2c_test_seed{seed}.json'
        require(path.is_file(), f'Missing dataset manifest: {path}')
        m = read(path)
        for key, want in (('dataset_id', f'p2c_test_seed{seed}'), ('seed', seed),
                          ('split', 'test'), ('n_scenes', 512)):
            require(m.get(key) == want, f'Unexpected {key} in {path}')
        require(m.get('main_py_sha256') == old_main_hash,
                f'Dataset {seed} was not generated with the current original main.py; STOP')
        sample = Path(m['sample_file'])
        if not sample.is_absolute():
            sample = ROOT / sample
        require(sample.is_file(), f'Missing generated sample {sample}')
        require(sha(sample) == m['sample_sha256'], f'Sample hash mismatch: {sample}')
        config = ROOT / 'configs_main.yaml'
        require(config.is_file() and sha(config) == m.get('config_sha256'),
                f'Configuration changed since generation: {seed}')
        manifests[str(seed)] = {'manifest_sha256': sha(path),
                                'sample_sha256': sha(sample)}

    lock = read(OLD_LOCK)
    require(lock.get('version') == 'p2c_qwen8_serial_v1',
            f'Unexpected old protocol version: {lock.get("version")}')
    expected_lock_main = lock.get('source_sha256', {}).get(str(MAIN.resolve()))
    require(expected_lock_main == old_main_hash,
            'Original lock does not match original main.py; STOP')
    require(lock.get('seeds') == list(SEEDS), 'Unexpected test seeds in protocol')
    require(lock.get('scenes_per_seed') == 512 and lock.get('chunk_size') == 32,
            'Unexpected test size or chunk size in protocol')
    for seed in SEEDS:
        exp = lock.get('test_datasets', {}).get(str(seed), {})
        require(exp.get('sample_sha256') == manifests[str(seed)]['sample_sha256'] and
                exp.get('manifest_sha256') == manifests[str(seed)]['manifest_sha256'],
                f'Protocol lock does not match test data {seed}')

    # This migration only applies before the first formal scene is scored.
    for summary in RUN_ROOT.glob('p2c_test_seed*/**/summary.json'):
        data = read(summary)
        require(data.get('n_scored', 0) == 0 and data.get('n_attempted', 0) == 0,
                f'Some scenes already evaluated ({summary}); do NOT reset a scored protocol')
    require(FAILED.is_dir(), f'Expected first failed MAPPO chunk missing: {FAILED}')
    failed_summary = read(FAILED / 'summary.json')
    require(failed_summary.get('n_scored') == 0 and failed_summary.get('n_attempted') == 0,
            'Failed chunk has scored/attempted scenes; require manual review')
    return source, patched, manifests, old_main_hash


def apply():
    original, patched, manifests, old_sha = preflight()
    PROV.mkdir(parents=True, exist_ok=True)

    # Archive first. Refuse to overwrite anything.
    PROV_MAIN.write_bytes(original)
    require(sha(PROV_MAIN) == old_sha, 'Cannot verify archived original main.py')
    shutil.copy2(OLD_LOCK, PROV_LOCK)
    require(sha(OLD_LOCK) == sha(PROV_LOCK), 'Cannot verify archived original lock')

    # Preserve failed chunk and never mix failed/partial data with scored output.
    shutil.move(str(FAILED), str(PROV_FAILED))
    require(not FAILED.exists() and PROV_FAILED.is_dir(), 'Failed chunk not archived correctly')

    # Atomically replace ONLY the approved conditional expression.
    tmp = MAIN.with_name('main_p2c_hotfix_tmp.py')
    require(not tmp.exists(), f'Unexpected temp file {tmp}')
    tmp.write_bytes(patched)
    require(sha(tmp) == hashlib.sha256(patched).hexdigest(), 'New source failed hash check')
    os.replace(tmp, MAIN)
    new_sha = sha(MAIN)
    require(new_sha == hashlib.sha256(patched).hexdigest(), 'Source unexpectedly changed')

    proof = {
        'fix_id': 'numpy_shortest_path_equality_v1',
        'reason': 'Replace ambiguous numpy-array != condition in _get_k_link_cost with array_equal',
        'applied_utc': datetime.now(timezone.utc).isoformat(),
        'generation_main_sha256': old_sha,
        'evaluation_main_sha256': new_sha,
        'original_protocol_lock_sha256': sha(PROV_LOCK),
        'test_dataset_hashes': manifests,
        'no_test_scenes_scored_before_fix': True,
        'dataset_manifests_and_pickles_unchanged': True,
    }
    with ATTESTATION.open('w', encoding='utf-8') as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)

    # Archive the old lock, so the revised queue creates its own audited new lock.
    # The old lock stays available at PROV_LOCK.
    OLD_LOCK.unlink()
    print('P2-C EVALUATION HOTFIX APPLIED AND AUDITED')
    print('original main.py:', old_sha)
    print('evaluation main.py:', new_sha)
    print('preserved 3 datasets; generation source SHA stays in original manifests')
    print('original protocol lock:', PROV_LOCK)
    print('failed chunk:', PROV_FAILED)
    print('evaluation provenance:', ATTESTATION)
    print('NEXT: run python p2c_qwen8_queue.py --mode plan, then --mode all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='Read-only preflight; no changes')
    group.add_argument('--apply', action='store_true', help='Apply the exact one-line fix with backups')
    args = parser.parse_args()
    if args.check:
        _, patched, manifests, old_sha = preflight()
        print('READ-ONLY PREFLIGHT OK: 3 valid datasets, 0 completed scenes, old lock verified')
        print('Main.py original SHA:', old_sha)
        print('Approved one-line patch SHA:', hashlib.sha256(patched).hexdigest())
        print('Original sample SHA:', *(f'{k}:{v["sample_sha256"][:12]}' for k,v in manifests.items()))
    else:
        apply()


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print('HOTFIX REFUSED:', type(exc).__name__, str(exc), file=sys.stderr)
        raise SystemExit(2)
