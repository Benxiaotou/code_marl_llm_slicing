#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-C evaluation-only adapter, preserving the original generator provenance.

The original p1c_runner.py is NOT edited. This adapter checks the original
sample SHA256, configuration SHA256, original generating main.py source SHA256,
AND exact approved patched evaluation main.py source SHA256. Then it delegates
all evaluation and output writing to the unmodified p1c_runner.
"""
from pathlib import Path
import json
import sys

import p1c_runner as base
from p1b.core import ContractError, file_sha256, read_json

PROVENANCE = Path('p2c_runs/qwen8_test_v1/provenance')
NEEDLE_OLD = b'if k_net_new_shortest_paths[idx_flow] != k_net.shortest_paths_node[idx_flow]:'
NEEDLE_NEW = b'if not np.array_equal(k_net_new_shortest_paths[idx_flow], k_net.shortest_paths_node[idx_flow]):'
TEST_IDS = {'p2c_test_seed6601', 'p2c_test_seed6701', 'p2c_test_seed6801'}

original_validator = base.checked_eval_manifest


def approved_test_manifest(root, manifest_path, library, config, *, check_files=True):
    # Original checks: exact manifest and strict training/test independence.
    manifest = original_validator(root, manifest_path, library, config, check_files=False)
    if not check_files:
        return manifest
    if manifest.get('dataset_id') not in TEST_IDS or manifest.get('split') != 'test':
        raise ContractError('Evaluation hotfix is restricted to the frozen P2-C test datasets')
    sample = Path(manifest['sample_file'])
    if not sample.is_absolute():
        sample = root / sample
    if not sample.is_file() or file_sha256(sample) != manifest.get('sample_sha256'):
        raise ContractError('P2-C test sample missing or SHA256 mismatch')
    actual_config = file_sha256(root / config)
    if actual_config != manifest.get('config_sha256'):
        raise ContractError('Environment configuration differs from generation')
    original_main = root / PROVENANCE / 'main_generation.py'
    attestation_path = root / PROVENANCE / 'eval_hotfix_attestation.json'
    if not original_main.is_file() or not attestation_path.is_file():
        raise ContractError('Missing audited P2-C evaluation hotfix provenance')
    attestation = read_json(attestation_path)
    if attestation.get('fix_id') != 'numpy_shortest_path_equality_v1':
        raise ContractError('Unrecognized P2-C evaluation hotfix')
    if (file_sha256(original_main) != manifest.get('main_py_sha256')
            or attestation.get('generation_main_sha256') != manifest.get('main_py_sha256')):
        raise ContractError('Archived generation main.py differs from dataset manifest')
    current_main = root / 'main.py'
    if (file_sha256(current_main) != attestation.get('evaluation_main_sha256')
            or original_main.read_bytes().count(NEEDLE_OLD) != 1
            or current_main.read_bytes() != original_main.read_bytes().replace(
                NEEDLE_OLD, NEEDLE_NEW, 1)):
        raise ContractError('Evaluation main.py is not the single approved path equality fix')
    # Library/model/action checks remain entirely in the original p1c_runner.
    return manifest


def main():
    base.checked_eval_manifest = approved_test_manifest
    base.run(base.get_args())


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ContractError) as exc:
        print('P2-C EVAL ADAPTER STOPPED:', type(exc).__name__, str(exc), file=sys.stderr)
        raise
