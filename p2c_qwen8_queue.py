#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-C eight-method serial GEANT evaluation with immutable held-out datasets.

Use Python from the original project's ``marl`` environment, in its project root.
Standard library only. The 8-method protocol, seeds, chunks, and model are fixed
here, not inferred from validation-set outcomes.

  python p2c_qwen8_queue.py --mode plan
  python p2c_qwen8_queue.py --mode all       # generate + evaluate in one run
  python p2c_qwen8_queue.py --mode prepare
  python p2c_qwen8_queue.py --mode execute
  python p2c_qwen8_queue.py --mode aggregate

Never edits the original environment, policies, bank contents, or prior runs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

VERSION = 'p2c_qwen8_serial_v1_2_numpy_path_hotfix'
SEEDS = (6601, 6701, 6801)
N_SCENES = 512
CHUNK_SIZE = 32
MODEL = 'qwen3-next-80b-a3b-instruct'
TEMPERATURE = 0.1
MAX_TOKENS = 4096
API_TIMEOUT_S = 120
RUN_SEED = 1
OUTPUT_REL = Path('p2c_runs') / 'qwen8_test_v1'
HOTFIX_PROVENANCE = OUTPUT_REL / 'provenance'
OLD_PATH_TEST = b'if k_net_new_shortest_paths[idx_flow] != k_net.shortest_paths_node[idx_flow]:'
NEW_PATH_TEST = b'if not np.array_equal(k_net_new_shortest_paths[idx_flow], k_net.shortest_paths_node[idx_flow]):'
DATASET_DIR = Path('p1b_data')
ROLES = ('agent_k1', 'agent_k2', 'agent_k3', 'agent_g')
RERANK = {
    'pool_size': 20, 'risk_weight': 0.35,
    'max_abs_distance_gap': 0.10, 'max_distance_ratio': 1.35,
    'anchor_nearest': True, 'g_retrieval_unchanged': True,
}
BASE_LIB = 'p1b_libraries/train_seed2601_v1'
BASE_IR = 'p1d_libraries/train_seed2601_ir_v1'
STD500 = 'p2b_libraries/p2b_standard_500_v2'
STD1000 = 'p2b_libraries/p2b_standard_1000_v2'
BAL1000 = 'p2b_libraries/p2b_balanced_1000_v2'
IR_STD500 = 'p2b_ir_libraries/p2b_standard_500_ir_v2'
IR_STD1000 = 'p2b_ir_libraries/p2b_standard_1000_ir_v2'
IR_BAL1000 = 'p2b_ir_libraries/p2b_balanced_1000_ir_v2'
CHECKPOINT = 'marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth'

@dataclass(frozen=True)
class Job:
    tag: str
    method: str
    library: str
    ir: str = ''

# First compare non-LLM methods, then the mechanism and bank-size ablations.
JOBS = (
    Job('mappo', 'mappo', BASE_LIB),
    Job('knn_std1000', 'knn', STD1000),
    Job('icl_std1000', 'icl', STD1000),
    Job('iraware_std1000', 'ir_icl', STD1000, IR_STD1000),
    Job('p2a_std1000', 'ir_rerank', STD1000, IR_STD1000),
    Job('p2a_base152', 'ir_rerank', BASE_LIB, BASE_IR),
    Job('p2a_std500', 'ir_rerank', STD500, IR_STD500),
    Job('p2a_bal1000', 'ir_rerank', BAL1000, IR_BAL1000),
)

# Direction: (new, reference), so delta = new - reference.
PAIRS = (
    ('main_vs_teacher', 'p2a_std1000', 'mappo'),
    ('main_vs_knn', 'p2a_std1000', 'knn_std1000'),
    ('llm_vs_action_copy', 'icl_std1000', 'knn_std1000'),
    ('ir_feedback', 'iraware_std1000', 'icl_std1000'),
    ('rerank_increment', 'p2a_std1000', 'iraware_std1000'),
    ('bank_500_vs_152', 'p2a_std500', 'p2a_base152'),
    ('bank_1000_vs_152', 'p2a_std1000', 'p2a_base152'),
    ('bank_1000_vs_500', 'p2a_std1000', 'p2a_std500'),
    ('quality_standard_vs_balanced_1000', 'p2a_std1000', 'p2a_bal1000'),
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8-sig') as stream:
        return json.load(stream)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def dataset_id(seed: int) -> str:
    return f'p2c_test_seed{seed}'


def dataset_manifest(root: Path, seed: int) -> Path:
    return root / DATASET_DIR / (dataset_id(seed) + '.json')


def source_sample(root: Path, dataset: dict) -> Path:
    raw = Path(dataset['sample_file'])
    return raw if raw.is_absolute() else root / raw


def eval_hotfix_approved(root: Path, expected_original_sha: str) -> bool:
    """Recognize exactly one audited runtime fix; preserve ORIGINAL data provenance."""
    p = root / HOTFIX_PROVENANCE
    original = p / 'main_generation.py'
    proof = p / 'eval_hotfix_attestation.json'
    actual = root / 'main.py'
    if not (original.is_file() and proof.is_file() and actual.is_file()):
        return False
    try:
        attestation = read_json(proof)
        before = original.read_bytes()
        after = actual.read_bytes()
        return (attestation.get('fix_id') == 'numpy_shortest_path_equality_v1'
                and sha256(original) == expected_original_sha
                and attestation.get('generation_main_sha256') == expected_original_sha
                and attestation.get('evaluation_main_sha256') == sha256(actual)
                and before.count(OLD_PATH_TEST) == 1
                and after == before.replace(OLD_PATH_TEST, NEW_PATH_TEST, 1))
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def check_dataset(root: Path, seed: int) -> dict:
    path = dataset_manifest(root, seed)
    fail_if(not path.is_file(), f'Missing test manifest {path}; use --mode prepare')
    manifest = read_json(path)
    for key, expected in (
        ('dataset_id', dataset_id(seed)), ('split', 'test'),
        ('seed', seed), ('n_scenes', N_SCENES),
    ):
        fail_if(manifest.get(key) != expected,
                f'{path}: expected {key}={expected!r}, found {manifest.get(key)!r}')
    sample = source_sample(root, manifest)
    fail_if(not sample.is_file(), f'Test sample file missing: {sample}')
    actual_sha = sha256(sample)
    fail_if(actual_sha != manifest.get('sample_sha256'),
            f'{path}: sample SHA256 differs from manifest; STOP')
    for filename, field in (('main.py', 'main_py_sha256'),
                            ('configs_main.yaml', 'config_sha256')):
        if manifest.get(field):
            if filename == 'main.py' and eval_hotfix_approved(root, manifest[field]):
                continue  # original generation file verified; runtime one-line fix is audited
            fail_if(sha256(root / filename) != manifest[field],
                    f'{filename} differs from original generation version without approved hotfix; STOP')
    return {
        'dataset_id': dataset_id(seed), 'seed': seed, 'manifest': manifest,
        'manifest_sha256': sha256(path),
        'sample_sha256': actual_sha, 'sample_path': str(sample.resolve()),
    }


def overlay_manifest(root: Path, relative: str) -> Path:
    path = root / relative
    choices = [path / 'ir_manifest.json', path / 'manifest.json']
    available = [p for p in choices if p.is_file()]
    fail_if(len(available) != 1,
            f'Expected one IR manifest for {relative}, found {available}')
    return available[0]


def check_sources(root: Path) -> dict:
    for filename in ('p1b_generate.py', 'p1c_runner.py', 'p2c_eval_runner.py',
                     'p1a_runner.py',
                     'main.py', 'configs_main.yaml', CHECKPOINT):
        fail_if(not (root / filename).is_file(), f'Missing {filename} in project root')
    libs = {}
    overlays = {}
    for job in JOBS:
        if job.library not in libs:
            location = root / job.library
            mf = location / 'library_manifest.json'
            fail_if(not mf.is_file(), f'Missing experience bank manifest: {mf}')
            meta = read_json(mf)
            fail_if(bool(meta.get('development_only', False)),
                    f'Experience bank is development-only: {job.library}')
            role_hashes = {}
            for role in ROLES:
                role_file = location / 'role_libraries' / (role + '.jsonl')
                fail_if(not role_file.is_file(), f'Missing role library {role_file}')
                role_hashes[role] = sha256(role_file)
            libs[job.library] = {
                'meta': meta, 'manifest_sha256': sha256(mf),
                'role_sha256': role_hashes,
            }
        if job.ir and job.ir not in overlays:
            mf = overlay_manifest(root, job.ir)
            overlays[job.ir] = {'manifest_sha256': sha256(mf)}
    return {
        'libraries': libs, 'ir_overlays': overlays,
        'checkpoint_sha256': sha256(root / CHECKPOINT),
    }


def code_file_list(root: Path) -> list[Path]:
    candidates = [root / 'p1b_generate.py', root / 'p1c_runner.py',
                  root / 'p1a_runner.py', root / 'main.py',
                  root / 'configs_main.yaml', root / 'p2c_eval_runner.py',
                  Path(__file__).resolve(),
                  root / HOTFIX_PROVENANCE / 'main_generation.py',
                  root / HOTFIX_PROVENANCE / 'eval_hotfix_attestation.json']
    for module in ('p1a', 'p1b', 'p1c', 'p1d', 'p2a'):
        folder = root / module
        if folder.is_dir():
            candidates.extend(folder.rglob('*.py'))
    return sorted(set(path.resolve() for path in candidates))


def fingerprint_sources(root: Path, checked: dict) -> dict:
    source_hashes = {str(path): sha256(path) for path in code_file_list(root)}
    source_hashes[str((root / CHECKPOINT).resolve())] = checked['checkpoint_sha256']
    for job in JOBS:
        libdir = root / job.library
        source_hashes[str((libdir / 'library_manifest.json').resolve())] = checked['libraries'][job.library]['manifest_sha256']
        for role in ROLES:
            role_file = libdir / 'role_libraries' / f'{role}.jsonl'
            source_hashes[str(role_file.resolve())] = checked['libraries'][job.library]['role_sha256'][role]
        if job.ir:
            p = overlay_manifest(root, job.ir)
            source_hashes[str(p.resolve())] = checked['ir_overlays'][job.ir]['manifest_sha256']
            # Freeze IR overlay data too. Preserve all *.jsonl / *.json inputs.
            for path in (root / job.ir).rglob('*'):
                if path.is_file() and path.suffix.lower() in ('.json', '.jsonl', '.csv'):
                    source_hashes[str(path.resolve())] = sha256(path)
    return dict(sorted(source_hashes.items()))


def check_all_datasets(root: Path, checked: dict) -> dict:
    datasets = {seed: check_dataset(root, seed) for seed in SEEDS}
    hashes = [d['sample_sha256'] for d in datasets.values()]
    fail_if(len(set(hashes)) != len(SEEDS),
            'Some final test seeds generated identical sample files')
    for lib in checked['libraries'].values():
        fail_if(lib['meta'].get('training_sample_sha256') in hashes,
                'Test SHA256 equals a training dataset SHA256; STOP')
    # Earlier dev validation SHA, not for use as final test.
    fail_if('7d92e97808f3269dcf11b3275c1cf85f6daae50d1678f412b7c29e347ad79c86' in hashes,
            'Test sample reuses the P2-B val5601 dataset; STOP')
    return datasets


def generation_orphans(root: Path, seed: int) -> list[Path]:
    """Find unverified outputs where the unchanged generator may have stopped.

    p1b_generate.py defaults to data/GEANT/<dataset_id>.pkl, NOT p1b_data.
    A missing JSON manifest means a partial pickle is NOT a valid test dataset.
    """
    name = dataset_id(seed)
    candidates = list((root / DATASET_DIR).glob(name + '*'))
    # The original generator's documented default output and typical temp files.
    candidates += list((root / 'data' / 'GEANT').glob(name + '.pkl*'))
    return sorted(set(p for p in candidates if p.exists()))


def run_generation(root: Path, seed: int) -> None:
    """One generator at a time. Tee to a durable log; never hide failures."""
    cmd = [sys.executable, '-u', 'p1b_generate.py',
           '--name', dataset_id(seed), '--split', 'test',
           '--seed', str(seed), '--count', str(N_SCENES)]
    log = root / OUTPUT_REL / 'logs' / 'generation' / (dataset_id(seed) + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    event(root, 'generation_start', seed=seed, detail='cmd=' + repr(cmd))
    print(f'\nGENERATE {dataset_id(seed)} (one-time, {N_SCENES} scenes)', flush=True)
    print(f'Generation log: {log}', flush=True)
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    proc = None
    with log.open('a', encoding='utf-8') as sink:
        sink.write('\n### GENERATE START ' + now_utc() + ' ###\n')
        sink.write('COMMAND=' + repr(cmd) + '\n')
        sink.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=root, env=env,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8', errors='replace', bufsize=1)
            assert proc.stdout is not None
            line_count = 0
            for line in proc.stdout:
                line_count += 1
                sink.write(line)
                # Keep massive GEANT topology prints in the log instead of the terminal.
                if (line.startswith('Generating ') or 'Traceback' in line or
                    'Error' in line or 'saved' in line.lower()):
                    print(line.rstrip(), flush=True)
                elif line_count % 2000 == 0:
                    print(f'[{dataset_id(seed)}] generator still running; '
                          f'full output in {log}', flush=True)
                if line_count % 100 == 0:
                    sink.flush()
            code = proc.wait()
            sink.write(f'\nEXIT_CODE={code} AT {now_utc()}\n')
            sink.flush()
        except KeyboardInterrupt:
            sink.write('\nINTERRUPTED AT ' + now_utc() + '\n')
            sink.flush()
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            event(root, 'generation_interrupted', seed=seed,
                  detail=f'log={log}; partial outputs must be inspected')
            raise
        finally:
            if proc is not None and proc.stdout is not None:
                proc.stdout.close()
    if code:
        event(root, 'generation_failed', seed=seed,
              detail=f'exit={code}; log={log}')
        raise RuntimeError(f'Generation failed for {dataset_id(seed)}, exit={code}; '
                           f'log={log}. Preserve incomplete data for inspection.')
    event(root, 'generation_process_completed', seed=seed, detail=f'log={log}')


def prepare(root: Path) -> None:
    """Generate each missing dataset, verify each one, then return to the queue."""
    check_sources(root)
    (root / DATASET_DIR).mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        manifest_file = dataset_manifest(root, seed)
        if manifest_file.exists():
            existing = check_dataset(root, seed)
            print(f'REUSE VERIFIED {dataset_id(seed)}, SHA={existing["sample_sha256"]}',
                  flush=True)
            continue
        # A partial pickle is not a checkpoint. Never overwrite it silently.
        orphans = generation_orphans(root, seed)
        fail_if(bool(orphans),
                f'INCOMPLETE generation artifacts for seed {seed}: {orphans}. '
                'No validated manifest. Inspect and archive them before restarting '
                'this seed; p1b_generate.py does not expose scene-level resumption.')
        run_generation(root, seed)
        generated = check_dataset(root, seed)
        event(root, 'generation_verified', seed=seed,
              detail=f'sample_sha256={generated["sample_sha256"]}')
        print('VERIFIED:', dataset_id(seed), generated['sample_sha256'], flush=True)
    checked = check_sources(root)
    check_all_datasets(root, checked)
    print('\nALL THREE TEST DATASETS VERIFIED; model evaluations may begin.',
          flush=True)


def run_all(root: Path) -> None:
    """Single command: sequential dataset generation + 384 evaluation chunks."""
    print('P2-C ALL-IN-ONE: generate missing scenes -> verify -> '
          'run 384 serial chunks -> aggregate', flush=True)
    prepare(root)
    execute(root)  # already aggregates upon full success


def protocol(root: Path, checked: dict, datasets: dict) -> dict:
    return {
        'version': VERSION, 'seeds': list(SEEDS), 'scenes_per_seed': N_SCENES,
        'chunk_size': CHUNK_SIZE, 'qwen_model': MODEL,
        'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS,
        'api_timeout_s': API_TIMEOUT_S, 'run_seed': RUN_SEED,
        'rerank_config': RERANK,
        'jobs': [asdict(job) for job in JOBS],
        'test_datasets': {str(seed): {
            'manifest_sha256': datasets[seed]['manifest_sha256'],
            'sample_sha256': datasets[seed]['sample_sha256'],
        } for seed in SEEDS},
        'source_sha256': fingerprint_sources(root, checked),
    }


def verify_lock(root: Path, intended: dict, *, create: bool) -> None:
    path = root / OUTPUT_REL / 'protocol_lock.json'
    if path.exists():
        fail_if(read_json(path) != intended,
                f'Protocol/data/source changed since first execution: {path}. STOP. Do not mix runs.')
    else:
        fail_if(not create, f'Protocol lock missing: {path}; execute first')
        write_json(path, intended)
        print('PROTOCOL LOCKED:', path, flush=True)


def file_stat_snapshot(paths: list[Path]) -> dict:
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def all_blocks(root: Path):
    for seed in SEEDS:
        for job in JOBS:
            for start in range(0, N_SCENES, CHUNK_SIZE):
                count = min(CHUNK_SIZE, N_SCENES - start)
                out = (root / OUTPUT_REL / dataset_id(seed) / job.tag /
                       f'chunk_{start:04d}_{count:03d}')
                yield seed, job, start, count, out


def command_for(job: Job, seed: int, start: int, count: int, out: Path) -> list[str]:
    cmd = [sys.executable, '-u', 'p2c_eval_runner.py',
           '--library', job.library,
           '--dataset-manifest', str(DATASET_DIR / f'{dataset_id(seed)}.json'),
           '--method', job.method,
           '--start', str(start), '--count', str(count),
           '--seed', str(RUN_SEED), '--on-error', 'abort',
           '--output-dir', str(out)]
    if job.ir:
        cmd += ['--ir-history', job.ir]
    if job.method in ('icl', 'ir_icl', 'ir_rerank'):
        cmd += ['--examples', '3', '--model', MODEL,
                '--temperature', str(TEMPERATURE),
                '--max-tokens', str(MAX_TOKENS),
                '--api-timeout', str(API_TIMEOUT_S)]
    if job.method == 'ir_rerank':
        cmd += ['--risk-pool', '20', '--risk-weight', '0.35',
                '--risk-max-gap', '0.10', '--risk-max-ratio', '1.35']
    return cmd


def parse_bool(value: str) -> int:
    if value in ('True', 'true', '1'):
        return 1
    if value in ('False', 'false', '0'):
        return 0
    raise RuntimeError(f'Invalid boolean value {value!r}')


def parse_float(value: str, label: str) -> float:
    num = float(value)
    fail_if(not math.isfinite(num), f'Nonfinite {label}: {value!r}')
    return num


def verify_chunk(root: Path, seed: int, job: Job, start: int, count: int,
                 out: Path, checked: dict, datasets: dict) -> list[dict]:
    fail_if(not out.is_dir(), f'Missing chunk output: {out}')
    for filename in ('manifest.json', 'summary.json', 'metrics.csv', 'records.jsonl'):
        fail_if(not (out / filename).is_file(), f'Incomplete {out}, missing {filename}')
    manifest = read_json(out / 'manifest.json')
    summary = read_json(out / 'summary.json')
    d = datasets[seed]
    expected = {
        'method': job.method,
        'mock_not_real_llm': False,
        'eval_dataset_id': dataset_id(seed),
        'eval_dataset_split': 'test',
        'eval_dataset_sha256': d['sample_sha256'],
        'eval_generation_seed': seed,
        'scene_ids': list(range(start, start + count)),
        'seed': RUN_SEED,
        'on_error': 'abort',
        'fallback_used': False, 'guard_used': False,
        'task_finetuning': False, 'test_time_global_state_access': False,
        'cross_agent_communication': False,
        'config_sha256': sha256(root / 'configs_main.yaml'),
        'main_py_sha256': sha256(root / 'main.py'),
    }
    for key, expected_value in expected.items():
        fail_if(manifest.get(key) != expected_value,
                f'{out}: manifest {key} not as expected')
    lib = checked['libraries'][job.library]
    for key, expected_value in (
        ('library_training_id', lib['meta']['dataset_id']),
        ('library_train_sha256', lib['meta']['training_sample_sha256']),
        ('library_manifest_sha256', lib['manifest_sha256']),
        ('library_role_sha256', lib['role_sha256']),
    ):
        fail_if(manifest.get(key) != expected_value, f'{out}: wrong {key}')
    if job.ir:
        fail_if(manifest.get('ir_history_manifest_sha256') !=
                checked['ir_overlays'][job.ir]['manifest_sha256'],
                f'{out}: wrong IR overlay manifest SHA')
        fail_if(manifest.get('ir_history_data_present') is not True,
                f'{out}: missing IR-history data in run manifest')
    else:
        fail_if(manifest.get('ir_history_manifest_sha256') is not None,
                f'{out}: unexpected IR overlay')
    if job.method in ('icl', 'ir_icl', 'ir_rerank'):
        for key, value in (('model', MODEL), ('examples', 3),
                           ('temperature', TEMPERATURE),
                           ('max_tokens', MAX_TOKENS), ('stream', True)):
            fail_if(manifest.get(key) != value,
                    f'{out}: wrong Qwen configuration {key}')
    elif job.method == 'knn':
        for key, value in (('model', 'no_llm_knn'), ('examples', 1)):
            fail_if(manifest.get(key) != value, f'{out}: {key} differs')
    elif job.method == 'mappo':
        fail_if(manifest.get('model') != 'frozen_mappo',
                f'{out}: wrong MAPPO model')
        fail_if(manifest.get('checkpoint_sha256') != checked['checkpoint_sha256'],
                f'{out}: wrong frozen teacher checkpoint SHA')
    if job.method == 'ir_rerank':
        fail_if(manifest.get('policy') != 'qwen_ir_risk_reranked_coop_icl_local_p2a_v1',
                f'{out}: unexpected rerank policy label')
        for key, value in RERANK.items():
            fail_if((manifest.get('p2a_rerank_config') or {}).get(key) != value,
                    f'{out}: wrong rerank parameter {key}')
    for key, value in (
        ('n_attempted', count), ('n_scored', count),
        ('n_failed_policy', 0), ('fallback_rate', 0.0),
        ('eval_dataset_id', dataset_id(seed)), ('split', 'test'),
        ('method', job.method), ('mock_not_real_llm', False),
    ):
        fail_if(summary.get(key) != value,
                f'{out}: failed/incomplete summary field {key}: {summary.get(key)!r}')
    with (out / 'metrics.csv').open('r', encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fail_if(len(rows) != count, f'{out}: {len(rows)} metric rows, expected {count}')
    for index, row in enumerate(rows):
        sid = start + index
        fail_if(int(row['original_scene_id']) != sid or row.get('status') != 'ok'
                or bool(row.get('failed_role')),
                f'{out}: failed/out-of-order metric at scene {sid}')
        for col in ('g_done', 'cooperative_valid', 'individual_rationality'):
            parse_bool(row[col])
        for col in ('V', 'V_g', 'V_k1', 'V_k2', 'V_k3',
                    'cost_sum', 'policy_latency_s'):
            parse_float(row[col], col)
    with (out / 'records.jsonl').open('r', encoding='utf-8-sig') as stream:
        line_num = 0
        for line in stream:
            if not line.strip():
                continue
            rec = json.loads(line)
            fail_if(rec.get('original_scene_id') != start + line_num
                    or rec.get('status') != 'ok'
                    or rec.get('fallback_used', False),
                    f'{out}: invalid raw record at row {line_num}')
            line_num += 1
    fail_if(line_num != count, f'{out}: {line_num} raw records, expected {count}')
    for output_name, csv_name, boolean in (
        ('acceptance_rate_scored_only', 'g_done', True),
        ('cooperative_valid_rate_scored_only', 'cooperative_valid', True),
        ('V_scored_only', 'V', False),
        ('cost_sum_scored_only', 'cost_sum', False),
    ):
        mean = sum(parse_bool(row[csv_name]) if boolean else float(row[csv_name])
                   for row in rows) / count
        fail_if(not math.isclose(mean, float(summary[output_name]),
                                 rel_tol=1e-9, abs_tol=1e-8),
                f'{out}: CSV vs summary mismatch on {output_name}')
    return rows


def event(root: Path, action: str, *, job: Job | None = None,
          seed: int | None = None, start: int | None = None, detail: str = '') -> None:
    path = root / OUTPUT_REL / 'queue_events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {'utc': now_utc(), 'event': action,
             'job': job.tag if job is not None else None,
             'seed': seed, 'start': start, 'detail': detail}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + '\n')


def load_context(root: Path, *, allow_lock_creation: bool) -> tuple[dict, dict, dict]:
    checked = check_sources(root)
    datasets = check_all_datasets(root, checked)
    intended = protocol(root, checked, datasets)
    verify_lock(root, intended, create=allow_lock_creation)
    return checked, datasets, intended


def execute(root: Path) -> None:
    checked, datasets, intended = load_context(root, allow_lock_creation=True)
    # This is a cheap within-session sentinel. Heavy SHA-256 checks are performed
    # before execution, again on resume, and by the original runner per chunk.
    protected_paths = [Path(p) for p in intended['source_sha256']]
    protected_paths.extend(dataset_manifest(root, seed) for seed in SEEDS)
    protected_paths.extend(Path(datasets[seed]['sample_path']) for seed in SEEDS)
    quick_before = file_stat_snapshot(protected_paths)
    batches = list(all_blocks(root))
    print(f'P2-C SERIAL: {len(batches)} chunks; 8 methods × 3 seeds × 512 scenes', flush=True)
    print('Qwen model:', MODEL, '| model calls:', 6 * len(SEEDS) * N_SCENES * 4, flush=True)
    for batch_i, (seed, job, start, count, out) in enumerate(batches, start=1):
        fail_if(file_stat_snapshot(protected_paths) != quick_before,
                'Protected source/library/sample file changed while queue running: STOP')
        tag = f'{dataset_id(seed)}/{job.tag}/{start:04d}-{start+count-1:04d}'
        if out.exists():
            verify_chunk(root, seed, job, start, count, out, checked, datasets)
            print(f'[{batch_i}/{len(batches)}] SKIP VERIFIED {tag}', flush=True)
            continue
        cmd = command_for(job, seed, start, count, out)
        log = root / OUTPUT_REL / 'logs' / dataset_id(seed) / job.tag / f'{start:04d}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        event(root, 'start', job=job, seed=seed, start=start,
              detail=f'cmd={cmd!r}, log={log}')
        print(f'[{batch_i}/{len(batches)}] RUN {tag}', flush=True)
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        with log.open('a', encoding='utf-8') as sink:
            sink.write(f'\n### START {now_utc()} ###\nCOMMAND={cmd!r}\n')
            sink.flush()
            with subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding='utf-8', errors='replace', bufsize=1) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sink.write(line)
                    sink.flush()
                    if ('FAILED' in line or 'Traceback' in line or 'Error' in line
                        or line.startswith('Results:')
                        or (line.startswith('method=') and (
                            f' scene={start} ' in line or
                            f' scene={start+count-1} ' in line))):
                        print(line.rstrip(), flush=True)
                rc = proc.wait()
        if rc != 0:
            event(root, 'process_failed', job=job, seed=seed, start=start,
                  detail=f'return_code={rc}, log={log}')
            raise RuntimeError(f'STOP at {tag}, exit={rc}. Partial files PRESERVED: {out}. '
                               f'Log: {log}. See README for safe manual repair.')
        try:
            verify_chunk(root, seed, job, start, count, out, checked, datasets)
        except Exception as exc:
            event(root, 'validation_failed', job=job, seed=seed, start=start,
                  detail=str(exc))
            raise
        event(root, 'completed', job=job, seed=seed, start=start)
        print(f'[{batch_i}/{len(batches)}] PASSED {tag}', flush=True)
    print(f'\nSUCCESS: ALL {len(batches)} chunks fully scored and verified.', flush=True)
    aggregate(root, checked=checked, datasets=datasets)


def export_csv(path: Path, rows: list[dict]) -> None:
    fail_if(not rows, f'Refusing to write empty CSV {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    print('WROTE:', path, flush=True)


def quantiles(data: list[float]) -> tuple[float, float]:
    sorted_data = sorted(data)
    return (sorted_data[int(0.025 * len(data))],
            sorted_data[min(len(data) - 1, int(0.975 * len(data)))])


def paired_bootstrap(pairs: list[tuple[dict, dict]], *, reps: int, seed: int,
                     stratified: bool) -> dict:
    """Paired case-resampling CI; stratifies pooled tests by preset generator seed."""
    rng = random.Random(seed)
    groups: dict[int, list[tuple[dict, dict]]] = {}
    for a, b in pairs:
        groups.setdefault(a['dataset_seed'], []).append((a, b))
    sample_groups = list(groups.values()) if stratified else [pairs]
    diffs = {label: [] for label in ('coop_pp', 'cost', 'V')}
    denom = len(pairs)
    for _ in range(reps):
        coop = cost = utility = 0.0
        for group in sample_groups:
            size = len(group)
            for _ in range(size):
                a, b = group[rng.randrange(size)]
                coop += a['coop'] - b['coop']
                cost += a['cost'] - b['cost']
                utility += a['V'] - b['V']
        diffs['coop_pp'].append(100.0 * coop / denom)
        diffs['cost'].append(cost / denom)
        diffs['V'].append(utility / denom)
    return {name + '_ci95': quantiles(values) for name, values in diffs.items()}


def aggregate(root: Path, *, checked: dict | None = None,
              datasets: dict | None = None) -> None:
    if checked is None or datasets is None:
        checked, datasets, _ = load_context(root, allow_lock_creation=False)
    all_rows = []
    for seed, job, start, count, out in all_blocks(root):
        fail_if(not out.exists(), f'Missing chunk {out}. Resume --mode execute first.')
        rows = verify_chunk(root, seed, job, start, count, out, checked, datasets)
        for row in rows:
            all_rows.append({
                'dataset_seed': seed, 'dataset_id': dataset_id(seed),
                'scene_id': int(row['original_scene_id']),
                'job': job.tag, 'method': job.method,
                'accept': parse_bool(row['g_done']),
                'coop': parse_bool(row['cooperative_valid']),
                'ir_valid': parse_bool(row['individual_rationality']),
                'V': float(row['V']), 'V_g': float(row['V_g']),
                'V_k1': float(row['V_k1']), 'V_k2': float(row['V_k2']),
                'V_k3': float(row['V_k3']), 'cost': float(row['cost_sum']),
                'latency_s': float(row['policy_latency_s']),
            })
    fail_if(len(all_rows) != len(JOBS) * len(SEEDS) * N_SCENES,
            'Wrong total number of scored records')
    by_index = {(r['dataset_seed'], r['scene_id'], r['job']): r for r in all_rows}
    fail_if(len(by_index) != len(all_rows), 'Duplicate (seed, scene_id, method) pair')
    summary = []
    for seed in (*SEEDS, 'pooled'):
        for job in JOBS:
            rows = [r for r in all_rows
                    if r['job'] == job.tag and (seed == 'pooled' or r['dataset_seed'] == seed)]
            fail_if(len(rows) != (N_SCENES * len(SEEDS) if seed == 'pooled' else N_SCENES),
                    f'Incomplete summary data for {job.tag}: seed={seed}')
            n = len(rows)
            summary.append({
                'seed': seed, 'method': job.method, 'job': job.tag, 'n_scenes': n,
                'acceptance_rate': sum(r['accept'] for r in rows) / n,
                'cooperative_valid_rate': sum(r['coop'] for r in rows) / n,
                'ir_valid_rate': sum(r['ir_valid'] for r in rows) / n,
                'V_mean': sum(r['V'] for r in rows) / n,
                'V_g_mean': sum(r['V_g'] for r in rows) / n,
                'V_k1_mean': sum(r['V_k1'] for r in rows) / n,
                'V_k2_mean': sum(r['V_k2'] for r in rows) / n,
                'V_k3_mean': sum(r['V_k3'] for r in rows) / n,
                'cost_mean': sum(r['cost'] for r in rows) / n,
                'cost_mean_coop_success': (sum(r['cost'] for r in rows if r['coop']) /
                                           sum(r['coop'] for r in rows)
                                           if any(r['coop'] for r in rows) else ''),
                'latency_s_mean': sum(r['latency_s'] for r in rows) / n,
            })
    paired = []
    for seed in (*SEEDS, 'pooled'):
        scene_keys = [(x, i) for x in (SEEDS if seed == 'pooled' else (seed,))
                      for i in range(N_SCENES)]
        for label, new, reference in PAIRS:
            couples = [(by_index[(s, i, new)], by_index[(s, i, reference)])
                       for s, i in scene_keys]
            n = len(couples)
            wins = sum(a['coop'] and not b['coop'] for a, b in couples)
            losses = sum(b['coop'] and not a['coop'] for a, b in couples)
            delta = lambda field: sum(a[field] - b[field] for a, b in couples) / n
            ci = paired_bootstrap(couples, reps=2000,
                                  seed=20261102 + len(paired),
                                  stratified=(seed == 'pooled'))
            paired.append({
                'seed': seed, 'comparison': label,
                'new_method': new, 'reference_method': reference,
                'n_paired': n, 'coop_wins': wins, 'coop_losses': losses,
                'coop_delta_pp': 100 * delta('coop'),
                'coop_delta_ci95_lower_pp': ci['coop_pp_ci95'][0],
                'coop_delta_ci95_upper_pp': ci['coop_pp_ci95'][1],
                'accept_delta_pp': 100 * delta('accept'),
                'cost_delta_mean': delta('cost'),
                'cost_delta_ci95_lower': ci['cost_ci95'][0],
                'cost_delta_ci95_upper': ci['cost_ci95'][1],
                'V_delta_mean': delta('V'),
                'V_delta_ci95_lower': ci['V_ci95'][0],
                'V_delta_ci95_upper': ci['V_ci95'][1],
            })
    dest = root / OUTPUT_REL
    export_csv(dest / 'p2c_all_scene_metrics.csv', all_rows)
    export_csv(dest / 'p2c_summary_by_seed_and_pooled.csv', summary)
    export_csv(dest / 'p2c_paired_comparisons.csv', paired)
    print(f'DONE: {len(SEEDS)} seeds, {len(JOBS)} methods, paired/stratified scene-bootstrap CIs.', flush=True)
    print('NOTE: bootstrap CIs condition on sampled scene distribution and do not measure Qwen sampling variability.', flush=True)


def plan(root: Path) -> None:
    checked = check_sources(root)
    print('EXPERIMENT:', VERSION)
    print('Model:', MODEL, 'temperature=', TEMPERATURE, 'max_tokens=', MAX_TOKENS)
    print('Datasets:', ', '.join(dataset_id(seed) for seed in SEEDS),
          '| scenes/seed:', N_SCENES)
    print('Jobs:', ', '.join(job.tag for job in JOBS))
    print('Chunks:', len(list(all_blocks(root))),
          '| Qwen scene decisions:', 6 * len(SEEDS) * N_SCENES,
          '| Qwen role requests:', 6 * len(SEEDS) * N_SCENES * 4)
    for seed in SEEDS:
        if dataset_manifest(root, seed).exists():
            info = check_dataset(root, seed)
            print('VERIFIED EXISTING:', dataset_id(seed), info['sample_sha256'])
        else:
            leftovers = generation_orphans(root, seed)
            if leftovers:
                print('BLOCKED: unverified interrupted generation artifacts:', leftovers)
            else:
                print('NEED TO GENERATE:', dataset_id(seed))
    lock = root / OUTPUT_REL / 'protocol_lock.json'
    if lock.exists():
        print('NOTE: protocol lock exists; --mode execute will require an exact match:', lock)
    print('PLAN OK. No dataset generation, model calls, or output writes performed.')


def main() -> int:
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--mode', choices=('plan', 'all', 'prepare', 'execute', 'aggregate'),
                     default='plan')
    args = arg.parse_args()
    root = Path.cwd().resolve()
    fail_if(not (root / 'p1c_runner.py').is_file(),
            'Launch in path/to/code_marl_llm_slicing with conda marl activated')
    if args.mode == 'plan':
        plan(root)
    elif args.mode == 'all':
        run_all(root)
    elif args.mode == 'prepare':
        prepare(root)
    elif args.mode == 'execute':
        execute(root)
    else:
        aggregate(root)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nINTERRUPTED. Completed datasets and evaluation chunks are preserved. '
              'An unfinished dataset may not be resumable; inspect the files.',
              file=sys.stderr, flush=True)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError,
            json.JSONDecodeError) as error:
        print('\nSTOPPED:', type(error).__name__, str(error),
              file=sys.stderr, flush=True)
        raise SystemExit(2)
