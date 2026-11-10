"""Strict evaluation adapter for the existing P2-B validation data only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import p1c_runner as runner
from p1b.core import ContractError, file_sha256, read_json

DATASET_ID = 'p2b_val_seed5601'
SAMPLE_SHA = '7d92e97808f3269dcf11b3275c1cf85f6daae50d1678f412b7c29e347ad79c86'
PROVENANCE = Path('p2c_runs/qwen8_test_v1/provenance')
OLD = b'if k_net_new_shortest_paths[idx_flow] != k_net.shortest_paths_node[idx_flow]:'
NEW = b'if not np.array_equal(k_net_new_shortest_paths[idx_flow], k_net.shortest_paths_node[idx_flow]):'
ORIGINAL_VALIDATOR = runner.checked_eval_manifest


def verify_environment(root: Path, manifest: dict) -> dict:
    if (manifest.get('dataset_id') != DATASET_ID or manifest.get('split') != 'validation'
            or manifest.get('seed') != 5601 or manifest.get('n_scenes') != 200
            or manifest.get('sample_sha256') != SAMPLE_SHA):
        raise ContractError('Only the fixed P2-B val5601 validation dataset is allowed')
    sample = Path(manifest['sample_file'])
    if not sample.is_absolute():
        sample = root / sample
    if not sample.is_file() or file_sha256(sample) != SAMPLE_SHA:
        raise ContractError('val5601 sample missing or changed')
    if file_sha256(root / 'configs_main.yaml') != manifest['config_sha256']:
        raise ContractError('Configuration differs from dataset generation')
    archived = root / PROVENANCE / 'main_generation.py'
    attestation = read_json(root / PROVENANCE / 'eval_hotfix_attestation.json')
    original = archived.read_bytes()
    current = (root / 'main.py').read_bytes()
    if (attestation.get('fix_id') != 'numpy_shortest_path_equality_v1'
            or file_sha256(archived) != manifest['main_py_sha256']
            or attestation.get('generation_main_sha256') != manifest['main_py_sha256']
            or attestation.get('evaluation_main_sha256') != file_sha256(root / 'main.py')
            or original.count(OLD) != 1 or original.replace(OLD, NEW, 1) != current):
        raise ContractError('Environment is not exactly the previously audited NumPy equality fix')
    return {'sample': str(sample), 'sample_sha256': SAMPLE_SHA,
            'generation_main_sha256': manifest['main_py_sha256'],
            'evaluation_main_sha256': file_sha256(root / 'main.py'),
            'compatibility': 'exact_numpy_shortest_path_equality_v1'}


def checked_manifest(root, manifest_path, library, config, *, check_files=True):
    if config != 'configs_main.yaml':
        raise ContractError('Only the unchanged project configuration is permitted')
    manifest = ORIGINAL_VALIDATOR(root, manifest_path, library, config, check_files=False)
    verify_environment(Path(root), manifest)
    return manifest


def main():
    args = runner.get_args()
    if args.start < 0 or args.start + args.count > 100:
        raise ContractError('This experiment is restricted to scene IDs 0..99')
    runner.checked_eval_manifest = checked_manifest
    teacher = None
    if args.method == 'mappo':
        from p1b_collect import _prepare_teacher_conf
        from p1b.teacher import P0FrozenTeacher
        manifest = read_json(ROOT / args.dataset_manifest)
        checked = verify_environment(ROOT, manifest)
        conf = _prepare_teacher_conf(ROOT, args.config)
        # Actor-construction environment reads the same existing 200 scenes,
        # avoiding the unrelated default 600-scene data. No scenes are generated.
        conf.p0_sample_file = checked['sample']
        conf.p0_sample_cnt = 200
        runtime = (ROOT / args.output_dir / 'teacher_runtime').resolve()
        # Keep runtime files outside the chunk directory until the runner creates it.
        runtime = runtime.parent.parent / 'teacher_runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        for attr in ('path_reward_save', 'path_env_save', 'path_model_save',
                     'path_model_train_log', 'path_F_save', 'log_dir', 'model_dir'):
            target = runtime / attr
            target.mkdir(parents=True, exist_ok=True)
            setattr(conf, attr, str(target))
        teacher = P0FrozenTeacher(conf, ROOT / args.checkpoint)
    runner.run(args, root=ROOT, teacher_override=teacher)


if __name__ == '__main__':
    main()
