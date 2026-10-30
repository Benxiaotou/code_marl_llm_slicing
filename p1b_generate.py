"""Generate NEW independent GEANT sample pickle using the ORIGINAL environment.

No changes to main.py. No training or MAPPO model calls. The generator uses
original CustomMultiAgentEnv.__init__'s existing sample-generation branch.
"""
import argparse
import datetime
import os
import random
from pathlib import Path

import numpy as np

from p1b.core import file_sha256, save_json


def get_args(argv=None):
    p = argparse.ArgumentParser(description='P1-B independent GEANT sample generation')
    p.add_argument('--name', required=True, help='Unique dataset ID, e.g. p1b_train_seed2601')
    p.add_argument('--count', required=True, type=int)
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    p.add_argument('--config', default='configs_main.yaml')
    p.add_argument('--sample-file', default='', help='Default: data/GEANT/<name>.pkl')
    p.add_argument('--manifest-dir', default='p1b_data')
    return p.parse_args(argv)


def generate(args, root=None, env_class_override=None):
    root = Path(root or Path.cwd()).resolve()
    if args.count < 1 or args.seed < 0:
        raise ValueError('count must be positive and seed nonnegative')
    if not args.name or not all(c.isascii() and (c.isalnum() or c in '_-') for c in args.name):
        raise ValueError('name must contain only ASCII letters, digits, underscore or dash')
    name = args.name
    if args.name.lower().startswith('geant_3_12_600'):
        raise ValueError('Do not use existing original 600-sample filename')
    sample_path = (root / (args.sample_file or ('data/GEANT/'+name+'.pkl'))).resolve()
    manifest_path = (root / args.manifest_dir / (name+'.json')).resolve()
    original_file = (root / 'data/GEANT/GEANT_3_12_600_sample_save.pkl').resolve()
    if sample_path == original_file:
        raise ValueError('Refusing to write the original GEANT 600-scene dataset')
    if sample_path.exists() or manifest_path.exists():
        raise FileExistsError('Dataset/manifest already exists. Choose a NEW name and path')
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # The original generator uses NumPy global RNG; seed before env init.
    random.seed(args.seed)
    np.random.seed(args.seed)
    if env_class_override is None:
        from p1a_runner import _config, _compat_pickle_main_classes, make_mini_env_type
        main, conf = _config(root, root / args.config)
        _compat_pickle_main_classes(main)
        cls = make_mini_env_type(main.CustomMultiAgentEnv, sample_path, args.count)
    else:
        cls, conf = env_class_override, None
    print('Generating {} {} samples with seed {}: {}'.format(
        args.count, args.split, args.seed, sample_path), flush=True)
    try:
        env = cls(conf)
        if not sample_path.is_file():
            raise RuntimeError('Original environment did not generate the requested dataset')
        if env.sample_cnt != args.count or len(env.sample_dict) != args.count:
            raise RuntimeError('Generated dataset size inconsistent')
        if set(env.sample_dict) != set(range(args.count)):
            raise RuntimeError('Generated dataset keys are not 0..count-1')
    except Exception:
        # Keep a failed generation file for forensic diagnosis; never silently reuse it.
        raise
    result = {
        'schema': 'p1b_dataset_v1', 'dataset_id': name, 'split': args.split,
        'n_scenes': args.count, 'seed': args.seed, 'generator': 'original_GEANT_CustomMultiAgentEnv',
        'sample_file': str(sample_path), 'sample_sha256': file_sha256(sample_path),
        'main_py_sha256': file_sha256(root / 'main.py') if (root / 'main.py').is_file() else None,
        'config_sha256': file_sha256(root / args.config) if (root / args.config).is_file() else None,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z'),
        'source_note': 'Independent generated samples, not original 600-scene development data',
    }
    save_json(manifest_path, result)
    print('Dataset manifest: '+str(manifest_path), flush=True)
    return result


if __name__ == '__main__':
    generate(get_args())
