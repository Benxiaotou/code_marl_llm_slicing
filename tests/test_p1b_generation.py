import argparse
import json
import tempfile
import unittest
from pathlib import Path

from p1b_generate import generate
from p1b.core import read_json


class FakeGenerator:
    def __init__(self, config):
        # The actual GEANT initializer creates and serializes a sample_dict.
        self.sample_cnt=4
        self.sample_dict={k:tuple() for k in range(4)}
        FakeGenerator.out.write_bytes(b'fake-but-reproducible-dataset')


class GenerationTests(unittest.TestCase):
    def test_new_data_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            FakeGenerator.out = root/'data/GEANT/new_seed11.pkl'
            args=argparse.Namespace(name='new_seed11',count=4,seed=11,
                                    split='train',config='configs_main.yaml',
                                    sample_file='data/GEANT/new_seed11.pkl',
                                    manifest_dir='p1b_data')
            m=generate(args,root=root,env_class_override=FakeGenerator)
            self.assertEqual(m['dataset_id'],'new_seed11')
            self.assertEqual(m['split'],'train')
            self.assertEqual(m['n_scenes'],4)
            self.assertEqual(len(m['sample_sha256']),64)
            self.assertEqual(read_json(root/'p1b_data/new_seed11.json')['seed'],11)
            with self.assertRaises(FileExistsError):
                generate(args,root=root,env_class_override=FakeGenerator)

    def test_reject_source_original_filename(self):
        with tempfile.TemporaryDirectory() as t:
            a=argparse.Namespace(name='new_seed22',count=4,seed=11,split='train',
                                  config='configs_main.yaml',
                                  sample_file='data/GEANT/GEANT_3_12_600_sample_save.pkl',
                                  manifest_dir='p1b_data')
            with self.assertRaises(ValueError):generate(a,root=t,env_class_override=FakeGenerator)


if __name__=='__main__':unittest.main()
