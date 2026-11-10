import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import val5601_eval as compat
import val5601_100 as experiment
from p1b.core import ContractError


class ValidationExperimentTests(unittest.TestCase):
    def fixture(self, root):
        provenance = root / compat.PROVENANCE
        provenance.mkdir(parents=True)
        original = b'# fixture\n' + compat.OLD + b'\n    pass\n'
        current = original.replace(compat.OLD, compat.NEW, 1)
        (provenance / 'main_generation.py').write_bytes(original)
        (root / 'main.py').write_bytes(current)
        (root / 'configs_main.yaml').write_text('fixed: true\n')
        sample = root / 'sample.pkl'
        sample.write_bytes(b'not actually deserialized')
        generation_hash = hashlib.sha256(original).hexdigest()
        attestation = {'fix_id': 'numpy_shortest_path_equality_v1',
                       'generation_main_sha256': generation_hash,
                       'evaluation_main_sha256': hashlib.sha256(current).hexdigest()}
        (provenance / 'eval_hotfix_attestation.json').write_text(json.dumps(attestation))
        manifest = {'dataset_id': compat.DATASET_ID, 'split': 'validation', 'seed': 5601,
                    'n_scenes': 200, 'sample_file': str(sample),
                    'sample_sha256': compat.SAMPLE_SHA, 'main_py_sha256': generation_hash,
                    'config_sha256': compat.file_sha256(root / 'configs_main.yaml')}
        actual_digest = compat.file_sha256
        def digest(path):
            return compat.SAMPLE_SHA if Path(path) == sample else actual_digest(path)
        return manifest, digest

    def test_only_exact_audited_environment_change_is_allowed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, digest = self.fixture(root)
            with patch.object(compat, 'file_sha256', side_effect=digest):
                self.assertEqual(compat.verify_environment(root, manifest)['compatibility'],
                                 'exact_numpy_shortest_path_equality_v1')
                with (root / 'main.py').open('ab') as fp:
                    fp.write(b'# unrelated modification\n')
                with self.assertRaises(ContractError):
                    compat.verify_environment(root, manifest)

    def test_final_test_data_and_different_validation_data_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, digest = self.fixture(root)
            with patch.object(compat, 'file_sha256', side_effect=digest):
                for updates in ({'dataset_id': 'p2c_test_seed6601', 'split': 'test'},
                                {'split': 'train'}, {'seed': 7601},
                                {'n_scenes': 100}, {'sample_sha256': 'changed'}):
                    with self.assertRaises(ContractError):
                        compat.verify_environment(root, {**manifest, **updates})

    def test_configuration_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest, digest = self.fixture(root)
            (root / 'configs_main.yaml').write_text('fixed: false\n')
            with patch.object(compat, 'file_sha256', side_effect=digest):
                with self.assertRaisesRegex(ContractError, 'Configuration'):
                    compat.verify_environment(root, manifest)

    def test_non_overwriting_and_partial_chunk_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lock.json'
            experiment.save_new(path, {'fixed': True})
            with self.assertRaises(FileExistsError):
                experiment.save_new(path, {'fixed': False})
            self.assertTrue(json.loads(path.read_text())['fixed'])
            with self.assertRaisesRegex(RuntimeError, 'Incomplete chunk'):
                experiment.read_chunk(Path(folder), 'semantic_balance', 0, 5, {})

    def test_all_methods_have_identical_predeclared_scene_ids(self):
        for method in experiment.METHODS:
            actual = [scene for start, count in experiment.blocks(method)
                      for scene in range(start, start + count)]
            self.assertEqual(actual, list(range(100)))
        self.assertNotIn('teacher_scale', experiment.METHODS)


if __name__ == '__main__':
    unittest.main()
