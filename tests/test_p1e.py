"""Tests for the exact 2x2 historical-IR-data vs risk-instruction ablation."""
import json
import tempfile
import unittest
from pathlib import Path

from p1a.local_obs import AGENTS
from p1b.core import ContractError, read_json, read_jsonl, save_json
from p1c.policy import RetrievalPolicy
from p1c.retrieval import LocalHistory
from p1d.ir import IRHistory, build_ir_overlay
from p1d.policy import IRAblationPolicy, IRRetrievalPolicy
from p1c_runner import get_args, run
from p1c_compare import compare
from p1e_ablation_report import make_report
from test_p1c import sample_library, sample_obs, FakeEnv, FakeClient, FakeTeacher
from test_p1d import fake_joint_source


class TestIRFactorialPrompt(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.libdir = sample_library(self.root)
        self.joint_source = fake_joint_source(self.libdir)
        self.overlaydir = self.root / 'ir'
        build_ir_overlay(self.libdir, self.overlaydir)
        self.lib = LocalHistory(self.libdir)
        self.overlay = IRHistory(self.overlaydir, self.lib)

    def tearDown(self):
        self.tmp.cleanup()

    def variants(self):
        kw = dict(client=FakeClient(), examples=3, seed=1)
        return {
            'icl': RetrievalPolicy(self.lib, 'icl', **kw),
            'ir_prompt_only': IRAblationPolicy(self.lib, 'ir_prompt_only', **kw),
            'ir_data_only': IRAblationPolicy(self.lib, 'ir_data_only', ir_history=self.overlay, **kw),
            'ir_icl': IRRetrievalPolicy(self.lib, self.overlay, **kw),
        }

    def test_exact_factor_independence_for_each_role_and_retrieval(self):
        obs = {role: sample_obs(role, .35) for role in AGENTS}
        decisions = {}
        for method, policy in self.variants().items():
            policy.set_scene(0)
            actions, decisions[method] = policy.act(obs)
            self.assertEqual(set(actions), set(AGENTS))
        for role in AGENTS:
            a = decisions['icl'][role]
            b = decisions['ir_prompt_only'][role]
            c = decisions['ir_data_only'][role]
            d = decisions['ir_icl'][role]
            ids = a['retrieved_ids_for_audit_only']
            for entry in (b, c, d):
                self.assertEqual(entry['retrieved_ids_for_audit_only'], ids)
                self.assertEqual(entry['distances_for_audit_only'], a['distances_for_audit_only'])
            am, bm, cm, dm = (x['messages'] for x in (a, b, c, d))
            # The two content dimensions are EXACTLY separable.
            self.assertEqual(am[0], cm[0])
            self.assertEqual(bm[0], dm[0])
            self.assertEqual(am[1], bm[1])
            self.assertEqual(cm[1], dm[1])
            self.assertNotEqual(am[0], bm[0]); self.assertNotEqual(am[1], cm[1])
            self.assertNotIn('own_historical_ir_margin', json.dumps(bm))
            self.assertNotIn('own_historical_adjustment_cost', json.dumps(bm))
            self.assertNotIn('IR-aware historical reasoning', json.dumps(cm))
            self.assertNotIn('private_offline', json.dumps(dm))
            self.assertNotIn('experience_id', json.dumps(dm))
            self.assertNotIn('cost_sum', json.dumps(dm))
            self.assertNotIn('V_k2_0', json.dumps(dm))
            self.assertIn('own_historical_', json.dumps(cm))
            if role == 'agent_g':
                self.assertIn('own_historical_new_slice_utility', json.dumps(cm))
                self.assertNotIn('own_historical_ir_margin', json.dumps(cm))
            else:
                self.assertIn('own_historical_ir_margin', json.dumps(cm))
                self.assertNotIn('own_historical_new_slice_utility', json.dumps(cm))

    def test_prompt_only_works_without_any_ir_overlay_at_runtime(self):
        # Proves prompt-only does not need any access to historical IR data.
        policy = IRAblationPolicy(self.lib, 'ir_prompt_only', client=FakeClient(), examples=3)
        self.joint_source.unlink()
        for f in self.overlaydir.rglob('*'):
            if f.is_file(): f.unlink()
        policy.set_scene(4)
        actions, rows = policy.act({r: sample_obs(r, .2) for r in AGENTS})
        self.assertEqual(len(actions), 4)
        self.assertFalse('own_historical_' in json.dumps(rows['agent_k1']['messages']))

    def test_data_only_does_not_need_offline_joint_records(self):
        self.joint_source.unlink()
        policy = IRAblationPolicy(self.lib, 'ir_data_only', client=FakeClient(),
                                  ir_history=IRHistory(self.overlaydir, self.lib))
        policy.set_scene(0)
        actions, rows = policy.act({r: sample_obs(r, .35) for r in AGENTS})
        self.assertEqual(len(actions), 4)
        self.assertIn('own_historical_adjustment_cost', json.dumps(rows['agent_k1']['messages']))

    def test_policy_refuses_wrong_factor_or_extra_overlay(self):
        with self.assertRaises(ValueError):
            IRAblationPolicy(self.lib, 'ir_data_only', client=FakeClient())
        with self.assertRaises(ValueError):
            IRAblationPolicy(self.lib, 'ir_prompt_only', client=FakeClient(), ir_history=self.overlay)
        with self.assertRaises(ValueError):
            IRAblationPolicy(self.lib, 'ir_icl', client=FakeClient(), ir_history=self.overlay)
        with self.assertRaises(ValueError):
            IRRetrievalPolicy(self.lib, None, client=FakeClient())


class TestIRFactorialComparison(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.libdir = sample_library(self.root)
        fake_joint_source(self.libdir)
        build_ir_overlay(self.libdir, self.root / 'ir_overlay')
        (self.root / 'configs_main.yaml').write_text('#test')
        (self.root / 'main.py').write_text('#test')
        self.eval_manifest = {'schema': 'p1b_dataset_v1', 'split': 'validation',
                              'dataset_id': 'p1c_val_seed4601',
                              'sample_sha256': 'valhash', 'n_scenes': 8, 'seed': 4601}
        self.methods = ('knn', 'icl', 'ir_prompt_only', 'ir_data_only',
                        'ir_icl', 'mappo', 'random_icl', 'zero')

    def tearDown(self):
        self.tmp.cleanup()

    def run_method(self, method):
        argv = ['--library', 'p1b_libraries/train_v1', '--dataset-manifest', 'unused.json',
                '--method', method, '--start', '0', '--count', '4', '--seed', '1',
                '--output-dir', method]
        if method in ('ir_icl', 'ir_data_only'):
            argv += ['--ir-history', 'ir_overlay']
        args = get_args(argv)
        return run(args, root=self.root, env_override=FakeEnv(),
                   eval_manifest_override=self.eval_manifest,
                   client_override=FakeClient(),
                   teacher_override=(FakeTeacher() if method == 'mappo' else None))

    def populated(self):
        for method in self.methods:
            summary = self.run_method(method)
            self.assertEqual(summary['n_scored'], 4)
        return {method: self.root / method for method in self.methods}

    def test_end_to_end_eight_way_and_report(self):
        dirs = self.populated()
        res, rows = compare(dirs, bootstrap=100, seed=29)
        self.assertEqual(res['n_paired'], 4)
        self.assertEqual(len(rows), 4)
        for tag in ('ir_prompt_only_vs_icl', 'ir_data_only_vs_icl',
                    'ir_icl_vs_ir_prompt_only', 'ir_icl_vs_ir_data_only'):
            self.assertIn(tag, res['pairwise_comparisons'])
        effects = res['ir_ablation_factorial']['per_metric']
        for metric in ('admission', 'cooperative', 'V', 'cost_sum', 'policy_latency_s'):
            self.assertIn('factor_interaction', effects[metric])
        # All stubs return the same actions: performance differences must be zero.
        self.assertAlmostEqual(effects['V']['factor_interaction']['mean_delta'], 0.0)
        for key, has_data, has_risk in (
                ('icl', False, False), ('ir_prompt_only', False, True),
                ('ir_data_only', True, False), ('ir_icl', True, True)):
            manifest = read_json(self.root / key / 'manifest.json')
            if key != 'icl':
                self.assertEqual(manifest['ir_risk_prompt_present'], has_risk)
                self.assertEqual(manifest['ir_history_data_present'], has_data)
        report = make_report(res)
        self.assertIn('IR-Prompt-only', report)
        self.assertIn('IR-Data-only', report)
        self.assertIn('interaction', report.lower())

    def test_comparison_detects_changed_retrieval_or_prompt(self):
        dirs = self.populated()
        records_file = self.root / 'ir_prompt_only' / 'records.jsonl'
        data = list(read_jsonl(records_file))
        original = data[0]['per_role']['agent_k1']['retrieved_ids_for_audit_only']
        data[0]['per_role']['agent_k1']['retrieved_ids_for_audit_only'] = ['forged']
        records_file.write_text(''.join(json.dumps(row) + '\n' for row in data), encoding='utf-8')
        with self.assertRaises(ContractError): compare(dirs, bootstrap=10)
        data[0]['per_role']['agent_k1']['retrieved_ids_for_audit_only'] = original
        data[0]['per_role']['agent_k1']['messages'][1]['content'] += ' tampered!'
        records_file.write_text(''.join(json.dumps(row) + '\n' for row in data), encoding='utf-8')
        with self.assertRaises(ContractError): compare(dirs, bootstrap=10)

    def test_comparison_detects_ir_overlay_mismatch(self):
        dirs = self.populated()
        meta_path = self.root / 'ir_data_only' / 'manifest.json'
        meta = read_json(meta_path)
        meta['ir_history_manifest_sha256'] = 'forged'
        save_json(meta_path, meta)
        with self.assertRaises(ContractError): compare(dirs, bootstrap=10)

    def test_factorial_algebra_with_synthetic_v_delta(self):
        dirs = self.populated()
        path = self.root / 'ir_icl' / 'records.jsonl'
        data = list(read_jsonl(path))
        data[0]['metrics']['V'] += 8.0
        path.write_text(''.join(json.dumps(row) + '\n' for row in data), encoding='utf-8')
        res, rows = compare(dirs, bootstrap=40)
        self.assertAlmostEqual(res['ir_ablation_factorial']['per_metric']['V']
                               ['factor_interaction']['mean_delta'], 2.0)
        self.assertAlmostEqual(res['ir_ablation_factorial']['per_metric']['V']
                               ['full_minus_baseline']['mean_delta'], 2.0)

    def test_argument_and_dataset_checks(self):
        for method, flag in (('ir_prompt_only', 'ir_overlay'), ('ir_data_only', '')):
            args = get_args(['--library', 'p1b_libraries/train_v1',
                             '--dataset-manifest', 'unused.json',
                             '--method', method, '--ir-history', flag,
                             '--output-dir', 'invalid'])
            with self.assertRaises(ValueError):
                run(args, root=self.root, env_override=FakeEnv(),
                    eval_manifest_override=self.eval_manifest,
                    client_override=FakeClient())
        self.eval_manifest['split'] = 'train'
        with self.assertRaises(ContractError): self.run_method('ir_prompt_only')

    def test_report_rejects_incomplete_factorial(self):
        with self.assertRaises(ContractError):
            make_report({'schema': 'p1c_comparison_v1', 'method_means': {}})


if __name__ == '__main__':
    unittest.main()
