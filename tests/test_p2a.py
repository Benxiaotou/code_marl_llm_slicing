"""P2-A contracts: provenance, role locality, reranking, paired comparisons."""
import json
import tempfile
import unittest
from pathlib import Path

from p1a.local_obs import AGENTS
from p1b.core import ContractError, read_json, read_jsonl
from p1c.retrieval import LocalHistory
from p1d.ir import IRHistory, build_ir_overlay
from p1d.policy import IRRetrievalPolicy
from p2a.rerank import IRRiskReranker, RerankConfig, _midrank_percentiles
from p2a.policy import IRRiskRerankPolicy
from p2a_preview import preview
from p1c_runner import get_args, run
from p2a_compare import compare_with_rerank
from test_p1c import sample_library, sample_obs, FakeEnv, FakeClient, FakeTeacher
from test_p1d import fake_joint_source


class RerankFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.libdir = sample_library(self.root, n=8)
        self.joint = fake_joint_source(self.libdir)
        self.overlaydir = self.root / 'history'
        build_ir_overlay(self.libdir, self.overlaydir)
        self.library = LocalHistory(self.libdir)
        self.overlay = IRHistory(self.overlaydir, self.library)
        self.obs = {role: sample_obs(role, .34) for role in AGENTS}
    def tearDown(self):
        self.tmp.cleanup()


class TestP2ALocalSelection(RerankFixtures):
    def test_training_percentiles_handle_ties(self):
        a = _midrank_percentiles([10, 10, 10])
        self.assertEqual(a.tolist(), [0.5, 0.5, 0.5])
        a = _midrank_percentiles([1, 2, 2, 8])
        self.assertEqual(a.tolist(), [0.0, 0.5, 0.5, 1.0])

    def test_exact_anchor_and_g_unchanged(self):
        ranker = IRRiskReranker(self.library, self.overlay)
        for role in AGENTS:
            base = self.library.query(role, self.obs[role], top_k=3)
            selected, trace = ranker.query(role, self.obs[role], top_k=3)
            self.assertEqual(len(selected), 3)
            self.assertEqual(selected[0]['experience_id'], base[0]['experience_id'])
            self.assertEqual(trace['selected_ids'][0], trace['baseline_top_ids'][0])
            self.assertEqual(len(set(trace['selected_ids'])),3)
            self.assertLessEqual(len(trace['candidate_top_ids']), len(self.library.cases[role]))
            if role == 'agent_g':
                self.assertEqual(trace['selected_ids'], trace['baseline_top_ids'])
                self.assertEqual(selected, base)

    def test_historical_cost_and_margin_risks_use_own_role_only(self):
        r = IRRiskReranker(self.library, self.overlay)
        self.assertNotIn('agent_g', r.training_risk)
        for role in AGENTS[:3]:
            arr = list(r.training_risk[role].values())
            self.assertTrue(all(0 <= x <= 1 for x in arr))
        # Deleting the OFFLINE joint source must not affect runtime reranking.
        self.joint.unlink()
        r2 = IRRiskReranker(self.library, IRHistory(self.overlaydir, self.library))
        self.assertEqual(r.training_risk, r2.training_risk)

    def test_prompt_content_is_same_template_only_examples_change(self):
        old = IRRetrievalPolicy(self.library, self.overlay, client=FakeClient(), examples=3)
        new = IRRiskRerankPolicy(self.library, self.overlay, client=FakeClient(), examples=3)
        old.set_scene(0); new.set_scene(0)
        _, a = old.act(self.obs)
        _, b = new.act(self.obs)
        for role in AGENTS:
            self.assertEqual(a[role]['messages'][0], b[role]['messages'][0])
            self.assertEqual(a[role]['retrieved_ids_for_audit_only'][0],
                             b[role]['retrieved_ids_for_audit_only'][0])
            tr = b[role]['p2a_rerank_trace_for_audit_only']
            self.assertEqual(tr['selected_ids'], b[role]['retrieved_ids_for_audit_only'])
            self.assertNotIn('global_state', json.dumps(b[role]['messages']))
            self.assertNotIn('cost_sum', json.dumps(b[role]['messages']))
            if role == 'agent_g':
                self.assertEqual(a[role]['messages'], b[role]['messages'])
                self.assertEqual(a[role]['distances_for_audit_only'],
                                 b[role]['distances_for_audit_only'])
            else:
                self.assertIn('own_historical_adjustment_cost', json.dumps(b[role]['messages']))

    def test_reranking_actually_uses_history_risk_to_swap_analogues(self):
        # A controlled artificial training library: all candidates are LOCAL
        # neighbours, but the safer historical cases are ranks 4 and 5.
        from types import SimpleNamespace
        ids = ['fake_train::{:04d}'.format(i) for i in range(6)]
        candidates = [{'experience_id': cid, 'distance': .3 + .01 * idx,
                       'ratios': [.3,.3,.3,.3], 'summary': {}}
                      for idx, cid in enumerate(ids)]
        class FakeLocal:
            cases = {role: candidates for role in AGENTS}
            def query(self, role, observation, top_k=3):
                return candidates[:top_k]
        overlay = {cid: {'own_historical_adjustment_cost': float(10-i),
                         'own_historical_ir_margin': float(1 + i*10),
                         'own_historical_relative_ir_margin': float(.01 + i*.1)}
                   for i, cid in enumerate(ids)}
        train = SimpleNamespace(rows={role: overlay for role in AGENTS[:3]})
        ranker = IRRiskReranker(FakeLocal(), train, config=RerankConfig(
            pool_size=6, risk_weight=.85, max_abs_distance_gap=.1,
            max_distance_ratio=1.5))
        result, audit = ranker.query('agent_k1', [], top_k=3)
        self.assertEqual(result[0]['experience_id'], ids[0])
        self.assertEqual([r['experience_id'] for r in result],
                         [ids[0], ids[5], ids[4]])
        self.assertTrue(audit['changed_vs_nearest'])
        self.assertEqual(audit['eligible_count'],6)

    def test_extreme_parameters_fallback_and_validated(self):
        for kwargs in ({'pool_size': 2}, {'risk_weight': -0.1},
                       {'risk_weight': 1.1}, {'max_abs_distance_gap': 0},
                       {'max_distance_ratio': 1}):
            with self.assertRaises(ContractError): RerankConfig(**kwargs)
        cfg = RerankConfig(pool_size=3, risk_weight=1, max_abs_distance_gap=.001,
                           max_distance_ratio=1.0001)
        ranker = IRRiskReranker(self.library, self.overlay, config=cfg)
        selected, meta = ranker.query('agent_k2', self.obs['agent_k2'], top_k=3)
        self.assertEqual(len(selected),3)
        self.assertEqual(meta['eligible_count'],3)
        with self.assertRaises(ContractError):ranker.query('agent_k2',self.obs['agent_k2'],top_k=4)

    def test_preview_does_not_call_step_or_model(self):
        env = FakeEnv()
        def fail_step(*unused):
            raise AssertionError('preview must never call GEANT step')
        env.step = fail_step
        summary, rows = preview(self.library, self.overlay, env, range(4), examples=3)
        self.assertEqual(summary['n_scenes'],4)
        self.assertEqual(len(rows),16)
        self.assertEqual(summary['changes_by_role']['agent_g'],0)
        self.assertEqual(len(summary['nearest_distance_quantiles_by_role']),4)


class TestP2APairedComparison(RerankFixtures):
    def setUp(self):
        super().setUp()
        (self.root / 'configs_main.yaml').write_text('#fake')
        (self.root / 'main.py').write_text('#fake')
        self.eval_manifest = {'schema':'p1b_dataset_v1','split':'validation',
                              'dataset_id':'p1c_val_seed4601','sample_sha256':'test_val_hash',
                              'n_scenes':8,'seed':4601}
        self.methods = ('knn','icl','ir_icl','ir_rerank','mappo','ir_prompt_only',
                        'ir_data_only','random_icl','zero')

    def run_method(self, method):
        argv=['--library','p1b_libraries/train_v1','--dataset-manifest','ignore.json',
              '--method',method,'--start','0','--count','4','--seed','1',
              '--output-dir',method]
        if method in ('ir_icl','ir_data_only','ir_rerank'):
            argv.extend(['--ir-history','history'])
        args=get_args(argv)
        return run(args,root=self.root,env_override=FakeEnv(),
                   eval_manifest_override=self.eval_manifest,
                   client_override=FakeClient(),
                   teacher_override=FakeTeacher() if method=='mappo' else None)

    def prepare(self):
        for method in self.methods:
            result=self.run_method(method)
            self.assertEqual(result['n_scored'],4)
        return ({m:self.root/m for m in self.methods if m!='ir_rerank'},
                self.root/'ir_rerank')

    def test_end_to_end_nine_way_comparison(self):
        baselines, risk=self.prepare()
        summary, rows=compare_with_rerank(baselines,risk,bootstrap=60)
        self.assertEqual(summary['n_paired'],4)
        self.assertEqual(len(summary['methods']),9)
        self.assertIn('ir_rerank_vs_ir_icl',summary['pairwise_comparisons'])
        self.assertIn('ir_ablation_factorial',summary)
        self.assertEqual(summary['p2a_retrieval_audit']['changed_scene_count_per_role']['agent_g'],0)
        self.assertIn('ir_rerank_V',rows[0])
        meta=read_json(risk/'manifest.json')
        self.assertTrue(meta['ir_risk_prompt_present'])
        self.assertTrue(meta['ir_history_data_present'])
        self.assertEqual(meta['examples'],3)
        self.assertEqual(meta['p2a_rerank_config']['pool_size'],20)
        self.assertEqual(meta['model'],'qwen3-next-80b-a3b-instruct')

    def test_refuse_changed_current_state_or_changed_g(self):
        baselines, risk=self.prepare()
        path=risk/'records.jsonl'
        records=list(read_jsonl(path))
        records[0]['per_role']['agent_g']['retrieved_ids_for_audit_only']=['invalid']
        path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
        with self.assertRaises(ContractError):compare_with_rerank(baselines,risk,bootstrap=30)

    def test_legacy_p1d_manifest_without_factor_flags_is_accepted(self):
        # Real P1-D runs from before P1-E lack the factor flags entirely.
        baselines, risk = self.prepare()
        path = baselines['ir_icl'] / 'manifest.json'
        meta = read_json(path)
        meta.pop('ir_risk_prompt_present')
        meta.pop('ir_history_data_present')
        path.write_text(json.dumps(meta), encoding='utf-8')
        result, rows = compare_with_rerank(baselines, risk, bootstrap=30)
        self.assertEqual(result['n_paired'], 4)
        self.assertEqual(result['p2a_retrieval_audit']['changed_scene_count_per_role']['agent_g'], 0)
        self.assertEqual(len(rows), 4)

    def test_explicitly_false_factor_in_newer_p1d_manifest_rejected(self):
        baselines, risk = self.prepare()
        path = baselines['ir_icl'] / 'manifest.json'
        meta = read_json(path)
        meta['ir_risk_prompt_present'] = False
        path.write_text(json.dumps(meta), encoding='utf-8')
        with self.assertRaises(ContractError):
            compare_with_rerank(baselines, risk, bootstrap=30)

    def test_refuse_changed_history_hash(self):
        baselines,risk=self.prepare()
        path=risk/'manifest.json'
        meta=read_json(path)
        meta['ir_history_manifest_sha256']='mismatched'
        path.write_text(json.dumps(meta),encoding='utf-8')
        with self.assertRaises(ContractError):compare_with_rerank(baselines,risk,bootstrap=30)

    def test_refuse_training_split(self):
        self.eval_manifest['split']='train'
        with self.assertRaises(ContractError):self.run_method('ir_rerank')


if __name__ == '__main__':
    unittest.main()
