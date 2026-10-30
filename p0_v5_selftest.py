# -*- coding: utf-8 -*-
"""P0-v5 offline tests; no XuanCe, Torch, LLM or API key required."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import p0_paper_eval as evalmod


def rec(scene=0, admitted=True):
    revenues = {"k1":100.0,"k2":90.0,"k3":110.0}
    costs = {"k1":3.0,"k2":2.0,"k3":5.0}
    comp = {"k1":6.0,"k2":4.0,"k3":8.0} if admitted else {
        "k1":0.0,"k2":0.0,"k3":0.0}
    g_revenue=40.0 if admitted else 0.0
    r={"original_scene_index":scene,"g_done":admitted,
       "revenue_g":g_revenue,
       "V_g":g_revenue-sum(comp.values()),
       "cost_sum":sum(costs.values())}
    for k in ("k1","k2","k3"):
        r["revenue_"+k]=revenues[k]
        r["cost_"+k]=costs[k]
        r["X_"+k]=comp[k]
        r["V_"+k+"_0"]=revenues[k]
        r["V_"+k]=revenues[k]-costs[k]+comp[k]
    r["V"]=sum(r["V_"+k] for k in ("k1","k2","k3"))+r["V_g"]
    return r


def standardized(data):
    return evalmod._standardize(data,"test.jsonl",1)


def data_with(rows, manifest=None):
    return {"path":"memory.jsonl",
            "manifest":manifest or {"dataset":"mini","seed":1,"main_sha256":"abc"},
            "rows":{r["scene"]:r for r in rows}}


class PaperEvaluationTests(unittest.TestCase):
    def test_valid_accepted_accounting_and_ir(self):
        r=standardized(rec())
        self.assertTrue(r["scored"])
        self.assertTrue(r["cooperation_valid"])
        self.assertTrue(r["ir"])
        self.assertTrue(all(r["ir_each"].values()))
        self.assertAlmostEqual(r["numbers"]["V"],330.0)

    def test_valid_original_env_failure_is_scored(self):
        r=standardized(rec(admitted=False))
        self.assertFalse(r["g_done"])
        self.assertTrue(r["scored"])
        self.assertFalse(r["cooperation_valid"])
        self.assertFalse(r["ir"])

    def test_rejected_without_commit_not_deleted(self):
        ok=standardized(rec(0,True))
        fail=standardized({"original_scene_index":1,
            "g_done":False,"status":"REJECTED_NO_VERIFIED_ACTION"})
        result=evalmod._aggregate("Guard",data_with([ok,fail]))
        self.assertEqual(result["attempted"],2)
        self.assertEqual(result["scored"],1)
        self.assertEqual(result["rejected_without_score"],1)
        self.assertEqual(result["acceptance_rate"],0.5)
        self.assertIsNone(result["means_all_attempted"]["V"])
        self.assertIsNotNone(
            result["means_of_scored_only_NOT_comparable_when_missing"]["V"])

    def test_llm_error_without_commit_not_deleted(self):
        r=standardized({"original_scene_index":3,"g_done":False,
                        "status":"LLM_ERROR_NOT_COMMITTED"})
        self.assertFalse(r["scored"])
        self.assertFalse(r["g_done"])

    def test_corrupt_sum_rejected(self):
        r=rec()
        r["V"]+=2
        with self.assertRaisesRegex(ValueError,"accounting mismatch"):
            standardized(r)

    def test_corrupt_compensation_rejected(self):
        r=rec()
        r["V_g"]+=0.2
        r["V"]+=0.2
        with self.assertRaisesRegex(ValueError,"accounting mismatch"):
            standardized(r)

    def test_corrupt_cost_rejected(self):
        r=rec()
        r["cost_sum"]+=0.2
        with self.assertRaisesRegex(ValueError,"accounting mismatch"):
            standardized(r)

    def test_explicit_fallback_must_not_be_rebranded(self):
        r=rec()
        r["used_fallback"]=True
        with self.assertRaisesRegex(ValueError,"fallback"):
            standardized(r)

    def test_mismatched_scene_sets_fail_strict(self):
        a=data_with([standardized(rec(0)),standardized(rec(1))])
        b=data_with([standardized(rec(0))])
        with self.assertRaisesRegex(ValueError,"Missing"):
            evalmod._check_sets({"A":a,"B":b},None)

    def test_paired_comparison(self):
        a=data_with([standardized(rec(0)),standardized(rec(1))])
        changed=rec(0)
        changed["cost_k1"]+=1
        changed["cost_sum"]+=1
        changed["V_k1"]-=1
        changed["V"]-=1
        b=data_with([standardized(changed),standardized(rec(1))])
        rows, stats=evalmod._paired({"MAPPO":a,"Fixed":b},[0,1],"MAPPO")
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(stats["Fixed"]["mean_delta_V_of_scored_pairs"],-0.5)

    def test_duplicate_original_indices_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"guard_records.jsonl"
            row=rec()
            p.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"duplicate"):
                evalmod._load_records_path(Path(td))

    def test_report_generation(self):
        a=data_with([standardized(rec(0)),standardized(rec(1))])
        b=data_with([standardized(rec(0)),standardized(rec(1,False))])
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/"report"
            report,out=evalmod.generate({"MAPPO":a,"RawQwen":b},dest,"MAPPO",{0,1})
            self.assertTrue((out/"metrics.json").is_file())
            self.assertTrue((out/"per_scene.csv").is_file())
            self.assertTrue((out/"report.md").is_file())
            self.assertEqual(report["methods"][1]["acceptance_rate"],0.5)

    def test_guard_runner_v5_summary_counts_failures(self):
        # AST-extract only the pure summarizer; does not import a MARL environment.
        path=Path(__file__).resolve().parent/"p0_guard_runner_v5.py"
        tree=ast.parse(path.read_text(encoding="utf-8"))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=="_summarize")
        namespace={}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),"exec"),namespace)
        rows=[
            {"policy":"fixed_anchor_guard","source_model":None,"g_done":True,
             "raw_check":{"accepted":False},"guard_changed":True,
             "V":330.0,"V_g":22.0,"cost_sum":10.0,"api_latency_s":0.0,
             "guard_latency_s":0.1,"env_step_s":0.03,"wall_time_s":0.13},
            {"policy":"fixed_anchor_guard","source_model":None,"g_done":False,
             "status":"REJECTED_NO_VERIFIED_ACTION","guard_changed":False,
             "raw_check":{"accepted":False},"api_latency_s":0.0,
             "guard_latency_s":0.2,"env_step_s":0.0,"wall_time_s":0.2},
        ]
        summary=namespace["_summarize"](rows)
        self.assertEqual(summary["n"],2)
        self.assertEqual(summary["n_scored"],1)
        self.assertEqual(summary["acceptance_rate"],0.5)
        self.assertIsNone(summary["V"])
        self.assertAlmostEqual(summary["guard_latency_s"],0.15)


if __name__=="__main__":
    unittest.main(verbosity=2)
