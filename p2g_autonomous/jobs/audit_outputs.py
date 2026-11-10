"""Offline consistency audits and paired development comparisons; no inference."""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "p2f_analysis"))
import val5601_counterfactual_replay as replay
from audit_qwen_full200 import paired, summarize

CROSS = ROOT / "p2g_autonomous/results/cross_blocks_20261104T152954_364235Z"
BLEND = ROOT / "p2g_autonomous/results/blend_node_20261104T153414_857013Z"
PILOT = ROOT / "p2g_autonomous/results/node_pilot_20261104T153832_383493Z"
SOURCE = ROOT / "p2f_analysis/qwen_full200/run_20261104T141915_396460Z"

def load(path):
    assert path.is_file() and path.stat().st_size < 100_000_000, str(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]

def normalize(rows):
    return [{**r, "original_scene_id": r.get("original_scene_id", r.get("scene"))} for r in rows]

def groups(rows, field):
    d = collections.defaultdict(list)
    for r in normalize(rows):
        d[str(r[field])].append(r)
    for v in d.values():
        v.sort(key=lambda r: r["original_scene_id"])
    return dict(d)

def metric_identities(row):
    m = row["metrics"]
    ir = (m["V_g"] >= -1e-8 and all(m["V_"+k] >= m["V_"+k+"_0"]-1e-8 for k in ("k1","k2","k3")))
    assert bool(m["individual_rationality"]) == ir
    assert bool(m["cooperative_valid"]) == (bool(m["g_done"]) and ir)
    assert math.isclose(m["V"], sum(m["revenue_"+k] for k in ("k1","k2","k3","g")) - m["cost_sum"], abs_tol=1e-7)

def write_new(path, value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def audit_cross():
    d = groups(load(CROSS / "records.jsonl"), "case")
    original = load(SOURCE / "full_original_records.jsonl")
    candidate = load(SOURCE / "full_link_calibration_records.jsonl")
    assert len(d) == 8
    for v in d.values():
        assert [r["original_scene_id"] for r in v] == list(range(200))
        for r in v: metric_identities(r)
    for name, ref in (("original", original), ("candidate", candidate)):
        for a,b in zip(d[name], ref):
            assert not replay.control_diff(b["metrics"], a["metrics"])
    out = {"checks_passed":True, "controls_reproduced":400, "rows_checked":1600,
           "comparisons":{k:paired(v,d["original"]) for k,v in d.items() if k!="original"},
           "API_calls":0}
    write_new(CROSS / "audit_paired.json",out)
    return out

def audit_blend():
    d = groups(load(BLEND/"records.jsonl"),"alpha")
    original=load(SOURCE/"full_original_records.jsonl")
    for v in d.values():
        assert len(v)==200
        for r in v: metric_identities(r)
    for a,b in zip(d["0.0"],original): assert not replay.control_diff(b["metrics"],a["metrics"])
    out={"checks_passed":True,"rows_checked":1000,
         "comparisons":{k:paired(v,d["0.0"]) for k,v in d.items() if k!="0.0"},"API_calls":0}
    write_new(BLEND/"audit_paired.json",out)
    return out

def audit_qwen(folder):
    from p1a.policy import parse_ratios,to_env_action
    rows=load(folder/"records.jsonl")
    d=groups(rows,"arm")
    expected=list(range(len(d["original"])))
    n=len(expected)
    assert set(d)=={"original","candidate_node","blended_25"}
    for v in d.values():
        assert [r["original_scene_id"] for r in v]==expected
        for r in v: metric_identities(r)
    events=load(folder/"requests.jsonl")
    starts={r["request_id"]:r for r in events if r["event"]=="started"}
    ends={r["request_id"]:r for r in events if r["event"]=="finished"}
    assert len(events)==n*16 and len(starts)==len(ends)==n*8
    assert set(starts)==set(ends)
    historic=replay.read_jsonl(ROOT/"p2b_runs/val5601_standard1000_p2a200/records.jsonl")
    prompt=(ROOT/"p2g_autonomous/jobs/node_calibration_prompt.txt").read_text(encoding="utf-8").strip()
    action_count=0
    for arm in ("original","candidate_node"):
        for r in d[arm]:
            scene=r["original_scene_id"]
            for role in replay.ROLES:
                e=r["per_role"][role]
                messages=[dict(m) for m in historic[scene]["per_role"][role]["messages"]]
                if arm=="candidate_node" and role in replay.K_ROLES:
                    messages[0]["content"]+="\n"+prompt
                assert e["messages"]==messages
                ph=hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()
                rid=f"{arm}:{scene}:{role}"
                assert ph==e["prompt_sha256"]==starts[rid]["messages_sha256"]
                assert e["raw_model_output"]==ends[rid]["response"]
                action=to_env_action(role,parse_ratios(e["raw_model_output"],role)).tolist()
                assert action==r["env_actions"][role]==e["env_action"]
                assert e["retrieved_ids_for_audit_only"]==historic[scene]["per_role"][role]["retrieved_ids_for_audit_only"]
                action_count+=1
    for o,c,b in zip(d["original"],d["candidate_node"],d["blended_25"]):
        for role in replay.ROLES:
            expected_action=np.array(o["env_actions"][role],dtype=float)
            if role in replay.K_ROLES:
                expected_action[:2]+=.25*(np.asarray(c["env_actions"][role][:2])-expected_action[:2])
            assert np.allclose(expected_action,b["env_actions"][role],atol=0,rtol=0)
    mappo=replay.read_jsonl(ROOT/"p2b_runs/val5601_mappo200/records.jsonl")
    mrows=[mappo[i] for i in expected]
    missing=sum(not r.get("usage") for r in ends.values())
    report={"checks_passed":True,"n":n,"model":"qwen3-next-80b-a3b-instruct",
            "request_success":len(ends),"request_failure_or_retry":0,
            "messages_and_actions_verified":action_count,"blend_verified":n,
            "usage_missing":missing,"token_count":"UNKNOWN" if missing else "see journal",
            "API_cost_usd_estimate":0,
            "summaries":{k:summarize(v) for k,v in {**d,"MAPPO":mrows}.items()},
            "comparisons":{
                "candidate_node_minus_original":paired(d["candidate_node"],d["original"]),
                "blended_25_minus_original":paired(d["blended_25"],d["original"]),
                "original_minus_MAPPO":paired(d["original"],mrows),
                "candidate_node_minus_MAPPO":paired(d["candidate_node"],mrows),
                "blended_25_minus_MAPPO":paired(d["blended_25"],mrows)},
            "limitations":"exploratory development estimates; repeated validation use, unadjusted multiple comparisons; two sampled full policies plus deterministic node blend, not fresh independent test"}
    if n==200:
        report["subgroups"]={name:{k:summarize(v[a:b]) for k,v in d.items()} for name,a,b in
                            (("pilot_overlap",0,40),("additional_development",40,200))}
    write_new(folder/"audit_paired.json",report)
    print("AUDITED",str(folder),json.dumps({"n":n,"summaries":report["summaries"],"blended_minus_original":report["comparisons"]["blended_25_minus_original"]["metrics"]}),flush=True)
    return report

def main():
    if not (CROSS/"audit_paired.json").exists(): audit_cross()
    if not (BLEND/"audit_paired.json").exists(): audit_blend()
    if not (PILOT/"audit_paired.json").exists(): audit_qwen(PILOT)
    # Job producer writes this small pointer once the current run completes.
    pointer=ROOT/"p2g_autonomous/latest_full_path.txt"
    if pointer.exists():
        folder=ROOT/pointer.read_text(encoding="utf-8").strip()
        assert folder.is_relative_to(ROOT) if sys.version_info>=(3,9) else str(folder).startswith(str(ROOT))
        if not (folder/"audit_paired.json").exists(): audit_qwen(folder)

if __name__=="__main__": main()
