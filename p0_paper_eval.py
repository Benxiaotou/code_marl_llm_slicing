# -*- coding: utf-8 -*-
"""P0-v5: paper-aligned, strictly paired audit of GEANT evaluation records.

Requires only the Python standard library. Reads p0_runner *_records.jsonl
and p0_guard_runner guard_records.jsonl; never calls the LLM or environment.
No missing/rejected scene is silently removed from the acceptance denominator.

Usage:
  python p0_paper_eval.py --run MAPPO=PATH --run FixedGuard=PATH \
      --expected-indices 0,1,2,299,599 --reference MAPPO \
      --output-dir p0_runs/p0_v5_audit_mini5

  python p0_paper_eval.py --run MAPPO=PATH --run LPonly=PATH \
      --expected-range 0:512 --reference MAPPO \
      --output-dir p0_runs/p0_v5_audit_512
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

SLICES = ("k1", "k2", "k3")
METRICS = (
    "V", "V_k1", "V_k2", "V_k3", "V_g",
    "V_k1_0", "V_k2_0", "V_k3_0",
    "X_k1", "X_k2", "X_k3",
    "cost_k1", "cost_k2", "cost_k3",
    "cost_sum", "revenue_k1", "revenue_k2", "revenue_k3", "revenue_g",
)
DISPLAY_MEANS = (
    "V", "V_k1", "V_k2", "V_k3", "V_g",
    "V_existing_mean", "cost_sum",
    "cost_k1", "cost_k2", "cost_k3",
    "X_k1", "X_k2", "X_k3",
)
TOL = 1e-6


def _number(v, name):
    if isinstance(v, bool):
        raise ValueError("Boolean in numeric field {}".format(name))
    x = float(v)
    if not math.isfinite(x):
        raise ValueError("Non-finite numeric field {}".format(name))
    return x


def _nearly_equal(actual, expected, name):
    if abs(actual - expected) > TOL * max(1.0, abs(expected), abs(actual)):
        raise ValueError("{} accounting mismatch: {:.10f} != {:.10f}".format(
            name, actual, expected))


def _wilson(success, n):
    if not n:
        return None
    z = 1.95996398454
    p = success / n
    den = 1.0 + z * z / n
    mid = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [max(0.0, mid-half), min(1.0, mid+half)]


def _load_records_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if not path.name.endswith(".jsonl"):
            raise ValueError("Expected a .jsonl file: {}".format(path))
        record_path = path
        directory = path.parent
    elif path.is_dir():
        guard = path / "guard_records.jsonl"
        if guard.exists():
            record_path = guard
        else:
            choices = list(path.glob("*_records.jsonl"))
            if len(choices) != 1:
                raise ValueError("{}: expected exactly one *_records.jsonl, got {}".format(
                    path, [x.name for x in choices]))
            record_path = choices[0]
        directory = path
    else:
        raise FileNotFoundError("Run directory/file does not exist: {}".format(path))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    records = {}
    for line_no, line in enumerate(record_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            raise ValueError("{}:{}: not a JSON object".format(record_path, line_no))
        idx = rec.get("original_scene_index", rec.get("scene_index"))
        if idx is None:
            raise ValueError("{}:{}: missing scene index".format(record_path, line_no))
        idx = int(idx)
        if idx in records:
            raise ValueError("{}: duplicate original_scene_index {}".format(record_path, idx))
        records[idx] = _standardize(rec, record_path, line_no)
    if not records:
        raise ValueError("No records: {}".format(record_path))
    return {"path": str(record_path), "manifest": manifest, "rows": records}


def _standardize(raw, path, line_no):
    status = raw.get("status", "SCORED")
    g_done_raw = raw.get("g_done")
    if g_done_raw is None:
        if status == "REJECTED_NO_VERIFIED_ACTION":
            g_done = False
        else:
            raise ValueError("{}:{}: no g_done/status rejection".format(path, line_no))
    elif isinstance(g_done_raw, bool):
        g_done = g_done_raw
    else:
        raise ValueError("{}:{}: g_done must be a JSON boolean".format(path, line_no))
    if raw.get("used_fallback") is True:
        # Do not quietly count a heuristic result as "Qwen-only".
        raise ValueError("{}:{}: contains a policy fallback. Label it as a separate baseline.".format(
            path, line_no))

    numeric = {}
    for name in METRICS:
        if name in raw and raw[name] is not None:
            numeric[name] = _number(raw[name], name)
    full = all(key in numeric for key in METRICS if key != "cost_sum")
    if status == "REJECTED_NO_VERIFIED_ACTION" and (g_done or full):
        raise ValueError("{}:{}: contradictory rejected record".format(path, line_no))
    if full:
        if "cost_sum" not in numeric:
            numeric["cost_sum"] = sum(numeric["cost_"+k] for k in SLICES)
        _nearly_equal(numeric["cost_sum"], sum(numeric["cost_"+k] for k in SLICES), "cost_sum")
        _nearly_equal(numeric["V"], sum(numeric["V_"+k] for k in SLICES) + numeric["V_g"], "V_total")
        for k in SLICES:
            _nearly_equal(numeric["V_"+k], numeric["revenue_"+k] -
                          numeric["cost_"+k] + numeric["X_"+k], "V_"+k)
        _nearly_equal(numeric["V_g"], numeric["revenue_g"] -
                      sum(numeric["X_"+k] for k in SLICES), "V_g")
        numeric["V_existing_mean"] = sum(numeric["V_"+k] for k in SLICES)/3.0

    if g_done and not full:
        raise ValueError("{}:{}: accepted scene has incomplete metric fields".format(path, line_no))
    ir = None
    ir_each = {k: None for k in SLICES}
    if full:
        ir_each = {k: (numeric["V_"+k] + 1e-8 >= numeric["V_"+k+"_0"]) for k in SLICES}
        ir = all(ir_each.values()) and numeric["V_g"] >= -1e-8
    return {
        "scene": int(raw.get("original_scene_index", raw.get("scene_index"))),
        "g_done": g_done,
        "status": status,
        "scored": full,
        "ir": ir,
        "ir_each": ir_each,
        "cooperation_valid": bool(g_done and ir),
        "guard_changed": raw.get("guard_changed"),
        "raw_admitted": (raw.get("raw_check") or {}).get("accepted") if isinstance(
                         raw.get("raw_check"), dict) else None,
        "numbers": numeric,
    }


def _mean(xs):
    return sum(xs)/len(xs) if xs else None


def _aggregate(name, data):
    rows = [data["rows"][k] for k in sorted(data["rows"])]
    n = len(rows)
    scored = [r for r in rows if r["scored"]]
    admitted = [r for r in rows if r["g_done"]]
    assessed_ir = [r for r in admitted if r["ir"] is not None]
    cooperatively_valid = [r for r in rows if r["cooperation_valid"]]
    safe_means = {}
    available_means = {}
    for metric in DISPLAY_MEANS:
        available = [r["numbers"][metric] for r in scored]
        available_means[metric] = _mean(available)
        # All-attempted mean is UNDEFINED when even one attempted scene is not scored.
        safe_means[metric] = _mean(available) if len(scored)==n else None
    per_k = {}
    for k in SLICES:
        valid_rows = [r for r in assessed_ir if r["ir_each"][k] is not None]
        per_k[k] = (sum(1 for r in valid_rows if r["ir_each"][k]) / len(valid_rows)
                    if valid_rows else None)
    return {
        "method": name,
        "attempted": n,
        "scored": len(scored),
        "rejected_without_score": n-len(scored),
        "admitted": len(admitted),
        "acceptance_rate": len(admitted)/n,
        "acceptance_95pct_wilson": _wilson(len(admitted), n),
        "cooperative_valid": len(cooperatively_valid),
        "cooperative_valid_rate_all": len(cooperatively_valid)/n,
        "ir_given_admission": (
            len(cooperatively_valid)/len(assessed_ir)
            if assessed_ir and len(assessed_ir)==len(admitted) else None
        ),
        "per_existing_slice_ir_given_admission": per_k,
        "means_all_attempted": safe_means,
        "means_of_scored_only_NOT_comparable_when_missing": available_means,
        "record_file": data["path"],
    }


def _check_sets(runs, expect):
    actual = {label:set(data["rows"]) for label,data in runs.items()}
    if expect is None:
        expect = actual[next(iter(actual))]
    for label, scenes in actual.items():
        if scenes != expect:
            missing = sorted(expect - scenes)
            extra = sorted(scenes - expect)
            raise ValueError(
                "{} scenes do not match expected sample set (expected {}; got {}). "
                "Missing={} Extra={}".format(label,len(expect),len(scenes),
                                              missing[:12],extra[:12]))
    return sorted(expect)


def _warnings(runs):
    msgs = []
    fields = ("dataset","seed","main_sha256")
    for field in fields:
        values = [(name,dat["manifest"].get(field)) for name,dat in runs.items()]
        known = [(name,v) for name,v in values if v is not None]
        if len(set(str(v) for name,v in known)) > 1:
            msgs.append("Manifest mismatch for {}: {}".format(field,known))
        if len(known) != len(values):
            msgs.append("Manifest {} missing for some methods; compare run settings manually".format(field))
    vals = {name: dat["manifest"].get("source_kind") for name,dat in runs.items()}
    if any(v is None for v in vals.values()):
        msgs.append("Some runs lack source_kind; explicitly label replay versus live LLM")
    return msgs


def _paired(runs, ordered_scenes, reference):
    ref = runs[reference]["rows"]
    output = []
    summary = {}
    for label, data in runs.items():
        if label == reference:
            continue
        delta_utility=[]
        delta_cost=[]
        for scene in ordered_scenes:
            a, b = ref[scene], data["rows"][scene]
            d_v = d_cost = None
            if a["scored"] and b["scored"]:
                d_v = b["numbers"]["V"] - a["numbers"]["V"]
                d_cost = b["numbers"]["cost_sum"] - a["numbers"]["cost_sum"]
                delta_utility.append(d_v)
                delta_cost.append(d_cost)
            output.append({
                "original_scene_index":scene,
                "reference":reference,
                "method":label,
                "reference_g_done":int(a["g_done"]),
                "method_g_done":int(b["g_done"]),
                "delta_V":d_v, "delta_cost_sum":d_cost,
                "reference_ir":a["ir"], "method_ir":b["ir"],
            })
        summary[label] = {
            "delta_acceptance_rate":(
                sum(int(data["rows"][i]["g_done"])-int(ref[i]["g_done"])
                    for i in ordered_scenes)/len(ordered_scenes)),
            "paired_metric_coverage":len(delta_utility),
            "expected_pairs":len(ordered_scenes),
            "mean_delta_V_of_scored_pairs":_mean(delta_utility),
            "mean_delta_cost_of_scored_pairs":_mean(delta_cost),
            "full_pairing_for_utility":len(delta_utility)==len(ordered_scenes),
        }
    return output, summary


def generate(runs, output_dir, reference="MAPPO", expected=None):
    if reference not in runs:
        raise ValueError("Reference '{}' not in {}".format(reference,list(runs)))
    if len(runs)<2:
        raise ValueError("Need at least two methods")
    scene_ids = _check_sets(runs, expected)
    summaries = [_aggregate(label,data) for label,data in runs.items()]
    warns = _warnings(runs)
    details, pair_summary = _paired(runs,scene_ids,reference)
    report = {
        "paper_metric_context": {
            "paper": "Pei Xinglong thesis, Chapter 5, Fig. 5.7",
            "paper_reported_acceptance_rate":0.8398,
            "paper_evaluation_window_in_original_code":512,
            "note":"Paper figures are context only, NOT a matched-scene statistical comparison.",
            "sample_count":len(scene_ids),
            "sample_indices":scene_ids,
            "ir_definition":"g_done and each V_ki >= V_ki_0 and V_g >= 0; IR is diagnostic, not retroactively added to original admission condition.",
        },
        "methods":summaries,
        "reference":reference,
        "paired_vs_reference":pair_summary,
        "comparability_warnings":warns,
    }
    out=Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory not empty: {}".format(out))
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),
                                    encoding="utf-8")
    fields = ("original_scene_index","method","g_done","status","scored",
              "IR_existing_all_and_g","cooperation_valid","V","V_k1",
              "V_k2","V_k3","V_g","V_k1_0","V_k2_0","V_k3_0",
              "cost_sum","X_k1","X_k2","X_k3")
    with (out/"per_scene.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=fields)
        wr.writeheader()
        for idx in scene_ids:
            for label, dat in runs.items():
                r=dat["rows"][idx]
                line={"original_scene_index":idx,"method":label,
                      "g_done":int(r["g_done"]),"status":r["status"],
                      "scored":int(r["scored"]),"IR_existing_all_and_g":r["ir"],
                      "cooperation_valid":int(r["cooperation_valid"])}
                for k in METRICS:
                    if k in fields:
                        line[k]=r["numbers"].get(k)
                wr.writerow(line)
    with (out/"paired_vs_{}.csv".format(reference)).open("w",newline="",encoding="utf-8-sig") as f:
        fields2 = ("original_scene_index","reference","method","reference_g_done",
                   "method_g_done","delta_V","delta_cost_sum","reference_ir","method_ir")
        wr=csv.DictWriter(f,fieldnames=fields2)
        wr.writeheader()
        wr.writerows(details)
    readme = [
        "# P0-v5 GEANT paper-metric audit",
        "",
        "This is a matched-scene comparison, **not** a reproduction of the paper's original sample distribution.",
        "",
        "All N attempts count toward acceptance rate and cooperative-valid rate.",
        "V/cost averages across all N attempts are **null** when any refused scene lacks original-env metrics.",
        "The *_scored_only fields are descriptive ONLY and must not be used to claim better overall utility.",
        "",
        "| Method | N | Admitted | Accept % | Coop-valid % | Mean V (all) | Mean V_g (all) | Mean cost (all) | IR of admitted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(x):
        return "N/A" if x is None else "{:.3f}".format(x)
    for s in summaries:
        m=s["means_all_attempted"]
        readme.append("| {} | {} | {} | {:.2f} | {:.2f} | {} | {} | {} | {} |".format(
            s["method"],s["attempted"],s["admitted"],100*s["acceptance_rate"],
            100*s["cooperative_valid_rate_all"],fmt(m["V"]),fmt(m["V_g"]),
            fmt(m["cost_sum"]),fmt(None if s["ir_given_admission"] is None else 100*s["ir_given_admission"])))
    readme.extend(["","## Reference","Paper Fig. 5.7: CoMarl admission rate 83.98% (0.8398);",
                   "original evaluation code averages the last 512 steps.",
                   "Comparing a different 512-scene selection cannot establish exact paper reproduction.",
                   "","## Methodological alerts"])
    readme.extend(["- "+x for x in warns] if warns else ["- No metadata mismatch found among declared manifest fields."])
    for s in summaries:
        if s["rejected_without_score"]:
            readme.append("- {} has {} rejected scenes without comparable V/cost; whole-sample utility is N/A.".format(
                s["method"],s["rejected_without_score"]))
    readme.extend(["","## Paired deltas vs {}".format(reference),"",
                   "Mean delta V is reported from paired scored scenes only; check metric coverage before interpretation."])
    for name, d in pair_summary.items():
        readme.append("- {}: d_accept={:.2f} pp; dV={} (paired {}/{}); d_cost={}".format(
            name,d["delta_acceptance_rate"]*100,
            fmt(d["mean_delta_V_of_scored_pairs"]),d["paired_metric_coverage"],
            d["expected_pairs"],fmt(d["mean_delta_cost_of_scored_pairs"])))
    (out/"report.md").write_text("\n".join(readme)+"\n",encoding="utf-8")
    return report,out


def parse_expected(args):
    if args.expected_indices and args.expected_range:
        raise ValueError("Choose only one of --expected-indices / --expected-range")
    if args.expected_indices:
        ids=[int(s.strip()) for s in args.expected_indices.split(",")]
        if len(ids)!=len(set(ids)):
            raise ValueError("Duplicate indices in --expected-indices")
        return set(ids)
    if args.expected_range:
        parts=args.expected_range.split(":")
        if len(parts)!=2:
            raise ValueError("expected-range must be START:STOP (STOP excluded)")
        start,stop=map(int,parts)
        if not (0<=start<stop):
            raise ValueError("Invalid --expected-range")
        return set(range(start,stop))
    return None


def main(argv=None):
    p=argparse.ArgumentParser(description="Original-paper GEANT metric audit. No LLM, environment, or MARL dependencies.")
    p.add_argument("--run",action="append",required=True,metavar="METHOD=RUN_DIRECTORY",
                   help="Repeat for each model/baseline, e.g. --run MAPPO=.\\p0_runs\\mappo_run")
    p.add_argument("--reference",default="MAPPO",help="Reference name used for paired scene differences")
    p.add_argument("--expected-indices",default=None,help="Original scene IDs, comma separated")
    p.add_argument("--expected-range",default=None,help="START:STOP (exclusive STOP), e.g. 0:512")
    p.add_argument("--output-dir",required=True)
    args=p.parse_args(argv)
    runs={}
    for val in args.run:
        if "=" not in val:
            p.error("--run must be METHOD=PATH")
        label,path=val.split("=",1)
        label=label.strip()
        if not label or "/" in label or "\\" in label or not path.strip():
            p.error("Invalid method label/path for "+val)
        if label in runs:
            p.error("Duplicate method label: "+label)
        runs[label]=_load_records_path(path.strip())
    report,out=generate(runs,args.output_dir,args.reference,parse_expected(args))
    print("P0-v5 paper-aligned metric audit, {} original scenes".format(
          report["paper_metric_context"]["sample_count"]))
    print("{:22s} {:>7s} {:>9s} {:>12s} {:>12s} {:>11s}".format(
          "Method","N","accept%","IR+admit%","V(all)","cost(all)"))
    for s in report["methods"]:
        m=s["means_all_attempted"]
        def f(x):
            return "N/A" if x is None else "{:.3f}".format(x)
        print("{:22.22s} {:7d} {:9.2f} {:12.2f} {:>12s} {:>11s}".format(
            s["method"],s["attempted"],100*s["acceptance_rate"],
            100*s["cooperative_valid_rate_all"],f(m["V"]),f(m["cost_sum"])))
    if report["comparability_warnings"]:
        print("WARNINGS:")
        for w in report["comparability_warnings"]:
            print("  -",w)
    for s in report["methods"]:
        if s["rejected_without_score"]:
            print("INCOMPLETE:",s["method"],"has",s["rejected_without_score"],
                  "attempts without V; all-attempted mean is N/A, NOT success-only mean.")
    print("Output:",out)
    return 0


if __name__=="__main__":
    sys.exit(main())
