"""Pool final independent replay records and write compact final report."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def metric_summary(rows):
 keys=("g_done","cooperative_valid","individual_rationality","V","V_g","cost_sum")
 out={"n":len(rows)}
 out["accepted"]=sum(bool(r["metrics"]["g_done"]) for r in rows)
 out["cooperative"]=sum(bool(r["metrics"]["cooperative_valid"]) for r in rows)
 out["ir_valid"]=sum(bool(r["metrics"]["individual_rationality"]) for r in rows)
 out["accepted_ir_invalid_n"]=sum(bool(r["metrics"]["g_done"]) and not bool(r["metrics"]["individual_rationality"]) for r in rows)
 for k in ("V","V_g","cost_sum"): out[k+"_mean"]=float(np.mean([r["metrics"][k] for r in rows]))
 return out
def paired(a,b):
 assert len(a)==len(b)
 out={}
 for k in ("g_done","cooperative_valid","V","V_g","cost_sum"):
  x=np.asarray([float(r["metrics"][k]) for r in a]); y=np.asarray([float(r["metrics"][k]) for r in b]); d=x-y
  z={"mean_difference":float(d.mean())}
  if k in ("g_done","cooperative_valid"):
   wins=int(((x==1)&(y==0)).sum());losses=int(((x==0)&(y==1)).sum());z.update({"wins":wins,"losses":losses})
  out[k]=z
 return out
def main():
 folder=sorted((ROOT/"p2g_autonomous/results").glob("formal_test_replay_*"),key=lambda p:p.stat().st_mtime)[-1]
 arms={k:[] for k in ("MAPPO","Qwen_original","Frozen_link_mean3_a0.5")}
 for seed in (6601,6701,6801):
  with (folder/f"records_{seed}.jsonl").open(encoding="utf-8") as f:
   for line in f:
    r=json.loads(line); arms[r["arm"]].append(r)
 assert all(len(v)==1536 for v in arms.values())
 summaries={k:metric_summary(v) for k,v in arms.items()}
 comparisons={"candidate_minus_qwen":paired(arms["Frozen_link_mean3_a0.5"],arms["Qwen_original"]),
              "candidate_minus_mappo":paired(arms["Frozen_link_mean3_a0.5"],arms["MAPPO"]),
              "qwen_minus_mappo":paired(arms["Qwen_original"],arms["MAPPO"])}
 final={"status":"COMPLETE","folder":str(folder.relative_to(ROOT)),"pooled_n":1536,
        "summaries":summaries,"comparisons":comparisons,"controls_reproduced":3072,
        "candidate":"frozen link_mean3 alpha=.5 postprocessor",
        "API_calls":0,"API_cost_usd_estimate":0.0,
        "test_used_for_tuning":False,
        "interpretation":"Independent P2-C test replay of cached Qwen/MAPPO actions. Candidate improves Qwen cooperation, V and cost pooled; it is not statistically established superior to MAPPO on every endpoint.",
        "limitations":"Candidate postprocesses cached Qwen P2-A actions; no fresh Qwen calls on P2-C. Test data were not used for tuning. Per-seed results remain primary because seed pairing is by identical scene IDs within each seed."}
 (folder/"pooled_summary.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(final,separators=(",",":")),flush=True)
if __name__=="__main__":main()
