"""Train-local link calibration with reset/control checks on three cached Qwen draws."""
from __future__ import annotations
import contextlib, datetime as dt, json, sys, os, shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"p2f_analysis"))
import val5601_counterfactual_replay as replay
from audit_qwen_full200 import paired,summarize
from cross_action_blocks import score
MODEL="qwen3-next-80b-a3b-instruct"
SOURCES={
 "historical":ROOT/"p2b_runs/val5601_standard1000_p2a200/records.jsonl",
 "fresh_linkrun":ROOT/"p2f_analysis/qwen_full200/run_20261104T141915_396460Z/full_original_records.jsonl",
 "fresh_noderun":ROOT/"p2g_autonomous/results/node_full_20261104T161150_600593Z/records.jsonl",
}
METHODS=("nearest","mean3","link_mean3")
ALPHAS=(.25,.5,1.)
def load_sources():
 out={}
 for label,path in SOURCES.items():
  assert path.is_file() and path.stat().st_size<100_000_000
  rows={}
  for line in path.open(encoding="utf-8"):
   row=json.loads(line)
   if label=="fresh_noderun" and row["arm"]!="original":continue
   scene=row.get("original_scene_id",row.get("scene"))
   rows[scene]={**row,"original_scene_id":scene}
  assert sorted(rows)==list(range(200))
  out[label]=rows
 return out

def teacher_targets(lib,role,observation):
 from p1c.retrieval import local_numeric_summary
 nearest=lib.query(role,observation,top_k=3)
 # A fixed unweighted average avoids introducing distance-temperature tuning.
 targets={"nearest":np.asarray(nearest[0]["ratios"][2:],dtype=float),
          "mean3":np.mean([x["ratios"][2:] for x in nearest],axis=0)}
 summary=local_numeric_summary(role,observation)
 indices=[i for i,k in enumerate(lib.keys[role]) if k.startswith(("link_control_","overlap_links_","all_link_","link_count_"))]
 assert indices
 curr=np.asarray([summary[k] for k in lib.keys[role]],dtype=float)
 dist=np.sqrt(np.mean(((lib.features[role][:,indices]-curr[indices])/lib.scales[role][indices])**2,axis=1))
 selected=sorted(range(len(dist)),key=lambda j:(float(dist[j]),lib.cases[role][j]["experience_id"]))[:3]
 targets["link_mean3"]=np.mean([lib.cases[role][j]["ratios"][2:] for j in selected],axis=0)
 info={"nearest_ids":[x["experience_id"] for x in nearest],
       "link_mean3_ids":[lib.cases[role][j]["experience_id"] for j in selected]}
 return targets,info

def modified_actions(original,targets,method,alpha):
 actions={r:list(original[r]) for r in replay.ROLES}
 for role in replay.K_ROLES:
  ov=np.asarray(original[role],dtype=float)
  teacher=2*np.asarray(targets[role][method])-1
  ov[2:]+=alpha*(teacher-ov[2:])
  actions[role]=ov.tolist()
 return actions

def main():
 cached=load_sources()
 manifest_path=ROOT/"p1b_data/p2b_val_seed5601.json"
 manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
 import p1a_runner as base,p1c_runner
 from p2e_prompt.val5601_eval import verify_environment
 from p1c.retrieval import LocalHistory
 mm,_=base._config(ROOT,ROOT/"configs_main.yaml");base._compat_pickle_main_classes(mm)
 environment=verify_environment(ROOT,manifest)
 lib=LocalHistory(ROOT/"p2b_libraries/p2b_standard_1000_v2");lib.assert_independent(manifest)
 out=ROOT/"p2g_autonomous/results"/("teacher_multicohort_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
 out.mkdir(parents=True,exist_ok=False);runtime=out/"env_runtime";runtime.mkdir()
 hashes={str(p.relative_to(ROOT)):replay.sha256(p) for p in
         (Path(__file__),ROOT/"main.py",ROOT/"configs_main.yaml",*SOURCES.values())}
 protocol={"schema":"train_local_link_multicohort_v1","sources":{k:str(v.relative_to(ROOT)) for k,v in SOURCES.items()},
           "methods":METHODS,"alphas":ALPHAS,"environment":environment,"source_hashes":hashes,
           "library_manifest_sha256":lib.manifest_sha256,"library_role_sha256":lib.library_hashes,
           "seed":1,"dataset":"p2b_val_seed5601","split":"validation","scene_ids":list(range(200)),
           "primary":["cooperative_valid","V"],"secondary":["g_done","cost_sum","V_g"],
           "API_calls":0,"limitation":"three Qwen draws on identical already-used development scenes, not 600 independent scenes"}
 (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
 rows={label:{"original":[]} for label in SOURCES};targets_all={}
 with (out/"env_stdout.log").open("x",encoding="utf-8") as log:
  with contextlib.redirect_stdout(log):
   env=p1c_runner.create_eval_env(ROOT,manifest,"configs_main.yaml")
   env.path_reward_save=str(runtime);env.path_env_save=str(runtime)
   for scene in range(200):
    obs=base.load_scene_local_observations(env,scene)
    targets_all[scene]={}
    for role in replay.K_ROLES:
     targets_all[scene][role],_=teacher_targets(lib,role,obs[role])
   # Controls are checked both before and after interventions.
   for label,source in cached.items():
    for scene in range(200):
     m=score(env,source[scene]["env_actions"],scene)
     diffs=replay.control_diff(source[scene]["metrics"],m)
     if diffs:raise RuntimeError("CONTROL_MISMATCH "+label+" "+str(scene)+" "+repr(diffs))
     rows[label]["original"].append({"original_scene_id":scene,"metrics":m})
  print("CONTROLS_PASSED",600,str(out),flush=True)
  with (out/"records.jsonl").open("x",encoding="utf-8") as sink:
   for label,source in cached.items():
    for method in METHODS:
     for alpha in ALPHAS:
      name=method+"_a"+str(alpha)
      values=[]
      with contextlib.redirect_stdout(log):
       for scene in range(200):
        actions=modified_actions(source[scene]["env_actions"],targets_all[scene],method,alpha)
        m=score(env,actions,scene)
        row={"original_scene_id":scene,"cohort":label,"method":name,"metrics":m,"env_actions":actions}
        values.append(row);sink.write(json.dumps(row,allow_nan=False)+"\n")
      rows[label][name]=values;sink.flush();os.fsync(sink.fileno())
    # A post-intervention reference ensures candidate scoring never polluted a
    # supposedly reloaded scene (the failure mode of earlier discarded scripts).
    with contextlib.redirect_stdout(log):
     for scene in (0,39,100,199):
      m=score(env,source[scene]["env_actions"],scene)
      assert not replay.control_diff(source[scene]["metrics"],m)
    print("COHORT_COMPLETE",label,flush=True)
 summaries={label:{name:summarize(v) for name,v in arms.items()} for label,arms in rows.items()}
 comparisons={label:{name:paired(v,arms["original"]) for name,v in arms.items() if name!="original"} for label,arms in rows.items()}
 for relative,h in hashes.items():assert replay.sha256(ROOT/relative)==h
 result={"status":"COMPLETE","summaries":summaries,"comparisons":comparisons,"controls":600,"post_controls":12,
         "API_calls":0,"API_cost_usd_estimate":0,"source_hashes_unchanged":True}
 (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
 lines=["# Own TRAIN link-action calibration: development-only","",
        "Controls:600 before intervention,12 after. Three action draws on the same200 validation scenes.","",
        "| Cohort | Method | Admitted | Cooperative | V | Cost |","|---|---|---:|---:|---:|---:|"]
 for label,arms in summaries.items():
  for name,s in arms.items():
   lines.append(f"| {label} | {name} | {s['accepted']} | {s['cooperative']} | {s['V_mean']:.4f} | {s['cost_mean']:.4f} |")
 (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 print("OUTPUT",str(out),flush=True)
 print(json.dumps({c:{m:{k:v[k] for k in ("cooperative","V_mean","cost_mean")} for m,v in arms.items()} for c,arms in summaries.items()}),flush=True)

if __name__=="__main__":main()
