"""Independent P2-C replay of frozen link-mean3 alpha=.5 postprocessor."""
from __future__ import annotations
import contextlib, datetime as dt, json, sys, random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"p2f_analysis"))
import val5601_counterfactual_replay as replay
from audit_qwen_full200 import summarize,paired
from teacher_link_multicohort import teacher_targets
SEEDS=(6601,6701,6801); TAG="p2a_std1000"; ALPHA=.5

def load_chunks(folder):
 rows={}
 for chunk in sorted(folder.glob("chunk_*")):
  p=chunk/"records.jsonl"
  if not p.is_file():continue
  with p.open(encoding="utf-8") as f:
   for line in f:
    r=json.loads(line); scene=int(r["original_scene_id"])
    if scene in rows:raise ValueError("duplicate scene")
    rows[scene]=r
 if sorted(rows)!=list(range(512)):raise ValueError(f"expected512 got{len(rows)}")
 return rows

def main():
 import p1a_runner as base,p1c_runner
 from p1c.retrieval import LocalHistory
 from p2e_prompt.val5601_eval import verify_environment
 lib=LocalHistory(ROOT/"p2b_libraries/p2b_standard_1000_v2")
 out=ROOT/"p2g_autonomous/results"/("formal_test_replay_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"));out.mkdir(parents=True,exist_ok=False)
 pooled={}; provenance={}
 for seed in SEEDS:
  manifest=json.loads((ROOT/f"p1b_data/p2c_test_seed{seed}.json").read_text(encoding="utf-8"))
  main,_=base._config(ROOT,ROOT/"configs_main.yaml");base._compat_pickle_main_classes(main)
  # verify exact dataset hash/config/generation provenance without val-only adapter
  assert manifest["split"]=="test" and manifest["n_scenes"]==512
  assert replay.sha256(Path(manifest["sample_file"]))==manifest["sample_sha256"]
  assert replay.sha256(ROOT/"configs_main.yaml")==manifest["config_sha256"]
  q=load_chunks(ROOT/f"p2c_runs/qwen8_test_v1/{'p2c_test_seed'+str(seed)}/{TAG}")
  m=load_chunks(ROOT/f"p2c_runs/qwen8_test_v1/{'p2c_test_seed'+str(seed)}/mappo")
  import p1c_runner
  runtime=out/f"env_runtime_{seed}";runtime.mkdir()
  with (out/f"env_stdout_{seed}.log").open("x",encoding="utf-8") as log:
   with contextlib.redirect_stdout(log): env=p1c_runner.create_eval_env(ROOT,manifest,"configs_main.yaml")
  env.path_reward_save=str(runtime);env.path_env_save=str(runtime)
  arms={"MAPPO":[],"Qwen_original":[],"Frozen_link_mean3_a0.5":[]}
  controls={"MAPPO":0,"Qwen_original":0}
  with (out/f"records_{seed}.jsonl").open("x",encoding="utf-8") as sink:
   for scene in range(512):
    obs=base.load_scene_local_observations(env,scene)
    targets={}
    for role in replay.K_ROLES:targets[role],_=teacher_targets(lib,role,obs[role])
    qa=q[scene]["env_actions"];ma=m[scene]["env_actions"]
    ca={r:list(qa[r]) for r in replay.ROLES}
    for role in replay.K_ROLES:
     vec=np.asarray(ca[role],dtype=float);tv=2*np.asarray(targets[role]["link_mean3"],dtype=float)-1
     vec[2:]+=.5*(tv-vec[2:]);ca[role]=vec.tolist()
    for label,source,actions in (("MAPPO",m[scene],ma),("Qwen_original",q[scene],qa),("Frozen_link_mean3_a0.5",None,ca)):
     from p1a_runner import load_scene_local_observations,_get_step_metrics
     load_scene_local_observations(env,scene);random.seed(1+scene);np.random.seed(1+scene)
     before=len(env.g_done_list);_,rew,_,_,_=env.step({r:np.asarray(v,dtype=np.float32) for r,v in actions.items()})
     met=_get_step_metrics(env,before,rew)
     if source is not None:
      diff=replay.control_diff(source["metrics"],met)
      if diff:raise RuntimeError(f"CONTROL_MISMATCH seed{seed} {label} scene{scene} {diff[:2]}")
      controls[label]+=1
     row={"seed":seed,"original_scene_id":scene,"arm":label,"metrics":met}
     arms[label].append(row);sink.write(json.dumps(row,allow_nan=False)+"\n")
    sink.flush()
  sums={k:summarize(v) for k,v in arms.items()}
  comps={"candidate_minus_qwen":paired(arms["Frozen_link_mean3_a0.5"],arms["Qwen_original"]),
         "candidate_minus_mappo":paired(arms["Frozen_link_mean3_a0.5"],arms["MAPPO"]),
         "qwen_minus_mappo":paired(arms["Qwen_original"],arms["MAPPO"])}
  pooled.update({str(seed):{"summaries":sums,"comparisons":comps,"controls":controls}})
  provenance[str(seed)]={"manifest":str((ROOT/f"p1b_data/p2c_test_seed{seed}.json").relative_to(ROOT)),
                         "qwen_dir":str((ROOT/f"p2c_runs/qwen8_test_v1/p2c_test_seed{seed}/{TAG}").relative_to(ROOT)),
                         "n":512}
 result={"status":"COMPLETE","schema":"p2c_frozen_link_mean3_replay_v1","seeds":list(SEEDS),"candidate":"mean of 3 nearest TRAIN link ratios, alpha=.5 applied to cached Qwen P2-A link dimensions","summaries":pooled,"provenance":provenance,"API_calls":0,"API_cost_usd_estimate":0.0,"test_used_for_tuning":False,"limitations":"P2-C actions are cached frozen Qwen/MAPPO controls; candidate is deterministic postprocessing, no new Qwen inference"}
 (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
 print("FORMAL_COMPLETE",str(out),json.dumps({s:{k:{x:v[x] for x in ("accepted","cooperative","V_mean","cost_mean")} for k,v in pooled[s]["summaries"].items()} for s in pooled},separators=(",",":")),flush=True)
if __name__=="__main__":main()
