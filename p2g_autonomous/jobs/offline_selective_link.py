"""Offline selective link calibration by local retrieval distance."""
from __future__ import annotations
import contextlib, datetime as dt, json, sys, random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"p2f_analysis"))
import val5601_counterfactual_replay as replay
SOURCE=ROOT/"p2f_analysis/qwen_full200/run_20261104T141915_396460Z"
CASES=(("alpha_0p1_all",.1,float("inf")),("alpha_0p25_d0p2",.25,.2),("alpha_0p25_d0p3",.25,.3),("alpha_0p25_d0p5",.25,.5),("alpha_0p25_d0p75",.25,.75))
ROLES=replay.K_ROLES
def load(path):
 d={}
 with path.open(encoding="utf-8") as f:
  for line in f:
   r=json.loads(line); d[int(r["original_scene_id"])]=r
 return d
def main():
 q=load(SOURCE/"full_original_records.jsonl"); man=json.loads((ROOT/"p1b_data/p2b_val_seed5601.json").read_text(encoding="utf-8"))
 import p1a_runner,p1c_runner; import p2e_prompt.val5601_eval as adapter
 from p1c.retrieval import LocalHistory
 main,_=p1a_runner._config(ROOT,ROOT/"configs_main.yaml"); p1a_runner._compat_pickle_main_classes(main); adapter.verify_environment(ROOT,man)
 lib=LocalHistory(ROOT/"p2b_libraries/p2b_standard_1000_v2")
 out=ROOT/"p2g_autonomous/results"/("offline_selective_link_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")); out.mkdir(parents=True,exist_ok=False); runtime=out/"env_runtime"; runtime.mkdir()
 with (out/"env_stdout.log").open("x",encoding="utf-8") as log:
  with contextlib.redirect_stdout(log): env=p1c_runner.create_eval_env(ROOT,man,"configs_main.yaml")
 env.path_reward_save=str(runtime); env.path_env_save=str(runtime)
 rows={name:[] for name,_,_ in CASES}; dist_counts={name:0 for name,_,_ in CASES}
 from p1a_runner import load_scene_local_observations,_get_step_metrics
 for scene in range(200):
  obs=load_scene_local_observations(env,scene); teacher={}; dist={}
  for role in ROLES:
   m=lib.query(role,obs[role],top_k=1)[0]; teacher[role]=(2*np.asarray(m["ratios"],dtype=float)-1).tolist(); dist[role]=float(m["distance"])
  for name,alpha,threshold in CASES:
   actions={r:list(q[scene]["env_actions"][r]) for r in replay.ROLES}
   for role in ROLES:
    if dist[role] <= threshold:
     vec=np.asarray(actions[role],dtype=float); tv=np.asarray(teacher[role]); vec[2:]=vec[2:]+alpha*(tv[2:]-vec[2:]); actions[role]=vec
     dist_counts[name]+=1
   # Every case must reload the original scene before stepping; cases are paired
   # counterfactuals, never sequential environment transitions.
   load_scene_local_observations(env,scene)
   random.seed(1+scene); np.random.seed(1+scene); before=len(env.g_done_list)
   _,rew,_,_,_=env.step({r:np.asarray(v,dtype=np.float32) for r,v in actions.items()})
   rows[name].append(_get_step_metrics(env,before,rew))
 def agg(v): return replay.aggregate([{"metrics":x} for x in v])
 data={"status":"COMPLETE","summaries":{k:agg(v) for k,v in rows.items()},"modified_role_scene_counts":dist_counts,"api_calls":0,"api_cost_usd_estimate":0.0,"source":"fresh Qwen original actions; nearest own TRAIN teacher link only when distance gate passes","limitations":"offline development diagnostic; no current MAPPO action input"}
 (out/"summary.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"output":str(out),"summaries":data["summaries"],"modified":dist_counts},separators=(",",":")),flush=True)
if __name__=="__main__": main()
