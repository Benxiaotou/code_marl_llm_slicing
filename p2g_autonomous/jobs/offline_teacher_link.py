"""Offline link-action calibration from each role's nearest TRAIN teacher case."""
from __future__ import annotations
import contextlib, datetime as dt, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"p2f_analysis"))
import val5601_counterfactual_replay as replay
SOURCE=ROOT/"p2f_analysis/qwen_full200/run_20261104T141915_396460Z"
ALPHAS=(0.25,0.5,1.0)
ROLES=replay.ROLES; K_ROLES=replay.K_ROLES

def load(path):
 d={}
 with path.open(encoding="utf-8") as f:
  for line in f:
   r=json.loads(line); d[int(r["original_scene_id"])]=r
 return d
def aggregate(rows):
 return replay.aggregate([{"metrics":r} for r in rows])
def main():
 q=load(SOURCE/"full_original_records.jsonl")
 man=json.loads((ROOT/"p1b_data/p2b_val_seed5601.json").read_text(encoding="utf-8"))
 import p1a_runner, p1c_runner
 import p2e_prompt.val5601_eval as adapter
 from p1c.retrieval import LocalHistory
 main,_=p1a_runner._config(ROOT,ROOT/"configs_main.yaml"); p1a_runner._compat_pickle_main_classes(main)
 adapter.verify_environment(ROOT,man)
 lib=LocalHistory(ROOT/"p2b_libraries/p2b_standard_1000_v2")
 out=ROOT/"p2g_autonomous/results"/("offline_teacher_link_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
 out.mkdir(parents=True,exist_ok=False); runtime=out/"env_runtime"; runtime.mkdir()
 with (out/"env_stdout.log").open("x",encoding="utf-8") as log:
  with contextlib.redirect_stdout(log): env=p1c_runner.create_eval_env(ROOT,man,"configs_main.yaml")
 env.path_reward_save=str(runtime); env.path_env_save=str(runtime)
 rows={a:[] for a in ALPHAS}; retrieved=[]
 from p1a_runner import load_scene_local_observations,_get_step_metrics
 import random
 for scene in range(200):
  obs=load_scene_local_observations(env,scene)
  teacher={}
  for role in K_ROLES:
   match=lib.query(role,obs[role],top_k=1)[0]
   teacher[role]=(2*np.asarray(match["ratios"],dtype=float)-1.0).tolist()
   retrieved.append({"scene":scene,"role":role,"experience_id":match["experience_id"],"distance":match["distance"]})
  for alpha in ALPHAS:
   actions={}
   for role in K_ROLES:
    vec=np.asarray(q[scene]["env_actions"][role],dtype=float)
    tv=np.asarray(teacher[role],dtype=float)
    vec[2:]=vec[2:]+alpha*(tv[2:]-vec[2:])
    actions[role]=vec
   actions["agent_g"]=np.asarray(q[scene]["env_actions"]["agent_g"],dtype=float)
   random.seed(1+scene); np.random.seed(1+scene)
   before=len(env.g_done_list); _,rew,_,_,_=env.step(actions)
   m=_get_step_metrics(env,before,rew); rows[alpha].append(m)
 data={"status":"COMPLETE","cases":list(ALPHAS),"summaries":{str(a):aggregate(v) for a,v in rows.items()},
       "api_calls":0,"api_cost_usd_estimate":0.0,"retrieved":retrieved,
       "source":"fresh Qwen original actions; nearest own TRAIN teacher link dimensions only",
       "limitations":"offline diagnostic; validation data already used for development; no current MAPPO action input"}
 (out/"summary.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"output":str(out),"summaries":data["summaries"]},separators=(",",":")),flush=True)
if __name__=="__main__": main()
