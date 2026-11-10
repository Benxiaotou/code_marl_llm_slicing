"""Offline distance-weighted three-neighbor link calibration probe."""
from __future__ import annotations
import contextlib,datetime as dt,json,sys,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"p2f_analysis"))
import val5601_counterfactual_replay as replay
from audit_qwen_full200 import summarize,paired
SOURCE=ROOT/"p2f_analysis/qwen_full200/run_20261104T141915_396460Z/full_original_records.jsonl"
def main():
 q={int(json.loads(x)["original_scene_id"]):json.loads(x) for x in SOURCE.open(encoding="utf-8")}
 man=json.loads((ROOT/"p1b_data/p2b_val_seed5601.json").read_text(encoding="utf-8"))
 import p1a_runner as base,p1c_runner
 from p1c.retrieval import LocalHistory
 from p2e_prompt.val5601_eval import verify_environment
 mm,_=base._config(ROOT,ROOT/"configs_main.yaml");base._compat_pickle_main_classes(mm);verify_environment(ROOT,man);lib=LocalHistory(ROOT/"p2b_libraries/p2b_standard_1000_v2")
 out=ROOT/"p2g_autonomous/results"/("distance_weighted_link_dev_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"));out.mkdir(parents=True,exist_ok=False);runtime=out/"env_runtime";runtime.mkdir()
 with (out/"env_stdout.log").open("x",encoding="utf-8") as log:
  with contextlib.redirect_stdout(log):env=p1c_runner.create_eval_env(ROOT,man,"configs_main.yaml")
 env.path_reward_save=str(runtime);env.path_env_save=str(runtime);rows=[]
 from p1a_runner import load_scene_local_observations,_get_step_metrics
 for scene in range(200):
  obs=load_scene_local_observations(env,scene);actions={r:list(q[scene]["env_actions"][r]) for r in replay.ROLES}
  for role in replay.K_ROLES:
   ms=lib.query(role,obs[role],top_k=3);dist=np.asarray([max(float(x["distance"]),.05) for x in ms]);w=(1/dist);w/=w.sum();target=np.sum(np.asarray([x["ratios"][2:] for x in ms])*w[:,None],axis=0)
   vec=np.asarray(actions[role],dtype=float);vec[2:]+=.5*(2*target-1-vec[2:]);actions[role]=vec
  load_scene_local_observations(env,scene);random.seed(1+scene);np.random.seed(1+scene);before=len(env.g_done_list);_,rew,_,_,_=env.step({r:np.asarray(v,dtype=np.float32) for r,v in actions.items()});rows.append({"original_scene_id":scene,"metrics":_get_step_metrics(env,before,rew)})
 original=[{"original_scene_id":i,"metrics":q[i]["metrics"]} for i in range(200)]
 data={"status":"COMPLETE","summary":summarize(rows),"comparison_vs_original":paired(rows,original),"API_calls":0,"API_cost_usd_estimate":0.0,"definition":"inverse local-distance weighting over three nearest TRAIN link ratios, alpha=.5","limitations":"development-only probe"}
 (out/"summary.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8");print(json.dumps({"output":str(out),"summary":data["summary"],"delta":data["comparison_vs_original"]["metrics"]},separators=(",",":")),flush=True)
if __name__=="__main__":main()
