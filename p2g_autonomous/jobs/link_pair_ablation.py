"""Offline paired-role link calibration ablation."""
from __future__ import annotations
import contextlib,datetime as dt,json,sys,random
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"p2f_analysis"))
import val5601_counterfactual_replay as replay
from audit_qwen_full200 import summarize,paired
from teacher_link_multicohort import teacher_targets
SOURCE=ROOT/"p2f_analysis/qwen_full200/run_20261104T141915_396460Z/full_original_records.jsonl"
CASES={"k2_k3":("agent_k2","agent_k3"),"k1_k3":("agent_k1","agent_k3"),"k1_k2":("agent_k1","agent_k2")}
def main():
 q={int(json.loads(x)["original_scene_id"]):json.loads(x) for x in SOURCE.open(encoding="utf-8")}
 man=json.loads((ROOT/"p1b_data/p2b_val_seed5601.json").read_text(encoding="utf-8"))
 import p1a_runner as base,p1c_runner
 from p1c.retrieval import LocalHistory
 from p2e_prompt.val5601_eval import verify_environment
 mm,_=base._config(ROOT,ROOT/"configs_main.yaml");base._compat_pickle_main_classes(mm);verify_environment(ROOT,man);lib=LocalHistory(ROOT/"p2b_libraries/p2b_standard_1000_v2")
 out=ROOT/"p2g_autonomous/results"/("link_pair_ablation_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"));out.mkdir(parents=True,exist_ok=False);runtime=out/"env_runtime";runtime.mkdir()
 with (out/"env_stdout.log").open("x",encoding="utf-8") as log:
  with contextlib.redirect_stdout(log):env=p1c_runner.create_eval_env(ROOT,man,"configs_main.yaml")
 env.path_reward_save=str(runtime);env.path_env_save=str(runtime);rows={k:[] for k in CASES}
 from p1a_runner import load_scene_local_observations,_get_step_metrics
 with (out/"records.jsonl").open("x",encoding="utf-8") as sink:
  for scene in range(200):
   obs=load_scene_local_observations(env,scene);target={}
   for role in replay.K_ROLES:target[role],_=teacher_targets(lib,role,obs[role])
   for name,selected in CASES.items():
    actions={r:list(q[scene]["env_actions"][r]) for r in replay.ROLES}
    for role in selected:
     vec=np.asarray(actions[role],dtype=float);tv=2*np.asarray(target[role]["link_mean3"],dtype=float)-1;vec[2:]+=.5*(tv-vec[2:]);actions[role]=vec.tolist()
    load_scene_local_observations(env,scene);random.seed(1+scene);np.random.seed(1+scene);before=len(env.g_done_list);_,rew,_,_,_=env.step({r:np.asarray(v,dtype=np.float32) for r,v in actions.items()});m=_get_step_metrics(env,before,rew)
    rows[name].append({"original_scene_id":scene,"metrics":m});sink.write(json.dumps({"case":name,"original_scene_id":scene,"metrics":m},allow_nan=False)+"\n")
 summary={k:summarize(v) for k,v in rows.items()};original=[{"original_scene_id":i,"metrics":q[i]["metrics"]} for i in range(200)]
 data={"status":"COMPLETE","summary":summary,"comparisons_vs_original":{k:paired(v,original) for k,v in rows.items()},"API_calls":0,"API_cost_usd_estimate":0.0}
 (out/"summary.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8");print(json.dumps({"output":str(out),"summary":summary},separators=(",",":")),flush=True)
if __name__=="__main__":main()
