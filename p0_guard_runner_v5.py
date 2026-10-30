"""P0-v4: Qwen + verified feasibility shield, or offline replay of Qwen actions.

Independent entrypoint: does not modify or replace p0_runner.py or main.py.
Only a candidate that PASSED the original environment's g admission step on a
fresh shadow scenario will be executed in the actual evaluation environment.
"""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
from p0_guard import HardAdmissionGuard, NoFeasibleAction
from p0_policy import AGENTS, LLMPolicy, ratios_to_actions, ratio_actions
from p0_runner import prepare_config, stage_scene

# The fixed anchor is defined IN CODE, not taken from historical Qwen responses.
# All 15 effective ratios are constant across scenes; Guard remains scene-aware.
FIXED_ANCHOR_RATIOS = {
    "agent_k1": [0.2, 0.3, 0.2, 0.3],
    "agent_k2": [0.2, 0.3, 0.2, 0.3],
    "agent_k3": [0.2, 0.3, 0.2, 0.3],
    "agent_g": [0.2, 0.2, 0.2],
}



def _jsonl_append(fp, data):
    fp.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')
    fp.flush()


def _export_actions(action):
    return {k:[float(x) for x in np.asarray(action[k]).reshape(-1)] for k in AGENTS}


def _export_ratios(action):
    return {k:[float((x+1.)/2.) for x in np.asarray(action[k]).reshape(-1)] for k in AGENTS}


def _get_record_actions(path):
    by_id = {}
    with open(path,'r',encoding='utf-8') as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = row.get('original_scene_index',row.get('scene_index'))
            idx = int(idx)
            if idx in by_id:
                raise ValueError('Duplicate original scene index in raw records: {}'.format(idx))
            if row.get('used_fallback'):
                raise ValueError('Refusing to replay a fallback/heuristic-labeled record at scene {}'.format(idx))
            if not row.get('raw_response'):
                raise ValueError('Replay requires an actual Qwen raw_response for scene {}'.format(idx))
            by_id[idx] = row
    if not by_id:
        raise ValueError('Replay file has no Qwen actions')
    return by_id


def _summarize(rows):
    """Do not omit failed/refused scenes from admission denominators.

    If one or more uncommitted scenes lack actual environment utility,
    the whole-sample V and cost mean MUST be None, rather than a
    misleading success-only estimate.
    """
    if not rows:
        raise ValueError("No evaluated scenes")
    n = len(rows)
    admitted = sum(int(r.get("g_done", False)) for r in rows)
    raw_admitted = sum(int((r.get("raw_check") or {}).get("accepted", False))
                       for r in rows)
    changed = sum(int(r.get("guard_changed", False)) for r in rows)
    scored = [r for r in rows if all(k in r and r[k] is not None
                                    for k in ("V","V_g","cost_sum"))]
    mean_keys = ("V","V_g","cost_sum","api_latency_s",
                 "guard_latency_s","env_step_s","wall_time_s")
    means = {}
    for k in mean_keys:
        values = [float(r[k]) for r in rows if r.get(k) is not None]
        means[k] = (sum(values)/n if len(values)==n else None)
    source_model = rows[0].get("source_model")
    summary = {
        "n":n,
        "n_scored":len(scored),
        "n_unscored":n-len(scored),
        "policy":rows[0]["policy"],
        "source_model":source_model,
        "acceptance_rate":admitted/n,
        "raw_proposal_acceptance_rate":raw_admitted/n,
        "changed_action_rate":changed/n,
        **means,
    }
    if source_model is not None:
        summary["qwen_raw_acceptance_rate"] = raw_admitted/n
        summary["repair_rate"] = changed/n
    return summary

def cli():
    p=argparse.ArgumentParser(description='P0-v4 Qwen + original-env verified LP constraint repair')
    p.add_argument('--project-root',default='.')
    p.add_argument('--dataset',choices=('mini','full'),default='mini')
    p.add_argument('--sample-file',default=None)
    p.add_argument('--mini-indices-json',default='P0_small_samples.indices.json')
    p.add_argument('--indices',default=None,help='Comma-separated CURRENT dataset indices; overrides --count')
    p.add_argument('--count',type=int,default=None)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--output-dir',default=None)
    p.add_argument('--raw-records',default=None,help='Replay recorded Qwen JSON responses, NO API needed')
    p.add_argument('--solver-only',action='store_true',help='Non-LLM LP-only baseline')
    p.add_argument('--fixed-anchor',action='store_true',help='NON-LLM fixed numeric anchor + Guard; no log replay or API')
    p.add_argument('--on-infeasible',choices=('abort','record'),default='abort',
                   help='record: count as rejected and continue; never commit an unverified action')
    p.add_argument('--model',default='qwen3-next-80b-a3b-instruct')
    p.add_argument('--prompt-version',default='p0_v3b')
    p.add_argument('--api-url',default=None)
    p.add_argument('--api-timeout',type=float,default=20)
    p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--api-stream',dest='api_stream',action='store_true',default=True)
    p.add_argument('--no-api-stream',dest='api_stream',action='store_false')
    p.add_argument('--constraint-margin',type=float,default=0.001)
    return p.parse_args()


def main():
    args=cli()
    root=Path(args.project_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    if not (root/'main.py').is_file():
        raise FileNotFoundError('main.py not found in the project root')
    size=5 if args.dataset=='mini' else 600
    sample_file=(Path(args.sample_file).resolve() if args.sample_file else
                 (root/'P0_small_samples.pkl' if args.dataset=='mini' else None))
    if sample_file is not None and not sample_file.exists():
        raise FileNotFoundError('Scene file not found: '+str(sample_file))
    if args.indices is not None:
        indices=[int(s.strip()) for s in args.indices.split(',')]
    else:
        indices=list(range(args.count if args.count is not None else 2))
    if args.count is not None:
        indices=indices[:args.count]
    if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=size:
        raise ValueError('Invalid scene indices')
    original_ids={i:i for i in range(size)}
    if args.dataset=='mini':
        mapping=root/args.mini_indices_json
        if mapping.is_file():
            with mapping.open(encoding='utf-8') as fp:
                original_ids={int(k):int(v) for k,v in json.load(fp).items()}
            if set(original_ids)!=set(range(size)):
                raise ValueError('Mini indices mapping must cover 0..4')
    if args.dataset=='full' and sample_file is None:
        original_full=root/'data'/'GEANT'/'GEANT_3_12_600_sample_save.pkl'
        if not original_full.is_file():
            raise FileNotFoundError('Original 600-scene dataset not found')

    if sum((bool(args.solver_only), bool(args.fixed_anchor), bool(args.raw_records))) > 1:
        raise ValueError('--solver-only, --fixed-anchor, --raw-records are mutually exclusive')
    raw_records=_get_record_actions(args.raw_records) if args.raw_records else None
    if raw_records is None and not args.solver_only and not args.fixed_anchor and not os.environ.get('LLM_API_KEY'):
        raise ValueError('LLM_API_KEY must be configured for LIVE Qwen mode')
    when=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    kind=('lp_only_baseline' if args.solver_only else
          ('fixed_anchor_guard' if args.fixed_anchor else
          ('recorded_qwen_guard' if raw_records is not None else 'qwen_guard')))
    source_model = None if args.solver_only or args.fixed_anchor else args.model
    out=Path(args.output_dir).resolve() if args.output_dir else root/'p0_runs'/(kind+'_'+when)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory is not empty. Choose a new --output-dir: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    config=prepare_config(root,sample_file,size,out,len(indices))
    from main import CustomMultiAgentEnv, InitVnetworkTrain
    # P0_FULL_PICKLE_LEGACY_ALIAS_V1
    # Original dataset was serialized with main.py executed as __main__.
    sys.modules["__main__"].InitVnetworkTrain = InitVnetworkTrain
    env=CustomMultiAgentEnv(config)
    if config.topology_file!='GEANT':
        raise ValueError('This runner is only for the original GEANT experiment')
    env.is_eval=True
    policy = None if raw_records is not None or args.solver_only or args.fixed_anchor else LLMPolicy(
        model=args.model,base_url=args.api_url,timeout=args.api_timeout,
        max_tokens=args.max_tokens,fail_mode='abort',stream=args.api_stream)
    guard=HardAdmissionGuard(args.constraint_margin)
    manifest={
        'policy':kind,
        'source_model':source_model,
        'prompt_version':None if args.solver_only or args.fixed_anchor else args.prompt_version,
        'raw_records_path':str(args.raw_records) if args.raw_records else None,
        'fixed_anchor_ratios':FIXED_ANCHOR_RATIOS if args.fixed_anchor else None,
        'on_infeasible':args.on_infeasible,
        'original_scenes':[int(original_ids[i]) for i in indices],
        'dataset':args.dataset,
        'seed':args.seed,
        'main_sha256':hashlib.sha256((root/'main.py').read_bytes()).hexdigest(),
        'p0_policy_sha256':hashlib.sha256((root/'p0_policy.py').read_bytes()).hexdigest(),
        'source_kind':('solver_only_no_llm' if args.solver_only else
                       ('fixed_anchor_no_llm' if args.fixed_anchor else
                       ('recorded_qwen_response' if raw_records is not None else 'live_qwen'))),
        'evaluation_method':'Candidate proposal -> original-env shadow admission check -> '
                            'LP ratio projection if needed -> original-env shadow recheck -> original-env final step',
        'disclosure':('LP-ONLY BASELINE; NO LLM WAS CALLED. ' if args.solver_only else
                       ('FIXED ANCHOR BASELINE; NO LLM WAS CALLED. ' if args.fixed_anchor else
                       'REPAIRED RESULTS ARE QWEN+CONSTRAINT-SOLVER, NOT QWEN-ONLY. ')) +
                     'The repair reads original full scene graphs and is privileged versus the 240-dim MARL observation.',
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    logs=[]
    try:
        with (out/'guard_records.jsonl').open('w',encoding='utf-8') as fp:
            for seq,idx in enumerate(indices):
                orig=int(original_ids[idx])
                seed=args.seed+orig
                random.seed(seed)
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                except ImportError:
                    pass
                obs=stage_scene(env,idx)
                base_python_rng=random.getstate()
                base_numpy_rng=np.random.get_state()
                t0=time.perf_counter()
                if args.solver_only:
                    raw_text=None
                    raw_action=ratio_actions({k:[0.0]*(3 if k=='agent_g' else 4) for k in AGENTS})
                    llm_latency=0.0
                    source_api_latency=0.0
                    prompt_hash=None
                elif args.fixed_anchor:
                    raw_text = None
                    raw_action = ratio_actions(FIXED_ANCHOR_RATIOS)
                    llm_latency = 0.0
                    source_api_latency = 0.0
                    prompt_hash = None
                elif raw_records is not None:
                    rec=raw_records.get(orig)
                    if rec is None:
                        raise ValueError('No recorded Qwen response for original scene '+str(orig))
                    raw_text=rec['raw_response']
                    raw_action=ratios_to_actions(raw_text)
                    llm_latency=0.0
                    source_api_latency=float(rec.get('api_latency_s',0.0))
                    prompt_hash=rec.get('prompt_hash')
                else:
                    try:
                        decision=policy.act(obs)
                    except Exception as exc:
                        if args.on_infeasible == 'abort':
                            raise
                        elapsed=float(time.perf_counter()-t0)
                        failed={
                            'policy':kind,'source_model':source_model,
                            'scene_index':int(idx),'original_scene_index':orig,'seq':seq,
                            'status':'LLM_ERROR_NOT_COMMITTED','g_done':False,
                            'raw_check':{'accepted':False},
                            'guard_changed':False,
                            'reason':'{}: {}'.format(type(exc).__name__,str(exc)[:600]),
                            'api_latency_s':elapsed,'live_api_latency_s':elapsed,
                            'guard_latency_s':0.0,'env_step_s':0.0,'wall_time_s':elapsed,
                            'note':'No verified action was executed; V/cost unavailable.'
                        }
                        logs.append(failed)
                        _jsonl_append(fp,failed)
                        print('scene={} LLM_ERROR_NOT_COMMITTED after {:.2f}s '
                              '(counts as acceptance failure)'.format(orig,elapsed),flush=True)
                        continue
                    raw_action=decision.action
                    raw_text=decision.raw_response
                    llm_latency=float(decision.latency_s)
                    source_api_latency=llm_latency
                    prompt_hash=decision.prompt_hash
                guard_start = time.perf_counter()
                try:
                    result=guard.check_and_repair(env,idx,raw_action,seed)
                except NoFeasibleAction as exc:
                    # A rejection IS an attempted scene and must count in acceptance.
                    # No new-slice admission is claimed and no unsafe action is
                    # committed to the original evaluation environment.
                    rejection = {
                        'policy':kind, 'source_model':source_model,
                        'scene_index':int(idx),'original_scene_index':orig,'seq':seq,
                        'status':'REJECTED_NO_VERIFIED_ACTION','g_done':False,
                        'raw_check':{'accepted':False},
                        'guard_changed':False,
                        'raw_response':raw_text,'raw_action':_export_actions(raw_action),
                        'prompt_hash':prompt_hash,
                        'api_latency_s':float(source_api_latency),
                        'live_api_latency_s':float(llm_latency),
                        'guard_latency_s':float(time.perf_counter()-guard_start),
                        'env_step_s':0.0,
                        'wall_time_s':float(time.perf_counter()-t0),
                        'reason':str(exc),
                        'note':'Original environment was NOT stepped; V and cost are NOT measured for this scene.'
                    }
                    logs.append(rejection)
                    _jsonl_append(fp,rejection)
                    print('scene={} REJECTED_NOT_COMMITTED (counts as failure in admission; '
                          'V=NA), guard_time={:.3f}s'.format(
                          orig,rejection['guard_latency_s']),flush=True)
                    if args.on_infeasible=='abort':
                        raise RuntimeError('Scene {} failed hard admission gate: {}'.format(orig,exc)) from exc
                    continue
                # Preserve exact original evaluation-seed state.
                random.setstate(base_python_rng)
                np.random.set_state(base_numpy_rng)
                step_start=time.perf_counter()
                _, rewards, _, _, _ = env.step(result.action)
                env_latency=time.perf_counter()-step_start
                if not bool(env.g_done_list[-1]):
                    raise RuntimeError('URGENT: original environment final step disagrees with shadow verification!')
                metrics={name:float(history[-1]) for name,history in env.dict_eval_index.items()}
                if not all(math.isfinite(v) for v in metrics.values()):
                    raise RuntimeError('Nonfinite metric in committed environment evaluation')
                item={
                    'policy':kind,'source_model':source_model,
                    'prompt_version':None if args.solver_only or args.fixed_anchor else args.prompt_version,
                    'scene_index':int(idx),'original_scene_index':orig,'seq':seq,
                    'g_done':True,
                    'raw_check':result.raw_check,'guard_check':result.checked,
                    'guard_changed':bool(result.guard_changed),
                    'raw_response':raw_text,'prompt_hash':prompt_hash,
                    'raw_action':_export_actions(raw_action),'action':_export_actions(result.action),
                    'raw_ratio':_export_ratios(raw_action),'ratio':_export_ratios(result.action),
                    'guard_details':result.diagnostics,'guard_dry_runs':int(result.dry_runs),
                    'api_latency_s':float(source_api_latency),
                    'live_api_latency_s':float(llm_latency),
                    'guard_latency_s':float(result.elapsed_s),
                    'env_step_s':float(env_latency),
                    'wall_time_s':float(time.perf_counter()-t0),
                    'cost_sum':float(metrics['cost_k1']+metrics['cost_k2']+metrics['cost_k3']),
                    **metrics,
                }
                logs.append(item)
                _jsonl_append(fp,item)
                print('scene={} raw_admitted={} verified_final={} repaired={} V={:.3f} '
                      'cost={:.3f} guard_time={:.3f}s{}'.format(
                         orig,int(result.raw_check['accepted']),int(item['g_done']),
                         int(result.guard_changed),item['V'],item['cost_sum'],
                         result.elapsed_s,' [LP ONLY]' if args.solver_only else (
                             ' [FIXED]' if args.fixed_anchor else (
                             ' [REPLAY]' if raw_records is not None else ''))),flush=True)
        summary=_summarize(logs)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        print('\nOutputs:',out)
        print(json.dumps(summary,indent=2,ensure_ascii=False))
    finally:
        if hasattr(env,'close'):
            env.close()

if __name__=='__main__':
    main()
