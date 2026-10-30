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
    mean_keys = ('V','V_g','cost_sum','api_latency_s','guard_latency_s','env_step_s','wall_time_s')
    summary = dict(
        n=len(rows),
        policy=rows[0]['policy'],
        source_model=rows[0]['source_model'],
        acceptance_rate=sum(int(x['g_done']) for x in rows)/len(rows),
        raw_proposal_acceptance_rate=sum(int(x['raw_check']['accepted']) for x in rows)/len(rows),
        changed_action_rate=sum(int(x['guard_changed']) for x in rows)/len(rows),
        **{key:sum(float(x[key]) for x in rows)/len(rows) for key in mean_keys}
    )
    if summary['source_model'] is not None:
        summary['qwen_raw_acceptance_rate'] = summary['raw_proposal_acceptance_rate']
        summary['repair_rate'] = summary['changed_action_rate']
    return summary


def cli():
    p=argparse.ArgumentParser(description='P0-v4 Qwen + original-env verified LP constraint repair')
    p.add_argument('--project-root',default='.')
    p.add_argument('--dataset',choices=('mini','full'),default='mini')
    p.add_argument('--sample-file',default=None)
    p.add_argument('--mini-indices-json',default='P0_small_samples.indices.json')
    p.add_argument('--indices',default='0,1',help='Current mini sample indices; 0,1 means original scenes 0 and 1')
    p.add_argument('--count',type=int,default=None)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--output-dir',default=None)
    p.add_argument('--raw-records',default=None,help='Replay recorded Qwen JSON responses, NO API needed')
    p.add_argument('--solver-only',action='store_true',help='Non-LLM LP-only baseline; controls for the solver contribution')
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

    if args.solver_only and args.raw_records:
        raise ValueError('--solver-only and --raw-records are mutually exclusive')
    raw_records=_get_record_actions(args.raw_records) if args.raw_records else None
    if raw_records is None and not args.solver_only and not os.environ.get('LLM_API_KEY'):
        raise ValueError('LLM_API_KEY must be configured for LIVE Qwen mode')
    when=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    kind=('lp_only_baseline' if args.solver_only else
          ('recorded_qwen_guard' if raw_records is not None else 'qwen_guard'))
    source_model = None if args.solver_only else args.model
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
    policy = None if raw_records is not None or args.solver_only else LLMPolicy(
        model=args.model,base_url=args.api_url,timeout=args.api_timeout,
        max_tokens=args.max_tokens,fail_mode='abort',stream=args.api_stream)
    guard=HardAdmissionGuard(args.constraint_margin)
    manifest={
        'policy':kind,
        'source_model':source_model,
        'prompt_version':None if args.solver_only else args.prompt_version,
        'raw_records_path':str(args.raw_records) if args.raw_records else None,
        'original_scenes':[int(original_ids[i]) for i in indices],
        'dataset':args.dataset,
        'seed':args.seed,
        'main_sha256':hashlib.sha256((root/'main.py').read_bytes()).hexdigest(),
        'p0_policy_sha256':hashlib.sha256((root/'p0_policy.py').read_bytes()).hexdigest(),
        'source_kind':('solver_only_no_llm' if args.solver_only else
                       ('recorded_qwen_response' if raw_records is not None else 'live_qwen')),
        'evaluation_method':'Candidate proposal -> original-env shadow admission check -> '
                            'LP ratio projection if needed -> original-env shadow recheck -> original-env final step',
        'disclosure':('LP-ONLY BASELINE; NO LLM WAS CALLED. ' if args.solver_only else
                       'REPAIRED RESULTS ARE QWEN+CONSTRAINT-SOLVER, NOT QWEN-ONLY. ') +
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
                    decision=policy.act(obs)
                    raw_action=decision.action
                    raw_text=decision.raw_response
                    llm_latency=float(decision.latency_s)
                    source_api_latency=llm_latency
                    prompt_hash=decision.prompt_hash
                try:
                    result=guard.check_and_repair(env,idx,raw_action,seed)
                except NoFeasibleAction as exc:
                    rejection={
                        'policy':kind, 'scene_index':int(idx),'original_scene_index':orig,
                        'status':'REJECTED_NO_VERIFIED_ACTION',
                        'raw_response':raw_text, 'raw_action':_export_actions(raw_action),
                        'reason':str(exc),'note':'Original evaluation env.step NOT called.'
                    }
                    _jsonl_append(fp,rejection)
                    raise RuntimeError('Scene {} failed hard admission gate. Inspect {}: {}'.format(
                                       orig,out/'guard_records.jsonl',exc)) from exc
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
                    'prompt_version':None if args.solver_only else args.prompt_version,
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
                         result.elapsed_s,' [LP ONLY]' if args.solver_only else (' [REPLAY]' if raw_records is not None else '')),flush=True)
        summary=_summarize(logs)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        print('\nOutputs:',out)
        print(json.dumps(summary,indent=2,ensure_ascii=False))
    finally:
        if hasattr(env,'close'):
            env.close()

if __name__=='__main__':
    main()
