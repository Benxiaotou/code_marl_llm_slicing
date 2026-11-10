#!/usr/bin/env python3
"""P2-D: zero-API, paired offline counterfactual replay on seed7601 scenes 0..9.

Uses the already-working P1-C frozen-MAPPO runner to reset the *real* GEANT
environment, but intercepts ONLY the environment's one joint-action step and
replaces its actions with previously recorded values. No LLM client is built.

This is a *local-only* script. Never use this to claim causal generalization;
control replays must reproduce previously recorded outcomes before crossover
metrics are reported as valid.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROLES = ('agent_k1', 'agent_k2', 'agent_k3', 'agent_g')
CASES = {
    'MM': {'agent_k1':'MAPPO','agent_k2':'MAPPO','agent_k3':'MAPPO','agent_g':'MAPPO'},
    'GG': {'agent_k1':'GPT','agent_k2':'GPT','agent_k3':'GPT','agent_g':'GPT'},
    'MG': {'agent_k1':'MAPPO','agent_k2':'MAPPO','agent_k3':'MAPPO','agent_g':'GPT'},
    'GM': {'agent_k1':'GPT','agent_k2':'GPT','agent_k3':'GPT','agent_g':'MAPPO'},
}
NUMERIC_METRICS = ('V','V_g','V_k1','V_k2','V_k3','V_k1_0','V_k2_0',
                   'V_k3_0','X_k1','X_k2','X_k3','cost_k1','cost_k2',
                   'cost_k3','revenue_k1','revenue_k2','revenue_k3',
                   'revenue_g','cost_sum','reward_mean')
BOOL_METRICS = ('g_done','individual_rationality','cooperative_valid')


def sha256_file(path: Path) -> str:
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20),b''):
            sha.update(block)
    return sha.hexdigest()


def dump_json(path: Path, data) -> None:
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf-8')


def _decode_rows(payload: bytes, label: str) -> list[dict]:
    rows=[]
    for i,line in enumerate(payload.decode('utf-8-sig').splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        if not isinstance(row,dict) or row.get('status')!='ok':
            raise ValueError(f'{label} line {i}: not a successful original record')
        rows.append(row)
    return rows


def _by_scene(rows: list[dict], label: str) -> dict[int,dict]:
    out={}
    for r in rows:
        sid=r.get('original_scene_id')
        if not isinstance(sid,int) or sid in out:
            raise ValueError(f'{label}: invalid/duplicate scene id: {sid!r}')
        if r.get('fallback_used'):
            raise ValueError(f'{label} scene {sid}: original fallback used')
        if not isinstance(r.get('metrics'),dict):
            raise ValueError(f'{label} scene {sid}: missing original metrics')
        if not isinstance(r.get('env_actions'),dict):
            raise ValueError(f'{label} scene {sid}: missing original env actions')
        for role in ROLES:
            vec=r['env_actions'].get(role)
            if not isinstance(vec,list) or len(vec)!=4 or any(
                    isinstance(v,bool) or not isinstance(v,(float,int)) or
                    not math.isfinite(v) or not -1.00001 <= v <= 1.00001
                    for v in vec):
                raise ValueError(f'{label} scene {sid}: malformed action {role}: {vec!r}')
        out[sid]=r
    return out


def load_archive(path: Path) -> tuple[dict[int,dict],dict[int,dict],dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Archive missing: {path}')
    z=zipfile.ZipFile(path)
    # The user's Compress-Archive produced THREE SAME-NAMED records.jsonl
    # entries; ZipFile.read('records.jsonl') would only get the LAST one.
    records=[]
    for entry in z.infolist():
        if Path(entry.filename).name.lower()=='records.jsonl':
            parsed=_decode_rows(z.open(entry).read(), f'{entry.filename} crc={entry.CRC:08x}')
            records.append((entry,parsed))
    if len(records)!=3:
        raise ValueError(f'Expected 3 records.jsonl members (MAPPO 10, GPT 5+5), found {len(records)}')
    mappo_candidates=[rs for _,rs in records if len(rs)==10]
    gpt_candidates=[rs for _,rs in records if len(rs)==5]
    if len(mappo_candidates)!=1 or len(gpt_candidates)!=2:
        raise ValueError('Archive structure unexpected: expected 10 MAPPO rows and two sets of 5 GPT rows')
    mm=_by_scene(mappo_candidates[0], 'MAPPO')
    gg=_by_scene(gpt_candidates[0]+gpt_candidates[1], 'GPT')
    expected=set(range(10))
    if set(mm)!=expected or set(gg)!=expected:
        raise ValueError(f'Expected identical scene IDs 0..9, got MAPPO {sorted(mm)}, GPT {sorted(gg)}')
    for i in expected:
        for role in ROLES:
            details=mm[i]['per_role'][role]
            if details.get('messages') is not None:
                raise ValueError('The 10-row record does not look like frozen MAPPO')
            details=gg[i]['per_role'][role]
            if not isinstance(details.get('raw_model_output'),str):
                raise ValueError(f'Scene {i}/{role}: GPT raw output missing')
        for k in ('V_k1_0','V_k2_0','V_k3_0'):
            if not math.isclose(float(mm[i]['metrics'][k]),float(gg[i]['metrics'][k]),
                                rel_tol=1e-7,abs_tol=1e-5):
                raise ValueError(f'Scene {i}: different pre-cooperation baseline {k}, not paired')
    stats={
        'zip_sha256':sha256_file(path),
        'mappo_scenes':len(mm), 'gpt_scenes':len(gg),
        'scene_ids':sorted(expected),
        'gpt_all_agent_g_fourth_values':[gg[i]['env_actions']['agent_g'][3] for i in range(10)],
        'recorded_mappo_accept_count':sum(bool(mm[i]['metrics']['g_done']) for i in mm),
        'recorded_gpt_accept_count':sum(bool(gg[i]['metrics']['g_done']) for i in gg),
        'members':[{'name':entry.filename,'size':entry.file_size,'crc32':f'{entry.CRC:08x}',
                    'scenes':len(rows)} for entry,rows in records]
    }
    return mm,gg,stats


def chosen_actions(mm: dict[int,dict], gg: dict[int,dict],case: str,scene: int)->dict[str,list[float]]:
    if case not in CASES: raise KeyError(case)
    return {role:list((mm if source=='MAPPO' else gg)[scene]['env_actions'][role])
            for role,source in CASES[case].items()}


def _convert_like(values,template):
    # Keep the shape/dtype/container expected by the original GEANT env.step.
    import numpy as np
    if isinstance(template,np.ndarray):
        candidate=np.asarray(values,dtype=template.dtype)
        if candidate.shape!=template.shape:
            raise ValueError(f'Action shape mismatch {candidate.shape} vs {template.shape}')
        return candidate
    if isinstance(template,(tuple,list)):
        if len(values)!=len(template):
            raise ValueError('Action vector length differs from environment template')
        return type(template)(float(v) for v in values)
    try:
        import torch
        if isinstance(template,torch.Tensor):
            candidate=torch.as_tensor(values,dtype=template.dtype,device=template.device)
            if candidate.shape!=template.shape: raise ValueError('Torch action shape mismatch')
            return candidate
    except ImportError:
        pass
    raise TypeError(f'Unsupported action component type {type(template).__name__}; stop rather than guess')


def rewrite_joint_action(template, replacements: dict[str,list[float]],roles=ROLES):
    """Return new joint action in original shape; never mutate template."""
    import numpy as np
    if isinstance(template,dict):
        if set(template)!=set(roles):
            raise ValueError(f'Environment action dict roles {list(template)} != expected {roles}')
        return {role:_convert_like(replacements[role],template[role]) for role in template}
    if isinstance(template,np.ndarray):
        expected=np.asarray([replacements[r] for r in roles],dtype=template.dtype)
        if expected.shape!=template.shape:
            raise ValueError(f'Joint action array shape {template.shape} != {expected.shape}')
        return expected
    try:
        import torch
        if isinstance(template,torch.Tensor):
            expected=torch.as_tensor([replacements[r] for r in roles],
                                     dtype=template.dtype,device=template.device)
            if expected.shape!=template.shape:
                raise ValueError(f'Joint torch action shape {template.shape} != {expected.shape}')
            return expected
    except ImportError:
        pass
    if isinstance(template,(tuple,list)):
        if len(template)!=len(roles): raise ValueError('Unexpected number of agents')
        seq=[_convert_like(replacements[role],template[i]) for i,role in enumerate(roles)]
        return type(template)(seq)
    raise TypeError(f'Unknown joint action structure {type(template).__name__}; stop rather than guess')


def replay_one(env,policy,scene,seed,cached,case,base):
    """Run original base.run_one_scene, intercepting exactly 1 real env.step."""
    chosen=chosen_actions(*cached,case,scene)
    unpatched=env.step
    hits=[]
    base_actions=[]
    def substitute(*args,**kwargs):
        if hits:
            raise RuntimeError('Expected exactly one env.step per scene; extra step rejected')
        keys=('actions','action','action_dict','agent_actions','action_dicts')
        if args:
            if not isinstance(args[0],(dict,list,tuple)):
                import numpy as np
                if not isinstance(args[0],np.ndarray):
                    raise TypeError('Unsupported first positional env.step arg')
            if len(args)>1:
                raise TypeError(f'env.step got {len(args)} positional args, expected 1; stop')
            original=args[0]
            updated=rewrite_joint_action(original,chosen)
            call_args=(updated,)
            call_kwargs=kwargs
        else:
            present=[k for k in keys if k in kwargs]
            if len(present)!=1:
                raise TypeError(f'Expected 1 action keyword, got keys {list(kwargs)}')
            key=present[0]
            original=kwargs[key]
            updated=rewrite_joint_action(original,chosen)
            call_args=()
            call_kwargs=dict(kwargs)
            call_kwargs[key]=updated
        base_actions.append(original)
        hits.append(1)
        return unpatched(*call_args,**call_kwargs)
    try:
        env.step=substitute
        policy.set_scene(scene)
        row=base.run_one_scene(env,policy,scene,scene,seed)
    finally:
        env.step=unpatched
    if len(hits)!=1:
        raise RuntimeError(f'Scene {scene}, {case}: env.step called {len(hits)} times, expected exactly 1')
    if row.get('status')!='ok':
        raise RuntimeError(f'Scene {scene}, {case}: row status={row.get("status")}; abort')
    # Make clear that the action actually scored was swapped, not the teacher's
    # proposal emitted by base.run_one_scene. Do not preserve private prompts.
    teacher_proposal=row.get('env_actions')
    row['env_actions']=chosen
    row['case']=case
    row['action_sources']=dict(CASES[case])
    row['action_provenance']='archived environment actions, NO new inference'
    row['policy_proposal_for_audit_only']=teacher_proposal
    row['per_role']={role:{'source':CASES[case][role],
                           'env_action':chosen[role]}
                     for role in ROLES}
    row['policy_latency_s']=None
    row['latency_note']='Teacher policy latency is irrelevant; only original env.score compared'
    return row


def control_diff(expected: dict, got: dict, *, atol:float=1e-4,rtol:float=1e-6)->list[str]:
    diffs=[]
    for k in BOOL_METRICS:
        if bool(expected[k])!=bool(got[k]):
            diffs.append(f'{k}: expected={expected[k]} replay={got[k]}')
    for k in NUMERIC_METRICS:
        if k not in expected or k not in got:continue
        a,b=float(expected[k]),float(got[k])
        if not math.isclose(a,b,abs_tol=atol,rel_tol=rtol):
            diffs.append(f'{k}: expected={a:.9f}, replay={b:.9f}, delta={b-a:.9f}')
    return diffs


def csv_write(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def compact_metric(record,case):
    m=record['metrics']
    return {'case':case,'scene':record['original_scene_id'],
            'accept':int(bool(m['g_done'])),
            'cooperative':int(bool(m['cooperative_valid'])),
            'IR':int(bool(m['individual_rationality'])),
            'V':float(m['V']), 'V_g':float(m['V_g']),
            'cost_sum':float(m['cost_sum']),
            'IR_margin_k1':float(m['V_k1'])-float(m['V_k1_0']),
            'IR_margin_k2':float(m['V_k2'])-float(m['V_k2_0']),
            'IR_margin_k3':float(m['V_k3'])-float(m['V_k3_0'])}


def aggregate(records):
    from statistics import mean
    return {case:{'n':len(items),
                  'accept_rate':mean(int(r['metrics']['g_done']) for r in items),
                  'cooperative_valid_rate':mean(int(r['metrics']['cooperative_valid']) for r in items),
                  'V_mean':mean(r['metrics']['V'] for r in items),
                  'V_g_mean':mean(r['metrics']['V_g'] for r in items),
                  'cost_sum_mean':mean(r['metrics']['cost_sum'] for r in items)}
            for case,items in records.items()}


def ensure_legacy_pickle_compat(root: Path, config: str) -> None:
    """Match p1c_runner's known-working initialization order.

    Existing GEANT samples may contain pickle references to
    __main__.InitVnetworkTrain. Creating the frozen teacher before setting up
    the original evaluation environment leaves that name undefined.
    Register the legacy classes using the original project helper before
    XuanCe constructs its internal environment.
    """
    from p1a_runner import _config, _compat_pickle_main_classes
    env_main, _ = _config(root, root / config)
    _compat_pickle_main_classes(env_main)
    entrypoint = sys.modules.get('__main__')
    if entrypoint is None or not hasattr(entrypoint, 'InitVnetworkTrain'):
        raise RuntimeError(
            'Legacy pickle setup did not expose __main__.InitVnetworkTrain; '
            'inspect p1a_runner._compat_pickle_main_classes before replaying.')


def parse(argv=None):
    p=argparse.ArgumentParser(description='Offline MAPPO/GPT cached joint-action crossover (seed7601 scenes0-9)')
    p.add_argument('--archive',default='GPT61_MAPPO_10scene_diagnosis.zip')
    p.add_argument('--project-root',default='.')
    p.add_argument('--dataset-manifest',default='p1b_data/gpt61_dev_seed7601.json')
    p.add_argument('--library',default='p2b_libraries/p2b_standard_1000_v2')
    p.add_argument('--config',default='configs_main.yaml')
    p.add_argument('--checkpoint',default='marl_models/mappo/dir_model_1/512000_GEANT_mappo_model_save.pth')
    p.add_argument('--output-dir',default='gpt61_runs/dev7601_cross_replay_v1')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--preflight',action='store_true',help='only inspect duplicate-name ZIP and cached actions, no GEANT load')
    p.add_argument('--control-atol',type=float,default=1e-4)
    return p.parse_args(argv)


def main(argv=None):
    args=parse(argv)
    if args.control_atol<0 or not math.isfinite(args.control_atol):
        raise ValueError('--control-atol must be finite and nonnegative')
    root=Path(args.project_root).resolve()
    archive=Path(args.archive)
    if not archive.is_absolute():archive=(root/archive).resolve()
    mm,gg,info=load_archive(archive)
    print('ARCHIVE PREFLIGHT:',json.dumps(info,ensure_ascii=False,indent=2))
    if args.preflight:
        print('PREFLIGHT PASS: 10 MAPPO + 10 GPT matched scenes and complete recorded 4D environment actions.')
        return 0

    # Import *only* the user's existing P1-C environment and frozen teacher,
    # never ChatCompletionsClient; no API call occurs on this path.
    sys.path.insert(0,str(root))
    import p1c_runner as p1c
    import p1a_runner as base
    from p1c.retrieval import LocalHistory
    from p1a.local_obs import AGENTS
    if tuple(AGENTS)!=ROLES:
        raise RuntimeError(f'Role order changed, expected {ROLES}, got {tuple(AGENTS)}')
    if args.seed!=1:
        print('WARNING: seed differs from original MAPPO run seed=1; controls will detect disagreement.')
    library=LocalHistory(root/args.library)
    manifest=p1c.checked_eval_manifest(root,root/args.dataset_manifest,library,args.config)
    if manifest.get('dataset_id')!='gpt61_dev_seed7601' or manifest.get('n_scenes',0)<10:
        raise ValueError('Wrong evaluation dataset: requires gpt61_dev_seed7601 with >= 10 scenes')
    for source in (mm,gg):
        if set(source)!=set(range(10)):raise ValueError('Not exactly scenes 0..9')

    out=root/args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Will not overwrite output: {out}')
    # We only instantiate frozen teacher; p1c_runner's mappo branch has no GPT/Qwen endpoint.
    policy_args=p1c.get_args([
        '--library',args.library,'--dataset-manifest',args.dataset_manifest,
        '--method','mappo','--start','0','--count','10',
        '--seed',str(args.seed),'--config',args.config,
        '--checkpoint',args.checkpoint,'--output-dir',args.output_dir,
    ])
    # Original P1-C creates the evaluation environment BEFORE its teacher.
    # Install the same __main__ pickle class aliases before the teacher's
    # internal XuanCe make_envs() loads historical sample files.
    ensure_legacy_pickle_compat(root, args.config)
    policy,client,teacher=p1c.build_policy(policy_args,library,root)
    if client is not None or teacher is None:
        raise RuntimeError('Unexpected API client on mappo-only replay path; abort')
    out.mkdir(parents=True,exist_ok=False) if not out.exists() else None
    protocol={'schema':'p2d_cached_cross_replay_v1_1','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
              'dataset_id':manifest['dataset_id'],'dataset_sample_sha256':manifest['sample_sha256'],
              'archive_sha256':info['zip_sha256'],'main_py_sha256':sha256_file(root/'main.py'),
              'config_sha256':sha256_file(root/args.config),
              'checkpoint_sha256':sha256_file(root/args.checkpoint),
              'seed':args.seed,'scene_ids':list(range(10)),
              'no_api_calls':True,'no_guard':True,'no_new_actions':True,
              'legacy_pickle_compat_installed':True,
              'cases':CASES,'controls_must_match':True,
              'control_atol':args.control_atol,
              'note':'Only GEANT env.step replays archived joint actions. Frozen teacher policy used to invoke original runner; proposal NOT scored.'}
    dump_json(out/'protocol.json',protocol)
    all_results={}
    errors=[]
    try:
        # Run *both* previously observed control joint-actions, then crossovers.
        # Validate controls before interpreting crossovers.
        for case in ('MM','GG','MG','GM'):
            env=p1c.create_eval_env(root,manifest,args.config)
            result=[]
            with (out/f'{case}_records.jsonl').open('w',encoding='utf-8') as sink:
                for scene in range(10):
                    row=replay_one(env,policy,scene,args.seed,(mm,gg),case,base)
                    result.append(row)
                    sink.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                    sink.flush()
                    if case in ('MM','GG'):
                        expected=(mm if case=='MM' else gg)[scene]['metrics']
                        diff=control_diff(expected,row['metrics'],atol=args.control_atol)
                        if diff:
                            errors.append({'case':case,'scene':scene,'diff':diff})
                            dump_json(out/'CONTROL_FAILED.json',{'errors':errors,'message':'Do NOT interpret any crossovers until controls match.'})
                            raise RuntimeError(f'CONTROL MISMATCH {case} scene {scene}: '+ '; '.join(diff[:5]))
                    metrics=row['metrics']
                    print(f'{case} scene={scene} accept={int(metrics["g_done"])} cooperative={int(metrics["cooperative_valid"])} V={metrics["V"]:.4f} cost={metrics["cost_sum"]:.4f}',flush=True)
            all_results[case]=result
            if case in ('MM','GG'):
                print(f'CONTROL {case}: all 10 scenes reproduced (metric tolerance {args.control_atol})',flush=True)
        s=aggregate(all_results)
        dump_json(out/'summary.json',{'status':'VALIDATED','cases':s,
                                      'n_scenes':10,'no_new_model_calls':True,
                                      'control_match_MM':True,'control_match_GG':True})
        rows=[]
        for case,items in all_results.items():
            rows.extend(compact_metric(r,case) for r in items)
        csv_write(out/'scene_metrics.csv',rows,['case','scene','accept','cooperative','IR','V','V_g','cost_sum',
                                                'IR_margin_k1','IR_margin_k2','IR_margin_k3'])
        comparison=[]
        keyed={case:{r['original_scene_id']:r for r in items} for case,items in all_results.items()}
        for i in range(10):
            d={'scene':i}
            for case in ('MM','GG','MG','GM'):
                met=keyed[case][i]['metrics']
                d.update({f'{case}_accept':int(met['g_done']),
                          f'{case}_cooperative':int(met['cooperative_valid']),
                          f'{case}_V':met['V'],f'{case}_Vg':met['V_g'],
                          f'{case}_cost':met['cost_sum']})
            comparison.append(d)
        fields=['scene']+[f'{case}_{metric}' for case in ('MM','GG','MG','GM')
                          for metric in ('accept','cooperative','V','Vg','cost')]
        csv_write(out/'paired_crossover_by_scene.csv',comparison,fields)
        print('\nVALIDATED CROSSOVER SUMMARY (n=10):',json.dumps(s,ensure_ascii=False,indent=2))
        print('OUTPUT:',out)
    finally:
        if teacher is not None and hasattr(teacher,'close'):teacher.close()
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('REPLAY ABORTED: '+str(exc),file=sys.stderr)
        raise
