"""Predeclared untouched3x512 confirmation, all12 arms, internal Qwen only."""
import contextlib
import gc
import json
import os
from pathlib import Path
import shutil

import common as c
v = c.v


def generate_one(out, seed):
    import p1a_runner as base
    from p1b_generate import generate, get_args
    assert seed in c.NEW_SEEDS
    name = 'p2m_test_seed' + str(seed)
    sample = out/'datasets'/(name+'.pkl')
    with (out/('generation_'+str(seed)+'.log')).open('x',encoding='utf8') as log, contextlib.redirect_stdout(log):
        main, conf = v.config_for(out/('generator_runtime_'+str(seed)))
        cls = base.make_mini_env_type(main.CustomMultiAgentEnv, sample, c.COUNT)
        args = get_args(['--name',name,'--count',str(c.COUNT),'--seed',str(seed),
                        '--split','test','--sample-file',str(sample),'--manifest-dir',str(out/'datasets')])
        m = generate(args,root=c.ROOT,env_class_override=lambda _: cls(conf))
    assert m['seed']==seed and m['split']=='test' and m['n_scenes']==c.COUNT
    print('DATASET_GENERATED',seed,c.COUNT,'bytes',sample.stat().st_size,flush=True)
    gc.collect()
    return m


def run():
    from trace_service import windows_ram_gib
    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    from p2a.policy import IRRiskRerankPolicy
    from p1b.teacher import P0FrozenTeacher
    assert os.environ.get('P1A_MODEL')==v.MODEL and os.environ.get('LLM_BASE_URL') and os.environ.get('LLM_API_KEY')
    assert c.NEW_SEEDS == c.formal.SEEDS and c.COUNT == c.formal.COUNT and c.CALLS == c.formal.CALLS
    ram = windows_ram_gib()
    assert ram >= 12 and shutil.disk_usage(c.ROOT).free >= 12*1024**3
    hashes = c.frozen_sources()
    library=LocalHistory(v.LIBRARY); ir=IRHistory(v.IR_LIBRARY,library)
    policy=IRRiskRerankPolicy(library,ir,client=object(),examples=3,seed=1)
    out=c.create_run('fresh'); protocol=c.protocol('untouched_test_confirmation',hashes,ram)
    protocol.update(expected_scores=12*c.TOTAL,expected_MAPPO_controls=c.TOTAL,expected_post_controls=36,
                    fresh_Qwen_calls=c.CALLS,split='test',independent_final_test=True)
    v.write_json(out/'protocol.json',protocol)
    print('PAPER_FRESH_OUTPUT',out,flush=True)
    print('PREFLIGHT',json.dumps({'RAM_GiB':ram,'scenes':c.TOTAL,'Qwen_calls':c.CALLS,'extra_fee_usd':0}),flush=True)
    teacher=None; phase='generation'
    try:
        c.formal.status(out,phase)
        manifests=[generate_one(out,s) for s in c.NEW_SEEDS]
        for m in manifests:
            assert m['main_py_sha256']==hashes['main.py'] and m['config_sha256']==hashes['configs_main.yaml']
            library.assert_independent(m)
        assert len({m['sample_sha256'] for m in manifests})==3
        v.write_json(out/'datasets.json',manifests)
        phase='preparation'; c.formal.status(out,phase)
        with (out/'preparation_stdout.log').open('x',encoding='utf8') as log:
            with contextlib.redirect_stdout(log):
                _,conf=v.config_for(out/'teacher_runtime',Path(manifests[0]['sample_file']),c.COUNT)
                conf.render=False; conf.test_mode=True; conf.device='cpu'; conf.parallels=1
                teacher=P0FrozenTeacher(conf,v.CHECKPOINT)
            v.historical_preflight(policy,teacher,out,log)
            print('HISTORICAL_PREFLIGHT_PASSED','12 prompts,3 MAPPO actions,6 metrics',flush=True)
            prepared=c.formal.prepare_inputs(manifests,library,ir,policy,teacher,out,log)
        teacher.close(); teacher=None; gc.collect()
        assert len(prepared)==c.TOTAL
        v.write_json(out/'preflight.json',{'passed':True,'prepared_scenes':c.TOTAL,'prepared_prompts':c.CALLS,'API_calls':0})
        c.verify_sources(hashes)
        phase='Qwen_inference'; c.formal.status(out,phase,expected_requests=c.CALLS,completed_requests=0)
        print('QWEN_INFERENCE_STARTED',c.CALLS,'requests,no retries,no GPT',flush=True)
        responses=c.formal.infer_all(prepared,out)
        print('QWEN_INFERENCE_COMPLETE',len(responses),flush=True)
        phase='evaluation'; c.formal.status(out,phase)
        methods={a:{} for a in c.ARMS}; controls=posts=0
        with (out/'records.jsonl').open('x',encoding='utf8') as sink, \
             (out/'controls.jsonl').open('x',encoding='utf8') as check:
            for m in manifests:
                seed=m['seed']; print('FRESH_SEED_STARTED',seed,flush=True)
                with (out/('env_'+str(seed)+'.log')).open('x',encoding='utf8') as log, contextlib.redirect_stdout(log):
                    env=v.make_env(m,out/('runtime_'+str(seed)))
                    for item in (x for x in prepared if x['dataset_seed']==seed):
                        scene=item['original_scene_id']; key=seed,scene
                        qa={r:responses[str(seed)+':'+str(scene)+':'+r]['env_action'] for r in v.ROLES}
                        bases={'MAPPO':item['MAPPO_actions'],'kNN':c.knn_action(library,item['observations']),'Qwen':qa}
                        acts=c.actions(bases,item['observations'],item['link_targets'])
                        for arm,action in acts.items():
                            metrics=v.score(env,action,scene)
                            if arm=='MAPPO_raw':
                                assert not v.replay.control_diff(item['MAPPO_metrics'],metrics)
                                controls+=1
                            row={'dataset_seed':seed,'original_scene_id':scene,'arm':arm,'metrics':metrics,'env_actions':action}
                            methods[arm][key]=row; v.journal(sink,row)
                    for arm in c.ARMS:
                        row=methods[arm][seed,0]; metrics=v.score(env,row['env_actions'],0)
                        assert not v.replay.control_diff(row['metrics'],metrics)
                        v.journal(check,{'kind':'post','dataset_seed':seed,'original_scene_id':0,'arm':arm,'metrics':metrics}); posts+=1
                    del env
                gc.collect(); print('FRESH_SEED_COMPLETE',seed,'512scenes,12arms',flush=True)
        assert controls==c.TOTAL and posts==36
        phase='analysis_and_audit'; c.formal.status(out,phase)
        c.verify_sources(hashes)
        for m in manifests:
            assert v.replay.sha256(Path(m['sample_file']))==m['sample_sha256']
        result=c.stats(methods,c.NEW_SEEDS)
        usages=[r['usage'] for r in responses.values()]
        result.update(status='SCORED_AWAITING_AUDIT',kind=protocol['kind'],n_paired=c.TOTAL,
            API_calls=len(responses),API_cost_usd_estimate=0,MAPPO_controls=controls,post_controls=posts,
            new_scores=12*c.TOTAL,parameters_changed=False,independent_final_test=True,
            token_usage={'requests_without_usage':sum(not u for u in usages),
                         'reported_totals':{k:sum(int(u.get(k,0)) for u in usages if u)
                                            for k in ('prompt_tokens','completion_tokens','total_tokens')}})
        v.write_json(out/'scored_summary.json',result)
        from audit import audit
        audit(out)
        result['status']='COMPLETE'; v.write_json(out/'summary.json',result); c.report(out,result)
        c.formal.status(out,'COMPLETE',completed_requests=len(responses),audit_passed=True)
        print('PAPER_FRESH_COMPLETE',out,flush=True)
        print(json.dumps({k:result['comparisons'][k]['V'] for k in c.CLAIMS}),flush=True)
    except Exception as exc:
        c.formal.status(out,'STOPPED',failed_phase=phase,error_type=type(exc).__name__)
        v.write_json(out/'STOPPED.json',{'phase':phase,'error_type':type(exc).__name__,
            'message':str(exc) if phase!='Qwen_inference' else 'See request journal; no automatic retries.',
            'outputs_preserved':True})
        raise
    finally:
        if teacher is not None:
            teacher.close()


if __name__=='__main__':
    run()
