"""Audit exact saved actions, prompts, response journals, metrics and hashes."""
from pathlib import Path
import math

import numpy as np

import common as c
v=c.v


def audit(out):
    from p1c.retrieval import LocalHistory
    from teacher_link_multicohort import teacher_targets
    out=Path(out).resolve(); assert out.parent==c.ROOT/'p2m_paper'
    protocol=c.read(out/'protocol.json',5_000_000); summary=c.read(out/'scored_summary.json',500_000)
    assert summary['status']=='SCORED_AWAITING_AUDIT' and not summary['parameters_changed']
    assert protocol['claim_family_size']==3 and protocol['all12_arms_run_in_both_stages_regardless_of_old_results']
    c.verify_sources(protocol['source_hashes'])
    fresh=protocol['kind']=='untouched_test_confirmation'
    seeds=c.NEW_SEEDS if fresh else c.OLD_SEEDS
    expected={(s,i) for s in seeds for i in range(c.COUNT)}
    methods={a:{} for a in c.ARMS}
    rows=list(c.lines(out/'records.jsonl'))
    if not fresh:
        rows+=list(c.lines(out/'reused_records.jsonl'))
    assert len(rows)==12*c.TOTAL
    for r in rows:
        key=r['dataset_seed'],r['original_scene_id']; arm=r['arm']
        assert key not in methods[arm]; methods[arm][key]=r; v.validate_metrics(r['metrics'])
    assert all(set(m)==expected for m in methods.values())
    library=LocalHistory(v.LIBRARY)
    prompts_checked=requests=0; knn_controls=0
    if fresh:
        from p1d.ir import IRHistory
        from p2a.policy import IRRiskRerankPolicy
        from p1a.policy import parse_ratios,to_env_action
        policy=IRRiskRerankPolicy(library,IRHistory(v.IR_LIBRARY,library),client=object(),examples=3,seed=1)
        prepared=list(c.lines(out/'prepared.jsonl'))
        responses=list(c.lines(out/'responses.jsonl')); assert len(responses)==c.CALLS
        lookup={r['request_id']:r for r in responses}; assert len(lookup)==c.CALLS
        events=list(c.lines(out/'requests.jsonl')); assert len(events)==2*c.CALLS
        starts={e['request_id']:e for e in events if e['event']=='started'}
        ends={e['request_id']:e for e in events if e['event']=='finished'}
        assert len(starts)==len(ends)==c.CALLS and starts.keys()==ends.keys()==lookup.keys()
        assert {e['event'] for e in events}=={'started','finished'}
        for item in prepared:
            seed=item['dataset_seed']; scene=item['original_scene_id']; qa={}
            for role in v.ROLES:
                matches=policy.library.query(role,item['observations'][role],top_k=3)
                messages=policy._make_messages(role,item['observations'][role],matches)
                assert messages==item['per_role'][role]['messages']
                assert [x['experience_id'] for x in matches]==item['per_role'][role]['retrieved_ids']
                rid=str(seed)+':'+str(scene)+':'+role; r=lookup[rid]
                assert r['model']==v.MODEL and r['finish_reason']=='stop'
                assert v.message_hash(messages)==r['prompt_sha256']==starts[rid]['prompt_sha256']
                ratios=parse_ratios(r['raw_model_output'],role); assert ratios==r['ratios']
                qa[role]=to_env_action(role,ratios).tolist(); assert qa[role]==r['env_action']
                prompts_checked+=1
            item['bases']={'MAPPO':item['MAPPO_actions'],'kNN':c.knn_action(library,item['observations']),'Qwen':qa}
            assert not v.replay.control_diff(item['MAPPO_metrics'],methods['MAPPO_raw'][seed,scene]['metrics'])
        requests=len(responses)
        manifests=c.read(out/'datasets.json'); assert len(manifests)==3
        assert all(m['split']=='test' and m['n_scenes']==c.COUNT and m['seed'] in seeds for m in manifests)
        assert c.read(out/'historical_preflight.json')['passed'] and c.read(out/'preflight.json')['passed']
    else:
        for x in protocol['input_metadata']:
            assert v.replay.sha256(c.ROOT/x['path'])==x['sha256']
        prepared=list(c.lines(out/'contexts.jsonl'))
        manifests=[c.read(c.ROOT/'p1b_data'/('p2c_test_seed'+str(s)+'.json')) for s in seeds]
        for item in prepared:
            key=item['dataset_seed'],item['original_scene_id']
            assert not v.replay.control_diff(item['kNN_original_metrics'],methods['kNN_raw'][key]['metrics'])
            knn_controls+=1
        for old in c.lines(c.OLD/'records.jsonl'):
            arm=c.OLD_ARMS[old['arm']]; key=old['dataset_seed'],old['original_scene_id']
            assert old['env_actions']==methods[arm][key]['env_actions']
            assert not v.replay.control_diff(old['metrics'],methods[arm][key]['metrics'])
    assert len(prepared)==c.TOTAL
    for m in manifests:
        library.assert_independent(m)
        assert v.replay.sha256(Path(m['sample_file']))==m['sample_sha256']
    seen=set()
    for item in prepared:
        key=item['dataset_seed'],item['original_scene_id']; assert key in expected and key not in seen; seen.add(key)
        obs=item['observations']; targets={}
        for role in v.K:
            t,info=teacher_targets(library,role,obs[role]); targets[role]=t['link_mean3'].tolist()
            assert targets[role]==item['link_targets'][role]
            assert info['link_mean3_ids']==item['link_target_TRAIN_ids'][role]
        assert c.knn_action(library,obs)==item['bases']['kNN']
        acts=c.actions(item['bases'],obs,targets)
        for arm in c.ARMS:
            assert acts[arm]==methods[arm][key]['env_actions'],(arm,key)
    checks=list(c.lines(out/'controls.jsonl'))
    assert len(checks)==(36 if fresh else 60)
    assert sum(r['kind']=='post' for r in checks)==36
    for r in checks:
        key=r['dataset_seed'],r['original_scene_id']
        assert not v.replay.control_diff(r['metrics'],methods[r['arm']][key]['metrics'])
    for arm,source in methods.items():
        computed=v.summarize([source[k] for k in sorted(source)])
        for field,val in computed.items():
            assert math.isclose(val,summary['summaries'][arm][field],abs_tol=1e-9)
        for seed in seeds:
            computed=v.summarize([source[seed,i] for i in range(c.COUNT)])
            for field,val in computed.items():
                assert math.isclose(val,summary['per_seed'][str(seed)][arm][field],abs_tol=1e-9)
    for contrast in summary['comparisons'].values():
        a,b=contrast['arms']
        for field in ('V','cost_sum','V_g','g_done','cooperative_valid'):
            delta=np.array([methods[a][k]['metrics'][field]-methods[b][k]['metrics'][field] for k in sorted(expected)],dtype=float)
            assert math.isclose(delta.mean(),contrast[field]['mean_delta'],abs_tol=1e-9)
            lo,hi=contrast[field]['CI95']; assert math.isfinite(lo) and math.isfinite(hi) and lo<=hi
    assert summary['API_calls']==requests
    result={'passed':True,'paired_scenes':c.TOTAL,'arms':12,'metric_rows_checked':12*c.TOTAL,
            'actions_reconstructed':12*c.TOTAL,'kNN_historical_controls':knn_controls,
            'historical_probes':0 if fresh else 24,'post_controls':36,
            'prompts_and_responses_reconstructed':prompts_checked,'fresh_requests':requests,
            'source_and_dataset_hashes_unchanged':True,'API_calls_during_audit':0}
    v.write_json(out/'audit.json',result)
    return result
