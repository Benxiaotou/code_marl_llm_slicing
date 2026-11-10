"""Additional fixed ablations on existing P2-C1536; no API inference."""
import contextlib
import gc
import json
import shutil

import numpy as np

import common as c
v = c.v


def run():
    from trace_service import windows_ram_gib
    from p1c.retrieval import LocalHistory
    from teacher_link_multicohort import teacher_targets
    import p1a_runner as base
    ram = windows_ram_gib()
    assert ram >= 8 and shutil.disk_usage(c.ROOT).free >= 12 * 1024**3
    hashes = c.frozen_sources()
    # The full fresh-test design is fixed in common.py before reading new scores.
    old_protocol = c.read(c.OLD / 'protocol.json', 5_000_000)
    for rel, h in old_protocol['source_hashes'].items():
        assert v.replay.sha256(c.ROOT / rel) == h, rel
    assert c.read(c.OLD / 'summary.json')['status'] == 'COMPLETE'
    assert c.read(c.OLD / 'audit.json')['passed']
    library = LocalHistory(v.LIBRARY)
    inputs = [c.fingerprint(c.OLD / name) for name in ('protocol.json','summary.json','audit.json','records.jsonl')]
    datasets = {}
    for seed in c.OLD_SEEDS:
        p = c.ROOT / 'p1b_data' / ('p2c_test_seed' + str(seed) + '.json')
        m = c.read(p); assert m['seed'] == seed and m['n_scenes'] == c.COUNT
        sample = c.fingerprint(m['sample_file']); assert sample['sha256'] == m['sample_sha256']
        inputs.extend([c.fingerprint(p), sample]); library.assert_independent(m); datasets[seed] = m
    prior = {a: {} for a in c.OLD_ARMS.values()}
    for row in c.lines(c.OLD / 'records.jsonl'):
        name = c.OLD_ARMS[row['arm']]; key = row['dataset_seed'], row['original_scene_id']
        assert key not in prior[name]
        prior[name][key] = {**row, 'arm': name}
    knn = {}
    job = next(j for j in c.cached.JOBS if j.tag == 'knn_std1000')
    for seed in c.OLD_SEEDS:
        for start, folder in zip(range(0, c.COUNT, 32), c.cached.record_paths(seed, job)):
            for name in ('manifest.json', 'summary.json', 'records.jsonl'):
                inputs.append(c.fingerprint(folder / name))
            meta = c.read(folder / 'manifest.json')
            c.cached.frozen_meta_matches(meta, datasets[seed], job, start)
            assert meta['library_manifest_sha256'] == library.manifest_sha256
            assert meta['library_role_sha256'] == library.library_hashes
            summary = c.read(folder / 'summary.json')
            assert summary['n_scored'] == summary['n_attempted'] == 32 and summary['n_failed_policy'] == 0
            seen = []
            for r in c.lines(folder / 'records.jsonl'):
                i = r['original_scene_id']; key = seed, i
                assert r['status'] == 'ok' and not r['fallback_used'] and key not in knn
                knn[key] = r; seen.append(i)
            assert seen == list(range(start, start+32))
    expected = {(s, i) for s in c.OLD_SEEDS for i in range(c.COUNT)}
    assert set(knn) == expected and all(set(rows) == expected for rows in prior.values())
    out = c.create_run('cached')
    protocol = c.protocol('existing_test_attribution', hashes, ram)
    protocol.update(input_metadata=inputs, expected_new_scores=8*c.TOTAL,
                    reused_scores=4*c.TOTAL, expected_kNN_controls=c.TOTAL,
                    expected_historical_probe_controls=24, expected_post_controls=36)
    v.write_json(out / 'protocol.json', protocol)
    print('PAPER_CACHED_OUTPUT', out, flush=True)
    methods = {a: dict(prior.get(a, {})) for a in c.ARMS}
    controls = probes = posts = 0
    try:
        with (out/'reused_records.jsonl').open('x',encoding='utf8') as reuse:
            for rows in prior.values():
                for key in sorted(rows):
                    v.journal(reuse, rows[key])
        with (out/'records.jsonl').open('x',encoding='utf8') as sink, \
             (out/'contexts.jsonl').open('x',encoding='utf8') as contexts, \
             (out/'controls.jsonl').open('x',encoding='utf8') as check:
            for seed in c.OLD_SEEDS:
                print('CACHED_SEED_STARTED', seed, flush=True)
                with (out/('env_'+str(seed)+'.log')).open('x',encoding='utf8') as log, contextlib.redirect_stdout(log):
                    env = v.make_env(datasets[seed], out/('runtime_'+str(seed)))
                    for scene in range(c.COUNT):
                        key = seed, scene; obs = base.load_scene_local_observations(env, scene)
                        qrow = prior['Qwen_both'][key]
                        for r in v.K:
                            assert np.array_equal(obs[r], qrow['own_observations'][r])
                        targets = {}; target_ids = {}
                        for r in v.K:
                            t, info = teacher_targets(library, r, obs[r])
                            targets[r] = t['link_mean3'].tolist(); target_ids[r] = info['link_mean3_ids']
                            assert targets[r] == qrow['link_targets'][r] and target_ids[r] == qrow['link_TRAIN_ids'][r]
                        ka = c.knn_action(library, obs)
                        assert ka == knn[key]['env_actions']
                        bases = {'MAPPO': prior['MAPPO_raw'][key]['env_actions'],
                                 'kNN': ka, 'Qwen': prior['Qwen_raw'][key]['env_actions']}
                        action = c.actions(bases, obs, targets)
                        for a in prior:
                            assert action[a] == prior[a][key]['env_actions']
                        v.journal(contexts, {'dataset_seed': seed,'original_scene_id': scene,
                            'observations': {r: obs[r].tolist() for r in v.ROLES},
                            'bases': bases,'link_targets': targets,'link_target_TRAIN_ids': target_ids,
                            'kNN_original_metrics': knn[key]['metrics']})
                        for arm in c.ARMS:
                            if arm in prior:
                                if scene in (0,c.COUNT-1):
                                    m = v.score(env, action[arm], scene)
                                    assert not v.replay.control_diff(prior[arm][key]['metrics'], m)
                                    v.journal(check, {'kind':'historical_probe','dataset_seed':seed,'original_scene_id':scene,'arm':arm,'metrics':m})
                                    probes += 1
                                continue
                            m = v.score(env, action[arm], scene)
                            if arm == 'kNN_raw':
                                assert not v.replay.control_diff(knn[key]['metrics'], m)
                                controls += 1
                            row = {'dataset_seed':seed,'original_scene_id':scene,'arm':arm,'metrics':m,'env_actions':action[arm]}
                            methods[arm][key] = row; v.journal(sink, row)
                    for arm in c.ARMS:
                        row = methods[arm][seed,0]; m = v.score(env,row['env_actions'],0)
                        assert not v.replay.control_diff(row['metrics'], m)
                        v.journal(check,{'kind':'post','dataset_seed':seed,'original_scene_id':0,'arm':arm,'metrics':m}); posts += 1
                    del env
                gc.collect(); print('CACHED_SEED_COMPLETE', seed, flush=True)
        assert controls == c.TOTAL and probes == 24 and posts == 36
        c.verify_sources(hashes)
        for x in inputs:
            assert v.replay.sha256(c.ROOT/x['path']) == x['sha256']
        result = c.stats(methods, c.OLD_SEEDS)
        result.update(status='SCORED_AWAITING_AUDIT',kind=protocol['kind'],n_paired=c.TOTAL,
            API_calls=0,API_cost_usd_estimate=0,kNN_controls=controls,historical_probes=probes,post_controls=posts,
            new_scores=8*c.TOTAL,reused_scores=4*c.TOTAL,parameters_changed=False,independent_final_test=False)
        v.write_json(out/'scored_summary.json',result)
        from audit import audit
        audit(out)
        result['status']='COMPLETE'; v.write_json(out/'summary.json',result); c.report(out,result)
        print('PAPER_CACHED_COMPLETE',out,flush=True)
        print(json.dumps({k:result['comparisons'][k]['V'] for k in c.CLAIMS}),flush=True)
    except Exception as exc:
        v.write_json(out/'STOPPED.json',{'error_type':type(exc).__name__,'error':str(exc),'API_calls':0})
        raise


if __name__ == '__main__':
    run()
