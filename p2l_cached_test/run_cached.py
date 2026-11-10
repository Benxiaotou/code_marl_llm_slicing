"""Fixed node policy on existing P2-C scenes, with archived method comparison."""
from __future__ import annotations
import contextlib
import datetime as dt
import gc
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'p2j_validation'))
import run_validation as v
from p2c_qwen8_queue import JOBS

SEEDS = (6601, 6701, 6801)
COUNT = 512
TOTAL = 1536
CACHE_ROOT = ROOT / 'p2c_runs/qwen8_test_v1'
PRIOR = ROOT / 'p2g_autonomous/results/formal_test_replay_20261104T165841_092219Z'
NEW = 'Frozen_link_node'
LINK = 'Frozen_link'
NAMES = {
    'mappo': 'MAPPO', 'knn_std1000': 'kNN_Standard1000',
    'icl_std1000': 'Qwen_ICL_Standard1000', 'iraware_std1000': 'Qwen_IR_Standard1000',
    'p2a_std1000': 'Qwen_P2A_Standard1000', 'p2a_base152': 'Qwen_P2A_Base152',
    'p2a_std500': 'Qwen_P2A_Standard500', 'p2a_bal1000': 'Qwen_P2A_Balanced1000',
}


def read_json(path, max_bytes=100000):
    assert path.is_file() and path.stat().st_size <= max_bytes, str(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def record_paths(seed, job):
    return [CACHE_ROOT / ('p2c_test_seed' + str(seed)) / job.tag /
            ('chunk_%04d_032' % start) for start in range(0, COUNT, 32)]


def metadata(path):
    assert path.is_file(), str(path)
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'format': path.suffix}


def frozen_meta_matches(meta, dataset, job, start):
    expected = {'method': job.method, 'eval_dataset_id': dataset['dataset_id'],
        'eval_dataset_split': 'test', 'eval_dataset_sha256': dataset['sample_sha256'],
        'eval_generation_seed': dataset['seed'], 'scene_ids': list(range(start, start + 32)),
        'seed': 1, 'config_sha256': v.replay.sha256(ROOT / 'configs_main.yaml'),
        'main_py_sha256': v.replay.sha256(ROOT / 'main.py'), 'mock_not_real_llm': False}
    for key, value in expected.items():
        assert meta.get(key) == value, (job.tag, start, key)
    for key in ('fallback_used', 'guard_used', 'task_finetuning', 'test_time_global_state_access',
                'cross_agent_communication'):
        assert meta.get(key) is False
    if job.tag == 'mappo':
        assert meta['checkpoint_sha256'] == 'fa36695c78feb28b3506fbd2858e6efbc2c8f44e1f98fd7fe0282acf191ebf03'
    elif job.method in ('icl', 'ir_icl', 'ir_rerank'):
        assert meta['model'] == v.MODEL and meta['temperature'] == .1 and meta['max_tokens'] == 4096


def load_history(datasets, library):
    """Read exact32-scene chunks; retain only needed metrics/actions, not logs."""
    rows = {NAMES[j.tag]: {} for j in JOBS}
    fingerprints = {}; meta_info = []
    for seed in SEEDS:
        for job in JOBS:
            for start, folder in zip(range(0, COUNT, 32), record_paths(seed, job)):
                for filename in ('manifest.json', 'summary.json', 'records.jsonl'):
                    p = folder / filename; info = metadata(p)
                    assert info['bytes'] < 20_000_000
                    meta_info.append(info); fingerprints[info['path']] = v.replay.sha256(p)
                meta = read_json(folder / 'manifest.json')
                frozen_meta_matches(meta, datasets[seed], job, start)
                if job.tag == 'p2a_std1000':
                    assert meta['library_manifest_sha256'] == library.manifest_sha256
                    assert meta['library_role_sha256'] == library.library_hashes
                summary = read_json(folder / 'summary.json')
                assert summary['n_scored'] == summary['n_attempted'] == 32
                assert summary['n_failed_policy'] == 0 and summary['fallback_rate'] == 0
                scene_ids = []
                with (folder / 'records.jsonl').open(encoding='utf-8') as f:
                    for line in f:
                        r = json.loads(line); i = r['original_scene_id']
                        assert r['status'] == 'ok' and not r['fallback_used']
                        assert i in range(start, start + 32) and (seed, i) not in rows[NAMES[job.tag]]
                        v.validate_metrics(r['metrics'])
                        row = {'dataset_seed': seed, 'original_scene_id': i,
                               'arm': NAMES[job.tag], 'metrics': r['metrics']}
                        if job.tag in ('mappo', 'p2a_std1000'):
                            row['env_actions'] = r['env_actions']
                            for role in v.ROLES:
                                a = np.asarray(r['env_actions'][role])
                                assert a.shape == (4,) and np.isfinite(a).all() and abs(a).max() <= 1.000001
                        rows[NAMES[job.tag]][seed, i] = row
                        scene_ids.append(i)
                assert scene_ids == list(range(start, start + 32))
    prior = {s: {} for s in SEEDS}
    for seed in SEEDS:
        path = PRIOR / ('records_' + str(seed) + '.jsonl')
        info = metadata(path); meta_info.append(info); fingerprints[info['path']] = v.replay.sha256(path)
        assert info['bytes'] < 5_000_000
        old = {}
        for line in path.open(encoding='utf-8'):
            r = json.loads(line); i = r['original_scene_id']
            assert r['seed'] == seed and (r['arm'], i) not in old
            old[r['arm'], i] = r['metrics']
        assert len(old) == COUNT * 3
        for i in range(COUNT):
            for arm, tag in (('MAPPO', 'mappo'), ('Qwen_original', 'p2a_std1000')):
                assert not v.replay.control_diff(rows[NAMES[tag]][seed, i]['metrics'], old[arm, i])
            prior[seed][i] = old['Frozen_link_mean3_a0.5', i]
    expected = {(s, i) for s in SEEDS for i in range(COUNT)}
    for source in rows.values():
        assert source.keys() == expected
        for key in expected:
            for k in ('k1', 'k2', 'k3'):
                assert math.isclose(source[key]['metrics']['V_' + k + '_0'],
                    rows['MAPPO'][key]['metrics']['V_' + k + '_0'], abs_tol=1e-7)
    return rows, prior, fingerprints, meta_info


def ordered(rows):
    return [rows[s, i] for s in SEEDS for i in range(COUNT)]


def comparisons(methods):
    """Paired comparisons for every existing formal method; no model selection."""
    candidate = ordered(methods[NEW])
    return {name: v.paired(candidate, ordered(rows)) for name, rows in methods.items() if name != NEW}


def write_report(out, result):
    lines = ['# 已有1536场景固定方案回放及横向比较', '',
        '使用已有P2-C种子6601、6701、6801，每种子512场景；不生成数据、不调用API。',
        '节点参数固定为ratio增量上限0.025、额外预测超额上限自身容量1%；链路校正alpha0.5。',
        '新节点与链路方案共享原P2-A Standard1000缓存动作；其他8种方法使用各自原正式记录。', '',
        '| 方法 | 平均V | 接纳率 | 合作有效率 | 平均成本 | 新节点方案减该方法的ΔV | 配对95%区间 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for name, s in result['summaries'].items():
        if name == NEW:
            extra = '— | —'
        else:
            c = result['comparisons'][name]['V']; ci = c['paired_seed_stratified_CI95']
            extra = f"{c['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(f"| {name} | {s['V_mean']:.4f} | {100*s['acceptance_rate']:.2f}% | "
                     f"{100*s['cooperative_rate']:.2f}% | {s['cost_mean']:.4f} | {extra} |")
    lines += ['', '## 各种子', '', '| 种子 | 方法 | V | 接纳数 | 合作有效数 |', '|---|---|---:|---:|---:|']
    for seed, methods in result['per_seed'].items():
        for name, s in methods.items():
            lines.append(f"| {seed} | {name} | {s['V_mean']:.4f} | {s['admitted']} | {s['cooperative']} |")
    lines += ['', '## 解释与完整性', '',
        '这些测试集此前已查看过，本次属于固定方案在已有正式测试集上的追加评估；不是全新未使用最终测试。',
        '本次不以结果调参或选择新参数；横向比较是描述性分析，不宣称所有9个比较均获多重比较校正。',
        '区间按生成种子分层，对相同(seed,scene)配对重采样10000次。',
        'MAPPO、原P2-A Qwen、链路方案各1536场重评分复现；其他6个既有方法只汇总原正式评分，不重新推理或评分。',
        '原模型、指标、迁移和路由未改。链路收入按路径负载计账，不能解释为端到端吞吐量。',
        '源文件与历史记录哈希未变，测试样本哈希与原清单一致。API调用0，新增费用$0。', '',
        '复现（新输出目录、无API）：', '', '```powershell', '& .\\p2l_cached_test\\run_cached.ps1', '```', '']
    with (out / 'RESULTS.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    import p1a_runner as base
    from trace_service import windows_ram_gib
    from p1c.retrieval import LocalHistory
    from teacher_link_multicohort import teacher_targets
    from p2c_qwen8_queue import eval_hotfix_approved
    ram = windows_ram_gib()
    assert ram >= 8 and shutil.disk_usage(ROOT).free >= 2 * 1024**3
    selected = read_json(v.SELECTED)
    assert selected['max_ratio_increment'] == .025 and selected['extra_projected_overload_budget'] == .01
    assert v.replay.sha256(ROOT / 'p2i_node_boundary/bounded_release.py') == selected['policy_source_sha256']
    prior_protocol = read_json(ROOT / 'p2j_validation/run_20261105T014642_470733Z/protocol.json')
    for relative, digest in prior_protocol['source_sha256'].items():
        assert v.replay.sha256(ROOT / relative) == digest
    library = LocalHistory(v.LIBRARY)
    datasets = {}; dataset_metadata = []
    for seed in SEEDS:
        path = ROOT / 'p1b_data' / ('p2c_test_seed' + str(seed) + '.json')
        m = read_json(path); sample = Path(m['sample_file'])
        assert m['seed'] == seed and m['n_scenes'] == COUNT and m['split'] == 'test'
        info = metadata(sample); assert info['bytes'] < 2_000_000_000
        dataset_metadata.extend([metadata(path), info])
        assert v.replay.sha256(sample) == m['sample_sha256']
        assert v.replay.sha256(ROOT / 'configs_main.yaml') == m['config_sha256']
        assert eval_hotfix_approved(ROOT, m['main_py_sha256'])
        library.assert_independent(m); datasets[seed] = m
    history, old_link, input_hashes, meta_info = load_history(datasets, library)
    out = ROOT / 'p2l_cached_test' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name('audit_cached.py'),
                 Path(__file__).resolve().with_name('run_cached.ps1')):
        input_hashes[str(path.relative_to(ROOT))] = v.replay.sha256(path)
    for relative, digest in prior_protocol['source_sha256'].items():
        input_hashes[relative] = digest
    for seed in SEEDS:
        p = ROOT / 'p1b_data' / ('p2c_test_seed' + str(seed) + '.json')
        input_hashes[str(p.relative_to(ROOT))] = v.replay.sha256(p)
    protocol = {'schema': 'existing_P2C1536_frozen_node_comparison_v1', 'seeds': SEEDS,
        'scenes_per_seed': COUNT, 'n_paired': TOTAL, 'dataset_metadata': dataset_metadata,
        'input_metadata': meta_info, 'source_hashes': input_hashes, 'frozen_candidate': selected,
        'API_calls': 0, 'estimated_API_cost_usd': 0, 'memory_available_GiB': ram,
        'resource_plan': 'One512-scene environment at a time; require8GiB free RAM,2GiB disk; new outputs<100MiB.',
        'historical_methods': NAMES, 'new_method': NEW, 'frozen_link_method': LINK,
        'primary_comparison': 'Frozen_link_node minus MAPPO mean V',
        'secondary_comparisons': 'New candidate minus each other archived method; descriptive,no familywise adjustment.',
        'bootstrap': '10000 paired resamples stratified by dataset seed,RNG20261105',
        'expected_scores': 6144, 'expected_control_reproductions': 4608, 'expected_post_controls': 12,
        'parameters_changed': False, 'test_previously_inspected': True,
        'independent_new_final_test': False, 'new_inference': False,
        'information_boundary': 'Only own local observation and own TRAIN link targets enter deterministic postprocessing; no historical method actions or score feedback in candidate.',
        'limitation': 'Re-evaluation on previously inspected P2-C tests. No tuning or fresh Qwen calls.'}
    v.write_json(out / 'protocol.json', protocol)
    print('CACHED_TEST_OUTPUT', out, flush=True)
    print('PREFLIGHT_PASSED', 'all8 archived methods complete,1536 matched scenes,API0,RAM_GiB', round(ram, 2), flush=True)
    phase = 'replay'; controls = 0; post_controls = 0; scored = {name: {} for name in ('MAPPO', NAMES['p2a_std1000'], LINK, NEW)}
    try:
        with (out / 'historical_metrics.jsonl').open('x', encoding='utf-8') as historical_sink:
            for name, source in history.items():
                for row in ordered(source):
                    v.journal(historical_sink, {k: row[k] for k in ('dataset_seed', 'original_scene_id', 'arm', 'metrics')})
        with (out / 'records.jsonl').open('x', encoding='utf-8') as sink, \
                (out / 'post_controls.jsonl').open('x', encoding='utf-8') as post:
            for seed in SEEDS:
                print('SEED_STARTED', seed, flush=True)
                with (out / ('env_stdout_' + str(seed) + '.log')).open('x', encoding='utf-8') as log:
                    with contextlib.redirect_stdout(log):
                        env = v.make_env(datasets[seed], out / ('runtime_' + str(seed)))
                        for scene in range(COUNT):
                            key = seed, scene
                            obs = base.load_scene_local_observations(env, scene)
                            qa = history[NAMES['p2a_std1000']][key]['env_actions']
                            ma = history['MAPPO'][key]['env_actions']
                            targets = {}; ids = {}
                            for role in v.K:
                                target, info = teacher_targets(library, role, obs[role])
                                targets[role] = target['link_mean3']; ids[role] = info['link_mean3_ids']
                            link, node, adjustments = v.derived_actions(qa, obs, targets)
                            action_cases = [('MAPPO', ma, history['MAPPO'][key]['metrics']),
                                (NAMES['p2a_std1000'], qa, history[NAMES['p2a_std1000']][key]['metrics']),
                                (LINK, link, old_link[seed][scene]), (NEW, node, None)]
                            for label, action, expected in action_cases:
                                m = v.score(env, action, scene)
                                if expected is not None:
                                    diff = v.replay.control_diff(expected, m)
                                    if diff:
                                        raise RuntimeError('CONTROL_MISMATCH ' + str(key) + label + repr(diff))
                                    controls += 1
                                row = {'dataset_seed': seed, 'original_scene_id': scene, 'arm': label,
                                       'metrics': m, 'env_actions': action}
                                if label == NEW:
                                    row.update({'node_adjustment': adjustments, 'link_TRAIN_ids': ids,
                                        'link_targets': {r: list(map(float, a)) for r, a in targets.items()},
                                        'own_observations': {r: obs[r].tolist() for r in v.K}})
                                scored[label][key] = row; v.journal(sink, row)
                        for label, source in scored.items():
                            row = source[seed, 0]
                            m = v.score(env, row['env_actions'], 0)
                            assert not v.replay.control_diff(row['metrics'], m)
                            v.journal(post, {'dataset_seed': seed, 'original_scene_id': 0, 'arm': label, 'metrics': m})
                            post_controls += 1
                        del env
                gc.collect()
                print('SEED_COMPLETE', seed, '2048 scores,1536 controls', flush=True)
        assert controls == 4608 and post_controls == 12
        phase = 'analysis'
        methods = dict(history); methods[LINK] = scored[LINK]; methods[NEW] = scored[NEW]
        for relative, digest in input_hashes.items():
            assert v.replay.sha256(ROOT / relative) == digest
        for m in datasets.values():
            assert v.replay.sha256(Path(m['sample_file'])) == m['sample_sha256']
        summaries = {name: v.summarize(ordered(source)) for name, source in methods.items()}
        per_seed = {str(seed): {name: v.summarize([source[seed, i] for i in range(COUNT)])
                               for name, source in methods.items()} for seed in SEEDS}
        comps = comparisons(methods)
        per_seed_comps = {str(seed): {name: v.paired([methods[NEW][seed, i] for i in range(COUNT)],
            [methods[name][seed, i] for i in range(COUNT)]) for name in ('MAPPO', LINK)} for seed in SEEDS}
        result = {'status': 'SCORED_AWAITING_AUDIT', 'n_paired_scenes': TOTAL, 'seeds': SEEDS,
            'summaries': summaries, 'per_seed': per_seed, 'comparisons': comps,
            'per_seed_comparisons': per_seed_comps, 'controls_reproduced': controls,
            'post_controls': post_controls, 'new_policy_scores': TOTAL, 'total_scores': 6144,
            'archived_method_metric_rows': 12288, 'API_calls': 0, 'estimated_API_cost_usd': 0,
            'source_and_dataset_hashes_unchanged': True, 'test_previously_inspected': True,
            'parameters_changed': False, 'new_inference': False}
        v.write_json(out / 'scored_summary.json', result)
        phase = 'audit'
        from audit_cached import audit
        audit(out)
        result['status'] = 'COMPLETE'; v.write_json(out / 'summary.json', result)
        write_report(out, result)
        print('CACHED_TEST_COMPLETE', out, flush=True)
        print(json.dumps({'summaries': summaries, 'primary': comps['MAPPO']['V'],
                          'node_minus_link': comps[LINK]['V'], 'API_calls': 0}), flush=True)
    except Exception as exc:
        v.write_json(out / 'STOPPED.json', {'phase': phase, 'error': repr(exc), 'API_calls': 0})
        raise


if __name__ == '__main__':
    main()
