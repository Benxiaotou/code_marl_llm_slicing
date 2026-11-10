"""Fixed-grid development replay of bounded local node release. Zero API calls."""
from __future__ import annotations
import contextlib
import datetime as dt
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'p2h_v_decomposition'))
from trace_service import MULTI, ORIGINALS, load_rows, windows_ram_gib, replay
from bounded_release import ARMS, adjust_local

MAPPO = ROOT / 'p2b_runs/val5601_mappo200/records.jsonl'
MANIFEST = ROOT / 'p1b_data/p2b_val_seed5601.json'
ROLES = replay.ROLES
K = ROLES[:3]


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def stats(rows):
    out = replay.aggregate(rows)
    out['admitted'] = sum(bool(r['metrics']['g_done']) for r in rows)
    out['cooperative'] = sum(bool(r['metrics']['cooperative_valid']) for r in rows)
    return out


def paired(a, b):
    assert [r['original_scene_id'] for r in a] == [r['original_scene_id'] for r in b]
    indices = np.random.default_rng(5601).integers(0, len(a), (10000, len(a)))
    out = {}
    for key in ('V', 'cost_sum', 'V_g', 'g_done', 'cooperative_valid'):
        x = np.array([float(r['metrics'][key]) for r in a])
        y = np.array([float(r['metrics'][key]) for r in b])
        delta = x - y
        out[key] = {'mean_delta': float(delta.mean()),
                    'exploratory_paired_CI95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()}
        if key in ('g_done', 'cooperative_valid'):
            out[key]['win_ids'] = [i for i in range(len(a)) if x[i] and not y[i]]
            out[key]['loss_ids'] = [i for i in range(len(a)) if y[i] and not x[i]]
    return out


def validate_metrics(m):
    ir = m['V_g'] >= -1e-8 and all(m['V_' + k] >= m['V_' + k + '_0'] - 1e-8
                                  for k in ('k1', 'k2', 'k3'))
    assert ir == m['individual_rationality']
    assert m['cooperative_valid'] == (m['g_done'] and ir)
    assert math.isclose(m['V'], sum(m['revenue_' + k] for k in ('k1', 'k2', 'k3', 'g'))
                        - m['cost_sum'], abs_tol=1e-7)


def main():
    import p1a_runner as base
    import p1c_runner
    from p2e_prompt.val5601_eval import verify_environment

    inputs = [MANIFEST, MAPPO, MULTI, *(p for p, _ in ORIGINALS.values())]
    metadata = []
    for p in inputs:
        assert p.is_file() and p.stat().st_size < 100_000_000
        metadata.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                         'format': p.suffix})
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    sample = Path(manifest['sample_file'])
    assert sample.is_file() and sample.stat().st_size == 644537810
    ram = windows_ram_gib()
    if ram < 6 or shutil.disk_usage(ROOT).free < 1024**3:
        raise RuntimeError('Need at least6GiB available RAM and1GiB free disk')
    originals = {label: load_rows(p, arm) for label, (p, arm) in ORIGINALS.items()}
    candidates = {label: {} for label in ORIGINALS}
    with MULTI.open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['method'] == 'link_mean3_a0.5':
                label, i = r['cohort'], r['original_scene_id']
                assert i not in candidates[label]
                candidates[label][i] = {'metrics': r['metrics'], 'env_actions': r['env_actions']}
    assert all(sorted(v) == list(range(200)) for v in candidates.values())
    mappo = load_rows(MAPPO)
    out = ROOT / 'p2i_node_boundary' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    runtime = out / 'env_runtime'; runtime.mkdir()
    sources = [*inputs, ROOT / 'main.py', ROOT / 'configs_main.yaml', Path(__file__).resolve(),
               Path(__file__).resolve().with_name('bounded_release.py'),
               ROOT / 'p1a_runner.py', ROOT / 'p1c_runner.py',
               ROOT / 'p2h_v_decomposition/trace_service.py',
               ROOT / 'p2f_analysis/val5601_counterfactual_replay.py']
    hashes = {str(p.relative_to(ROOT)): replay.sha256(p) for p in sources}
    environment = verify_environment(ROOT, manifest)
    protocol = {'schema': 'bounded_role_local_node_release_v1', 'dataset': manifest['dataset_id'],
                'split': 'validation', 'scene_ids': list(range(200)), 'seed': 1,
                'cached_model': 'qwen3-next-80b-a3b-instruct', 'frozen_base': 'link_mean3_a0.5',
                'cohorts': list(ORIGINALS), 'arms': [dict(name=n, max_ratio_increment=d,
                    extra_projected_overload_budget=b) for n, d, b in ARMS],
                'policy_inputs': 'Only own240-vector and own cached action; no scene IDs, g observations, peer actions, or replay feedback.',
                'selection_rule': 'Recommend only if all3 draws have positive mean V delta versus frozen candidate, no admission losses and no net cooperation decline. Exploratory development screening only.',
                'expected_pre_controls': 800, 'expected_interventions': 3000, 'expected_post_controls': 12,
                'expected_total_scores': 3812, 'API_calls': 0, 'estimated_API_cost_usd': 0,
                'input_metadata': metadata, 'sample_bytes': sample.stat().st_size,
                'available_RAM_GiB': ram, 'memory_plan': 'One original200-scene environment; require6GiB free; observations/actions only cached separately.',
                'output_plan': 'New exclusive directory, runtime redirected here; expected structured output below30MiB.',
                'environment': environment, 'source_hashes': hashes,
                'limitations': ['Same200 repeatedly used development scenes, not600 independent scenes.',
                                'Local projected-overload bound is not a migration or admission guarantee.',
                                'No independent-test selection or global diagnostic inputs.',
                                'No outcome-based fallback to baseline actions.']}
    write_json(out / 'protocol.json', protocol)
    print('PREFLIGHT', json.dumps({'output': str(out), 'available_RAM_GiB': ram,
                                'scores': 3812, 'API_calls': 0}), flush=True)
    rows = {label: {} for label in ORIGINALS}
    observations = {}
    progress = {'phase': 'initializing', 'completed': 0}
    try:
        with (out / 'env_stdout.log').open('x', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log):
                mm, _ = base._config(ROOT, ROOT / 'configs_main.yaml')
                base._compat_pickle_main_classes(mm)
                env = p1c_runner.create_eval_env(ROOT, manifest, 'configs_main.yaml')
            env.path_reward_save = str(runtime); env.path_env_save = str(runtime)

            def score(actions, scene):
                with contextlib.redirect_stdout(log):
                    obs = base.load_scene_local_observations(env, scene)
                    if scene not in observations:
                        observations[scene] = {r: obs[r].copy() for r in K}
                    random.seed(1 + scene); np.random.seed(1 + scene)
                    before = len(env.g_done_list)
                    _, rewards, _, _, _ = env.step({r: np.asarray(v, dtype=np.float32)
                                                    for r, v in actions.items()})
                    metrics = base._get_step_metrics(env, before, rewards)
                validate_metrics(metrics)
                progress['completed'] += 1
                return metrics

            with (out / 'controls.jsonl').open('x', encoding='utf-8') as controls:
                for label, source in [('MAPPO', mappo)] + list(candidates.items()):
                    progress['phase'] = 'pre_control_' + label
                    for scene in range(200):
                        progress['scene'] = scene
                        m = score(source[scene]['env_actions'], scene)
                        diff = replay.control_diff(source[scene]['metrics'], m)
                        if diff:
                            raise RuntimeError('CONTROL_MISMATCH ' + label + '/' + str(scene) + repr(diff))
                        controls.write(json.dumps({'case': label, 'original_scene_id': scene,
                                                   'metrics': m}, allow_nan=False) + '\n')
                    controls.flush(); os.fsync(controls.fileno())
                print('CONTROLS_PASSED', 800, flush=True)
                with (out / 'records.jsonl').open('x', encoding='utf-8') as sink:
                    for label, source in candidates.items():
                        for name, delta, budget in ARMS:
                            progress['phase'] = label + '/' + name
                            values = []
                            for scene in range(200):
                                progress['scene'] = scene
                                actions = {r: list(v) for r, v in source[scene]['env_actions'].items()}
                                audits = {}
                                for role in K:
                                    actions[role], audits[role] = adjust_local(observations[scene][role],
                                        source[scene]['env_actions'][role], delta, budget)
                                    assert actions[role][2:] == source[scene]['env_actions'][role][2:]
                                assert actions['agent_g'] == source[scene]['env_actions']['agent_g']
                                m = score(actions, scene)
                                row = {'cohort': label, 'method': name, 'original_scene_id': scene,
                                       'metrics': m, 'env_actions': actions, 'local_adjustment': audits}
                                values.append(row)
                                sink.write(json.dumps(row, allow_nan=False) + '\n')
                            sink.flush(); os.fsync(sink.fileno())
                            rows[label][name] = values
                        progress['phase'] = 'post_control_' + label
                        for scene in (0, 39, 100, 199):
                            m = score(source[scene]['env_actions'], scene)
                            assert not replay.control_diff(source[scene]['metrics'], m)
                            controls.write(json.dumps({'case': label, 'phase': 'post',
                                'original_scene_id': scene, 'metrics': m}, allow_nan=False) + '\n')
                        controls.flush(); os.fsync(controls.fileno())
                        print('COHORT_COMPLETE', label, flush=True)
        assert progress['completed'] == 3812
        for relative, h in hashes.items():
            assert replay.sha256(ROOT / relative) == h
        baselines = {'MAPPO': [dict(original_scene_id=i, **mappo[i]) for i in range(200)]}
        summaries, comparisons = {}, {}
        for label in ORIGINALS:
            original = [dict(original_scene_id=i, **originals[label][i]) for i in range(200)]
            frozen = [dict(original_scene_id=i, **candidates[label][i]) for i in range(200)]
            summaries[label] = {'original_Qwen': stats(original), 'frozen_candidate': stats(frozen)}
            comparisons[label] = {}
            for name, _, _ in ARMS:
                values = rows[label][name]
                summaries[label][name] = stats(values)
                p = paired(values, frozen)
                protected = [i for i in range(200) if frozen[i]['metrics']['g_done'] and not mappo[i]['metrics']['g_done']]
                p['Qwen_only_admission_losses'] = [i for i in protected if not values[i]['metrics']['g_done']]
                p['MAPPO_only_admissions_rescued'] = [i for i in range(200) if mappo[i]['metrics']['g_done']
                    and not frozen[i]['metrics']['g_done'] and values[i]['metrics']['g_done']]
                comparisons[label][name] = {'versus_frozen': p,
                                           'versus_MAPPO': paired(values, baselines['MAPPO'])}
        eligible = [name for name, _, _ in ARMS if all(
            comparisons[label][name]['versus_frozen']['V']['mean_delta'] > 0 and
            not comparisons[label][name]['versus_frozen']['g_done']['loss_ids'] and
            comparisons[label][name]['versus_frozen']['cooperative_valid']['mean_delta'] >= 0
            for label in ORIGINALS)]
        # Average the3 draws within each scene before bootstrapping; never pool600.
        scene_averaged = {}
        for name, _, _ in ARMS:
            deltas = np.array([[rows[label][name][i]['metrics']['V'] - candidates[label][i]['metrics']['V']
                                for i in range(200)] for label in ORIGINALS]).mean(axis=0)
            idx = np.random.default_rng(5601).integers(0, 200, (10000, 200))
            scene_averaged[name] = {'n_development_scenes': 200, 'mean_V_delta': float(deltas.mean()),
                'exploratory_scene_paired_CI95': np.quantile(deltas[idx].mean(axis=1), [.025, .975]).tolist()}
        result = {'status': 'COMPLETE', 'scores_completed': 3812, 'pre_controls': 800,
                  'interventions': 3000, 'post_controls': 12, 'metrics_accounting_checked': 3812,
                  'source_hashes_unchanged': True, 'API_calls': 0, 'estimated_API_cost_usd': 0,
                  'MAPPO': stats(baselines['MAPPO']), 'summaries': summaries,
                  'comparisons': comparisons, 'eligible_under_preregistered_rule': eligible,
                  'scene_averaged_V_deltas': scene_averaged,
                  'limitation': 'Exploratory development selection only; no independent superiority claim.'}
        write_json(out / 'summary.json', result)
        lines = ['# 有上限的局部节点释放：开发集离线回放', '',
                 '固定5方案 × 3批缓存动作 × 同一200开发场景；未调用API。未改变环境、奖励或指标。', '',
                 'safe只利用本地静态空闲容量；buffer允许新增预测超额最多为本切片容量的1%或2%。',
                 '节点ratio增量最多0.025或0.05，对应额外释放最多为本切片容量的2%或4%。',
                 '该上限不保证全局接纳、迁移可行或成本不升。策略不读取失败场景编号或全局缺口。', '',
                 '| 批次 | 方案 | 接纳 | 合作有效 | V | 成本 | 相对冻结候选 ΔV | 接纳新增/损失 | 合作新增/损失 |',
                 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
        for label, arms in summaries.items():
            for name, s in arms.items():
                if name in comparisons[label]:
                    c = comparisons[label][name]['versus_frozen']
                    extra = f"{c['V']['mean_delta']:+.4f} | {len(c['g_done']['win_ids'])}/{len(c['g_done']['loss_ids'])} | {len(c['cooperative_valid']['win_ids'])}/{len(c['cooperative_valid']['loss_ids'])}"
                else:
                    extra = '— | — | —'
                lines.append(f"| {label} | {name} | {s['admitted']} | {s['cooperative']} | {s['V_mean']:.4f} | {s['cost_mean']:.4f} | {extra} |")
        lines += ['', 'MAPPO：接纳%d，合作有效%d，V %.4f。' % (result['MAPPO']['admitted'],
                  result['MAPPO']['cooperative'], result['MAPPO']['V_mean']), '',
                  '预先规定的筛选条件：三个批次均ΔV>0、无原接纳损失、合作有效数不下降。',
                  '满足条件的方案：' + (', '.join(eligible) if eligible else '无；不建议据此直接扩大正式实验。'), '',
                  '同一场景的三批动作先平均后计算探索性区间；200场景不能视为600独立样本。', '',
                  '| 方案 | 三批场景内平均 ΔV | 探索性95%区间 |', '|---|---:|---|']
        for name, s in scene_averaged.items():
            ci = s['exploratory_scene_paired_CI95']
            lines.append(f"| {name} | {s['mean_V_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        lines += ['', '逐批配对区间、接纳/合作得失场景、Qwen额外接纳保护情况和8场诊断修复情况见summary.json。',
                  '候选是缓存Qwen动作上的确定性后处理，结果不能代表重新调用模型的表现。',
                  '此前已查看的独立测试未在本任务读取，也不得参与后续调参。', '',
                  '验证：800次前控制、12次后控制复现；3000次干预；3812次收入/成本和IR关系检查。',
                  '源文件哈希未变。API调用0，新增费用$0。', '', '复现（创建全新输出目录）：', '',
                  '```powershell',
                  "& 'python' -B .\\p2i_node_boundary\\run_replay.py",
                  '```', '']
        with (out / 'RESULTS.md').open('x', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print('OUTPUT', out, flush=True)
        print(json.dumps({'eligible': eligible, 'scene_averaged_V_deltas': scene_averaged}), flush=True)
    except Exception as exc:
        write_json(out / 'STOPPED.json', {**progress, 'error': repr(exc), 'API_calls': 0})
        raise


if __name__ == '__main__':
    main()
