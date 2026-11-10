"""Predeclared 3x512 final-test evaluation of unchanged frozen candidates."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import contextlib
import datetime as dt
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'p2j_validation'))
import run_validation as v
ROOT = v.ROOT
SEEDS = (2026110511, 2026110512, 2026110513)
COUNT = 512
TOTAL = len(SEEDS) * COUNT
CALLS = TOTAL * 4
MODEL = v.MODEL
WORKERS = 8
PRIOR_PROTOCOL = ROOT / 'p2j_validation/run_20261105T014642_470733Z/protocol.json'
PRIMARY = 'Frozen_link_node mean V minus MAPPO on all1536 fixed untouched test scenes'


def status(out, phase, **extra):
    data = {'phase': phase, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
    temp = out / 'status.tmp'
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    os.replace(str(temp), str(out / 'status.json'))


def generate_one(out, seed):
    """Worker: original generator, independently seeded, exclusive test files."""
    import p1a_runner as base
    from p1b_generate import generate, get_args
    assert seed in SEEDS and out.parent == ROOT / 'p2k_formal'
    name = 'p2k_test_seed' + str(seed)
    sample = out / 'datasets' / (name + '.pkl')
    with (out / ('generation_' + str(seed) + '.log')).open('x', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            main, conf = v.config_for(out / ('generator_runtime_' + str(seed)))
            cls = base.make_mini_env_type(main.CustomMultiAgentEnv, sample, COUNT)
            args = get_args(['--name', name, '--count', str(COUNT), '--seed', str(seed),
                '--split', 'test', '--sample-file', str(sample), '--manifest-dir', str(out / 'datasets')])
            manifest = generate(args, root=ROOT, env_class_override=lambda _: cls(conf))
    assert manifest['split'] == 'test' and manifest['n_scenes'] == COUNT
    print('DATASET_GENERATED', seed, COUNT, 'bytes', sample.stat().st_size, flush=True)


def infer_all(prepared, out):
    from p1a.policy import ChatCompletionsClient, parse_ratios, to_env_action
    stop = threading.Event(); lock = threading.Lock(); local = threading.local()
    completed = {}
    with (out / 'requests.jsonl').open('x', encoding='utf-8') as ledger, \
            (out / 'responses.jsonl').open('x', encoding='utf-8') as responses:
        def call(item, role):
            if stop.is_set():
                raise RuntimeError('CANCELLED_BEFORE_REQUEST')
            rid = str(item['dataset_seed']) + ':' + str(item['original_scene_id']) + ':' + role
            messages = item['per_role'][role]['messages']
            if not hasattr(local, 'client'):
                local.client = ChatCompletionsClient(base_url=os.environ['LLM_BASE_URL'],
                    api_key=os.environ['LLM_API_KEY'], model=MODEL, temperature=.1,
                    max_tokens=4096, read_timeout=120, stream=True)
            with lock:
                v.journal(ledger, {'event': 'started', 'request_id': rid, 'model': MODEL,
                                  'prompt_sha256': v.message_hash(messages)})
            try:
                response = local.client.complete(role, messages)
                if response.finish_reason != 'stop':
                    raise ValueError('Incomplete model response')
                ratios = parse_ratios(response.text, role)
                entry = {'request_id': rid, 'model': MODEL, 'env_action': to_env_action(role, ratios).tolist(),
                         'ratios': ratios, 'raw_model_output': response.text, 'usage': response.usage,
                         'latency_s': response.latency_s, 'finish_reason': response.finish_reason,
                         'prompt_sha256': v.message_hash(messages)}
                with lock:
                    v.journal(responses, entry)
                    v.journal(ledger, {'event': 'finished', 'request_id': rid, 'usage': response.usage,
                                      'latency_s': response.latency_s})
                return entry
            except Exception as exc:
                stop.set()
                with lock:
                    v.journal(ledger, {'event': 'failed', 'request_id': rid,
                                      'exception_type': type(exc).__name__, 'no_retry': True})
                raise RuntimeError('QWEN_REQUEST_FAILED ' + rid) from None
        with futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            pending = [pool.submit(call, item, role) for item in prepared for role in v.ROLES]
            last_report = time.monotonic()
            try:
                for future in futures.as_completed(pending):
                    entry = future.result()
                    assert entry['request_id'] not in completed
                    completed[entry['request_id']] = entry
                    if time.monotonic() - last_report >= 45:
                        status(out, 'Qwen_inference', completed_requests=len(completed), expected_requests=CALLS)
                        print('REQUEST_PROGRESS', len(completed), '/', CALLS, flush=True)
                        last_report = time.monotonic()
            except Exception:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
    assert len(completed) == CALLS
    return completed


def prepare_inputs(manifests, library, ir, policy, teacher, out, log):
    import p1a_runner as base
    from teacher_link_multicohort import teacher_targets
    prepared = []
    with (out / 'prepared.jsonl').open('x', encoding='utf-8') as sink:
        for manifest in manifests:
            seed = manifest['seed']
            status(out, 'prepare_prompts_and_MAPPO', dataset_seed=seed)
            with contextlib.redirect_stdout(log):
                env = v.make_env(manifest, out / ('prepare_runtime_' + str(seed)))
                for scene in range(COUNT):
                    obs = base.load_scene_local_observations(env, scene)
                    role_inputs = {}; targets = {}; target_ids = {}
                    for role in v.ROLES:
                        matches = policy.library.query(role, obs[role], top_k=3)
                        messages = policy._make_messages(role, obs[role], matches)
                        role_inputs[role] = {'messages': messages, 'prompt_sha256': v.message_hash(messages),
                            'retrieved_ids': [x['experience_id'] for x in matches],
                            'rerank_trace': policy.adapter.last_trace[role]}
                        if role in v.K:
                            t, info = teacher_targets(library, role, obs[role])
                            targets[role] = t['link_mean3'].tolist(); target_ids[role] = info['link_mean3_ids']
                    ma = teacher.act(obs, original_scene_id=scene, seed=1)
                    ma = {r: list(map(float, ma[r])) for r in v.ROLES}
                    item = {'dataset_seed': seed, 'original_scene_id': scene,
                            'observations': {r: obs[r].tolist() for r in v.ROLES}, 'per_role': role_inputs,
                            'link_targets': targets, 'link_target_TRAIN_ids': target_ids,
                            'MAPPO_actions': ma, 'MAPPO_metrics': v.score(env, ma, scene)}
                    prepared.append(item); v.journal(sink, item)
                del env
            gc.collect()
            print('SEED_PREPARED', seed, COUNT, 'scenes', flush=True)
    return prepared


def evaluate(prepared, responses, manifests, out):
    arms = {name: [] for name in v.ARMS}; controls = 0
    with (out / 'evaluation_stdout.log').open('x', encoding='utf-8') as log, \
            (out / 'records.jsonl').open('x', encoding='utf-8') as sink, \
            (out / 'post_controls.jsonl').open('x', encoding='utf-8') as post:
        for manifest in manifests:
            seed = manifest['seed']; status(out, 'paired_evaluation', dataset_seed=seed)
            with contextlib.redirect_stdout(log):
                env = v.make_env(manifest, out / ('evaluation_runtime_' + str(seed)))
                for item in (r for r in prepared if r['dataset_seed'] == seed):
                    scene = item['original_scene_id']
                    qa = {role: responses[str(seed) + ':' + str(scene) + ':' + role]['env_action'] for role in v.ROLES}
                    link, node, audits = v.derived_actions(qa, item['observations'], item['link_targets'])
                    for label, action in zip(v.ARMS, (item['MAPPO_actions'], qa, link, node)):
                        m = v.score(env, action, scene)
                        if label == 'MAPPO':
                            assert not v.replay.control_diff(item['MAPPO_metrics'], m)
                        row = {'dataset_seed': seed, 'original_scene_id': scene, 'arm': label,
                               'metrics': m, 'env_actions': action}
                        if label == 'Frozen_link_node':
                            row['node_adjustment'] = audits
                        arms[label].append(row); v.journal(sink, row)
                for label in v.ARMS:
                    first = next(r for r in arms[label] if r['dataset_seed'] == seed)
                    m = v.score(env, first['env_actions'], 0)
                    assert not v.replay.control_diff(first['metrics'], m)
                    v.journal(post, {'dataset_seed': seed, 'original_scene_id': 0, 'arm': label, 'metrics': m})
                    controls += 1
                del env
            gc.collect()
            print('SEED_EVALUATED', seed, COUNT, 'paired scenes,4 arms', flush=True)
    return arms, controls


def analysis(arms):
    comparisons = {label: v.paired(arms['Frozen_link_node'], arms[base]) for label, base in
        (('node_minus_MAPPO', 'MAPPO'), ('node_minus_link', 'Frozen_link'), ('node_minus_Qwen', 'Qwen_original'))}
    per_seed = {}; per_seed_comparisons = {}
    for seed in SEEDS:
        subsets = {name: [r for r in rows if r['dataset_seed'] == seed] for name, rows in arms.items()}
        per_seed[str(seed)] = {name: v.summarize(rows) for name, rows in subsets.items()}
        per_seed_comparisons[str(seed)] = {label: v.paired(subsets['Frozen_link_node'], subsets[base]) for label, base in
            (('node_minus_MAPPO', 'MAPPO'), ('node_minus_link', 'Frozen_link'))}
    return {'summaries': {name: v.summarize(rows) for name, rows in arms.items()},
            'comparisons': comparisons, 'per_seed': per_seed, 'per_seed_comparisons': per_seed_comparisons}


def report(out, result):
    lines = ['# 固定方案正式测试：3 × 512 新场景', '',
        '三个预先固定新种子，各512场景；全部1536场景完成后统一分析。没有调参、择优或可选停止。',
        '每场原Qwen、链路方案、节点方案共享一组真实Qwen动作；MAPPO使用原冻结检查点。', '',
        '| 方法 | 平均V | 接纳数/1536 | 合作有效数/1536 | 平均成本 |', '|---|---:|---:|---:|---:|']
    for name, s in result['summaries'].items():
        lines.append(f"| {name} | {s['V_mean']:.4f} | {s['admitted']} | {s['cooperative']} | {s['cost_mean']:.4f} |")
    lines += ['', '| 配对比较 | ΔV | 按生成种子分层配对95%区间 |', '|---|---:|---|']
    for name, c in result['comparisons'].items():
        ci = c['V']['paired_seed_stratified_CI95']
        lines.append(f"| {name} | {c['V']['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ['', '## 分种子', '', '| 种子 | 方法 | V | 接纳 | 合作有效 |', '|---|---|---:|---:|---:|']
    for seed, methods in result['per_seed'].items():
        for name, s in methods.items():
            lines.append(f"| {seed} | {name} | {s['V_mean']:.4f} | {s['admitted']} | {s['cooperative']} |")
    primary = result['comparisons']['node_minus_MAPPO']['V']
    lo, hi = primary['paired_seed_stratified_CI95']
    conclusion = ('主要终点的95%配对区间完全高于0，支持在本生成分布和固定策略下平均V高于MAPPO。' if lo > 0 else
                  '主要终点的95%配对区间完全低于0，当前方案平均V低于MAPPO。' if hi < 0 else
                  '主要终点的95%配对区间包含0，尚不足以确认平均V优于MAPPO。')
    lines += ['', '## 解释边界', '', conclusion,
        '主要比较预先固定为节点方案减MAPPO的平均V；其他比较及指标属于次要分析。',
        '新测试集在相同GEANT生成分布内独立生成；不代表跨拓扑/流量分布泛化。每场仅一次Qwen动作抽样。',
        '已完成正式测试的数据不得再用于提示词、参数或模型选择后声称是未使用最终测试。',
        '原环境、奖励、迁移、路由与评价指标均未改变。路径加权链路收入不能当成端到端吞吐量。',
        'API：6144次内网Qwen请求，无失败/重试/回退，不调用GPT；用户无限内网额度下新增费用估计$0。',
        'usage缺失请求数：' + str(result['token_usage']['requests_without_usage']) + '。缺失token用量未知，不按0报告。',
        '审计：6144提示词和响应、3072派生动作、6144评分记录、1536 MAPPO复现、12后控制及源文件/数据哈希通过。', '',
        '复现命令（新目录、新数据、再次6144次Qwen请求）：', '', '```powershell', '& .\\p2k_formal\\run_formal.ps1', '```', '']
    with (out / 'RESULTS.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def run():
    from trace_service import windows_ram_gib
    from p1c.retrieval import LocalHistory
    from p1d.ir import IRHistory
    from p2a.policy import IRRiskRerankPolicy
    from p1b.teacher import P0FrozenTeacher
    assert os.environ.get('P1A_MODEL') == MODEL and os.environ.get('LLM_API_KEY') and os.environ.get('LLM_BASE_URL')
    ram = windows_ram_gib(); disk = shutil.disk_usage(ROOT).free
    assert ram >= 16 and disk >= 12 * 1024**3, 'Need16GiB available RAM and12GiB disk'
    assert PRIOR_PROTOCOL.is_file() and PRIOR_PROTOCOL.stat().st_size < 100000
    prior = json.loads(PRIOR_PROTOCOL.read_text(encoding='utf-8'))
    # Check every frozen source against the successful new validation run.
    for relative, h in prior['source_sha256'].items():
        p = ROOT / relative
        assert p.is_file() and v.replay.sha256(p) == h, relative
    selected = json.loads(v.SELECTED.read_text(encoding='utf-8'))
    assert selected['max_ratio_increment'] == .025 and selected['extra_projected_overload_budget'] == .01
    library = LocalHistory(v.LIBRARY); ir = IRHistory(v.IR_LIBRARY, library)
    policy = IRRiskRerankPolicy(library, ir, client=object(), examples=3, seed=1)
    out = ROOT / 'p2k_formal' / ('run_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    sources = dict(prior['source_sha256'])
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name('audit_formal.py'),
                 Path(__file__).resolve().with_name('run_formal.ps1'), PRIOR_PROTOCOL):
        sources[str(path.relative_to(ROOT))] = v.replay.sha256(path)
    protocol = {'schema': 'frozen_node_final_test_3x512_v1', 'split': 'test', 'new_generation_seeds': SEEDS,
        'scenes_per_seed': COUNT, 'n_paired': TOTAL, 'arms': v.ARMS, 'frozen_candidate': selected,
        'model': MODEL, 'temperature': .1, 'max_tokens': 4096, 'stream': True, 'timeout_s': 120,
        'max_concurrent_requests': WORKERS, 'expected_Qwen_calls': CALLS, 'automatic_retries': 0,
        'fallback': False, 'optional_stopping': False, 'parameter_changes_during_run': False,
        'primary_endpoint': PRIMARY, 'secondary_endpoints': ['g_done', 'cooperative_valid', 'cost_sum', 'V_g'],
        'secondary_comparisons': ['node_minus_link', 'node_minus_Qwen'],
        'bootstrap': '10000 paired samples stratified by generation seed; RNG20261105',
        'one_Qwen_draw_per_scene': True, 'shared_Qwen_draw_across_postprocessors': True,
        'available_RAM_GiB': ram, 'available_disk_bytes': disk, 'parallel_generation_processes': 3,
        'resource_plan': 'Three generator workers,require16GiB free; about5GB new binary samples; structured data<300MiB; require12GiB disk.',
        'source_sha256': sources, 'API_cost_usd_estimate': 0,
        'cost_basis': 'User unlimited internal Qwen quota,not independently verified',
        'information_boundary': 'Own local observation and own TRAIN history only; no test feedback or MAPPO actions in decision.',
        'generation_distribution': 'Original GEANT generator/config unchanged; new independent seeds under same distribution.'}
    v.write_json(out / 'protocol.json', protocol)
    status(out, 'generation', expected_test_scenes=TOTAL, expected_API_calls=CALLS)
    print('FORMAL_OUTPUT', out, flush=True)
    print('PREFLIGHT', json.dumps({'RAM_GiB': ram, 'disk_GiB': disk / 1024**3, 'scenes': TOTAL, 'Qwen_calls': CALLS}), flush=True)
    teacher = None; phase = 'generation'
    try:
        processes = []
        for seed in SEEDS:
            f = (out / ('generator_process_' + str(seed) + '.log')).open('x', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--generate', str(out), str(seed)],
                cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            processes.append((seed, proc, f))
        remaining = set(SEEDS)
        while remaining:
            for seed, proc, f in processes:
                if seed in remaining and proc.poll() is not None:
                    f.close(); remaining.remove(seed)
                    if proc.returncode:
                        # Preserve other generation jobs; no model calls have started.
                        for _, child, handle in processes:
                            if child.poll() is None:
                                child.wait()
                            if not handle.closed:
                                handle.close()
                        raise RuntimeError('GENERATOR_FAILED ' + str(seed))
                    print('GENERATION_COMPLETE', seed, flush=True)
            status(out, 'generation', seeds_complete=len(SEEDS) - len(remaining), total_seeds=len(SEEDS))
            if remaining:
                time.sleep(5)
        manifests = [json.loads((out / 'datasets' / ('p2k_test_seed' + str(s) + '.json')).read_text(encoding='utf-8')) for s in SEEDS]
        for m in manifests:
            assert m['split'] == 'test' and m['n_scenes'] == COUNT
            assert m['main_py_sha256'] == sources['main.py'] and m['config_sha256'] == sources['configs_main.yaml']
            library.assert_independent(m)
        v.write_json(out / 'datasets.json', manifests)
        phase = 'preflight_and_preparation'; status(out, phase)
        with (out / 'preparation_stdout.log').open('x', encoding='utf-8') as log:
            with contextlib.redirect_stdout(log):
                _, conf = v.config_for(out / 'teacher_runtime', Path(manifests[0]['sample_file']), COUNT)
                conf.render = False; conf.test_mode = True; conf.device = 'cpu'; conf.parallels = 1
                teacher = P0FrozenTeacher(conf, v.CHECKPOINT)
            v.historical_preflight(policy, teacher, out, log)
            print('HISTORICAL_PREFLIGHT_PASSED', '12 prompts,3 MAPPO actions,6 metrics', flush=True)
            prepared = prepare_inputs(manifests, library, ir, policy, teacher, out, log)
        teacher.close(); teacher = None; gc.collect()
        assert len(prepared) == TOTAL
        v.write_json(out / 'preflight.json', {'passed': True, 'new_test_scenes': TOTAL,
                     'prepared_prompts': CALLS, 'MAPPO_evaluations': TOTAL, 'API_calls': 0})
        print('NEW_PREFLIGHT_PASSED', TOTAL, 'scenes', CALLS, 'prompts', flush=True)
        phase = 'Qwen_inference'; status(out, phase, completed_requests=0, expected_requests=CALLS)
        responses = infer_all(prepared, out)
        print('QWEN_COMPLETE', len(responses), flush=True)
        phase = 'paired_evaluation'; status(out, phase)
        arms, controls = evaluate(prepared, responses, manifests, out)
        phase = 'final_analysis_and_audit'; status(out, phase)
        for relative, h in sources.items():
            assert v.replay.sha256(ROOT / relative) == h
        for m in manifests:
            assert v.replay.sha256(Path(m['sample_file'])) == m['sample_sha256']
        result = analysis(arms)
        usages = [r['usage'] for r in responses.values()]
        token_usage = {'requests_without_usage': sum(not u for u in usages),
            'reported_token_totals': {key: sum(int(u.get(key, 0)) for u in usages if u) for key in
                                      ('prompt_tokens', 'completion_tokens', 'total_tokens')}}
        result.update({'status': 'SCORED_AWAITING_AUDIT', 'n_paired_scenes': TOTAL, 'generation_seeds': SEEDS,
            'API_calls': CALLS, 'API_cost_usd_estimate': 0, 'token_usage': token_usage,
            'source_and_dataset_hashes_unchanged': True, 'MAPPO_controls_reproduced': TOTAL,
            'post_controls': controls, 'independent_final_test': True,
            'parameters_changed': False, 'fallback_or_retry': False})
        v.write_json(out / 'scored_summary.json', result)
        from audit_formal import audit
        audit(out)
        result['status'] = 'COMPLETE'
        v.write_json(out / 'summary.json', result)
        report(out, result)
        status(out, 'COMPLETE', completed_requests=CALLS, paired_scenes=TOTAL, audit_passed=True)
        print('FORMAL_COMPLETE', out, flush=True)
        print(json.dumps({'summaries': result['summaries'], 'primary': result['comparisons']['node_minus_MAPPO']['V']}), flush=True)
    except Exception as exc:
        status(out, 'STOPPED', failed_phase=phase, error_type=type(exc).__name__)
        v.write_json(out / 'STOPPED.json', {'phase': phase, 'error_type': type(exc).__name__,
            'message': str(exc) if phase != 'Qwen_inference' else 'See request journal; no automatic retries.',
            'outputs_preserved': True})
        raise
    finally:
        if teacher is not None:
            teacher.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--generate', nargs=2, metavar=('OUT', 'SEED'))
    args = p.parse_args()
    if args.generate:
        generate_one(Path(args.generate[0]).resolve(), int(args.generate[1]))
    else:
        run()
