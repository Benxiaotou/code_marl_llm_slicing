"""Passive task-demand telemetry on fixed cached actions; no model calls."""
import contextlib
import ctypes
import datetime as dt
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'p2m_paper/fresh_20261105T025939_435887Z'
SEEDS = (2026110511, 2026110512, 2026110513)
SCENES = tuple(range(16))
ARMS = ('MAPPO_raw', 'Qwen_raw', 'Qwen_both')
LABELS = {'MAPPO_raw': '原始MAPPO', 'Qwen_raw': '原始Qwen', 'Qwen_both': 'Qwen联合改进版'}


def write(path, obj):
    with path.open('x', encoding='utf-8') as sink:
        json.dump(obj, sink, ensure_ascii=False, indent=2, allow_nan=False)


def read(path, cap):
    assert path.is_file() and path.stat().st_size < cap
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def available_ram():
    class Mem(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong) for k in ('total', 'available', 'total_page', 'available_page',
                                             'total_virtual', 'available_virtual', 'extended')]
    value = Mem()
    value.length = ctypes.sizeof(value)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value))
    return value.available / 1024 ** 3


def tasks(net):
    count, compute, cache = 0, 0.0, 0.0
    for node in range(net.num_nodes):
        data = net.DG.nodes[node]
        items = data['tasks_info_list']
        assert len(items) == data['num_node_tasks']
        count += len(items)
        compute += math.fsum(float(t['dem_comp']) for t in items)
        cache += math.fsum(float(t['dem_cach']) for t in items)
    return {'task_count': count, 'compute': compute, 'cache': cache}


def traffic(net, tm_idx):
    """Original OD demand and path-weighted load, separately; no RNG/state changes."""
    tm = net.traffic_matrices[tm_idx] / 1e6
    od_total, path_load = 0.0, 0.0
    for pair in range(net.num_pairs):
        src, dst = net.pair_idx_to_sd[pair]
        demand = float(tm[src][dst])
        paths = net.shortest_paths_node[pair]
        if demand:
            assert paths
            od_total += demand
            path_load += demand * math.fsum(len(p) - 1 for p in paths) / len(paths)
    return {'od_Gbps': od_total, 'path_load_Gbps': path_load}


def assert_same(expected, got):
    assert set(expected) == set(got)
    for field, old in expected.items():
        new = got[field]
        if isinstance(old, bool):
            assert old == new, field
        else:
            assert math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-5), (field, old, new)


def summarize(rows):
    out = {}
    for arm in ARMS:
        rs = [r for r in rows if r['arm'] == arm]
        totals = {field: {'requested': math.fsum(r['requested'][field] for r in rs),
                          'served': math.fsum(r['served'][field] for r in rs)}
                  for field in ('task_count', 'compute', 'cache', 'od_Gbps', 'path_load_Gbps')}
        rates = {field: t['served'] / t['requested'] for field, t in totals.items()}
        admitted = sum(r['g_done'] for r in rs)
        conditional = {}
        for field in ('task_count', 'compute', 'cache'):
            denominator = math.fsum(r['requested'][field] for r in rs if r['g_done'])
            conditional[field] = totals[field]['served'] / denominator if denominator else None
        original_value = math.fsum(.01 * (r['requested']['compute'] + r['requested']['cache'])
                                   + 10 * r['requested']['path_load_Gbps'] for r in rs)
        served_value = math.fsum(r['revenue_g'] for r in rs)
        out[arm] = {'n': len(rs), 'admitted': admitted, 'acceptance_rate': admitted / len(rs),
                    'demand_satisfaction': rates, 'totals': totals,
                    'retention_given_admission': conditional,
                    'original_model_revenue_weighted_satisfaction': served_value / original_value,
                    'warning': 'Model-retained demand under original admission check; not measured completion/SLA.'}
    return out


def run(out):
    ram = available_ram()
    assert ram >= 8 and shutil.disk_usage(ROOT).free > 2 * 1024 ** 3
    manifests = read(SOURCE / 'datasets.json', 20_000)
    assert tuple(m['seed'] for m in manifests) == SEEDS
    main_hash = digest(ROOT / 'main.py')
    config_hash = digest(ROOT / 'configs_main.yaml')
    for m in manifests:
        p = Path(m['sample_file'])
        assert p.is_file() and 1_600_000_000 < p.stat().st_size < 1_700_000_000
        assert m['main_py_sha256'] == main_hash and m['config_sha256'] == config_hash
    protocol = {
        'kind': 'fixed48_scene_business_demand_telemetry_pilot', 'seeds': SEEDS,
        'original_scene_ids_per_seed': SCENES, 'selection': 'first16 per seed, fixed before pilot outcomes',
        'arms': ARMS, 'paired_scenes': 48, 'replays': 144, 'post_controls': 9,
        'minimum_RAM_GiB': 8, 'available_RAM_GiB': ram,
        'memory_plan': 'one512-scene pickle/environment at a time, about1.65GB on disk; no model/checkpoint/library load',
        'source': str(SOURCE.relative_to(ROOT)), 'records_sha256': digest(SOURCE / 'records.jsonl'),
        'main_sha256': main_hash, 'config_sha256': config_hash, 'pilot_sha256': digest(Path(__file__)),
        'API_calls': 0, 'API_cost_usd': 0, 'policy_changes': False,
        'preserved_metrics': ['unit successful admission adjustment cost', 'lower10% system utility increment'],
        'definitions': {
            'task': 'sum(g_done * new-slice retained task count) / sum(original new-slice task count)',
            'compute_cache': 'same ratio by task compute/cache demand, separately; not allocated capacity',
            'traffic': 'accepted original OD demand / all original OD demand; no partial traffic delivery model',
            'weighted': 'sum(revenue_g) / sum(.01*original_task_resource_demand + 10*original_path_load); original accounting weights',
        },
        'interpretation': 'exploratory diagnostic; task support, not actual task completion, latency or complete SLA satisfaction',
        'snapshot_metadata': [{'seed': m['seed'], 'path': m['sample_file'],
                               'bytes': Path(m['sample_file']).stat().st_size,
                               'recorded_sha256': m['sample_sha256']} for m in manifests],
    }
    write(out / 'protocol.json', protocol)
    print('PILOT_PREFLIGHT', json.dumps({'RAM_GiB': ram, 'scenes': 48, 'replays': 144}), flush=True)
    cached = {a: {} for a in ARMS}
    with (SOURCE / 'records.jsonl').open(encoding='utf-8') as src:
        for line in src:
            r = json.loads(line)
            if r['arm'] in ARMS and r['original_scene_id'] in SCENES:
                key = r['dataset_seed'], r['original_scene_id']
                assert key not in cached[r['arm']]
                cached[r['arm']][key] = r
    expected = {(seed, scene) for seed in SEEDS for scene in SCENES}
    assert all(set(rs) == expected for rs in cached.values())
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'p2j_validation'))
    import run_validation as v
    import p1a_runner as base
    import main
    base._compat_pickle_main_classes(main)
    rows, replay_checks, post_checks = [], 0, 0
    with (out / 'records.jsonl').open('x', encoding='utf-8') as sink:
        for manifest in manifests:
            seed = manifest['seed']
            print('SEED_LOADING', seed, flush=True)
            assert available_ram() >= 8
            assert digest(Path(manifest['sample_file'])) == manifest['sample_sha256']
            with (out / ('runtime_' + str(seed) + '.log')).open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log):
                env = v.make_env(manifest, out / ('runtime_' + str(seed)))
                original = env._get_g_adjustment
                observed = []

                def passive(*args, **kwargs):
                    net = kwargs['g_net'] if 'g_net' in kwargs else args[0]
                    requested = tasks(net)
                    requested.update(traffic(net, kwargs.get('tm_idx', 0)))
                    result = original(*args, **kwargs)
                    net_after, demand_n, demand_l, done = result
                    retained = tasks(net_after)
                    if done:
                        assert math.isclose(retained['compute'] + retained['cache'], demand_n, abs_tol=1e-6)
                        assert math.isclose(demand_l, requested['path_load_Gbps'], abs_tol=1e-6)
                        for field in ('task_count', 'compute', 'cache'):
                            assert 0 <= retained[field] <= requested[field] + 1e-6
                        retained.update({'od_Gbps': requested['od_Gbps'], 'path_load_Gbps': float(demand_l)})
                    else:
                        retained = {field: 0 for field in requested}
                    observed.append({'requested': requested, 'served': retained, 'g_done': bool(done)})
                    return result

                env._get_g_adjustment = passive
                for scene in SCENES:
                    request_controls = []
                    for arm in ARMS:
                        old = cached[arm][seed, scene]
                        observed.clear()
                        met = v.score(env, old['env_actions'], scene)
                        assert_same(old['metrics'], met)
                        assert len(observed) == 1
                        telemetry = observed[0]
                        assert telemetry['g_done'] == met['g_done']
                        served = telemetry['served']
                        assert math.isclose(met['revenue_g'], .01 * (served['compute'] + served['cache'])
                                            + 10 * served['path_load_Gbps'], abs_tol=1e-6)
                        snapshot_g = env.sample_dict[scene][3]
                        snapshot_tasks = tasks(snapshot_g)
                        for field in ('task_count', 'compute', 'cache'):
                            assert math.isclose(telemetry['requested'][field], snapshot_tasks[field], abs_tol=1e-6)
                        request_controls.append(telemetry['requested'])
                        row = {'dataset_seed': seed, 'original_scene_id': scene, 'arm': arm,
                               **telemetry, 'V': met['V'], 'revenue_g': met['revenue_g'],
                               'cached_metric_control_passed': True}
                        rows.append(row)
                        sink.write(json.dumps(row, allow_nan=False) + '\n')
                        sink.flush()
                        replay_checks += 1
                    assert request_controls[0] == request_controls[1] == request_controls[2]
                env._get_g_adjustment = original
                for arm in ARMS:
                    old = cached[arm][seed, SCENES[0]]
                    assert_same(old['metrics'], v.score(env, old['env_actions'], SCENES[0]))
                    post_checks += 1
                # Break the bound-method/closure references before next large environment.
                del original, env
            gc.collect()
            print('SEED_COMPLETE', seed, 'metric_controls', replay_checks, flush=True)
    assert len(rows) == 144 and replay_checks == 144 and post_checks == 9
    assert digest(ROOT / 'main.py') == main_hash and digest(ROOT / 'configs_main.yaml') == config_hash
    summary = {'status': 'COMPLETE', 'paired_scenes': 48, 'metrics': summarize(rows),
               'per_seed': {str(s): summarize([r for r in rows if r['dataset_seed'] == s]) for s in SEEDS},
               'checks': {'cached_metric_reproductions': replay_checks, 'unwrapped_post_controls': post_checks,
                          'same_requested_demand_across_arms': True, 'revenue_identity': True,
                          'original_snapshots_unchanged_in_memory': True, 'environment_sources_unchanged': True},
               'API_calls': 0, 'additional_API_cost_usd': 0}
    write(out / 'summary.json', summary)
    report = ['# 业务需求满足率48场离线回放试测', '',
              '种子2026110511/12/13，各取预先固定的前16场，共48场；每场三套原保存动作。',
              '这是小样本诊断，不是新独立正式测试，不改变原环境、奖励、补偿或动作。', '',
              '| 方案 | 接纳率 | 新任务承载率 | 计算需求满足率 | 缓存需求满足率 | OD流量接纳率 | 原收益权重满足率 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, met in summary['metrics'].items():
        rates = met['demand_satisfaction']
        report.append('| {} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} | {:.2%} |'.format(
            LABELS[arm], met['acceptance_rate'], rates['task_count'], rates['compute'], rates['cache'],
            rates['od_Gbps'], met['original_model_revenue_weighted_satisfaction']))
    report += ['', '## 口径', '',
               '新任务承载率=sum(接纳成功后保留的新切片任务数)/sum(新切片原始请求任务数)。失败场景分子为0。',
               '计算/缓存需求满足率分别按原始任务需求量加权，不能把获得的容量配额当成实际业务。',
               'OD流量接纳率按原始OD请求流量加权，原模型成功时提供全部给定OD流，失败时计0；不是实际测得的包交付率。',
               '原收益权重满足率按原模型0.01*(计算+缓存需求)+10*路径累计链路负载计价。它是请求价值加权的模型指标，不是无量纲物理吞吐量。',
               '原模型没有真实任务完成时间/端到端时延/完整SLA检查，因此更准确称“模型内任务承载率/需求保留率”。不能宣称真实业务完成率。',
               '所有需求分母在动作前记录，与原始内存快照逐场核对，三方案完全相同；不会把压缩后的需求用作分母。',
               '原环境fixed_tm_idx设置保留；使用其实际tm_idx，不改用别的流量样本。', '',
               '## 核查与文件', '',
               '144次回放的全部历史效益/代价/接纳等指标复现，9次移除采集包装后的对照复现。节点需求/链路负载/收益恒等式通过。',
               '只读原场景pickle，按清单核验SHA256，一次仅加载一个512场景环境。API调用0，新增费用0。',
               '新增本目录脚本、protocol.json、records.jsonl、summary.json及本报告，原数据/原结果未覆盖。',
               '查阅PROJECT_HANDOFF.md、AGENTS.md、main.py相关函数、p1a_runner.py、p2j_validation/run_validation.py、p2f_analysis/val5601_counterfactual_replay.py及原实验datasets.json/records.jsonl。', '',
               '复核命令（生成独立新输出目录，不调用API）：', '', '```powershell',
               "& 'python' -B .\\p2p_business_satisfaction\\run_pilot.py", '```', '']
    with (out / 'REPORT.md').open('x', encoding='utf-8') as sink:
        sink.write('\n'.join(report))
    print('PILOT_COMPLETE', str(out), flush=True)
    print(json.dumps({'metrics': summary['metrics'], 'checks': summary['checks']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    out = Path(__file__).resolve().parent / ('pilot_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(exist_ok=False)
    print('OUTPUT', out, flush=True)
    try:
        run(out)
    except Exception as exc:
        write(out / 'STOPPED.json', {'status': 'STOPPED', 'error_type': type(exc).__name__, 'error': str(exc), 'API_calls': 0})
        raise
