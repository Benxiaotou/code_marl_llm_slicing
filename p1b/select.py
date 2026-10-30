"""P1-B quality selection and conservative local-state features.

Candidate selection is CENTRALIZED and offline, using only TRAIN outcomes.
Every exported per-role experience is restricted to (local obs, own action).
"""
import math

import numpy as np

from p1a.local_obs import AGENTS, N_NODES, MAX_LINKS
from p1b.core import ContractError


def _safe_mean(x):
    x = list(x)
    return float(np.mean(x)) if x else 0.0


def local_features(observation, role):
    a = np.asarray(observation, dtype=np.float64)
    if a.shape != (240,) or not np.isfinite(a).all():
        raise ContractError('Invalid original 240-dimensional observation')
    nodes = a[:N_NODES*5].reshape(N_NODES, 5)
    links = a[N_NODES*5:N_NODES*5+MAX_LINKS*3].reshape(MAX_LINKS, 3)
    if role != 'agent_g':
        real_links = links[links[:,1] > 0]
        overlapping_nodes = nodes[nodes[:,0] > .5]
        overlapping_links = real_links[real_links[:,0] > .5]
        compute_pressure = _safe_mean(overlapping_nodes[:,2]) if len(overlapping_nodes) else 0.
        cache_pressure = _safe_mean(overlapping_nodes[:,4]) if len(overlapping_nodes) else 0.
        link_pressure = _safe_mean(overlapping_links[:,2]) if len(overlapping_links) else 0.
        return {
            'overlap_node_fraction': float(len(overlapping_nodes)/N_NODES),
            'overlap_link_fraction': float(len(overlapping_links)/max(1,len(real_links))),
            'compute_pressure': compute_pressure,
            'cache_pressure': cache_pressure,
            'link_pressure': link_pressure,
        }
    real_links = links[links[:,2] > 0]
    compute_shortage = np.maximum(0, nodes[:,1] + nodes[:,2] - 1)
    cache_shortage = np.maximum(0, nodes[:,3] + nodes[:,4] - 1)
    bw_shortage = np.maximum(0, real_links[:,1] + real_links[:,2] - 1) if len(real_links) else []
    return {
        'g_compute_shortage_mean': _safe_mean(compute_shortage),
        'g_cache_shortage_mean': _safe_mean(cache_shortage),
        'g_bandwidth_shortage_mean': _safe_mean(bw_shortage),
    }


def joint_features(joint):
    observations = joint['observations']
    per_role = {r: local_features(observations[r], r) for r in AGENTS}
    ks = [per_role[a] for a in AGENTS[:3]]
    g = per_role['agent_g']
    overlap = _safe_mean([(k['overlap_node_fraction'] + k['overlap_link_fraction'])/2 for k in ks])
    used = _safe_mean([(k['compute_pressure']+k['cache_pressure']+k['link_pressure'])/3 for k in ks])
    short = _safe_mean([g['g_compute_shortage_mean'], g['g_cache_shortage_mean'],
                        g['g_bandwidth_shortage_mean']])
    pressure = float(0.45 * used + 0.55 * min(1, short))
    # For selection only, not exposed in role-local examples.
    return {
        'overlap': overlap, 'pressure': pressure, 'utilization': used,
        'resource_shortage': short,
        'vector': np.array([pressure, overlap, used, short,
                           g['g_compute_shortage_mean'], g['g_cache_shortage_mean'],
                           g['g_bandwidth_shortage_mean']], dtype=np.float64),
    }


def _percentile_rank(values):
    n = len(values)
    if n == 1:
        return [0.5]
    order = sorted(range(n), key=lambda i: (float(values[i]), i))
    rank = [0] * n
    for pos, i in enumerate(order):
        rank[i] = pos / (n-1)
    return rank


def analyze_and_select(records, max_keep=400, cost_quantile=0.95, quality_weight=0.65,
                       diversity_weight=0.20):
    if max_keep < 1 or not (0 < cost_quantile <= 1):
        raise ValueError('max_keep>=1 and 0<cost_quantile<=1 required')
    if not (0 <= quality_weight <= 1) or not (0 <= diversity_weight <= 1):
        raise ValueError('weights must be in [0,1]')
    good = []
    reasons = {'failed_record': 0, 'admission_failed': 0,
               'individual_rationality_failed': 0, 'invalid_metrics': 0}
    for row in records:
        if row.get('status') != 'ok':
            reasons['failed_record'] += 1
            continue
        m = row.get('metrics') or {}
        try:
            valid_numbers = all(math.isfinite(float(m[k])) for k in (
                'V','V_k1','V_k2','V_k3','V_g','V_k1_0','V_k2_0','V_k3_0','cost_sum'))
        except (KeyError, TypeError, ValueError, OverflowError):
            valid_numbers = False
        if not valid_numbers or float(m['cost_sum']) < -1e-8:
            reasons['invalid_metrics'] += 1
            continue
        if not bool(m.get('g_done')):
            reasons['admission_failed'] += 1
            continue
        expected_ir = all(float(m['V_'+k]) >= float(m['V_'+k+'_0']) - 1e-8
                          for k in ('k1','k2','k3')) and float(m['V_g']) >= -1e-8
        if not expected_ir or not bool(m.get('individual_rationality')) or not bool(m.get('cooperative_valid')):
            reasons['individual_rationality_failed'] += 1
            continue
        features = joint_features(row)
        good.append({'row': row, 'features': features})
    report = {'n_input': len(records), 'n_cooperative_candidates': len(good),
              'hard_filter_rejections': reasons}
    if not good:
        return [], report
    # Quantile ranks on successful cooperative cases, not on all/easy cases.
    p_vals = [x['features']['pressure'] for x in good]
    overlap_vals = [x['features']['overlap'] for x in good]
    pressure_cutoffs = [float(np.quantile(p_vals, q)) for q in (1/3, 2/3)]
    overlap_cutoff = float(np.median(overlap_vals))
    groups = {}
    for candidate in good:
        f = candidate['features']
        p = f['pressure']
        pb = int(p > pressure_cutoffs[0]) + int(p > pressure_cutoffs[1])
        ob = int(f['overlap'] > overlap_cutoff)
        group = 'P{}-O{}'.format(pb, ob)
        candidate['stratum'] = group
        groups.setdefault(group, []).append(candidate)
    kept_by_stratum = {}
    removed_by_cost = {}
    for group, group_rows in groups.items():
        costs = [c['row']['metrics']['cost_sum'] for c in group_rows]
        threshold = float(np.quantile(costs, cost_quantile)) if len(costs) >= 3 else max(costs)
        subset = [c for c in group_rows if c['row']['metrics']['cost_sum'] <= threshold + 1e-10]
        if not subset:
            subset = [min(group_rows, key=lambda x: x['row']['metrics']['cost_sum'])]
        removed_by_cost[group] = len(group_rows) - len(subset)
        # Quality score relative to comparable pressure and overlap, not raw global V.
        v_rank = _percentile_rank([c['row']['metrics']['V'] for c in subset])
        cost_rank = _percentile_rank([c['row']['metrics']['cost_sum'] for c in subset])
        for i, c in enumerate(subset):
            c['quality_score'] = quality_weight * v_rank[i] + (1-quality_weight) * (1-cost_rank[i])
        kept_by_stratum[group] = subset
    all_candidates = [c for group in sorted(kept_by_stratum) for c in kept_by_stratum[group]]
    feature_mat = np.stack([c['features']['vector'] for c in all_candidates])
    std = feature_mat.std(axis=0)
    std[std < 1e-8] = 1.0
    means = feature_mat.mean(axis=0)
    for c in all_candidates:
        c['z'] = (c['features']['vector'] - means) / std
        c['nearest_selected_distance'] = float('inf')
    # Round-robin strata ensure coverage of hard/overlapping cases rather than
    # selecting only high-utility easy cases. Within each stratum, use quality
    # plus distance from already selected examples to avoid near duplicates.
    selected = []
    selected_ids = set()
    while len(selected) < max_keep:
        changed = False
        for group in sorted(kept_by_stratum):
            remaining = [c for c in kept_by_stratum[group]
                         if c['row']['joint_id'] not in selected_ids]
            if not remaining:
                continue
            def score(c):
                d = c['nearest_selected_distance']
                novelty = 1.0 if math.isinf(d) else d / (1.0 + d)
                return ((1-diversity_weight)*c['quality_score']
                        + diversity_weight * novelty)
            best = max(remaining, key=lambda c: (score(c), c['row']['joint_id']))
            selected.append(best)
            selected_ids.add(best['row']['joint_id'])
            # Incremental nearest-selected distance avoids O(N*K^2) recomputation.
            for candidate in all_candidates:
                if candidate['row']['joint_id'] in selected_ids:
                    continue
                d = float(np.linalg.norm(candidate['z'] - best['z']))
                candidate['nearest_selected_distance'] = min(
                    candidate['nearest_selected_distance'], d)
            changed = True
            if len(selected) >= max_keep:
                break
        if not changed:
            break
    report.update({
        'n_after_relative_cost_filter': len(all_candidates),
        'n_selected': len(selected), 'strata_candidates': {k:len(v) for k,v in groups.items()},
        'strata_selected': {k:sum(x['stratum']==k for x in selected) for k in sorted(groups)},
        'removed_by_cost_stratum': removed_by_cost,
        'pressure_cutoffs': pressure_cutoffs, 'overlap_median': overlap_cutoff,
        'selection_warning': 'Joint success does NOT establish per-agent causal optimality.',
    })
    return selected, report
