"""Deterministic, local-observation-only interpretations of action consequences.

The formulas and contiguous-repeat grouping are audited against upstream
main.py. No env object, physical graph, peer observation, MAPPO history,
optimization results or post-action feedback may enter this module.
"""
import json
import math

from p1a.local_obs import describe_local_observation

K_SHRINK_COEFF = 0.8
G_SHRINK_COEFF = 0.7
ACTION_LEVELS = (0.0, 0.25, 0.5, 0.8, 1.0)


def _r(x):
    return round(float(x), 5)


def _group_indices(m):
    """Imitate np.repeat([control_0,control_1],ceil(m/2))[:m]."""
    if m <= 0:
        return ([], [])
    rep = int(math.ceil(m / 2.0))
    return (list(range(min(rep, m))), list(range(rep, m)))


def _safe_max(utilizations, coeff):
    """Reference action for unchanged traffic/usage; not a policy constraint."""
    return max(0.0, min(1.0, min((1 - u) / coeff for u in utilizations)))


def _k_control_summary(name, targets, kind):
    if not targets:
        return {
            'control': name, 'resource_type': kind,
            'active': False, 'affected_local_indices': [],
            'reason': 'No overlapping elements; this control has no effect on this scene.'
        }
    if kind == 'node_compute_and_cache':
        utilization = [max(t['compute_utilization'], t['cache_utilization']) for t in targets]
    else:
        utilization = [t['bandwidth_utilization'] for t in targets]
    upper = _safe_max(utilization, K_SHRINK_COEFF)
    if kind == 'link_bandwidth' and all(u <= 1e-8 for u in utilization):
        upper = None  # Original link-load feature may be zero before traffic/rerouting evaluation.
    examples = []
    for r in ACTION_LEVELS:
        retained = 1.0 - K_SHRINK_COEFF * r
        examples.append({
            'r': _r(r),
            'capacity_retained_fraction_of_original': _r(retained),
            'elements_below_current_recorded_usage': sum(u > retained + 1e-8 for u in utilization),
        })
    ret = {
        'control': name, 'resource_type': kind, 'active': True,
        'affected_local_indices': [t['local_index'] for t in targets],
        'affected_count': len(targets),
        'original_capacity_shrink_fraction_per_unit_ratio': K_SHRINK_COEFF,
        'no_recorded_usage_shortfall_reference_max_ratio': None if upper is None else _r(upper),
        'max_recorded_utilization_of_affected_elements': _r(max(utilization)),
        'example_consequences': examples,
    }
    if kind == 'node_compute_and_cache':
        ret['affected_elements'] = [{
            'local_index': t['local_index'],
            'compute_utilization': _r(t['compute_utilization']),
            'cache_utilization': _r(t['cache_utilization']),
            'compute_capacity_fraction_of_physical': _r(t['compute_capacity_fraction_of_physical']),
            'cache_capacity_fraction_of_physical': _r(t['cache_capacity_fraction_of_physical']),
        } for t in targets]
    else:
        ret['affected_elements'] = [{
            'local_index': t['local_index'],
            'bandwidth_utilization_from_original_observation': _r(t['bandwidth_utilization']),
            'bandwidth_capacity_fraction_of_physical': _r(t['bandwidth_capacity_fraction_of_physical']),
        } for t in targets]
        ret['zero_link_utilization_snapshot'] = all(u <= 1e-8 for u in utilization)
        ret['warning'] = (
            'Original bandwidth utilization may be zero or not predict post-rerouting traffic; '
            'The threshold is a static snapshot reference, NOT a guarantee of no rerouting or feasible bandwidth.'
        )
    return ret


def _k_semantics(local):
    nodes = sorted((n for n in local['nodes'] if n['overlaps_with_new_slice']),
                   key=lambda n: n['local_index'])
    links = sorted((l for l in local['links'] if l['overlaps_with_new_slice']),
                   key=lambda l: l['local_index'])
    n_groups, l_groups = _group_indices(len(nodes)), _group_indices(len(links))
    controls = []
    for i in range(2):
        controls.append(_k_control_summary('node_control_{}'.format(i),
                                           [nodes[j] for j in n_groups[i]],
                                           'node_compute_and_cache'))
    for i in range(2):
        controls.append(_k_control_summary('link_control_{}'.format(i),
                                           [links[j] for j in l_groups[i]],
                                           'link_bandwidth'))
    return {
        'role': local['role'],
        'interpretation': 'Current local observation only; public action equations.',
        'allocation_rule': ('Sort only overlapping local indices ascending; '
                            'np.repeat([r0,r1],ceil(overlap_count/2))[:overlap_count]. '
                            'The groups are contiguous, not alternating.'),
        'ratio_to_capacity': 'remaining allocated capacity = original capacity * (1 - 0.8 * ratio)',
        'overlapping_node_count': len(nodes), 'overlapping_link_count': len(links),
        'controls': controls,
        'caveat': ('These thresholds are NOT globally optimal actions and do not forecast '
                   'migration, rerouting, compensation, g admission or other slices decisions. '
                   'Increasing shrink may free resources at cost to this slice.'),
    }


def _g_node_resource(local, resource):
    assert resource in ('compute', 'cache')
    rows = []
    for node in local['nodes']:
        a = node['physical_allocated_' + resource + '_fraction']
        q = node['requested_' + resource + '_fraction_of_physical_capacity']
        free = max(0.0, 1.0 - a)
        rows.append({
            'local_index': node['local_index'],
            'physical_free_fraction_before_peer_actions': _r(free),
            'original_requested_fraction_of_physical': _r(q),
            'original_request_exceeds_current_free': q > free + 1e-8,
            'gap_fraction_of_physical': _r(max(0, q - free)),
        })
    return {
        'resource': resource,
        'compression_ratio_effect': 'deployed requirement = original requirement * (1 - 0.7 * ratio)',
        'legacy_node_admission_test': ('In the audited original main.py, node admission compares '
                                       'current free against ORIGINAL, pre-compression requested capacity. '
                                       'Compression alone does not relax that preliminary test.'),
        'nodes_currently_failing_precompression_admission': sum(
            row['original_request_exceeds_current_free'] for row in rows),
        'nodes': rows,
    }


def _g_bandwidth(local):
    rows = []
    threshold = 0.0
    impossible = []
    for link in local['links']:
        a = link['physical_allocated_bandwidth_fraction']
        q = link['requested_bandwidth_fraction_of_physical_capacity']
        free = max(0.0, 1.0 - a)
        r_req = 0.0 if q <= 0 else (1 - free / q) / G_SHRINK_COEFF
        if r_req > 1 + 1e-8:
            impossible.append(link['local_index'])
        threshold = max(threshold, r_req)
        rows.append({
            'local_index': link['local_index'],
            'physical_free_fraction_before_peer_actions': _r(free),
            'original_requested_fraction_of_physical': _r(q),
            'minimum_compression_for_this_snapshot': _r(max(0.0, r_req)),
            'insufficient_even_at_max_compression_in_snapshot': r_req > 1 + 1e-8,
        })
    return {
        'resource': 'bandwidth',
        'compression_ratio_effect': 'allocated link bandwidth = original bandwidth * (1 - 0.7 * ratio)',
        'snapshot_required_ratio_for_all_links': _r(max(0.0, threshold)),
        'snapshot_feasible_within_ratio_0_to_1': not impossible,
        'links_unfillable_at_max_compression_before_peer_actions': impossible,
        'caveat': ('Only checks current free capacity on g-local links before peer releases; '
                   'not a forecast of peer actions or final traffic utilization <= 70%.'),
        'links': rows,
    }


def _g_semantics(local):
    return {
        'role': 'agent_g',
        'interpretation': 'g-local observations of its physical-node/link allocation fractions and g demand.',
        'controls': [
            _g_node_resource(local, 'compute'),
            _g_node_resource(local, 'cache'),
            _g_bandwidth(local),
        ],
        'ratio_grid': [{
            'r': _r(r),
            'original_demand_retained_fraction': _r(1.0 - G_SHRINK_COEFF * r),
        } for r in ACTION_LEVELS],
        'caveat': ('Physical availability shown is prior to k1/k2/k3 actions. '
                   'Do not claim global feasibility or know peer releases. '
                   'The environment also applies traffic load and service constraints.'),
    }


def describe_action_semantics(agent, observation):
    local = describe_local_observation(agent, observation)
    if agent == 'agent_g':
        return _g_semantics(local)
    return _k_semantics(local)


def make_semantic_messages(agent, observation):
    """Preserve P1-A raw prompt and ADD a read-only, deterministic explanation."""
    from p1a.local_obs import make_messages
    messages = make_messages(agent, observation)
    semantics = describe_action_semantics(agent, observation)
    messages[0]['content'] += (
        ' You are also provided exact deterministic calculations from YOUR OWN observation. '
        'Interpret numbers using the supplied action mapping and risk tables. '
        'Reference no-migration thresholds are not enforced and may be exceeded if justified; '
        'they do not guarantee joint admission or utility. Do not claim knowledge of peer actions. '
        'Use the same output JSON schema as before.'
    )
    messages[1]['content'] += (
        '\nLOCAL ACTION SEMANTICS (computed solely from the exact same local observation):\n'
        + json.dumps(semantics, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    )
    return messages
