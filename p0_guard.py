"""P0-v4: hard-gated, auditable capacity repair for Qwen joint actions.

This module reads the CURRENT ORIGINAL scenario, builds CPU/cache/link admission
inequalities using original code's same group mapping, and proposes an LP repair.
Every accepted action is re-executed on a fresh shadow copy using the ORIGINAL
CustomMultiAgentEnv.eval_env_step(), without modifying the live environment.

No MAPPO policy, reward imitation, or trained model is used here. Repair is a
separate conventional constraint solver, not an LLM-only result.
"""
from copy import copy
from dataclasses import dataclass
import math
import random
import time

import numpy as np
from scipy.optimize import linprog
from p0_policy import AGENTS

VARIABLE_KEYS = tuple((agent, dim) for agent in AGENTS[:3] for dim in range(4)) + (("agent_g", 2),)
NVAR = len(VARIABLE_KEYS)  # 13: 12 k variables and g's BW ratio


class NoFeasibleAction(RuntimeError):
    """Nothing was committed; never silently execute unverified actions."""


def _vec_from_action(action):
    result = []
    if not isinstance(action, dict) or set(action) != set(AGENTS):
        raise ValueError('Expected all four action keys')
    for agent in AGENTS:
        a = np.asarray(action[agent], dtype=float).reshape(-1)
        if a.shape != (4,) or not np.isfinite(a).all() or (a < -1.000001).any() or (a > 1.000001).any():
            raise ValueError('Invalid action from ' + agent)
        if agent != 'agent_g':
            result.extend((a[:4] + 1.0) / 2.0)
        else:
            result.append((a[2] + 1.0) / 2.0)
    return np.clip(np.asarray(result, dtype=float), 0.0, 1.0)


def _action_from_vec(vec, old_action):
    a = {}
    for i, name in enumerate(AGENTS[:3]):
        a[name] = (2.0 * np.asarray(vec[4*i:4*i+4]) - 1.0).astype(np.float32)
    original_g = np.asarray(old_action['agent_g'], dtype=np.float32).copy()
    original_g[2] = np.float32(2 * float(vec[12]) - 1)
    original_g[3] = np.float32(0)  # unused 4th action, neutral, no unrequested control
    a['agent_g'] = original_g
    return a


def _group_lookup(indices):
    """Same action grouping as main._get_scale_repeat_action (ceil(n/2) repeat)."""
    indices = sorted(indices)
    if not indices:
        return {}
    group_width = int(math.ceil(len(indices) / 2.0))
    return {idx: min(int(pos // group_width), 1) for pos, idx in enumerate(indices)}


def _g_bw_loads(gnet, tm_idx):
    """Replicate original g's fixed-shortest-path traffic distribution."""
    traffic = np.asarray(gnet.traffic_matrices[tm_idx], dtype=float) / 1e6
    loads = np.zeros(gnet.num_links, dtype=float)
    for pair_index in range(gnet.num_pairs):
        s, d = gnet.pair_idx_to_sd[pair_index]
        demand = float(traffic[s][d])
        if demand == 0:
            continue
        paths = gnet.shortest_paths_node[pair_index]
        if not paths:
            raise NoFeasibleAction('Missing shortest path in original g topology')
        for path in paths:
            portion = demand / len(paths)
            for p in range(len(path)-1):
                link = gnet.link_sd_to_idx[(path[p], path[p+1])]
                loads[link] += portion
    return loads


def _original_scene_constraints(env, margin=0.001):
    """A @ x <= b for 12 k release ratios + 1 g BW reduction ratio.

    We intentionally match the original environment's node-admission bug:
    compare against g's UNREDUCED CPU and cache requested capacities.
    """
    phys = env.DG
    gnet = env.g_net
    knets = (env.k1_net, env.k2_net, env.k3_net)
    knode_groups = []
    klink_groups = []
    for knet in knets:
        overlap_nodes, overlap_links, _ = env._get_overlap_node_link(knet, gnet)
        knode_groups.append(_group_lookup(overlap_nodes))
        klink_groups.append(_group_lookup(overlap_links))
    rows = []
    rhs = []
    names = []

    def append_constraints(coeffs, free, requested, description):
        # sum(coeffs_i * action_ratio_i) >= (requested - free + margin)
        rows.append(-np.asarray(coeffs, dtype=float))
        rhs.append(float(free - requested - margin))
        names.append(description)

    # CPU and cache for each requested g virtual node, using global physical-node IDs.
    for gi in range(gnet.num_nodes):
        vid = gnet.node_id_list[gi]
        pi = env.node_id_to_idx[vid]
        pg = phys.nodes[pi]
        gg = gnet.DG.nodes[gi]
        for resource, label in (('comp', 'CPU'), ('cach', 'CACHE')):
            coeffs = np.zeros(NVAR)
            for ki, knet in enumerate(knets):
                k_node = knet.node_id_to_idx.get(vid)
                if k_node is None:
                    continue
                slot = knode_groups[ki].get(k_node)
                if slot is None:
                    continue
                coeffs[4*ki + slot] += 0.8 * float(knet.DG.nodes[k_node][resource + '_capacity'])
            free = float(pg[resource + '_capacity'] - pg['allocated_' + resource])
            req = float(gg[resource + '_capacity'])
            append_constraints(coeffs, free, req, '{} g_node={} physical={}'.format(label,gi,pi))

    # Bandwidth capacity for each g virtual edge, directed physical edge ID.
    g_ub = 1.0
    tm_idx = 0 if env.fixed_tm_idx else env.sample_idx
    g_loads = _g_bw_loads(gnet, tm_idx)
    for gl in range(gnet.num_links):
        gs, gd = gnet.link_idx_to_sd[gl]
        ids, idd = gnet.node_id_list[gs], gnet.node_id_list[gd]
        ps, pd = env.node_id_to_idx[ids], env.node_id_to_idx[idd]
        gp = gnet.DG[gs][gd]
        pp = phys[ps][pd]
        orig_bw = float(gp['bandwidth'])
        if orig_bw <= 0:
            raise NoFeasibleAction('Original g bandwidth nonpositive at g link {}'.format(gl))
        coeffs = np.zeros(NVAR)
        for ki, knet in enumerate(knets):
            if ids not in knet.node_id_to_idx or idd not in knet.node_id_to_idx:
                continue
            ks, kd = knet.node_id_to_idx[ids], knet.node_id_to_idx[idd]
            if (ks, kd) not in knet.link_sd_to_idx:
                continue
            kl = knet.link_sd_to_idx[(ks,kd)]
            slot = klink_groups[ki].get(kl)
            if slot is None:
                continue
            coeffs[4*ki + 2 + slot] += 0.8 * float(knet.DG[ks][kd]['bandwidth'])
        coeffs[12] = 0.7 * orig_bw
        free = float(pp['bandwidth'] - pp['allocated_bandw'])
        append_constraints(coeffs, free, orig_bw, 'BW g_link={} physical={}->{}'.format(gl,ps,pd))
        # SLA condition: used_bw / adjusted_g_bw <= 0.7
        g_link_upper = (1.0 - float(g_loads[gl])/(0.7*orig_bw))/0.7
        g_ub = min(g_ub, g_link_upper)
    if g_ub < -1e-10:
        raise NoFeasibleAction('g exceeds link utilization cap even at bandwidth reduction=0')
    return np.asarray(rows), np.asarray(rhs), names, max(0., min(1., g_ub))


def _solve_projected_lp(env, candidate_action, margin):
    old = _vec_from_action(candidate_action)
    A, b, names, g_ub = _original_scene_constraints(env, margin=margin)
    # x = [13 ratios; 13 absolute changes], minimize deviation from Qwen.
    # Small cost regularization discourages needless releases but does not
    # intentionally override LLM output more than feasibility requires.
    n = NVAR
    obj = np.r_[np.full(n, 0.015), np.ones(n)]
    obj[12] = 0.015
    dist_A = np.zeros((2*n,2*n))
    dist_b = np.zeros(2*n)
    for i in range(n):
        dist_A[2*i, i] = 1.0
        dist_A[2*i, n+i] = -1.0
        dist_b[2*i] = old[i]
        dist_A[2*i+1,i] = -1.0
        dist_A[2*i+1,n+i] = -1.0
        dist_b[2*i+1] = -old[i]
    block_A = np.vstack((np.hstack((A,np.zeros((len(b),n)))),dist_A))
    block_b = np.r_[b,dist_b]
    bounds = [(0,1)]*(n-1) + [(0,g_ub)] + [(0,None)]*n
    solution = linprog(obj,A_ub=block_A,b_ub=block_b,bounds=bounds,method='highs')
    if not solution.success:
        return None, {'reason':str(solution.message), 'status':int(solution.status),
                      'g_bandwidth_reduction_upper_bound':float(g_ub)}
    x = np.clip(solution.x[:n],0,1)
    result = _action_from_vec(x, candidate_action)
    before_gap = np.maximum(A @ old - b,0)
    after_gap = np.maximum(A @ x - b,0)
    diagnostics = {'solver':'scipy.linprog.highs', 'margin':float(margin),
                   'g_bandwidth_reduction_upper_bound':float(g_ub),
                   'max_capacity_shortfall_before':float(np.max(before_gap)),
                   'max_capacity_shortfall_after':float(np.max(after_gap)),
                   'initial_top_shortfalls':sorted(
                       ({'constraint':str(names[i]),'absolute_shortfall':float(before_gap[i])}
                        for i in range(len(names)) if before_gap[i] > 1e-8),
                       key=lambda x:-x['absolute_shortfall'])[:8],
                   'ratio_l1_change':float(np.sum(np.abs(x-old)))}
    return result, diagnostics


def _verify_with_untouched_env(env, scene_index, action, seed):
    """Dry-run original env.step on an isolated copy; do not pollute live run.

    Return only numeric, finite results; a g_done True alone is insufficient if
    the original reward code has generated NaN/Inf on a degenerate scene.
    """
    from p0_runner import stage_scene
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    shadow = copy(env)
    shadow.g_done_list = []
    for field in ('reward_k1_list','reward_k2_list','reward_k3_list','reward_g_list'):
        setattr(shadow,field,[])
    shadow.dict_eval_index = {key:[] for key in env.dict_eval_index}
    shadow.timestep = 0
    shadow.eval_step_count = 0
    shadow.eval_episode_count = 0
    shadow.eval_max_step = max(100000, int(getattr(shadow,'eval_max_step',100000)))
    shadow.is_eval = True
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        stage_scene(shadow,int(scene_index))
        _observations, rewards, _terminated, _truncated, _infos = shadow.step(action)
        accepted = bool(shadow.g_done_list[-1])
        total_utility = float(shadow.dict_eval_index['V'][-1])
        all_finite = math.isfinite(total_utility)
        all_finite &= all(np.isfinite(float(rewards[a])) for a in AGENTS)
        all_finite &= all(math.isfinite(float(hist[-1])) for hist in shadow.dict_eval_index.values())
        return {'accepted':accepted and bool(all_finite), 'original_g_done':accepted,
                'metrics_finite':bool(all_finite), 'V':total_utility if math.isfinite(total_utility) else None}
    except Exception as exc:
        return {'accepted':False, 'original_g_done':False,'metrics_finite':False,
                'error':type(exc).__name__ + ': ' + str(exc)[:450]}
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


@dataclass
class GuardResult:
    action: dict
    raw_check: dict
    checked: dict
    guard_changed: bool
    diagnostics: dict
    elapsed_s: float
    dry_runs: int


class HardAdmissionGuard:
    def __init__(self, margin=0.001):
        self.margin = float(margin)
        if not 0.0 < self.margin <= 0.1:
            raise ValueError('margin must be in (0,0.1]')

    def check_and_repair(self, env, scene_index, candidate, seed):
        started = time.perf_counter()
        raw = _verify_with_untouched_env(env,scene_index,candidate,seed)
        if raw['accepted']:
            return GuardResult(candidate,raw,raw,False,
                               {'status':'qwen_action_already_feasible'},
                               time.perf_counter()-started,1)
        last_error = None
        diagnostics = {}
        count = 1
        # Normally just one LP solve and one replay. Retry with larger tiny
        # margins only for floating-point borderline mismatches.
        for margin in (self.margin, self.margin * 10., self.margin * 50.):
            proposal, diagnostics = _solve_projected_lp(env,candidate,margin)
            if proposal is None:
                last_error = 'Capacity LP infeasible: ' + str(diagnostics.get('reason'))
                break
            checked = _verify_with_untouched_env(env,scene_index,proposal,seed)
            count += 1
            if checked['accepted']:
                diagnostics = dict(diagnostics,status='verified_repair',dry_runs=count)
                return GuardResult(proposal,raw,checked,True,diagnostics,
                                   time.perf_counter()-started,count)
            last_error = 'LP passed capacity constraints but original step rejected: ' + str(checked)
        msg = ('GUARD FAIL-CLOSED: scene={} after {} dry runs; {}. '
               'No action was committed. Final LP audit: {}').format(
                   scene_index,count,last_error,diagnostics)
        raise NoFeasibleAction(msg)
