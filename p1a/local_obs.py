"""Lossless (up to JSON float representation) local observation decoding.

Based on CustomMultiAgentEnv in the original GEANT main.py:
- 12 nodes x 5 features
- max 36 links x 3 features (zero-padded)
- max 36 directed edge pairs x 2 features (zero-padded)
Total: 240 values per role. The model is NEVER given env, DG, or other agents' observations.
"""
import json
import math

import numpy as np

AGENTS = ("agent_k1", "agent_k2", "agent_k3", "agent_g")
N_NODES = 12
MAX_LINKS = 36
OBS_SIZE = N_NODES * 5 + MAX_LINKS * 3 + MAX_LINKS * 2


def _to_float(x):
    return float(x)


def describe_local_observation(agent, values):
    if agent not in AGENTS:
        raise ValueError("Unsupported role: {!r}".format(agent))
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or a.size != OBS_SIZE:
        raise ValueError("{}: expected {} flat observation values, got shape {}".format(
            agent, OBS_SIZE, tuple(a.shape)))
    if not np.all(np.isfinite(a)):
        raise ValueError("{}: non-finite observation".format(agent))
    if np.any(a < -1e-6) or np.any(a > 1 + 1e-6):
        raise ValueError("{}: original observation must lie in [0,1]".format(agent))
    node = a[:N_NODES * 5].reshape(N_NODES, 5)
    link_begin = N_NODES * 5
    link_end = link_begin + MAX_LINKS * 3
    links = a[link_begin:link_end].reshape(MAX_LINKS, 3)
    edges = a[link_end:].reshape(MAX_LINKS, 2)

    # The second/third link feature is strictly positive for real links in the
    # unmodified GEANT construction. Zero triplets at the end are padding.
    active = np.where(links[:, 2] > 0)[0] if agent == "agent_g" else np.where(links[:, 1] > 0)[0]
    n_links = len(active)
    if list(active) != list(range(n_links)):
        raise ValueError("{}: noncontiguous link padding".format(agent))
    if not np.allclose(links[n_links:], 0, atol=1e-8):
        raise ValueError("{}: malformed padded links".format(agent))
    if not np.allclose(edges[n_links:], 0, atol=1e-8):
        raise ValueError("{}: malformed padded topology".format(agent))

    node_list = []
    for idx, v in enumerate(node):
        if agent == "agent_g":
            if abs(v[0]) > 1e-7:
                raise ValueError("g node padding feature must be 0")
            node_list.append({
                "local_index": idx,
                "physical_allocated_compute_fraction": _to_float(v[1]),
                "requested_compute_fraction_of_physical_capacity": _to_float(v[2]),
                "physical_allocated_cache_fraction": _to_float(v[3]),
                "requested_cache_fraction_of_physical_capacity": _to_float(v[4]),
            })
        else:
            if min(abs(v[0]), abs(v[0] - 1)) > 1e-6:
                raise ValueError("Nonbinary overlap node flag")
            node_list.append({
                "local_index": idx,
                "overlaps_with_new_slice": int(round(v[0])),
                "compute_capacity_fraction_of_physical": _to_float(v[1]),
                "compute_utilization": _to_float(v[2]),
                "cache_capacity_fraction_of_physical": _to_float(v[3]),
                "cache_utilization": _to_float(v[4]),
            })

    link_list = []
    for idx in range(n_links):
        v = links[idx]
        edge = edges[idx]
        if agent == "agent_g":
            if abs(v[0]) > 1e-7:
                raise ValueError("g link padding feature must be 0")
            row = {
                "local_index": idx,
                "physical_allocated_bandwidth_fraction": _to_float(v[1]),
                "requested_bandwidth_fraction_of_physical_capacity": _to_float(v[2]),
            }
        else:
            if min(abs(v[0]), abs(v[0] - 1)) > 1e-6:
                raise ValueError("Nonbinary overlap link flag")
            row = {
                "local_index": idx,
                "overlaps_with_new_slice": int(round(v[0])),
                "bandwidth_capacity_fraction_of_physical": _to_float(v[1]),
                "bandwidth_utilization": _to_float(v[2]),
            }
        # Original code normalized local endpoints by max endpoint index.
        # Keeping them normalized avoids fabricating unrecoverable identifiers.
        row["local_edge_endpoints_normalized"] = [_to_float(edge[0]), _to_float(edge[1])]
        link_list.append(row)

    return {
        "role": agent,
        "observation_source": "original_per_agent_observation_only",
        "nodes": node_list,
        "links": link_list,
        "node_count": len(node_list),
        "link_count": len(link_list),
        "topology_encoding": "Directed local edge endpoints normalized by max local index",
    }


COMMON = (
    "You are a frozen language-model policy in a partially observable GEANT "
    "multi-agent network slicing experiment. Act only for your assigned role. "
    "The current state below is YOUR local observation, not the joint/global state. "
    "You do not know other slices' private current observations or chosen actions. "
    "Use only the provided local numbers and public action rules. "
    "Try to support admission of the new slice while avoiding unnecessary "
    "resource loss, migration and rerouting for existing slices. "
    "This is a single-shot decision: you cannot simulate, request global state, "
    "use a solver, negotiate with other agents, or request additional data. "
    "Do not invent unobserved absolute capacities, exact resource gaps, "
    "or the other agents' actions. Output only one JSON object with exactly "
    "the key 'ratios' and numeric values in [0,1]. No explanation."
)

K_RULES = (
    "You represent an EXISTING slice. Produce exactly 4 values: "
    "[node_control_0,node_control_1,link_control_0,link_control_1]. "
    "The original environment sorts indices of overlapping nodes and links "
    "in ascending local-index order, then repeats the two node controls across "
    "overlapping nodes and the two link controls across overlapping links "
    "(repeat factor=ceil(number_of_overlap_elements/2), truncate extras). "
    "For each controlled resource, ratio 0 releases no resources; ratio 1 "
    "requests the maximum allowed 80% reduction in that slice's allocated "
    "compute/cache capacity for nodes, or link bandwidth for links. "
    "A larger number may increase migration/rerouting cost."
)

G_RULES = (
    "You represent NEW slice g. Produce exactly 3 values: "
    "[compute_compression,cache_compression,bandwidth_compression]. "
    "The original environment uses the same compression value for all relevant "
    "g nodes or links of each resource type. A value of 0 keeps 100% requested "
    "capacity, and 1 reduces it to 30% (node/link scale = 1 - 0.7*ratio). "
    "Aggressive compression can sacrifice service utility or hurt load feasibility. "
    "Your observation includes physical ALLOCATED fractions and YOUR requested "
    "fractions; it does not show other slices' actions."
)


def make_messages(agent, observation):
    """Build ONLY from own observation and immutable, public environment rules."""
    local = describe_local_observation(agent, observation)
    system = COMMON + " " + (G_RULES if agent == "agent_g" else K_RULES)
    user = (
        "One current observation for your role (all node/link IDs are LOCAL):\n"
        + json.dumps(local, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\nReply as {\"ratios\":[...]} only."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
