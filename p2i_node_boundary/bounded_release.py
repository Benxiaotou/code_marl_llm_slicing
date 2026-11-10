"""Bounded node-only postprocessing from one role's local observation/action."""
import numpy as np

# (name, maximum ratio increase, maximum extra projected overload / own capacity)
ARMS = (
    ("safe_d025", .025, 0.),
    ("safe_d050", .050, 0.),
    ("buffer01_d025", .025, .01),
    ("buffer02_d025", .025, .02),
    ("buffer02_d050", .050, .02),
)


def adjust_local(observation, env_action, max_increment, overload_budget):
    """Increase ratios using own utilization; do not infer g's hidden request.

    Projected overload is max(0, utilization + .8 * ratio - 1).
    Its increase is bounded for every controlled compute/cache resource.
    This is a local static reference, not a migration feasibility guarantee.
    Links, padding dimensions and g actions are outside this function's scope.
    """
    obs = np.asarray(observation, dtype=float)
    action = np.asarray(env_action, dtype=float)
    if (obs.shape != (240,) or action.shape != (4,) or
            not np.isfinite(obs).all() or not np.isfinite(action).all() or
            np.any(obs < -1e-6) or np.any(obs > 1 + 1e-6) or
            np.any(action < -1 - 1e-6) or np.any(action > 1 + 1e-6)):
        raise ValueError("Invalid local observation/action")
    if not (0 <= max_increment <= .05 and 0 <= overload_budget <= .02):
        raise ValueError("Adjustment exceeds preregistered bound")
    nodes = obs[:60].reshape(12, 5)
    if np.any(np.minimum(abs(nodes[:, 0]), abs(nodes[:, 0] - 1)) > 1e-6):
        raise ValueError("Nonbinary overlap flag")
    selected = nodes[nodes[:, 0] > .5]
    boundary = (len(selected) + 1) // 2
    output = action.copy()
    audit = []
    for j, group in enumerate((selected[:boundary], selected[boundary:])):
        ratio = float(np.clip((action[j] + 1) / 2, 0, 1))
        if not len(group):
            audit.append({"group": j, "nodes": 0, "ratio_increment": 0.})
            continue
        util = group[:, [2, 4]]
        old_overload = np.maximum(0., util + .8 * ratio - 1)
        cap = float(np.min((1 - util + old_overload + overload_budget) / .8))
        # Small safety slack avoids creating overload from rounding at a limit.
        inc = min(max_increment, 1 - ratio, max(0., cap - ratio - 1e-6))
        if inc > 0:
            output[j] = 2 * (ratio + inc) - 1
        added = np.maximum(0., util + .8 * (ratio + inc) - 1) - old_overload
        assert float(added.max()) <= overload_budget + 1e-10
        audit.append({"group": j, "nodes": len(group), "ratio_increment": float(inc),
                      "max_extra_projected_overload": float(added.max())})
    return output.tolist(), audit
