"""Prevent silently applying upstream-specific semantic rules to a changed env."""
from pathlib import Path


RULE_MARKERS = {
    'k grouping np.repeat': 'np.repeat(action_k_n_l, control_scale)',
    'k node shrink coefficient': 'limit_n_allocate = 0.2',
    'k link shrink coefficient': 'limit_l_allocate = 0.2',
    'g node shrink coefficient': 'limit_n_allocate = 0.3',
    'g link shrink coefficient': 'limit_l_allocate = 0.3',
    'g compute legacy test': "res_p_comp >= g_net.DG.nodes[idx]['comp_capacity']",
    'g cache legacy test': "res_p_cach >= g_net.DG.nodes[idx]['cach_capacity']",
}


def verify_main_rules(path):
    source = Path(path).read_text(encoding='utf-8')
    missing = [name for name, marker in RULE_MARKERS.items() if marker not in source]
    if missing:
        raise RuntimeError(
            'Current main.py differs from upstream action rules: {}. '
            'Audit actual environment and update semantics before using P1-A+.'
            .format(', '.join(missing))
        )
    return sorted(RULE_MARKERS)
