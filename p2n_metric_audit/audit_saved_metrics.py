"""Read-only arithmetic audit of existing results; no simulator or API imports."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'p2m_paper/fresh_20261105T025939_435887Z/records.jsonl'
ARMS = ('MAPPO_raw', 'MAPPO_both', 'Qwen_both')
K = ('k1', 'k2', 'k3')


def main():
    assert RECORDS.is_file() and RECORDS.stat().st_size < 30_000_000
    rows = {a: {} for a in ARMS}
    checked = 0
    with RECORDS.open(encoding='utf-8') as source:
        for line in source:
            record = json.loads(line)
            m = record['metrics']
            assert math.isclose(m['V'], sum(m['V_' + k] for k in K + ('g',)), abs_tol=1e-7)
            assert math.isclose(m['V'], sum(m['revenue_' + k] for k in K + ('g',)) - m['cost_sum'], abs_tol=1e-7)
            ir = all(m['V_' + k] >= m['V_' + k + '_0'] - 1e-8 for k in K) and m['V_g'] >= -1e-8
            assert ir == m['individual_rationality']
            assert m['cooperative_valid'] == (m['g_done'] and ir)
            checked += 1
            if record['arm'] in rows:
                key = record['dataset_seed'], record['original_scene_id']
                assert key not in rows[record['arm']]
                rows[record['arm']][key] = m
    result = {'input': str(RECORDS.relative_to(ROOT)), 'input_bytes': RECORDS.stat().st_size,
              'all_rows_arithmetic_checked': checked, 'arms': {}, 'API_calls': 0}
    for arm, scenes in rows.items():
        assert len(scenes) == 1536
        metrics = list(scenes.values())
        accepted = [m for m in metrics if m['g_done']]
        failed = [m for m in accepted if not m['individual_rationality']]
        deficits = [sum(max(0, m['V_' + k + '_0'] - m['V_' + k]) for k in K) for m in failed]
        # Hypothetical monetary transfer feasibility ONLY. No changed actions,
        # compensation, metrics, simulator outcomes or original result files.
        topup_feasible = sum(m['V_g'] >= loss - 1e-8 for m, loss in zip(failed, deficits))
        assert all(m['V_g'] >= -1e-8 for m in accepted)
        result['arms'][arm] = {
            'n': len(metrics), 'accepted': len(accepted),
            'cooperative_valid': sum(m['cooperative_valid'] for m in metrics),
            'accepted_IR_fail': len(failed),
            'IR_pass_given_acceptance': sum(m['cooperative_valid'] for m in metrics) / len(accepted),
            'V_mean': statistics.mean(m['V'] for m in metrics),
            'V_g_mean': statistics.mean(m['V_g'] for m in metrics),
            'existing_V_per_slice_mean': statistics.mean(m['V_' + k] for m in metrics for k in K),
            'IR_fail_total_deficit_mean': statistics.mean(deficits),
            'IR_fail_total_deficit_median': statistics.median(deficits),
            'IR_fail_total_deficit_max': max(deficits),
            'IR_fail_deficit_at_most_1e_minus_6': sum(x <= 1e-6 for x in deficits),
            'IR_fail_g_surplus_covers_topup': topup_feasible,
        }
    q, m = rows['Qwen_both'], rows['MAPPO_both']
    assert set(q) == set(m)
    common = [key for key in q if q[key]['g_done'] and m[key]['g_done']]
    result['common_accepted_descriptive_only'] = {
        'n': len(common),
        'Qwen_valid': sum(q[key]['cooperative_valid'] for key in common),
        'MAPPO_valid': sum(m[key]['cooperative_valid'] for key in common),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
