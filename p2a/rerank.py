"""Auditable role-local IR-risk-aware historical demonstration reranking.

All input scores are computed from the audited TRAIN-only P1-D role overlay.
The CURRENT input to reranking is exclusively one role's 240D observation.
Never consult other roles' current observations, historical joint records,
GEANT environment outcomes or online simulation during action selection.

For existing slices, always retain the single nearest case (anchor), then
select the remaining examples from a tightly gated nearest-neighbour pool.
The incoming g agent uses exactly the original P1-C retrieval unchanged.
"""
import math
from dataclasses import dataclass

import numpy as np

from p1a.local_obs import AGENTS
from p1b.core import ContractError


@dataclass(frozen=True)
class RerankConfig:
    pool_size: int = 20
    risk_weight: float = 0.35
    max_abs_distance_gap: float = 0.10
    max_distance_ratio: float = 1.35

    def __post_init__(self):
        if (type(self.pool_size) is not int or self.pool_size < 3
                or not isinstance(self.risk_weight, (float, int))
                or not 0 <= self.risk_weight <= 1
                or not isinstance(self.max_abs_distance_gap, (float, int))
                or not 0 < self.max_abs_distance_gap <= 1
                or not isinstance(self.max_distance_ratio, (float, int))
                or not 1 < self.max_distance_ratio <= 3):
            raise ContractError('Invalid P2-A rerank config')
        if not all(math.isfinite(float(v)) for v in (
                self.risk_weight, self.max_abs_distance_gap, self.max_distance_ratio)):
            raise ContractError('Nonfinite P2-A rerank config')

    def as_dict(self):
        return {
            'pool_size': int(self.pool_size),
            'risk_weight': float(self.risk_weight),
            'max_abs_distance_gap': float(self.max_abs_distance_gap),
            'max_distance_ratio': float(self.max_distance_ratio),
            'anchor_nearest': True,
            'g_retrieval_unchanged': True,
            'risk_formula': '0.55*cost_percentile+0.30*(1-relative_margin_percentile)+0.15*(1-absolute_margin_percentile)',
            'distance_formula': '(distance-nearest_distance)/max(nearest_distance,0.15)',
            'score_formula': '(1-risk_weight)*distance_penalty+risk_weight*historical_risk',
        }


def _midrank_percentiles(values):
    """TRAIN-only empirical midrank scaled to [0,1], stable with ties."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all():
        raise ContractError('Invalid historical IR risk training vector')
    if len(arr) == 1 or np.all(arr == arr[0]):
        return np.full(len(arr), 0.5, dtype=np.float64)
    ordered = np.sort(arr)
    lower = np.searchsorted(ordered, arr, side='left')
    upper = np.searchsorted(ordered, arr, side='right')
    return (lower + upper - 1) / (2 * (len(arr) - 1))


class IRRiskReranker:
    """Reorder only similar OWN-role training examples without action repair.

    The historical risk score is a descriptive training statistic, NOT a
    calibrated estimate of the consequences of the current agent's action.
    """
    def __init__(self, library, ir_history, *, config=None):
        if ir_history is None:
            raise ContractError('IR risk reranking requires audited historical IR overlay')
        self.library = library
        self.ir_history = ir_history
        self.config = config or RerankConfig()
        self.training_risk = {}
        for role in AGENTS[:3]:
            cases = library.cases[role]
            history = ir_history.rows[role]
            if len(cases) < 3 or set(history) != {c['experience_id'] for c in cases}:
                raise ContractError('P2-A risk source not aligned with role library: ' + role)
            ids = [rec['experience_id'] for rec in cases]
            costs = [float(history[cid]['own_historical_adjustment_cost']) for cid in ids]
            margins = [float(history[cid]['own_historical_ir_margin']) for cid in ids]
            relatives = [float(history[cid]['own_historical_relative_ir_margin']) for cid in ids]
            cost_percentiles = _midrank_percentiles(costs)
            margin_percentiles = _midrank_percentiles(margins)
            relative_percentiles = _midrank_percentiles(relatives)
            risks = (.55 * cost_percentiles + .30 * (1 - relative_percentiles)
                     + .15 * (1 - margin_percentiles))
            if not np.isfinite(risks).all() or np.any(risks < -1e-9) or np.any(risks > 1 + 1e-9):
                raise ContractError('Invalid historical training risk')
            self.training_risk[role] = {cid: float(risk) for cid, risk in zip(ids, risks)}

    def query(self, role, observation, *, top_k=3):
        if role not in AGENTS or not isinstance(top_k, int) or top_k < 1:
            raise ContractError('Invalid role/top_k for risk-aware reranking')
        nearest = self.library.query(role, observation, top_k=top_k)
        base_ids = [rec['experience_id'] for rec in nearest]
        if role == 'agent_g' or top_k == 1:
            return nearest, {
                'schema': 'p2a_rerank_trace_v1', 'role': role,
                'strategy': 'unchanged_original_nearest' if role == 'agent_g' else 'single_anchor',
                'baseline_top_ids': base_ids, 'selected_ids': base_ids,
                'candidate_top_ids': base_ids, 'eligible_count': len(nearest),
                'selected_scores': [], 'eligible_scores': [], 'changed_vs_nearest': False,
            }
        if top_k > self.config.pool_size:
            raise ContractError('P2-A examples cannot exceed reranking candidate pool')
        candidate_count = min(self.config.pool_size, len(self.library.cases[role]))
        candidates = self.library.query(role, observation, top_k=candidate_count)
        nearest_distance = float(candidates[0]['distance'])
        # Eligibility is restricted BEFORE historical risk ranks are considered.
        # Include original top-k even if the gap gate would otherwise exclude one.
        max_allow = min(nearest_distance + self.config.max_abs_distance_gap,
                        nearest_distance * self.config.max_distance_ratio)
        eligible = [case for case in candidates
                    if case['distance'] <= max_allow + 1e-12]
        if len(eligible) < top_k:
            eligible = candidates[:top_k]
        eligibility_ids = {c['experience_id'] for c in eligible}
        if not set(base_ids).issubset(eligibility_ids):
            raise ContractError('Original nearest cases were removed by risk gate')
        scored = []
        for case in eligible:
            distance = float(case['distance'])
            gap = max(0., (distance - nearest_distance) / max(.15, nearest_distance))
            risk = self.training_risk[role][case['experience_id']]
            score = ((1 - self.config.risk_weight) * gap
                     + self.config.risk_weight * risk)
            scored.append((float(score), distance, case['experience_id'], case, float(risk)))
        scored.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        # Always preserve the original nearest demonstration in position 0.
        anchor_id = candidates[0]['experience_id']
        remaining = [row for row in scored if row[2] != anchor_id][:top_k - 1]
        selected = [candidates[0]] + [row[3] for row in remaining]
        if len(selected) != top_k or len({rec['experience_id'] for rec in selected}) != top_k:
            raise ContractError('Risk reranking did not return distinct requested cases')
        lookup = {s[2]: s for s in scored}
        selected_score = [{
            'experience_id': case['experience_id'],
            'local_distance': float(case['distance']),
            'historical_train_risk': self.training_risk[role][case['experience_id']],
            'rerank_score': float(lookup[case['experience_id']][0]),
        } for case in selected]
        trace = {
            'schema': 'p2a_rerank_trace_v1', 'role': role,
            'strategy': 'nearest_anchor_plus_local_risk_rerank',
            'baseline_top_ids': base_ids,
            'selected_ids': [rec['experience_id'] for rec in selected],
            'candidate_top_ids': [rec['experience_id'] for rec in candidates],
            'eligible_count': len(eligible),
            'selected_scores': selected_score,
            'eligible_scores': [{
                'experience_id': case['experience_id'],
                'local_distance': float(case['distance']),
                'historical_train_risk': self.training_risk[role][case['experience_id']],
                'rerank_score': float(lookup[case['experience_id']][0]),
            } for case in eligible],
            'changed_vs_nearest': [rec['experience_id'] for rec in selected] != base_ids,
        }
        return selected, trace
