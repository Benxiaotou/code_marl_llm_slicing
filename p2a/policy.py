"""Frozen Qwen P2-A policy: ONLY historical example selection changes."""

from p1d.policy import IRRetrievalPolicy
from .rerank import IRRiskReranker, RerankConfig


class _RiskQueryAdapter:
    """Provides the same query interface as P1-C but swaps training examples.

    The original LocalHistory instance is unchanged and remains responsible for
    loading, validating and scaling local observations.
    """
    def __init__(self, library, risk_reranker):
        self.original = library
        self.reranker = risk_reranker
        self.cases = library.cases
        self.last_trace = {}

    def query(self, role, observation, top_k=3):
        selected, trace = self.reranker.query(role, observation, top_k=top_k)
        self.last_trace[role] = trace
        return selected


class IRRiskRerankPolicy(IRRetrievalPolicy):
    def __init__(self, library, ir_history, *, client, examples=3, seed=1,
                 config=None):
        reranker = IRRiskReranker(library, ir_history, config=config or RerankConfig())
        self.reranker = reranker
        self.adapter = _RiskQueryAdapter(library, reranker)
        super().__init__(self.adapter, ir_history, client=client, examples=examples,
                         seed=seed)

    def act(self, observations):
        self.adapter.last_trace.clear()
        actions, entries = super().act(observations)
        for role, entry in entries.items():
            trace = self.adapter.last_trace[role]
            if entry['retrieved_ids_for_audit_only'] != trace['selected_ids']:
                raise RuntimeError('P2-A retrieval trace mismatch')
            entry['p2a_rerank_trace_for_audit_only'] = trace
        return actions, entries
