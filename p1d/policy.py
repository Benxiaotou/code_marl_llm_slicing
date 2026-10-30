"""Role-only IR historical data / risk-prompt 2x2 ablation policies.

All four conditions use identical P1-C nearest-local-example retrieval:
  icl            : neither factor
  ir_prompt_only : risk instruction only
  ir_data_only   : own-role historical feedback only
  ir_icl         : both factors (original P1-D prompt, byte-for-byte)

The runtime receives no actual current reward, other agents' observations,
private offline joint records, or access to the GEANT environment for planning.
"""
import json

from p1c.policy import RetrievalPolicy, make_icl_messages


# Preserve EXACT original P1-D risk instruction so existing P1-D logs are
# strictly comparable with newly generated prompt-only / data-only logs.
IR_RISK_INSTRUCTION = (
    '\nIR-aware historical reasoning: Protect existing slices\' '
    'individual rationality while considering new admission. '
    'Do not invent current utility or claim to know other agents\' '
    'current actions. Treat positive historical margins as '
    'contextual feedback, NOT guaranteed consequences. '
    'Prefer an action that balances cooperation with excessive '
    'resource reduction and migration/rerouting risk.'
)

_OUTPUT_SUFFIX = '\nOutput exactly one JSON object {"ratios":[...]} using the correct role dimensions.'


def _historical_feedback_block(role, matches, ir_history):
    # The IR-history module only serves historical, SAME-role annotations from
    # offline training cases, aligned with the identical nearest neighbours.
    own_history = ir_history.for_examples(role, matches)
    if role == 'agent_g':
        interpretation = ('Each number below is the new slice\'s OWN realized utility '
                          'in one historical joint decision. It is not the utility '
                          'of your CURRENT request or a predicted reward.')
    else:
        interpretation = ('Each historical IR margin is this SAME slice role\'s own '
                          'V_after minus V_before; positive values indicate that '
                          'the historical action JOINTLY with others respected that '
                          'slice\'s participation condition. Historical adjustment '
                          'costs are from that past joint outcome. These are not '
                          'predictions of your current cost or a causal attribution '
                          'to your individual action.')
    return ('\n\nFor the SAME ' + str(len(matches)) + ' historical examples above, '
            'additional training-only OWN-role feedback in exactly the same order:\n'
            + json.dumps(own_history, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':'))
            + '\n' + interpretation)


def make_ir_component_messages(role, observation, matches, *, ir_history=None,
                               risk_prompt=False, history_data=False):
    """Toggle P1-D's two factors independently without changing base ICL."""
    if history_data != (ir_history is not None):
        raise ValueError('History overlay is required if and only if history_data=True')
    messages = make_icl_messages(role, observation, matches, is_random=False)
    if risk_prompt:
        messages[0]['content'] += IR_RISK_INSTRUCTION
    if history_data:
        prefix, marker, suffix = messages[1]['content'].rpartition(_OUTPUT_SUFFIX)
        if not marker or suffix:
            raise RuntimeError('Base P1-C prompt format has changed; refuse uncertain IR insertion')
        messages[1]['content'] = (prefix + _historical_feedback_block(role, matches, ir_history)
                                  + marker)
    return messages


class IRRetrievalPolicy(RetrievalPolicy):
    """P1-D full IR-aware policy: BOTH historical data and extra IR instruction."""
    def __init__(self, library, ir_history, *, client, examples=3, seed=1):
        super().__init__(library, 'icl', client=client, examples=examples, seed=seed)
        if ir_history is None:
            raise ValueError('Full IR-aware policy requires audited role-only IR history')
        self.ir_history = ir_history

    def _make_messages(self, role, observation, matches):
        return make_ir_component_messages(role, observation, matches,
                                          ir_history=self.ir_history,
                                          risk_prompt=True, history_data=True)


class IRAblationPolicy(RetrievalPolicy):
    """Either IR instruction alone or IR history alone, NOT both."""
    METHODS = ('ir_prompt_only', 'ir_data_only')

    def __init__(self, library, method, *, client, ir_history=None, examples=3, seed=1):
        if method not in self.METHODS:
            raise ValueError('Only ir_prompt_only and ir_data_only are supported')
        if (method == 'ir_data_only') != (ir_history is not None):
            raise ValueError('IR data-only requires IR overlay; prompt-only forbids it')
        super().__init__(library, 'icl', client=client, examples=examples, seed=seed)
        self.ablation_method = method
        self.ir_history = ir_history

    def _make_messages(self, role, observation, matches):
        return make_ir_component_messages(
            role, observation, matches,
            ir_history=self.ir_history,
            risk_prompt=self.ablation_method == 'ir_prompt_only',
            history_data=self.ablation_method == 'ir_data_only')
