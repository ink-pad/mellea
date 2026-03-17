from ....backends import ModelOption
from ....backends.adapters import AdapterMixin, AdapterType, IntrinsicAdapter
from ....stdlib import functional as mfuncs
from ...context import ChatContext
from ..chat import Message
from .intrinsic import Intrinsic
from ._util import call_intrinsic


def policy_guardrails(context: ChatContext, backend: AdapterMixin, policy_text: str) -> str:
    """Checks whether text complied with specified policy.

    Uses the policy_guardrails LoRA adapter to judge whether the scenario
    described in the last message in ``context`` is compliant with the given ``policy_text``.

    :param context: Chat context containing the conversation to evaluate.
    :param backend: Backend instance that supports LoRA adapters.
    :param policy_text: Policy against with compliance is to be checked
    :return: Compliance as a "Yes/No/Ambiguous" label (Yes = compliant).
    """


    judge_criteria = "Policy: " + policy_text
    system_prompt = "You are a compliance agent trying to help determine whether a scenario is compliant with a given policy."

    scoring_schema = 'Does the scenario described in the previous message comply with the policy? Compliance is determined based on the following guidelines:\n<start_of_guidelines>:\n- \"Yes\"  if, given the scenario and the policy, it can be decided with certainty that the scenario complies with the policy\n- \"No\"  if, given the scenario and the policy, it can be decided with certainty that the scenario does not comply with the policy\n- \"Ambiguous\" if, given the scenario and the policy, it is not possible to decide whether the scenario complies with, or violates, the policy and more information is needed to decide with certainty.\n<end_of_guidelines>\n\nYour answer must be either "Yes", "No", or "Ambiguous". You MUST return your answer as a valid JSON object with the key "label". For example, if your answer is "Yes", respond as "{"label":"Yes"}".'

    judge_protocol = f"<guardian> {system_prompt}\n\n### Criteria: {judge_criteria}\n\n### Scoring Schema: {scoring_schema}"

    context = context.add(Message("user", judge_protocol))
    result_json = call_intrinsic("policy_guardrails", context, backend)
    return result_json["label"]



