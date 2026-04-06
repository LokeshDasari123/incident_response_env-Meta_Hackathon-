"""
graders/scoring/action_scorer.py
---------------------------------
Scores the remediation action selected by the agent.
Weight: 0.25
"""

from typing import Any, Dict, Tuple


def score_action(
    action: Dict[str, Any],
    rubric: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Score the remediation action.

    Returns:
        (score 0.0-1.0, feedback string)
    """
    predicted = (action.get("remediation_action") or "").strip().lower()
    a_rubric  = rubric.get("remediation_action", {})

    full_credit = [x.lower() for x in (a_rubric.get("full_credit") or [])]
    partial_map = {k.lower(): v for k, v in (a_rubric.get("partial_credit") or {}).items()}
    zero_credit = [x.lower() for x in (a_rubric.get("zero_credit") or [])]
    penalty_acts= [x.lower() for x in (a_rubric.get("penalty") or [])]

    if not predicted:
        return 0.0, "❌ No remediation_action provided."

    if predicted in full_credit:
        return 1.0, f"✅ Correct action: {predicted}"

    if predicted in partial_map:
        score = float(partial_map[predicted])
        return score, f"⚠️ Partial credit ({score:.0%}): {predicted} is acceptable but not optimal."

    if predicted in penalty_acts:
        return 0.0, f"❌ Penalized action: {predicted} could cause further damage."

    if predicted in zero_credit:
        return 0.0, f"❌ Ineffective action: {predicted} does not address root cause."

    return 0.1, f"⚠️ Unknown action '{predicted}'. Minimal credit."