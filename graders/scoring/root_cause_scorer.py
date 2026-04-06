"""
graders/scoring/root_cause_scorer.py
-------------------------------------
Scores whether the agent correctly identified the root cause service.
Weight: 0.35 — highest weight component.
"""

from typing import Any, Dict, Tuple


def score_root_cause(
    action: Dict[str, Any],
    rubric: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Score root cause identification.

    Returns:
        (score 0.0-1.0, feedback string)
    """
    predicted = (action.get("root_cause_service") or "").strip().lower()
    rc_rubric  = rubric.get("root_cause_service", {})
    exact      = (rc_rubric.get("exact_match") or "").lower()
    partials   = rc_rubric.get("partial_credit", {})

    if not predicted:
        return 0.0, "❌ No root_cause_service provided."

    # Exact match
    if predicted == exact:
        return 1.0, f"✅ Correct root cause: {predicted}"

    # Partial match (dict or list)
    if isinstance(partials, dict):
        for svc, credit in partials.items():
            if predicted == svc.lower():
                return float(credit), f"⚠️ Partial credit ({credit:.0%}): {predicted} is a downstream effect, not root cause."
    elif isinstance(partials, list):
        for svc in partials:
            if predicted == svc.lower():
                return 0.3, f"⚠️ Partial credit (30%): {predicted} is related but not the root cause."

    return 0.0, f"❌ Wrong root cause: '{predicted}'. Correct: '{exact}'."