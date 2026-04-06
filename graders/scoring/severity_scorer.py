"""
graders/scoring/severity_scorer.py
------------------------------------
Scores severity classification (P0/P1/P2/P3).
Weight: 0.20
"""

from typing import Any, Dict, Tuple


def score_severity(
    action: Dict[str, Any],
    rubric: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Score severity classification.

    Returns:
        (score 0.0-1.0, feedback string)
    """
    predicted = (action.get("severity") or "").strip().upper()
    s_rubric  = rubric.get("severity", {})
    exact     = (s_rubric.get("exact_match") or "").upper()
    partials  = {k.upper(): v for k, v in (s_rubric.get("partial_credit") or {}).items()}

    if not predicted:
        return 0.0, "❌ No severity provided."

    if predicted == exact:
        return 1.0, f"✅ Correct severity: {predicted}"

    if predicted in partials:
        score = float(partials[predicted])
        direction = "too low" if predicted > exact else "too high"
        return score, f"⚠️ Severity {direction}: got {predicted}, expected {exact}. Partial credit: {score:.0%}"

    return 0.0, f"❌ Wrong severity: {predicted}. Expected: {exact}"