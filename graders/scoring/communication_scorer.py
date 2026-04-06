"""
graders/scoring/communication_scorer.py
-----------------------------------------
Scores stakeholder message quality.
Weight: 0.10
"""

from typing import Any, Dict, Tuple


def score_communication(
    action: Dict[str, Any],
    rubric: Dict[str, Any],
) -> Tuple[float, str]:
    """
    Score stakeholder communication.

    Returns:
        (score 0.0-1.0, feedback string)
    """
    message  = (action.get("stakeholder_message") or "").strip()
    c_rubric = rubric.get("stakeholder_message", {})
    required = c_rubric.get("required", False)

    if not required:
        return 1.0, "✅ Stakeholder message not required for this severity."

    if not message:
        return 0.0, "❌ Stakeholder message required but not provided."

    must_contain = [k.lower() for k in (c_rubric.get("must_contain_any") or [])]
    quality_kw   = [k.lower() for k in (c_rubric.get("quality_bonus_keywords") or [])]
    msg_lower    = message.lower()

    # Check required keywords
    has_required = any(kw in msg_lower for kw in must_contain)
    if not has_required:
        return 0.3, f"⚠️ Message present but missing key context. Should mention: {must_contain}"

    # Base score
    score = 0.7

    # Quality bonuses
    bonus_hits = sum(1 for kw in quality_kw if kw in msg_lower)
    bonus = min(0.3, bonus_hits * 0.1)
    score = min(1.0, score + bonus)

    # Length check
    if len(message) < 30:
        score = max(0.3, score - 0.2)
        return score, f"⚠️ Message too brief ({len(message)} chars). Add more context."

    return score, f"✅ Stakeholder message adequate. Score: {score:.0%}"