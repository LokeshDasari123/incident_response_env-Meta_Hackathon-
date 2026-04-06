"""
graders/base_grader.py
-----------------------
Base grader. Uses weights defined in each scenario's metadata.json rubric.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from models.reward import IncidentReward, RewardBreakdown
from graders.scoring import (
    score_root_cause,
    score_action,
    score_severity,
    score_communication,
)


def _score_root_cause_type(action, rubric):
    predicted = (action.get("root_cause_type") or "unknown").strip().lower()
    rc_rubric = rubric.get("root_cause_type", {})
    exact     = (rc_rubric.get("exact_match") or "").lower()
    partials  = rc_rubric.get("partial_credit", {})
    if not exact:
        return 1.0, "root_cause_type not evaluated."
    if predicted == exact:
        return 1.0, f"Correct fault type: {predicted}"
    if isinstance(partials, dict):
        for k, v in partials.items():
            if predicted == k.lower():
                return float(v), f"Partial ({v:.0%}): {predicted}"
    elif isinstance(partials, list):
        for k in partials:
            if predicted == k.lower():
                return 0.3, f"Partial (30%): {predicted}"
    return 0.0, f"Wrong fault type: '{predicted}'. Expected: '{exact}'."


def _score_affected_services(action, rubric):
    aff_rubric   = rubric.get("affected_services", {})
    required     = [s.lower() for s in (aff_rubric.get("required") or [])]
    penalize     = [s.lower() for s in (aff_rubric.get("penalize_if_includes") or [])]
    penalty_each = float(aff_rubric.get("penalty_per_wrong", 0.10))
    predicted    = [s.lower() for s in (action.get("affected_services") or [])]
    if not required:
        return 1.0, "affected_services not evaluated."
    found        = sum(1 for r in required if r in predicted)
    coverage     = found / len(required)
    wrong_hits   = sum(1 for p in penalize if p in predicted)
    noise_penalty = min(1.0, wrong_hits * penalty_each)
    score        = max(0.0, coverage - noise_penalty)
    fb = f"Coverage {coverage:.0%}"
    if wrong_hits:
        fb += f", -{noise_penalty:.2f} for {wrong_hits} red herring(s)"
    return round(score, 4), fb


class BaseGrader(ABC):
    def __init__(self, rubric: Dict[str, Any], task_id: str) -> None:
        self.rubric  = rubric
        self.task_id = task_id

    def grade(self, action, step, max_steps):
        bd = RewardBreakdown()

        # Read weights from rubric
        w_rc  = float(self.rubric.get("root_cause_service", {}).get("weight", 0.30))
        w_rct = float(self.rubric.get("root_cause_type",    {}).get("weight", 0.10))
        w_svc = float(self.rubric.get("affected_services",  {}).get("weight", 0.10))
        w_sev = float(self.rubric.get("severity",           {}).get("weight", 0.15))
        w_act = float(self.rubric.get("remediation_action", {}).get("weight", 0.20))
        w_com = float(self.rubric.get("stakeholder_message",{}).get("weight", 0.10))
        w_spd = float(self.rubric.get("speed_bonus",        {}).get("weight", 0.05))

        # Normalise
        total_w = w_rc + w_rct + w_svc + w_sev + w_act + w_com + w_spd
        if total_w > 0:
            w_rc /= total_w; w_rct /= total_w; w_svc /= total_w
            w_sev /= total_w; w_act /= total_w; w_com /= total_w
            w_spd /= total_w

        # Score components
        rc_score,  rc_fb  = score_root_cause(action, self.rubric)
        rct_score, rct_fb = _score_root_cause_type(action, self.rubric)
        svc_score, svc_fb = _score_affected_services(action, self.rubric)
        act_score, a_fb   = score_action(action, self.rubric)
        sev_score, s_fb   = score_severity(action, self.rubric)
        com_score, c_fb   = score_communication(action, self.rubric)

        # Speed bonus
        speed_rubric = self.rubric.get("speed_bonus", {})
        full_by  = speed_rubric.get("full_bonus_within_steps", max_steps // 3)
        half_by  = speed_rubric.get("half_bonus_within_steps", max_steps // 2)
        spd_score = 1.0 if step <= full_by else (0.5 if step <= half_by else 0.0)

        # Weighted sum
        raw = (rc_score*w_rc + rct_score*w_rct + svc_score*w_svc +
               sev_score*w_sev + act_score*w_act + com_score*w_com +
               spd_score*w_spd)

        bd.root_cause_score    = rc_score
        bd.action_score        = act_score
        bd.severity_score      = sev_score
        bd.communication_score = com_score
        bd.speed_bonus         = spd_score

        bd.false_positive_penalty    = self._false_positive_penalty(action)
        bd.wrong_action_penalty      = self._wrong_action_penalty(action)
        bd.missed_escalation_penalty = self._missed_escalation_penalty(action)

        bd.raw_score     = round(raw, 4)
        bd.total_penalty = round(
            bd.false_positive_penalty  * 0.15 +
            bd.wrong_action_penalty    * 0.20 +
            bd.missed_escalation_penalty * 0.25, 4)
        bd.final_score = round(max(0.0, min(1.0, bd.raw_score - bd.total_penalty)), 4)

        bd.partial_credits = {
            "root_cause":        rc_fb,
            "root_cause_type":   rct_fb,
            "affected_services": svc_fb,
            "action":            a_fb,
            "severity":          s_fb,
            "communication":     c_fb,
        }
        bd.feedback = (
            f"RC:{rc_score:.0%} Type:{rct_score:.0%} "
            f"Svc:{svc_score:.0%} Act:{act_score:.0%} "
            f"Sev:{sev_score:.0%} Com:{com_score:.0%} "
            f"Pen:-{bd.total_penalty:.2f} → {bd.final_score:.2f}"
        )

        return IncidentReward(
            reward=bd.final_score, breakdown=bd,
            is_terminal=True, step=step, task_id=self.task_id,
        )

    def _false_positive_penalty(self, action):
        red_herrings = [r.lower() for r in
            (self.rubric.get("affected_services", {}).get("penalize_if_includes") or [])]
        return 1.0 if (action.get("root_cause_service") or "").lower() in red_herrings else 0.0

    def _wrong_action_penalty(self, action):
        penalty_acts = [x.lower() for x in
            (self.rubric.get("remediation_action", {}).get("penalty") or [])]
        return 1.0 if (action.get("remediation_action") or "").lower() in penalty_acts else 0.0

    def _missed_escalation_penalty(self, action):
        sev      = (action.get("severity") or "").upper()
        msg      = action.get("stakeholder_message") or ""
        required = self.rubric.get("severity", {}).get("exact_match", "") == "P0"
        return 1.0 if required and sev == "P0" and not msg.strip() else 0.0

    @abstractmethod
    def validate_rubric(self) -> bool: ...