"""graders/hard_grader.py — Hard task grader with SLA pressure."""
from typing import Any, Dict
from graders.base_grader import BaseGrader
from models.reward import IncidentReward
from scenarios import load_scenario


class HardGrader(BaseGrader):
    def __init__(self):
        scenario = load_scenario("hard")
        super().__init__(scenario.grader_rubric, "hard")
        self._sla_breach_step = scenario.sla_breach_step or 6

    def grade(self, action: Dict[str, Any], step: int, max_steps: int) -> IncidentReward:
        # Base grading
        result = super().grade(action, step, max_steps)

        # Apply SLA breach penalty on top
        sla_rubric = self.rubric.get("sla_pressure", {})
        sla_step   = sla_rubric.get("penalty_if_no_action_by_step", self._sla_breach_step)
        sla_amount = float(sla_rubric.get("penalty_amount", 0.15))

        if step > sla_step:
            bd = result.breakdown
            extra_pen = min(1.0, bd.missed_escalation_penalty + sla_amount)
            bd.missed_escalation_penalty = extra_pen
            bd.total_penalty = round(
                bd.false_positive_penalty  * 0.15 +
                bd.wrong_action_penalty    * 0.20 +
                bd.missed_escalation_penalty * 0.25, 4)
            bd.final_score = round(max(0.0, min(1.0, bd.raw_score - bd.total_penalty)), 4)
            bd.feedback += f" | SLA breached at step {sla_step} (-{sla_amount:.0%})"

            return IncidentReward(
                reward=bd.final_score, breakdown=bd,
                is_terminal=True, step=step, task_id="hard",
            )
        return result

    def _false_positive_penalty(self, action):
        red_herrings = [r.lower() for r in
            (self.rubric.get("affected_services", {}).get("penalize_if_includes") or [])]
        predicted_root = (action.get("root_cause_service") or "").lower()
        predicted_svcs = [s.lower() for s in (action.get("affected_services") or [])]
        penalty_per    = float(self.rubric.get("affected_services", {}).get("penalty_per_wrong", 0.15))
        hits = sum(1 for svc in red_herrings if svc == predicted_root or svc in predicted_svcs)
        return min(1.0, hits * penalty_per)

    def validate_rubric(self) -> bool:
        assert "root_cause_service" in self.rubric
        assert "sla_pressure" in self.rubric
        return True