"""graders/medium_grader.py"""
from typing import Any, Dict
from graders.base_grader import BaseGrader
from scenarios import load_scenario


class MediumGrader(BaseGrader):
    def __init__(self):
        scenario = load_scenario("medium")
        super().__init__(scenario.grader_rubric, "medium")

    def _false_positive_penalty(self, action: Dict[str, Any]) -> float:
        """Medium: penalize each red herring service flagged."""
        red_herrings = [
            r.lower()
            for r in (self.rubric.get("affected_services", {}).get("penalize_if_includes") or [])
        ]
        predicted_services = [
            s.lower()
            for s in (action.get("affected_services") or [])
        ]
        predicted_root = (action.get("root_cause_service") or "").lower()

        hits = sum(
            1 for svc in red_herrings
            if svc in predicted_services or svc == predicted_root
        )
        penalty_per = float(
            self.rubric.get("affected_services", {}).get("penalty_per_wrong", 0.10)
        )
        return min(1.0, hits * penalty_per)

    def validate_rubric(self) -> bool:
        assert "root_cause_service" in self.rubric
        assert "noise_filtering_bonus" in self.rubric or True
        return True