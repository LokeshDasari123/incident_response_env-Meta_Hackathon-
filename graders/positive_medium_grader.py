"""graders/positive_medium_grader.py"""
from graders.base_grader import BaseGrader
from scenarios import load_scenario


class PositiveMediumGrader(BaseGrader):
    def __init__(self):
        scenario = load_scenario("positive_medium")
        super().__init__(scenario.grader_rubric, "positive_medium")

    def validate_rubric(self) -> bool:
        assert "root_cause_service" in self.rubric
        assert "remediation_action" in self.rubric
        return True
