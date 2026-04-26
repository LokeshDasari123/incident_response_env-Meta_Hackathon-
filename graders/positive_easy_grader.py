"""graders/positive_easy_grader.py"""
from graders.base_grader import BaseGrader
from scenarios import load_scenario


class PositiveEasyGrader(BaseGrader):
    def __init__(self):
        scenario = load_scenario("positive_easy")
        super().__init__(scenario.grader_rubric, "positive_easy")

    def validate_rubric(self) -> bool:
        assert "root_cause_service" in self.rubric
        assert "remediation_action" in self.rubric
        return True
