"""tests/unit/test_graders.py"""
import pytest
from graders import load_grader


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_grader_loads(task_id):
    g = load_grader(task_id)
    assert g is not None
    assert g.task_id == task_id


def test_easy_correct_action():
    g = load_grader("easy")
    result = g.grade(
        action={
            "root_cause_service":  "payments-db",
            "root_cause_type":     "misconfiguration",
            "severity":            "P0",
            "affected_services":   ["payments-db", "payments-api", "checkout-ui"],
            "remediation_action":  "fix_config",
            "stakeholder_message": "Investigating payment delays. ETA 8 mins.",
            "confidence":          0.9,
        },
        step=2,
        max_steps=10,
    )
    assert result.reward >= 0.70
    assert result.breakdown.root_cause_score == 1.0
    assert result.breakdown.action_score == 1.0


def test_easy_wrong_root_cause():
    g = load_grader("easy")
    result = g.grade(
        action={
            "root_cause_service": "worker-node-4",
            "root_cause_type":    "resource_exhaustion",
            "severity":           "P2",
            "affected_services":  ["worker-node-4"],
            "remediation_action": "scale_up",
            "confidence":         0.5,
        },
        step=3,
        max_steps=10,
    )
    assert result.reward < 0.30
    assert result.breakdown.root_cause_score == 0.0


def test_scores_in_range():
    """All grader scores must be in [0.0, 1.0]."""
    for task_id in ["easy", "medium", "hard"]:
        g = load_grader(task_id)
        result = g.grade(
            action={
                "root_cause_service": "unknown",
                "root_cause_type":    "unknown",
                "severity":           "P3",
                "affected_services":  [],
                "remediation_action": "investigate_further",
                "confidence":         0.1,
            },
            step=1,
            max_steps=10,
        )
        assert 0.0 <= result.reward <= 1.0, f"{task_id}: reward out of range"