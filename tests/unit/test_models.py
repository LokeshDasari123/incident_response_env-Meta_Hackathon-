"""tests/unit/test_models.py"""
import pytest
from models.action      import IncidentAction, RootCauseType, SeverityLevel, RemediationAction
from models.observation import IncidentObservation
from models.reward      import IncidentReward, RewardBreakdown
from models.state       import IncidentState


def test_action_valid():
    a = IncidentAction(
        root_cause_service  = "payments-db",
        root_cause_type     = RootCauseType.MISCONFIGURATION,
        severity            = SeverityLevel.P0,
        affected_services   = ["payments-db", "payments-api"],
        remediation_action  = RemediationAction.FIX_CONFIG,
        stakeholder_message = "Investigating payment delays.",
        confidence          = 0.9,
    )
    assert a.root_cause_service == "payments-db"
    assert a.severity == SeverityLevel.P0

def test_action_deduplicates_services():
    a = IncidentAction(
        root_cause_service = "svc-a",
        root_cause_type    = RootCauseType.CRASH_LOOP,
        severity           = SeverityLevel.P1,
        affected_services  = ["svc-a", "svc-a", "svc-b"],
        remediation_action = RemediationAction.RESTART_SERVICE,
    )
    assert len(a.affected_services) == 2

def test_reward_compute():
    bd = RewardBreakdown(
        root_cause_score    = 1.0,
        action_score        = 1.0,
        severity_score      = 1.0,
        communication_score = 1.0,
        speed_bonus         = 1.0,
    )
    bd.compute()
    assert bd.final_score == pytest.approx(1.0, abs=0.01)

def test_reward_penalty_clamps_to_zero():
    bd = RewardBreakdown(
        root_cause_score         = 0.0,
        missed_escalation_penalty= 1.0,
    )
    bd.compute()
    assert bd.final_score == 0.0

def test_observation_valid():
    obs = IncidentObservation(
        step=1, max_steps=10, task_id="easy",
        episode_id="ep_001", done=False, reward=0.0,
    )
    assert obs.step == 1
    assert not obs.done