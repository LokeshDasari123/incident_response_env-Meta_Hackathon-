"""tests/e2e/test_full_episode.py — full episode simulation"""
import pytest
from envs.incident_env import IncidentResponseEnv
from models.action import IncidentAction, RootCauseType, SeverityLevel, RemediationAction

PERFECT_ACTIONS = {
    "easy": IncidentAction(
        root_cause_service="payments-db", root_cause_type=RootCauseType.MISCONFIGURATION,
        severity=SeverityLevel.P0, affected_services=["payments-db","payments-api","checkout-ui"],
        remediation_action=RemediationAction.FIX_CONFIG,
        stakeholder_message="Investigating payment delays caused by misconfiguration. ETA 8 mins.",
        confidence=0.95,
    ),
    "medium": IncidentAction(
        root_cause_service="user-service", root_cause_type=RootCauseType.NETWORK_PARTITION,
        severity=SeverityLevel.P1, affected_services=["user-service","auth-service","api-gateway","storefront-ui"],
        remediation_action=RemediationAction.FIX_CONFIG,
        stakeholder_message="Investigating login issues caused by DNS failure. ETA 5 mins.",
        confidence=0.90,
    ),
    "hard": IncidentAction(
        root_cause_service="payments-db", root_cause_type=RootCauseType.MEMORY_LEAK,
        severity=SeverityLevel.P0, affected_services=["payments-db","cache-service","order-service","api-gateway","storefront-ui"],
        remediation_action=RemediationAction.RESTART_SERVICE,
        stakeholder_message="P0: Revenue-impacting memory leak on payments-db. Escalating. ETA 10 mins.",
        confidence=0.85,
    ),
}

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_perfect_agent_scores_high(task_id):
    env = IncidentResponseEnv()
    env.reset(task_id)
    _, reward, done, _ = env.step(PERFECT_ACTIONS[task_id])
    env.close()
    assert reward >= 0.65, f"{task_id}: perfect agent scored only {reward:.3f}"

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_random_agent_scores_low(task_id):
    env = IncidentResponseEnv()
    env.reset(task_id)
    random_action = IncidentAction(
        root_cause_service="worker-node-4", root_cause_type=RootCauseType.UNKNOWN,
        severity=SeverityLevel.P3, affected_services=[],
        remediation_action=RemediationAction.INVESTIGATE_FURTHER,
    )
    _, reward, _, _ = env.step(random_action)
    env.close()
    assert reward < 0.30, f"{task_id}: random agent scored too high {reward:.3f}"