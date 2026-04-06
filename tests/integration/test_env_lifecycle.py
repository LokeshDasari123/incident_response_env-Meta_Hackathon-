"""tests/integration/test_env_lifecycle.py"""
import pytest
from envs.incident_env  import IncidentResponseEnv
from models.action      import IncidentAction, RootCauseType, SeverityLevel, RemediationAction


@pytest.fixture
def env():
    e = IncidentResponseEnv()
    yield e
    e.close()


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_reset_returns_observation(env, task_id):
    obs = env.reset(task_id=task_id)
    assert obs.task_id   == task_id
    assert obs.step      == 0
    assert not obs.done
    assert len(obs.alerts)   > 0
    assert len(obs.topology) > 0


def test_step_returns_valid_reward(env):
    env.reset(task_id="easy")
    action = IncidentAction(
        root_cause_service = "payments-db",
        root_cause_type    = RootCauseType.MISCONFIGURATION,
        severity           = SeverityLevel.P0,
        affected_services  = ["payments-db"],
        remediation_action = RemediationAction.FIX_CONFIG,
        stakeholder_message= "Investigating. ETA 5 mins.",
    )
    obs, reward, done, info = env.step(action)
    assert 0.0 <= reward <= 1.0
    assert isinstance(done, bool)
    assert "reward_breakdown" in info


def test_state_reflects_progress(env):
    env.reset(task_id="easy")
    s1 = env.state()
    assert s1.step == 0

    action = IncidentAction(
        root_cause_service = "payments-db",
        root_cause_type    = RootCauseType.MISCONFIGURATION,
        severity           = SeverityLevel.P0,
        affected_services  = ["payments-db"],
        remediation_action = RemediationAction.FIX_CONFIG,
        stakeholder_message= "On it.",
    )
    env.step(action)
    s2 = env.state()
    assert s2.step == 1
    assert s2.actions_taken == 1


def test_reset_clears_state(env):
    env.reset("easy")
    action = IncidentAction(
        root_cause_service = "x",
        root_cause_type    = RootCauseType.UNKNOWN,
        severity           = SeverityLevel.P3,
        affected_services  = [],
        remediation_action = RemediationAction.INVESTIGATE_FURTHER,
    )
    env.step(action)
    env.reset("easy")
    s = env.state()
    assert s.step         == 0
    assert s.actions_taken == 0


def test_hard_task_has_sla_pressure(env):
    obs = env.reset("hard")
    assert obs.sla_breach_in_steps is not None
    assert obs.time_pressure >= 0.0