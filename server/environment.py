"""
server/environment.py
---------------------
Thin wrapper that bridges FastAPI HTTP requests
to IncidentResponseEnv calls.
"""

import logging
from typing import Any, Dict

from envs.incident_env  import IncidentResponseEnv
from models.action      import IncidentAction

logger = logging.getLogger(__name__)


def handle_reset(env: IncidentResponseEnv, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST /reset → call env.reset(), return observation as dict."""
    task_id = payload.get("task_id", "easy")
    obs     = env.reset(task_id=task_id)
    return {
        "observation": obs.model_dump(),
        "reward":      0.0,
        "done":        False,
        "info":        {"task_id": task_id},
    }


def handle_step(env: IncidentResponseEnv, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step → parse action, call env.step(), return result."""
    try:
        raw = payload.get("action", {})

        # Sanitize enum fields — fall back to safe defaults if LLM returns
        # values outside our enum (e.g. "dns_failure", "configuration_error")
        valid_root_cause_types = {
            "misconfiguration","memory_leak","network_partition","crash_loop",
            "resource_exhaustion","auth_failure","dependency_failure","unknown"
        }
        valid_remediation_actions = {
            "rollback","restart_service","scale_up","fix_config",
            "increase_connection_pool","flush_cache","reroute_traffic",
            "escalate","investigate_further"
        }
        valid_severities = {"P0","P1","P2","P3"}

        if raw.get("root_cause_type") not in valid_root_cause_types:
            raw["root_cause_type"] = "unknown"
        if raw.get("remediation_action") not in valid_remediation_actions:
            raw["remediation_action"] = "investigate_further"
        if str(raw.get("severity","")).upper() not in valid_severities:
            raw["severity"] = "P2"

        # Ensure required fields have defaults
        raw.setdefault("root_cause_service", "unknown")
        raw.setdefault("root_cause_type",    "unknown")
        raw.setdefault("severity",           "P2")
        raw.setdefault("affected_services",  [])
        raw.setdefault("remediation_action", "investigate_further")

        action = IncidentAction(**raw)
    except Exception as exc:
        logger.warning(f"Invalid action payload: {exc}")
        raise ValueError(f"Invalid action: {exc}") from exc

    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward":      reward,
        "done":        done,
        "info":        info,
    }


def handle_state(env: IncidentResponseEnv) -> Dict[str, Any]:
    """GET /state → return current state as dict."""
    return env.state().model_dump()