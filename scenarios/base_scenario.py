"""
scenarios/base_scenario.py
--------------------------
Abstract base class for all incident scenarios.
Loads scenario + metadata JSON and provides clean accessors.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseScenario(ABC):
    """
    Base class for Easy / Medium / Hard scenarios.
    Each scenario wraps a scenario.json + metadata.json pair.
    """

    SCENARIO_DIR: Path = Path(__file__).parent

    def __init__(self, difficulty: str) -> None:
        self.difficulty = difficulty
        scenario_path = self.SCENARIO_DIR / difficulty / "scenario.json"
        metadata_path = self.SCENARIO_DIR / difficulty / "metadata.json"

        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self._scenario: Dict[str, Any]  = json.loads(scenario_path.read_text())
        self._metadata: Dict[str, Any]  = json.loads(metadata_path.read_text())

    # ── Scenario accessors ────────────────────────────────────────────────────
    @property
    def scenario_id(self) -> str:
        return self._scenario["scenario_id"]

    @property
    def name(self) -> str:
        return self._scenario["name"]

    @property
    def description(self) -> str:
        return self._scenario["description"]

    @property
    def fault_type(self) -> str:
        return self._scenario["fault_type"]

    @property
    def max_steps(self) -> int:
        return self._scenario["max_steps"]

    @property
    def sla_breach_step(self) -> Optional[int]:
        return self._scenario.get("sla_breach_step")

    @property
    def services(self) -> List[Dict[str, Any]]:
        return self._scenario["services"]

    @property
    def topology(self) -> List[Dict[str, Any]]:
        return self._scenario["topology"]

    @property
    def alerts(self) -> List[Dict[str, Any]]:
        return self._scenario["alerts"]

    @property
    def timeline(self) -> List[Dict[str, Any]]:
        return self._scenario["timeline"]

    # ── Metadata accessors ────────────────────────────────────────────────────
    @property
    def ground_truth(self) -> Dict[str, Any]:
        return self._metadata["ground_truth"]

    @property
    def grader_rubric(self) -> Dict[str, Any]:
        return self._metadata["grader_rubric"]

    @property
    def expected_scores(self) -> Dict[str, float]:
        return self._metadata.get("expected_scores", {})

    # ── Derived helpers ───────────────────────────────────────────────────────
    def get_alerts_for_agent(self) -> List[Dict[str, Any]]:
        """Return alerts without internal grader flags."""
        return [
            {k: v for k, v in alert.items() if k != "is_red_herring"}
            for alert in self.alerts
        ]

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Return current metrics per service for agent observation."""
        metrics = {}
        for svc in self.services:
            name = svc["name"]
            metrics[name] = {
                "cpu_utilization":    svc.get("incident_cpu",    svc.get("normal_cpu",    0.3)),
                "memory_utilization": svc.get("incident_mem",    svc.get("normal_mem",    0.4)),
                "http_rt":            svc.get("incident_http_rt",svc.get("normal_http_rt", None)),
                "consumer_rpc_rt":    svc.get("incident_consumer_rpc_rt", svc.get("normal_consumer_rpc_rt", None)),
                "provider_rpc_rt":    svc.get("incident_provider_rpc_rt", svc.get("normal_provider_rpc_rt", None)),
                "is_healthy":         not svc.get("is_root_cause", False) and not svc.get("crash_loop", False),
                "restart_count":      svc.get("restart_count", 0),
                "status":             svc.get("status", "healthy"),
            }
        return metrics

    def get_topology_for_agent(self) -> List[Dict[str, Any]]:
        """Return topology edges with current latencies injected."""
        result = []
        for edge in self.topology:
            current_lat = edge["avg_rt_ms"]
            # Inflate latency for affected edges
            for svc in self.services:
                if svc["name"] == edge["downstream"] and svc.get("incident_provider_rpc_rt"):
                    current_lat = svc["incident_provider_rpc_rt"]
                    break
            result.append({
                "upstream_service":    edge["upstream"],
                "downstream_service":  edge["downstream"],
                "rpc_type":            edge["rpc_type"],
                "avg_latency_ms":      edge["avg_rt_ms"],
                "current_latency_ms":  current_lat,
            })
        return result

    def get_red_herring_services(self) -> List[str]:
        return self.ground_truth.get("red_herring_services", [])

    @abstractmethod
    def validate(self) -> bool:
        """Validate scenario data integrity."""
        ...


class EasyScenario(BaseScenario):
    def __init__(self):
        super().__init__("easy")

    def validate(self) -> bool:
        assert len(self.services) >= 2
        assert self.ground_truth["root_cause_service"]
        assert self.ground_truth["severity"] == "P0"
        return True


class MediumScenario(BaseScenario):
    def __init__(self):
        super().__init__("medium")

    def validate(self) -> bool:
        assert len(self.services) >= 3
        assert self.ground_truth["root_cause_service"]
        assert len(self.get_red_herring_services()) >= 1
        return True


class HardScenario(BaseScenario):
    def __init__(self):
        super().__init__("hard")

    def validate(self) -> bool:
        assert len(self.services) >= 5
        assert self.ground_truth["root_cause_service"]
        assert self.sla_breach_step is not None
        assert len(self.get_red_herring_services()) >= 2
        return True


SCENARIO_MAP = {
    "easy":   EasyScenario,
    "medium": MediumScenario,
    "hard":   HardScenario,
}


def load_scenario(task_id: str) -> BaseScenario:
    """Factory: load scenario by task_id string."""
    if task_id not in SCENARIO_MAP:
        raise ValueError(f"Unknown task_id '{task_id}'. Must be one of {list(SCENARIO_MAP)}")
    return SCENARIO_MAP[task_id]()