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

    # ── Progressive Observation Methods ───────────────────────────────────────
    # These enable observations that evolve over time: the incident starts at
    # the root cause and cascades outward through the topology. Each service
    # degrades based on its distance from the root cause in the call graph.

    def _compute_cascade_order(self) -> Dict[str, int]:
        """
        BFS from root cause through reverse topology edges.

        Returns {service_name: onset_step} where onset_step is the number
        of hops from the root cause. Services not in the topology (red
        herrings, orphans) are NOT included in the result.

        The result is cached on the instance for the episode duration.
        """
        if hasattr(self, '_cascade_order_cache') and self._cascade_order_cache is not None:
            return self._cascade_order_cache

        rc = self.ground_truth.get("root_cause_service", "")

        # Build reverse adjacency: downstream → [upstreams]
        # Cascade propagates from downstream (root cause) to upstream (consumers)
        reverse_adj: Dict[str, List[str]] = {}
        for edge in self.topology:
            ds = edge["downstream"]
            us = edge["upstream"]
            reverse_adj.setdefault(ds, []).append(us)

        # BFS from root cause
        order: Dict[str, int] = {rc: 0}
        queue = [rc]
        while queue:
            current = queue.pop(0)
            for upstream in reverse_adj.get(current, []):
                if upstream not in order:
                    order[upstream] = order[current] + 1
                    queue.append(upstream)

        self._cascade_order_cache = order
        return order

    @staticmethod
    def _cascade_factor(step: int, onset_step: int, max_steps: int) -> float:
        """
        Degradation factor [0.0 → 1.0] for a service at a given step.

        - Root cause (onset=0): always 1.0 (already fully broken)
        - Cascade services: start at 0.0, jump to 0.2 at onset, then
          smooth ramp to 1.0 over ~30% of the episode length.

        The curve uses quadratic easing: fast initial degradation
        (visible to agent immediately), then gradual approach to 1.0.
        """
        if onset_step == 0:
            return 1.0  # Root cause: always fully degraded

        if step < onset_step:
            return 0.0  # Cascade hasn't reached this service yet

        steps_since = step - onset_step
        # Services fully degrade over ~30% of episode, minimum 2 steps
        window = max(2, int(max_steps * 0.3))
        progress = min(1.0, steps_since / window)

        # Quadratic ease-in: immediate 20% jump, then smooth curve to 100%
        initial_jump = 0.2
        remaining = 1.0 - initial_jump
        factor = initial_jump + remaining * (1.0 - (1.0 - progress) ** 2)
        return round(min(1.0, factor), 4)

    @staticmethod
    def _interpolate(normal: float, incident: float, factor: float) -> float:
        """Linear interpolation: normal → incident scaled by factor."""
        return round(normal + (incident - normal) * factor, 4)

    @staticmethod
    def _status_from_factor(factor: float) -> str:
        """Map degradation factor to human-readable status."""
        if factor < 0.1:
            return "healthy"
        if factor < 0.4:
            return "degraded"
        if factor < 0.75:
            return "critical"
        return "failing"

    def get_metrics_at_step(self, step: int, max_steps: int) -> Dict[str, Any]:
        """
        Progressive metrics: services degrade over time based on
        their position in the cascade chain.

        - Root cause: at full incident values from step 0
        - Direct dependents (1 hop): start degrading at step 1
        - Indirect dependents (2+ hops): delayed degradation
        - Red herrings: always show their incident values (static noise)
        """
        cascade_order = self._compute_cascade_order()
        metrics = {}

        for svc in self.services:
            name = svc["name"]
            onset = cascade_order.get(name)

            if onset is not None:
                # Service is in the cascade chain — progressive degradation
                factor = self._cascade_factor(step, onset, max_steps)
            else:
                # Not in cascade (red herring / orphan)
                # Show incident values (they have independent issues)
                factor = 1.0

            # CPU interpolation
            normal_cpu   = svc.get("normal_cpu", 0.3)
            incident_cpu = svc.get("incident_cpu", normal_cpu)
            cpu = self._interpolate(normal_cpu, incident_cpu, factor)

            # Memory interpolation
            normal_mem   = svc.get("normal_mem", 0.4)
            incident_mem = svc.get("incident_mem", normal_mem)
            mem = self._interpolate(normal_mem, incident_mem, factor)

            # HTTP response time
            normal_rt   = svc.get("normal_http_rt")
            incident_rt = svc.get("incident_http_rt")
            http_rt = self._interpolate(normal_rt, incident_rt, factor) \
                if normal_rt is not None and incident_rt is not None else None

            # Consumer RPC response time
            normal_crpc   = svc.get("normal_consumer_rpc_rt")
            incident_crpc = svc.get("incident_consumer_rpc_rt")
            consumer_rpc_rt = self._interpolate(normal_crpc, incident_crpc, factor) \
                if normal_crpc is not None and incident_crpc is not None else None

            # Provider RPC response time
            normal_prpc   = svc.get("normal_provider_rpc_rt")
            incident_prpc = svc.get("incident_provider_rpc_rt")
            provider_rpc_rt = self._interpolate(normal_prpc, incident_prpc, factor) \
                if normal_prpc is not None and incident_prpc is not None else None

            # Health / status
            is_healthy = factor < 0.75  # Unhealthy at 75%+ degradation
            restart_count = svc.get("restart_count", 0) if factor >= 0.9 else 0
            status = self._status_from_factor(factor)

            # Error rate: proportional to degradation, with some randomness
            error_rate = round(factor * 0.7, 4) if onset is not None and factor > 0 else 0.0

            metrics[name] = {
                "cpu_utilization":    cpu,
                "memory_utilization": mem,
                "http_rt":            http_rt,
                "consumer_rpc_rt":    consumer_rpc_rt,
                "provider_rpc_rt":    provider_rpc_rt,
                "is_healthy":         is_healthy,
                "restart_count":      restart_count,
                "status":             status,
                "error_rate":         error_rate,
            }

        return metrics

    def get_alerts_at_step(self, step: int, max_steps: int) -> List[Dict[str, Any]]:
        """
        Progressive alerts: alerts fire as the cascade reaches each service.

        - Root cause alerts: visible from step 0
        - Cascade alerts: visible from their service's onset step
        - Red herring alerts: always visible (they're independent noise)
        - Internal flags (is_red_herring) are stripped from output
        """
        cascade_order = self._compute_cascade_order()
        visible = []

        for alert in self.alerts:
            service = alert.get("service", "")
            is_rh = alert.get("is_red_herring", False)
            fired_at = alert.get("fired_at_step", 0)

            # Red herring alerts: always visible
            if is_rh:
                visible.append(
                    {k: v for k, v in alert.items() if k != "is_red_herring"}
                )
                continue

            # Calculate effective onset: max of explicitly defined fired_at and cascade onset
            onset = cascade_order.get(service)
            effective_onset = max(fired_at, onset if onset is not None else 0)

            if step >= effective_onset:
                visible.append(
                    {k: v for k, v in alert.items() if k != "is_red_herring"}
                )

        return visible

    def get_topology_at_step(self, step: int, max_steps: int) -> List[Dict[str, Any]]:
        """
        Progressive topology: edge latencies increase as the cascade
        reaches each downstream service.

        Structure (edges, services) stays constant — only latencies evolve.
        """
        cascade_order = self._compute_cascade_order()
        result = []

        for edge in self.topology:
            downstream = edge["downstream"]
            onset = cascade_order.get(downstream, 0)
            factor = self._cascade_factor(step, onset, max_steps)

            avg_lat = edge["avg_rt_ms"]

            # Find incident latency from the downstream service
            incident_lat = avg_lat
            for svc in self.services:
                if svc["name"] == downstream:
                    if svc.get("incident_provider_rpc_rt") is not None:
                        incident_lat = svc["incident_provider_rpc_rt"]
                    elif svc.get("incident_consumer_rpc_rt") is not None:
                        incident_lat = svc["incident_consumer_rpc_rt"]
                    elif svc.get("incident_http_rt") is not None:
                        incident_lat = svc["incident_http_rt"]
                    break

            current_lat = self._interpolate(avg_lat, incident_lat, factor)

            result.append({
                "upstream_service":   edge["upstream"],
                "downstream_service": edge["downstream"],
                "rpc_type":           edge["rpc_type"],
                "avg_latency_ms":     avg_lat,
                "current_latency_ms": round(current_lat, 2),
            })

        return result

    def get_timeline_at_step(self, step: int, max_steps: int) -> List[Dict[str, Any]]:
        """
        Progressive timeline: events accumulate as the cascade deepens.

        Static events (from scenario.json) are shown when step >= event.step.
        Dynamic events are generated as the cascade reaches new services:
          - "cascade_detected" when a new service starts degrading
          - "escalation_warning" at the episode midpoint
        """
        cascade_order = self._compute_cascade_order()

        # 1. Static events that have fired by this step
        events = [
            e for e in self.timeline
            if e.get("step", 0) <= step
        ]

        # 2. Dynamic cascade-reach events
        # Track which services already have cascade events in static timeline
        existing_cascade_services = {
            e["service"] for e in self.timeline
            if e.get("event_type") in ("cascade", "cascade_detected")
        }

        for svc_name, onset in cascade_order.items():
            if (onset > 0
                    and onset <= step
                    and svc_name not in existing_cascade_services):
                events.append({
                    "step":        onset,
                    "event_type":  "cascade_detected",
                    "service":     svc_name,
                    "description": (
                        f"Upstream failure cascading to {svc_name}. "
                        f"Performance degradation detected."
                    ),
                })

        # 3. Midpoint escalation warning (if not already in static timeline)
        midpoint = max_steps // 2
        has_mid_event = any(
            e.get("event_type") == "escalation_warning" for e in self.timeline
        )
        if step >= midpoint and not has_mid_event:
            events.append({
                "step":        midpoint,
                "event_type":  "escalation_warning",
                "service":     "incident-commander",
                "description": (
                    f"Incident unresolved for {midpoint} steps. "
                    f"Consider escalating to senior SRE."
                ),
            })

        # Sort chronologically by step
        events.sort(key=lambda e: e.get("step", 0))
        return events

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


class ExpertScenario(BaseScenario):
    def __init__(self):
        super().__init__("expert")

    def validate(self) -> bool:
        assert len(self.services) >= 7
        assert self.ground_truth["root_cause_service"]
        assert self.sla_breach_step is not None
        assert len(self.get_red_herring_services()) >= 3
        return True


SCENARIO_MAP = {
    "easy":   EasyScenario,
    "medium": MediumScenario,
    "hard":   HardScenario,
    "expert": ExpertScenario,
}


def load_scenario(task_id: str) -> BaseScenario:
    """Factory: load scenario by task_id string."""
    if task_id not in SCENARIO_MAP:
        raise ValueError(f"Unknown task_id '{task_id}'. Must be one of {list(SCENARIO_MAP)}")
    return SCENARIO_MAP[task_id]()