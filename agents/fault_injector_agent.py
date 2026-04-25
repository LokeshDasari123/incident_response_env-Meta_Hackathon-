"""
agents/fault_injector_agent.py
------------------------------
Competitive Fault Injection Agent — dynamically introduces secondary
failures, noise, and cascading events mid-episode.

Sees:  Current system state + responder's confidence level.
Acts:  Injects metric noise, secondary faults, delayed cascades.
Goal:  Make the incident harder when the responder is doing well.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from agents.base_agent  import BaseAgent
from agents.message_bus import Message, MessageBus, MessageType


class FaultInjectorAgent(BaseAgent):
    """
    Adversarial agent that dynamically injects secondary faults
    and noise into the environment. Has a limited action budget
    per episode that scales with curriculum difficulty.

    Injection types:
    - metric_noise:      Jitter healthy service metrics to look suspicious
    - secondary_fault:   Create a second failing service (unrelated to root cause)
    - cascade_extension: Make the cascade reach an additional service
    - alert_storm:       Inject multiple misleading alerts at once
    """

    ROLE = "fault_injector"

    INJECTION_TYPES = [
        "metric_noise",
        "secondary_fault",
        "cascade_extension",
        "alert_storm",
    ]

    def __init__(
        self,
        budget:     int   = 3,
        aggression: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__("fault_injector")
        self.budget     = budget       # max injections per episode
        self.aggression = aggression   # 0-1, how aggressively to inject
        self._rng       = random.Random(seed)
        self._used       = 0
        self._injections: List[Dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self._used       = 0
        self._injections = []

    def perceive(
        self,
        observation: Dict[str, Any],
        messages:    List[Message],
    ) -> Dict[str, Any]:
        """
        Fault injector sees system state and responder's confidence.
        It decides whether to inject based on how well the responder is doing.
        """
        # Extract responder confidence from messages
        responder_confidence = 0.5
        for msg in messages:
            if msg.sender == "responder" and msg.msg_type == MessageType.DIAGNOSIS_HYPOTHESIS:
                responder_confidence = msg.content.get("confidence", 0.5)

        return {
            "step":                 observation.get("step", 0),
            "metrics":              observation.get("metrics", {}),
            "current_score":        observation.get("current_score", 0.0),
            "responder_confidence": responder_confidence,
            "budget_remaining":     self.budget - self._used,
        }

    def act(self, filtered_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether and what to inject.
        Injects more aggressively when responder is doing well.
        """
        if self._used >= self.budget:
            return {"injection": None, "reason": "budget_exhausted"}

        step       = filtered_obs.get("step", 0)
        score      = filtered_obs.get("current_score", 0.0)
        confidence = filtered_obs.get("responder_confidence", 0.5)
        metrics    = filtered_obs.get("metrics", {})

        # Decision: inject if responder is doing well AND random check passes
        inject_probability = self.aggression * max(score, confidence)
        # Don't inject on step 0 or 1 — let the agent observe first
        if step < 2 or self._rng.random() > inject_probability:
            return {"injection": None, "reason": "skipped"}

        # Choose injection type
        injection_type = self._rng.choice(self.INJECTION_TYPES)
        injection = self._build_injection(injection_type, metrics, step)

        self._used += 1
        self._injections.append(injection)

        return {"injection": injection, "reason": f"responder score={score:.2f}"}

    def communicate(
        self,
        action: Dict[str, Any],
        obs:    Dict[str, Any],
        bus:    MessageBus,
        step:   int,
    ) -> Optional[Message]:
        """Announce fault injection (visible to all agents)."""
        injection = action.get("injection")
        if not injection:
            return None

        return bus.broadcast(
            sender   = self.agent_id,
            msg_type = MessageType.FAULT_ALERT,
            content  = {
                "injection_type": injection["type"],
                "target":         injection.get("target", "unknown"),
                "description":    injection.get("description", ""),
            },
            step     = step,
            priority = 2,
        )

    def get_injections(self) -> List[Dict[str, Any]]:
        """Return all injections made this episode."""
        return self._injections

    # ── Private helpers ───────────────────────────────────────────────

    def _build_injection(
        self,
        injection_type: str,
        metrics: Dict[str, Any],
        step: int,
    ) -> Dict[str, Any]:
        """Build a specific injection payload."""
        healthy_svcs = [
            svc for svc, m in metrics.items()
            if m.get("is_healthy", True) and m.get("status") == "healthy"
        ]
        target = self._rng.choice(healthy_svcs) if healthy_svcs else "unknown"

        if injection_type == "metric_noise":
            return {
                "type":        "metric_noise",
                "target":      target,
                "description": f"Injecting CPU/memory noise into {target}",
                "step":        step,
                "modifications": {
                    "cpu_delta":    round(self._rng.uniform(0.2, 0.4), 3),
                    "mem_delta":    round(self._rng.uniform(0.1, 0.3), 3),
                    "error_delta":  round(self._rng.uniform(0.1, 0.3), 3),
                },
            }

        elif injection_type == "secondary_fault":
            return {
                "type":        "secondary_fault",
                "target":      target,
                "description": f"Secondary fault injected in {target}",
                "step":        step,
                "modifications": {
                    "status":           "degraded",
                    "cpu_delta":        round(self._rng.uniform(0.3, 0.5), 3),
                    "mem_delta":        round(self._rng.uniform(0.2, 0.4), 3),
                    "http_rt_multiplier": round(self._rng.uniform(3, 8), 1),
                },
            }

        elif injection_type == "cascade_extension":
            return {
                "type":        "cascade_extension",
                "target":      target,
                "description": f"Cascade extended to {target}",
                "step":        step,
                "modifications": {
                    "status":      "degraded",
                    "error_delta": round(self._rng.uniform(0.2, 0.5), 3),
                },
            }

        else:  # alert_storm
            num_alerts = self._rng.randint(2, 4)
            fake_alerts = []
            for i in range(num_alerts):
                fake_svc = self._rng.choice(healthy_svcs) if healthy_svcs else target
                fake_alerts.append({
                    "alert_id":     f"ALT-FAKE-{step:02d}-{i}",
                    "service":      fake_svc,
                    "metric":       self._rng.choice(["cpu_utilization", "memory_utilization", "http_rt"]),
                    "current_value": round(self._rng.uniform(0.8, 0.99), 3),
                    "threshold":    0.85,
                    "severity":     self._rng.choice(["warning", "critical"]),
                    "fired_at_step": step,
                    "is_injected":  True,
                })
            return {
                "type":        "alert_storm",
                "target":      "multiple",
                "description": f"Alert storm: {num_alerts} misleading alerts injected",
                "step":        step,
                "fake_alerts": fake_alerts,
            }

    def apply_injection(
        self,
        injection: Dict[str, Any],
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]],
    ) -> tuple:
        """
        Apply the injection to the observation data.
        Returns (modified_metrics, modified_alerts).
        """
        if not injection:
            return metrics, alerts

        target = injection.get("target", "")
        mods   = injection.get("modifications", {})

        # Deep copy to avoid mutating originals
        import copy
        metrics = copy.deepcopy(metrics)
        alerts  = list(alerts)

        if injection["type"] in ("metric_noise", "secondary_fault", "cascade_extension"):
            if target in metrics:
                m = metrics[target]
                m["cpu_utilization"] = min(0.99, m.get("cpu_utilization", 0.3) + mods.get("cpu_delta", 0))
                m["memory_utilization"] = min(0.99, m.get("memory_utilization", 0.4) + mods.get("mem_delta", 0))
                m["error_rate"] = min(1.0, m.get("error_rate", 0) + mods.get("error_delta", 0))
                if "status" in mods:
                    m["status"]     = mods["status"]
                    m["is_healthy"] = mods["status"] == "healthy"
                if "http_rt_multiplier" in mods and m.get("http_rt"):
                    m["http_rt"] = round(m["http_rt"] * mods["http_rt_multiplier"], 1)

        elif injection["type"] == "alert_storm":
            alerts.extend(injection.get("fake_alerts", []))

        return metrics, alerts
