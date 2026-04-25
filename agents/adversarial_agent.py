"""
agents/adversarial_agent.py
---------------------------
Adversarial Agent — selectively hides, delays, or corrupts evidence
to make root cause identification harder.

Sees:  Ground truth (it knows the answer).
Acts:  Corrupts evidence, delays alerts, injects misleading signals.
Goal:  Maximize responder's time-to-resolution without making it unsolvable.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from agents.base_agent  import BaseAgent
from agents.message_bus import Message, MessageBus, MessageType


class AdversarialAgent(BaseAgent):
    """
    Evidence obfuscation agent that knows the ground truth and
    selectively corrupts the responder's information stream.

    Has a deception budget — limited corruptions per episode.
    The curriculum controller scales this budget.

    Deception tactics:
    - delay_alert:      Delay a key alert from being visible for N steps
    - swap_metric:      Swap metric labels between root cause and a healthy service
    - inject_red_herring: Add a plausible-but-wrong signal pointing at the wrong service
    - corrupt_message:  Intercept and modify a monitor agent message
    """

    ROLE = "adversary"

    TACTICS = [
        "delay_alert",
        "swap_metric",
        "inject_red_herring",
        "corrupt_message",
    ]

    def __init__(
        self,
        deception_budget: int   = 2,
        cunning:          float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__("adversary")
        self.deception_budget = deception_budget
        self.cunning          = cunning   # 0-1, probability of using deception
        self._rng             = random.Random(seed)
        self._used            = 0
        self._deceptions: List[Dict[str, Any]] = []
        self._ground_truth:   Optional[Dict[str, Any]] = None
        self._delayed_alerts: Dict[str, int]   = {}  # alert_id → visible_at_step

    def reset(self) -> None:
        super().reset()
        self._used            = 0
        self._deceptions      = []
        self._ground_truth    = None
        self._delayed_alerts  = {}

    def set_ground_truth(self, ground_truth: Dict[str, Any]) -> None:
        """Inject ground truth so the adversary knows the answer."""
        self._ground_truth = ground_truth

    def perceive(
        self,
        observation: Dict[str, Any],
        messages:    List[Message],
    ) -> Dict[str, Any]:
        """Adversary sees everything including ground truth."""
        return {
            "step":           observation.get("step", 0),
            "metrics":        observation.get("metrics", {}),
            "alerts":         observation.get("alerts", []),
            "topology":       observation.get("topology", []),
            "ground_truth":   self._ground_truth,
            "budget_remaining": self.deception_budget - self._used,
            "messages":       [m.to_dict() for m in messages],
        }

    def act(self, filtered_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Decide whether and how to deceive."""
        if self._used >= self.deception_budget:
            return {"deception": None, "reason": "budget_exhausted"}

        step = filtered_obs.get("step", 0)
        gt   = filtered_obs.get("ground_truth") or {}

        # Don't deceive on first step — let agent get initial data
        if step < 1 or self._rng.random() > self.cunning:
            return {"deception": None, "reason": "skipped"}

        tactic = self._rng.choice(self.TACTICS)
        deception = self._build_deception(tactic, filtered_obs, gt, step)

        if deception:
            self._used += 1
            self._deceptions.append(deception)

        return {"deception": deception, "reason": f"tactic={tactic}"}

    def communicate(
        self,
        action: Dict[str, Any],
        obs:    Dict[str, Any],
        bus:    MessageBus,
        step:   int,
    ) -> Optional[Message]:
        """
        The adversary may corrupt a pending monitor message instead
        of sending its own message.
        """
        deception = action.get("deception")
        if not deception:
            return None

        if deception["type"] == "corrupt_message":
            # Try to intercept a pending monitor message
            pending = bus._pending
            monitor_msgs = [m for m in pending if m.sender == "monitor"]
            if monitor_msgs:
                target_msg = self._rng.choice(monitor_msgs)
                corrupted_content = self._corrupt_content(
                    target_msg.content, deception
                )
                bus.intercept(target_msg, corrupted_content)

        # Send deception signal (visible in audit trail)
        return bus.broadcast(
            sender   = self.agent_id,
            msg_type = MessageType.DECEPTION,
            content  = {
                "tactic": deception["type"],
                "target": deception.get("target", "unknown"),
            },
            step     = step,
            priority = 1,
        )

    def apply_deception(
        self,
        deception: Optional[Dict[str, Any]],
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply deception to the observation before the responder sees it.
        Returns modified observation.
        """
        if not deception:
            return observation

        import copy
        obs = copy.deepcopy(observation)

        if deception["type"] == "delay_alert":
            # Remove a specific alert from visibility
            target_svc = deception.get("target", "")
            obs["alerts"] = [
                a for a in obs.get("alerts", [])
                if a.get("service") != target_svc
            ]

        elif deception["type"] == "swap_metric":
            # Swap metrics between root cause and a healthy service
            metrics = obs.get("metrics", {})
            src = deception.get("source", "")
            dst = deception.get("target", "")
            if src in metrics and dst in metrics:
                metrics[src], metrics[dst] = metrics[dst], metrics[src]

        elif deception["type"] == "inject_red_herring":
            # Add a misleading signal
            rh = deception.get("red_herring", {})
            target = rh.get("service", "unknown")
            if target in obs.get("metrics", {}):
                m = obs["metrics"][target]
                m["cpu_utilization"]    = max(m.get("cpu_utilization", 0.3), 0.92)
                m["memory_utilization"] = max(m.get("memory_utilization", 0.4), 0.88)
                m["status"]     = "critical"
                m["is_healthy"] = False
                # Add a fake alert
                obs.setdefault("alerts", []).append({
                    "alert_id":      f"ALT-ADV-{obs.get('step', 0):02d}",
                    "service":       target,
                    "metric":        "cpu_utilization",
                    "current_value": 0.95,
                    "threshold":     0.85,
                    "severity":      "critical",
                    "fired_at_step": obs.get("step", 0),
                })

        return obs

    def get_deceptions(self) -> List[Dict[str, Any]]:
        return self._deceptions

    # ── Private helpers ───────────────────────────────────────────────

    def _build_deception(
        self,
        tactic: str,
        obs: Dict[str, Any],
        gt: Dict[str, Any],
        step: int,
    ) -> Optional[Dict[str, Any]]:
        """Build a specific deception payload."""
        rc_service = gt.get("root_cause_service", "")
        metrics    = obs.get("metrics", {})
        healthy    = [
            s for s, m in metrics.items()
            if m.get("is_healthy", True) and s != rc_service
        ]

        if tactic == "delay_alert":
            return {
                "type":   "delay_alert",
                "target": rc_service,
                "step":   step,
                "description": f"Delaying alerts from {rc_service}",
            }

        elif tactic == "swap_metric":
            if healthy:
                swap_target = self._rng.choice(healthy)
                return {
                    "type":   "swap_metric",
                    "source": rc_service,
                    "target": swap_target,
                    "step":   step,
                    "description": f"Swapping metrics: {rc_service} ↔ {swap_target}",
                }

        elif tactic == "inject_red_herring":
            if healthy:
                rh_target = self._rng.choice(healthy)
                return {
                    "type": "inject_red_herring",
                    "target": rh_target,
                    "step":   step,
                    "red_herring": {
                        "service": rh_target,
                        "reason":  f"Adversary-injected false signal on {rh_target}",
                    },
                    "description": f"Injecting red herring at {rh_target}",
                }

        elif tactic == "corrupt_message":
            return {
                "type":   "corrupt_message",
                "target": "monitor",
                "step":   step,
                "corrupt_service": self._rng.choice(healthy) if healthy else rc_service,
                "description": "Corrupting monitor's anomaly report",
            }

        return None

    def _corrupt_content(
        self,
        original: Dict[str, Any],
        deception: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Modify a monitor message to point at the wrong service."""
        corrupted = dict(original)
        wrong_svc = deception.get("corrupt_service", "unknown")
        corrupted["top_service"]   = wrong_svc
        corrupted["anomaly_score"] = round(self._rng.uniform(0.7, 0.95), 3)
        corrupted["reason"]        = f"Critical anomaly detected on {wrong_svc}"
        return corrupted
