"""
agents/responder_agent.py
-------------------------
Primary Incident Responder Agent — the agent being trained / evaluated.

Sees:  Alerts, metrics, topology, timeline, messages from other agents, own memory.
Acts:  Produces IncidentAction (root cause, severity, remediation, stakeholder msg).
Comms: Shares diagnosis hypothesis for other agents to challenge.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional

from agents.base_agent  import BaseAgent
from agents.message_bus import Message, MessageBus, MessageType


class ResponderAgent(BaseAgent):
    """
    The primary agent being trained.
    Two modes:
    - rule_based: skill-based heuristic (for simulation training)
    - llm:        delegates to an LLM via prompt (for GRPO training)

    In both modes, the agent maintains persistent memory and
    integrates signals from the Monitor Agent.
    """

    ROLE = "responder"

    def __init__(
        self,
        mode:   str   = "rule_based",
        skill:  float = 0.5,
        seed: Optional[int] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        red_herrings: Optional[List[str]]      = None,
    ) -> None:
        super().__init__("responder")
        self.mode          = mode
        self.skill         = skill
        self._rng          = random.Random(seed)
        self._ground_truth = ground_truth or {}
        self._red_herrings = red_herrings or []
        self._monitor_signals: List[Dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self._monitor_signals = []

    def set_skill(self, skill: float) -> None:
        """Update skill level (used by curriculum / training progress)."""
        self.skill = max(0.0, min(1.0, skill))

    def perceive(
        self,
        observation: Dict[str, Any],
        messages:    List[Message],
    ) -> Dict[str, Any]:
        """
        Responder sees: alerts, metrics, topology, timeline,
        and messages from other agents. Does NOT see ground truth.
        """
        self._step = observation.get("step", 0)

        # Process monitor signals
        for msg in messages:
            if msg.msg_type == MessageType.ANOMALY_REPORT:
                self._monitor_signals.append(msg.content)
            # Store all messages as evidence
            self.receive_messages([msg])

        return {
            "step":             observation.get("step", 0),
            "max_steps":        observation.get("max_steps", 10),
            "task_id":          observation.get("task_id", "easy"),
            "alerts":           observation.get("alerts", []),
            "metrics":          observation.get("metrics", {}),
            "topology":         observation.get("topology", []),
            "timeline":         observation.get("timeline", []),
            "time_pressure":    observation.get("time_pressure", 0.0),
            "previous_actions": observation.get("previous_actions", []),
            "current_score":    observation.get("current_score", 0.0),
            "monitor_signals":  self._monitor_signals[-3:],
            "memory_summary":   self.memory.summarize(),
            "agent_messages":   [m.to_dict() for m in messages],
        }

    def act(self, filtered_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a diagnosis action. In rule_based mode, uses
        topology traversal with skill-based accuracy.
        """
        if self.mode == "llm":
            return self._act_llm(filtered_obs)
        return self._act_rule_based(filtered_obs)

    def communicate(
        self,
        action: Dict[str, Any],
        obs:    Dict[str, Any],
        bus:    MessageBus,
        step:   int,
    ) -> Optional[Message]:
        """Share diagnosis hypothesis with other agents."""
        return bus.broadcast(
            sender   = self.agent_id,
            msg_type = MessageType.DIAGNOSIS_HYPOTHESIS,
            content  = {
                "root_cause_service":  action.get("root_cause_service", "unknown"),
                "root_cause_type":     action.get("root_cause_type", "unknown"),
                "severity":            action.get("severity", "P2"),
                "confidence":          action.get("confidence", 0.5),
                "affected_services":   action.get("affected_services", []),
            },
            step     = step,
            priority = 2,
        )

    # ── Rule-based acting ─────────────────────────────────────────────

    def _act_rule_based(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Topology-inward traversal with skill-based noise.
        Integrates monitor signals for better accuracy.
        """
        gt       = self._ground_truth
        metrics  = obs.get("metrics", {})
        topology = obs.get("topology", [])
        signals  = obs.get("monitor_signals", [])

        # Effective skill: boosted by monitor signals
        signal_bonus = 0.05 * len(signals) if signals else 0.0
        eff_skill = min(0.97, self.skill + signal_bonus)

        # Pick root cause service
        all_svcs = list(metrics.keys()) or ["unknown"]
        wrong_rc = [s for s in all_svcs if s != gt.get("root_cause_service")]

        # Use monitor's top anomaly as a hint
        monitor_hint = None
        if signals:
            monitor_hint = signals[-1].get("top_service")
            # If monitor is right, boost the agent's chance
            if monitor_hint == gt.get("root_cause_service"):
                eff_skill = min(0.97, eff_skill + 0.1)

        rc = self._pick(
            gt.get("root_cause_service", "unknown"),
            wrong_rc or ["unknown"],
            eff_skill,
        )

        # Root cause type
        all_types = [
            "misconfiguration", "memory_leak", "network_partition",
            "crash_loop", "resource_exhaustion", "dependency_failure", "unknown",
        ]
        wrong_ft = [t for t in all_types if t != gt.get("root_cause_type")]
        ft = self._pick(
            gt.get("root_cause_type", "unknown"),
            wrong_ft,
            eff_skill * 0.95,
        )

        # Severity
        sev_map = {"P0": ["P1", "P2"], "P1": ["P0", "P2"], "P2": ["P1", "P3"]}
        wrong_sev = sev_map.get(gt.get("severity", "P2"), ["P1", "P2"])
        sev = self._pick(gt.get("severity", "P2"), wrong_sev, eff_skill * 0.90)

        # Affected services
        correct_affected = gt.get("affected_services", [rc])
        rh = self._red_herrings
        if eff_skill > 0.70:
            affected = correct_affected[:]
            if rh and self._rng.random() > eff_skill:
                affected.append(self._rng.choice(rh))
        elif eff_skill > 0.40:
            n = max(1, int(len(correct_affected) * eff_skill))
            affected = correct_affected[:n]
        else:
            affected = self._rng.sample(all_svcs, k=min(3, len(all_svcs)))

        # Remediation
        action_map = {
            "misconfiguration":    ["fix_config", "rollback", "restart_service"],
            "memory_leak":         ["restart_service", "scale_up", "investigate_further"],
            "network_partition":   ["fix_config", "reroute_traffic", "investigate_further"],
            "crash_loop":          ["restart_service", "rollback", "investigate_further"],
            "resource_exhaustion": ["scale_up", "fix_config", "restart_service"],
        }
        correct_act = gt.get("remediation_action", "investigate_further")
        alt_acts = [a for a in action_map.get(ft, ["investigate_further"]) if a != correct_act]
        act = self._pick(correct_act, alt_acts or ["investigate_further"], eff_skill * 0.88)

        # Stakeholder message
        needs_msg = sev in ("P0", "P1")
        msg = None
        if needs_msg and self.skill > 0.5:
            msg = (
                f"{rc} is experiencing {ft.replace('_', ' ')} causing cascade "
                f"to {len(affected)} services. Severity: {sev}. "
                f"Action: {act.replace('_', ' ')}. ETA: ~10 minutes."
            )
        elif needs_msg and self._rng.random() > 0.5:
            msg = f"Investigating {rc} issue."

        confidence = round(max(0.05, min(0.99,
            eff_skill + self._rng.gauss(0, 0.05)
        )), 2)

        # Update memory
        self.memory.add_hypothesis(rc, ft, confidence, f"Step {self._step}", self._step)

        action_dict = {
            "root_cause_service":  rc,
            "root_cause_type":     ft,
            "severity":            sev,
            "affected_services":   list(dict.fromkeys(affected)),
            "remediation_action":  act,
            "stakeholder_message": msg,
            "confidence":          confidence,
            "reasoning": (
                f"Traversed topology inward. {rc} shows highest degradation. "
                f"Pattern matches {ft}. Monitor signal: "
                f"{monitor_hint or 'none'}. "
                f"Cascade: {' → '.join(affected[:3])}."
            ),
        }
        return action_dict

    def _act_llm(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM mode: builds a prompt and returns it for external LLM invocation.
        The actual LLM call is handled by the training wrapper.
        """
        # Return the observation in a format suitable for prompt building
        return {
            "mode":       "llm",
            "prompt_data": obs,
            "memory":     self.memory.summarize(),
        }

    def _pick(self, correct: Any, alternatives: List[Any], p_correct: float) -> Any:
        """Pick correct answer with probability p_correct, else random alternative."""
        if self._rng.random() < p_correct:
            return correct
        return self._rng.choice(alternatives) if alternatives else correct
