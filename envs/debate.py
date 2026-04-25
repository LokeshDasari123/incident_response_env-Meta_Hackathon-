"""
envs/debate.py
==============
Multi-Agent Debate Engine for the Incident Response Environment.

Implements the Responder ↔ Challenger ↔ Commander debate pattern:
  1. Responder (external agent) submits initial diagnosis
  2. Challenger (this engine) generates an adversarial challenge
  3. Responder revises (next step's action)
  4. Commander (grader) evaluates improvement

The debate is embedded in the observation: after every step, the
observation includes a `debate_challenge` field. The agent's next
action is implicitly a response to the challenge. This is fully
backward compatible — agents that ignore the challenge still work.
"""

import random
from typing import Any, Dict, List, Optional, Tuple


# ── Challenge strategy pool ──────────────────────────────────────────────────
CHALLENGE_STRATEGIES = [
    "topology_challenge",
    "fault_type_challenge",
    "severity_challenge",
    "red_herring_bait",
    "cascade_completeness",
    "remediation_challenge",
    "evidence_demand",
]


class DebateEngine:
    """
    Generates adversarial challenges for the multi-agent debate loop.

    The engine uses scenario data (ground truth, topology, metrics) to
    create grounded challenges — not random noise. This forces the agent
    to reason about its diagnosis rather than pattern-match.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self._history: List[Dict[str, Any]] = []
        self._strategies_used: List[str] = []

    def generate_challenge(
        self,
        action: Dict[str, Any],
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        topology: List[Dict[str, Any]],
        ground_truth: Dict[str, Any],
        step: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Generate an adversarial challenge to the agent's last action.

        Returns a dict with:
          - challenge_text: The adversarial question/statement
          - strategy: Which strategy was used
          - hint_quality: How helpful the challenge is (0=misleading, 1=helpful)
        """
        # Pick strategy — avoid repeating the same one consecutively
        available = [s for s in CHALLENGE_STRATEGIES if s not in self._strategies_used[-2:]]
        if not available:
            available = CHALLENGE_STRATEGIES
        strategy = self.rng.choice(available)
        self._strategies_used.append(strategy)

        rc_predicted = action.get("root_cause_service", "unknown")
        rc_actual = ground_truth.get("root_cause_service", "")
        severity = action.get("severity", "P2")
        fault_type = action.get("root_cause_type", "unknown")
        affected = action.get("affected_services", [])
        remediation = action.get("remediation_action", "investigate_further")

        # ── Generate challenge based on strategy ─────────────────────────
        if strategy == "topology_challenge":
            challenge_text, hint = self._topology_challenge(
                rc_predicted, metrics, topology, rc_actual
            )
        elif strategy == "fault_type_challenge":
            challenge_text, hint = self._fault_type_challenge(
                fault_type, metrics, rc_predicted, ground_truth
            )
        elif strategy == "severity_challenge":
            challenge_text, hint = self._severity_challenge(
                severity, affected, alerts, ground_truth
            )
        elif strategy == "red_herring_bait":
            challenge_text, hint = self._red_herring_bait(
                rc_predicted, metrics, ground_truth
            )
        elif strategy == "cascade_completeness":
            challenge_text, hint = self._cascade_completeness(
                affected, topology, rc_predicted, ground_truth
            )
        elif strategy == "remediation_challenge":
            challenge_text, hint = self._remediation_challenge(
                remediation, fault_type, rc_predicted, ground_truth
            )
        else:  # evidence_demand
            challenge_text, hint = self._evidence_demand(
                rc_predicted, metrics, alerts, ground_truth
            )

        # Track challenge in history
        entry = {
            "step": step,
            "strategy": strategy,
            "challenge": challenge_text,
            "hint_quality": hint,
            "agent_rc": rc_predicted,
            "correct_rc": rc_actual,
            "was_correct": rc_predicted == rc_actual,
        }
        self._history.append(entry)

        return {
            "challenge_text": challenge_text,
            "strategy": strategy,
            "hint_quality": round(hint, 2),
            "challenger_role": "Adversarial SRE Reviewer",
        }

    # ── Strategy implementations ─────────────────────────────────────────────

    def _topology_challenge(
        self, rc: str, metrics: Dict, topology: List, actual_rc: str
    ) -> Tuple[str, float]:
        """Challenge root cause by pointing to an alternative service."""
        unhealthy = [
            s for s, m in metrics.items()
            if s != rc and not m.get("is_healthy", True)
        ]
        if unhealthy:
            alt = self.rng.choice(unhealthy)
            alt_m = metrics.get(alt, {})
            text = (
                f"CHALLENGER: You identified '{rc}' as root cause, but '{alt}' "
                f"shows status='{alt_m.get('status', 'degraded')}' with "
                f"CPU={alt_m.get('cpu_utilization', 0):.0%}, "
                f"Memory={alt_m.get('memory_utilization', 0):.0%}. "
                f"The topology shows traffic flows through '{alt}'. "
                f"Could '{alt}' be the actual origin? Trace the call graph "
                f"edges and justify your root cause selection."
            )
            # Hint quality: high if pointing at actual RC, low otherwise
            hint = 0.9 if alt == actual_rc else 0.2
        else:
            text = (
                f"CHALLENGER: You say '{rc}' is root cause, but all other "
                f"services appear healthy. If '{rc}' truly failed, why aren't "
                f"its upstream consumers showing degradation? This is "
                f"inconsistent. Re-examine the cascade chain."
            )
            hint = 0.5
        return text, hint

    def _fault_type_challenge(
        self, fault_type: str, metrics: Dict, rc: str, gt: Dict
    ) -> Tuple[str, float]:
        """Challenge the fault type classification."""
        actual_ft = gt.get("root_cause_type", "")
        all_types = [
            "misconfiguration", "memory_leak", "network_partition",
            "crash_loop", "resource_exhaustion", "dependency_failure",
        ]
        alternatives = [t for t in all_types if t != fault_type]
        alt = self.rng.choice(alternatives)

        rc_m = metrics.get(rc, {})
        text = (
            f"CHALLENGER: You classified this as '{fault_type}', but the "
            f"metric pattern shows CPU={rc_m.get('cpu_utilization', 0):.0%}, "
            f"Memory={rc_m.get('memory_utilization', 0):.0%}, "
            f"Error Rate={rc_m.get('error_rate', 0):.0%}. "
            f"This pattern is more consistent with '{alt}'. "
            f"A wrong fault type leads to wrong remediation. Reconsider."
        )
        hint = 0.8 if alt == actual_ft else 0.15
        return text, hint

    def _severity_challenge(
        self, severity: str, affected: List, alerts: List, gt: Dict
    ) -> Tuple[str, float]:
        """Challenge severity assessment."""
        actual_sev = gt.get("severity", "P1")
        n_affected = len(affected)
        critical_alerts = sum(1 for a in alerts if a.get("severity") == "critical")

        if severity == actual_sev:
            # Agent is correct — try to mislead
            wrong_sev = "P2" if severity == "P0" else "P0"
            text = (
                f"CHALLENGER: You set severity to {severity}, but with "
                f"only {n_affected} services affected and {critical_alerts} "
                f"critical alerts, this looks more like {wrong_sev}. "
                f"Over-escalation wastes resources. Justify your severity."
            )
            hint = 0.1  # Misleading — agent should stick with its answer
        else:
            text = (
                f"CHALLENGER: Severity {severity} seems wrong. "
                f"{n_affected} services are affected with {critical_alerts} "
                f"critical alerts firing. This should be {actual_sev}. "
                f"Wrong severity = wrong escalation path = longer MTTR."
            )
            hint = 0.9  # Helpful — agent should fix its severity
        return text, hint

    def _red_herring_bait(
        self, rc: str, metrics: Dict, gt: Dict
    ) -> Tuple[str, float]:
        """Try to lure agent toward a red herring service."""
        rh_services = gt.get("red_herring_services", [])
        actual_rc = gt.get("root_cause_service", "")

        if rh_services:
            rh = self.rng.choice(rh_services)
            rh_m = metrics.get(rh, {})
            text = (
                f"CHALLENGER: You overlooked '{rh}' — it shows "
                f"CPU={rh_m.get('cpu_utilization', 0):.0%}, "
                f"Memory={rh_m.get('memory_utilization', 0):.0%}. "
                f"These metrics look alarming. Could this be the real "
                f"root cause and '{rc}' is just a downstream victim? "
                f"Investigate '{rh}' before finalizing."
            )
            # This is a TRAP — low hint quality
            hint = 0.05 if rc == actual_rc else 0.3
        else:
            text = (
                f"CHALLENGER: Are you confident in '{rc}'? "
                f"The alert storm has multiple services in distress. "
                f"Consider whether you're seeing a symptom rather than "
                f"the cause. What specific evidence makes '{rc}' the origin?"
            )
            hint = 0.4
        return text, hint

    def _cascade_completeness(
        self, affected: List, topology: List, rc: str, gt: Dict
    ) -> Tuple[str, float]:
        """Challenge the completeness of affected_services list."""
        actual_affected = gt.get("affected_services", [])
        missing = [s for s in actual_affected if s not in affected]
        extra = [s for s in affected if s not in actual_affected and s != rc]

        if missing:
            text = (
                f"CHALLENGER: Your blast radius is incomplete. "
                f"You listed {len(affected)} affected services, but the "
                f"topology shows '{missing[0]}' depends on '{rc}' and "
                f"should be degraded. Missing services = missed escalations = "
                f"SLA breach. Review the call graph."
            )
            hint = 0.85
        elif extra:
            text = (
                f"CHALLENGER: You included '{extra[0]}' in affected services, "
                f"but it's not in the cascade chain from '{rc}'. Including "
                f"unrelated services creates noise in the incident response. "
                f"Clean up your blast radius."
            )
            hint = 0.7
        else:
            text = (
                f"CHALLENGER: Your affected_services list covers the cascade, "
                f"but are you sure about the ordering? The service closest to "
                f"the root cause should show the highest degradation. Does "
                f"your list reflect the correct propagation direction?"
            )
            hint = 0.3
        return text, hint

    def _remediation_challenge(
        self, action: str, fault_type: str, rc: str, gt: Dict
    ) -> Tuple[str, float]:
        """Challenge the remediation action choice."""
        actual_action = gt.get("remediation_action", "investigate_further")
        correct_mappings = {
            "misconfiguration": "fix_config",
            "memory_leak": "restart_service",
            "network_partition": "fix_config",
            "crash_loop": "restart_service",
            "resource_exhaustion": "scale_up",
        }
        expected = correct_mappings.get(fault_type, "investigate_further")

        if action == actual_action:
            text = (
                f"CHALLENGER: You chose '{action}' for a '{fault_type}' fault, "
                f"but won't that cause downtime? Consider if a less disruptive "
                f"action could resolve this. Justify your choice."
            )
            hint = 0.1  # Agent should stick with correct answer
        else:
            text = (
                f"CHALLENGER: '{action}' for a '{fault_type}' fault? "
                f"The SRE runbook for '{fault_type}' recommends '{expected}'. "
                f"Wrong remediation can worsen the incident. "
                f"Reconsider your action based on the fault type."
            )
            hint = 0.85
        return text, hint

    def _evidence_demand(
        self, rc: str, metrics: Dict, alerts: List, gt: Dict
    ) -> Tuple[str, float]:
        """Demand specific evidence for the diagnosis."""
        rc_m = metrics.get(rc, {})
        rc_alerts = [a for a in alerts if a.get("service") == rc]

        text = (
            f"CHALLENGER: Before we act on '{rc}' as root cause, provide "
            f"specific evidence:\n"
            f"  1. What metric anomaly do you see? (CPU={rc_m.get('cpu_utilization', 0):.0%}, "
            f"Mem={rc_m.get('memory_utilization', 0):.0%})\n"
            f"  2. How many alerts fired for '{rc}'? ({len(rc_alerts)} alerts)\n"
            f"  3. Where does '{rc}' sit in the topology? (upstream or downstream?)\n"
            f"  4. Could this be a cascading SYMPTOM rather than the ROOT CAUSE?\n"
            f"Answer these before finalizing your diagnosis."
        )
        hint = 0.5
        return text, hint

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score_debate_improvement(
        self,
        prev_action: Dict[str, Any],
        curr_action: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Tuple[float, str]:
        """
        Score whether the agent improved after being challenged.

        Returns (bonus, feedback_text):
          - bonus > 0: agent improved (max +0.08)
          - bonus = 0: no change
          - bonus < 0: agent got worse (max -0.03)
        """
        gt_rc = ground_truth.get("root_cause_service", "")
        gt_sev = ground_truth.get("severity", "")
        gt_ft = ground_truth.get("root_cause_type", "")
        gt_act = ground_truth.get("remediation_action", "")

        prev_correct = sum([
            prev_action.get("root_cause_service") == gt_rc,
            prev_action.get("severity") == gt_sev,
            prev_action.get("root_cause_type") == gt_ft,
            prev_action.get("remediation_action") == gt_act,
        ])
        curr_correct = sum([
            curr_action.get("root_cause_service") == gt_rc,
            curr_action.get("severity") == gt_sev,
            curr_action.get("root_cause_type") == gt_ft,
            curr_action.get("remediation_action") == gt_act,
        ])

        delta = curr_correct - prev_correct

        if delta > 0:
            bonus = min(0.08, delta * 0.03)
            feedback = (
                f"Debate improvement: +{delta} components corrected after "
                f"challenge (bonus: +{bonus:.2f})"
            )
        elif delta < 0:
            bonus = max(-0.03, delta * 0.015)
            feedback = (
                f"Debate regression: {delta} components worsened after "
                f"challenge (penalty: {bonus:.2f})"
            )
        else:
            bonus = 0.0
            feedback = "No change after challenge — agent maintained position."

        return round(bonus, 4), feedback

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history

    def get_debate_summary(self) -> Dict[str, Any]:
        """Summary stats for the episode's debate."""
        if not self._history:
            return {"rounds": 0, "strategies": [], "avg_hint": 0}

        return {
            "rounds": len(self._history),
            "strategies": [h["strategy"] for h in self._history],
            "avg_hint": round(
                sum(h["hint_quality"] for h in self._history) / len(self._history), 2
            ),
            "agent_accuracy_progression": [h["was_correct"] for h in self._history],
            "corrections": sum(
                1 for i in range(1, len(self._history))
                if not self._history[i - 1]["was_correct"]
                and self._history[i]["was_correct"]
            ),
        }
