"""
envs/multi_agent_env.py
-----------------------
Multi-Agent Incident Response Environment.

Wraps IncidentResponseEnv with 4-agent orchestration:
- Monitor Agent:        Cooperative — provides anomaly signals
- Fault Injector Agent: Competitive — injects secondary failures
- Adversarial Agent:    Competitive — corrupts evidence
- Responder Agent:      Primary     — produces the graded action

Episode flow per step:
    1. Environment produces raw observation
    2. Monitor perceives metrics → sends anomaly signals
    3. Adversarial agent may corrupt evidence
    4. Fault injector may inject secondary failures
    5. Responder perceives filtered observation + messages → produces action
    6. Environment grades responder action → reward
    7. All agents communicate results
"""

from __future__ import annotations

import math
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from envs.incident_env import IncidentResponseEnv
from models.action     import IncidentAction
from models.memory     import EpisodeMemory, InvestigationResult

from agents.message_bus         import MessageBus, MessageType
from agents.monitor_agent       import MonitorAgent
from agents.fault_injector_agent import FaultInjectorAgent
from agents.adversarial_agent   import AdversarialAgent
from agents.responder_agent     import ResponderAgent


# ── Investigation findings pool ───────────────────────────────────────────────
# When the responder uses "investigate_further", they get one of these findings
# tailored to the scenario's ground truth.
INVESTIGATION_FINDINGS = {
    "log_entry": [
        "ERROR: {service} — connection pool exhausted: 0/{max} available",
        "WARN: {service} — GC pause 12400ms, heap at 97%",
        "ERROR: {service} — NXDOMAIN lookup for {downstream}.prod.svc.cluster.local",
        "FATAL: {service} — OOMKilled by kernel (exit code 137)",
        "ERROR: {service} — ConfigMap reload failed: invalid YAML at line 42",
    ],
    "config_diff": [
        "ConfigMap {service}-config: max_connections changed 100 → 5 at {timestamp}",
        "Deployment {service}: memory_limit reduced 2Gi → 512Mi at {timestamp}",
        "NetworkPolicy {service}: egress rule DROP ALL added at {timestamp}",
    ],
    "trace_sample": [
        "Trace: {upstream} → {service} — timeout after 30000ms (p99 normally 120ms)",
        "Trace: {service} → {downstream} — connection refused (port 5432 unreachable)",
    ],
    "metric_deep_dive": [
        "{service} CPU: 12% → 94% in 3 minutes. Correlates with deploy at {timestamp}",
        "{service} memory: linear growth 400Mi → 1.8Gi over 30 minutes (leak pattern)",
        "{service} error_rate: 0.1% → 78% step function at {timestamp}",
    ],
}


class MultiAgentIncidentEnv:
    """
    Multi-agent wrapper around IncidentResponseEnv.

    Orchestrates 4 agents per step and provides:
    - Inter-agent communication via MessageBus
    - Persistent memory across steps
    - Dynamic fault injection / evidence corruption
    - Rich logging for evaluation
    """

    def __init__(
        self,
        monitor_reliability: float = 0.85,
        monitor_noise:       float = 0.1,
        fault_budget:        int   = 3,
        fault_aggression:    float = 0.5,
        adversary_budget:    int   = 2,
        adversary_cunning:   float = 0.5,
        seed: Optional[int]        = None,
    ) -> None:
        self._env  = IncidentResponseEnv()
        self._seed = seed
        self._rng  = random.Random(seed)

        # Agents
        self.monitor   = MonitorAgent(
            reliability=monitor_reliability,
            noise_level=monitor_noise,
            seed=seed,
        )
        self.injector  = FaultInjectorAgent(
            budget=fault_budget,
            aggression=fault_aggression,
            seed=seed,
        )
        self.adversary = AdversarialAgent(
            deception_budget=adversary_budget,
            cunning=adversary_cunning,
            seed=seed,
        )
        self.responder = ResponderAgent(
            mode="rule_based",
            seed=seed,
        )

        # Communication
        self.bus = MessageBus()

        # State
        self._step         = 0
        self._done         = False
        self._task_id      = "easy"
        self._episode_memory = EpisodeMemory()
        self._step_logs:     List[Dict[str, Any]] = []
        self._ground_truth:  Dict[str, Any] = {}

    def reset(
        self,
        task_id: str  = "easy",
        *,
        dynamic: bool = True,
        seed: Optional[int] = None,
        responder_skill: float = 0.5,
        ground_truth: Optional[Dict[str, Any]] = None,
        red_herrings: Optional[List[str]]      = None,
    ) -> Dict[str, Any]:
        """
        Reset environment and all agents.
        Returns the initial observation for the responder.
        """
        self._task_id = task_id
        self._step    = 0
        self._done    = False
        self._step_logs = []
        self._episode_memory = EpisodeMemory()

        # Reset core env
        obs = self._env.reset(task_id=task_id, dynamic=dynamic, seed=seed)
        obs_dict = obs.model_dump()

        # Ground truth from scenario
        if ground_truth:
            self._ground_truth = ground_truth
        elif self._env._scenario:
            self._ground_truth = self._env._scenario.ground_truth
        else:
            self._ground_truth = {}

        rh = red_herrings or self._ground_truth.get("red_herring_services", [])

        # Reset agents
        self.monitor.reset()
        self.injector.reset()
        self.adversary.reset()
        self.adversary.set_ground_truth(self._ground_truth)
        self.responder.reset()
        self.responder.set_skill(responder_skill)
        self.responder._ground_truth = self._ground_truth
        self.responder._red_herrings = rh
        self.bus.reset(intercept_budget=self.adversary.deception_budget)

        return obs_dict

    def step(
        self,
        external_action: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one multi-agent step.

        If external_action is provided (e.g., from LLM), use it instead
        of the responder agent's rule-based action.

        Returns: (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode done. Call reset().")

        self._step += 1

        # ── 1. Get raw observation from core env ─────────────────────────────
        raw_obs = self._env._build_observation(reward=0.0, done=False)
        obs_dict = raw_obs.model_dump()

        # ── 2. Monitor Agent perceives + acts ────────────────────────────────
        monitor_messages = self.bus.deliver("monitor")
        monitor_filtered = self.monitor.perceive(obs_dict, monitor_messages)
        monitor_action   = self.monitor.act(monitor_filtered)
        self.monitor.communicate(monitor_action, obs_dict, self.bus, self._step)

        # ── 3. Adversarial Agent perceives + acts ────────────────────────────
        adv_messages = self.bus.deliver("adversary")
        adv_filtered = self.adversary.perceive(obs_dict, adv_messages)
        adv_action   = self.adversary.act(adv_filtered)
        deception    = adv_action.get("deception")

        # Apply deception to observation before responder sees it
        resp_obs = self.adversary.apply_deception(deception, obs_dict)

        # Adversary communicates (may corrupt monitor messages)
        self.adversary.communicate(adv_action, obs_dict, self.bus, self._step)

        # ── 4. Fault Injector perceives + acts ───────────────────────────────
        inj_messages = self.bus.deliver("fault_injector")
        inj_filtered = self.injector.perceive(resp_obs, inj_messages)
        inj_action   = self.injector.act(inj_filtered)
        injection    = inj_action.get("injection")

        # Apply injection to observation
        if injection:
            resp_metrics, resp_alerts = self.injector.apply_injection(
                injection,
                resp_obs.get("metrics", {}),
                resp_obs.get("alerts", []),
            )
            resp_obs["metrics"] = resp_metrics
            resp_obs["alerts"]  = resp_alerts

        self.injector.communicate(inj_action, resp_obs, self.bus, self._step)

        # ── 5. Handle investigation results ──────────────────────────────────
        # If previous action was "investigate_further", generate a finding
        last_action = self._episode_memory.actions_taken[-1] if self._episode_memory.actions_taken else None
        if last_action and last_action["action"].get("remediation_action") == "investigate_further":
            finding = self._generate_investigation_finding(self._step)
            if finding:
                self._episode_memory.add_investigation_result(finding)
                resp_obs["investigation_results"] = [finding.model_dump()]

        # ── 6. Responder perceives + acts ────────────────────────────────────
        resp_messages = self.bus.deliver("responder")
        resp_filtered = self.responder.perceive(resp_obs, resp_messages)

        # Add memory context
        resp_filtered["memory_summary"] = self._episode_memory.summarize()

        if external_action:
            action_dict = external_action
        else:
            action_dict = self.responder.act(resp_filtered)

        # Responder communicates hypothesis
        self.responder.communicate(action_dict, resp_obs, self.bus, self._step)

        # ── 7. Grade responder action ────────────────────────────────────────
        try:
            action_obj = IncidentAction(
                root_cause_service  = action_dict.get("root_cause_service", "unknown"),
                root_cause_type     = action_dict.get("root_cause_type", "unknown"),
                severity            = action_dict.get("severity", "P2"),
                affected_services   = action_dict.get("affected_services", []),
                remediation_action  = action_dict.get("remediation_action", "investigate_further"),
                stakeholder_message = action_dict.get("stakeholder_message"),
                confidence          = float(action_dict.get("confidence", 0.5)),
                reasoning           = action_dict.get("reasoning"),
            )
            obs_result, reward, done, info = self._env.step(action_obj)
            obs_out = obs_result.model_dump()
        except Exception as e:
            reward = 0.0
            done   = True
            info   = {"error": str(e)}
            obs_out = resp_obs

        self._done = done

        # ── 8. Update memory ─────────────────────────────────────────────────
        self._episode_memory.add_action(action_dict, reward, self._step)
        self.responder.memory.add_action(action_dict, reward, self._step)

        # ── 9. Flush message bus ─────────────────────────────────────────────
        self.bus.flush()

        # ── 10. Build step log ───────────────────────────────────────────────
        step_log = {
            "step":           self._step,
            "task_id":        self._task_id,
            "reward":         reward,
            "done":           done,
            "action":         action_dict,
            "monitor_action": monitor_action,
            "injection":      injection,
            "deception":      deception,
            "messages":       self.bus.get_stats(),
            "memory_state":   {
                "hypotheses":   len(self._episode_memory.hypotheses),
                "evidence":     len(self._episode_memory.evidence_log),
                "investigations": len(self._episode_memory.investigation_results),
                "reward_trend": self._episode_memory.get_reward_trend(),
            },
            "agent_states":   {
                "responder": self.responder.get_state(),
                "monitor":   self.monitor.get_state(),
            },
        }
        self._step_logs.append(step_log)

        # Enrich info with multi-agent data
        info["multi_agent"] = {
            "monitor_anomalies":    len(monitor_action.get("anomalies", [])),
            "fault_injected":       injection is not None,
            "deception_applied":    deception is not None,
            "messages_this_step":   self.bus.get_stats()["total_messages"],
            "memory_hypotheses":    len(self._episode_memory.hypotheses),
            "investigation_results": len(self._episode_memory.investigation_results),
        }

        return obs_out, reward, done, info

    def get_step_logs(self) -> List[Dict[str, Any]]:
        """Return all step logs for the current episode."""
        return self._step_logs

    def get_episode_summary(self) -> Dict[str, Any]:
        """Comprehensive episode summary for evaluation."""
        rewards = [sl["reward"] for sl in self._step_logs]
        return {
            "task_id":         self._task_id,
            "total_steps":     len(self._step_logs),
            "best_reward":     max(rewards) if rewards else 0.0,
            "avg_reward":      sum(rewards) / len(rewards) if rewards else 0.0,
            "final_reward":    rewards[-1] if rewards else 0.0,
            "injections":      len(self.injector.get_injections()),
            "deceptions":      len(self.adversary.get_deceptions()),
            "message_stats":   self.bus.get_stats(),
            "memory_state": {
                "total_hypotheses":      len(self._episode_memory.hypotheses),
                "total_evidence":        len(self._episode_memory.evidence_log),
                "investigation_results": len(self._episode_memory.investigation_results),
            },
            "reward_history":  rewards,
            "reward_trend":    self._episode_memory.get_reward_trend(),
        }

    def close(self) -> None:
        self._env.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate_investigation_finding(self, step: int) -> Optional[InvestigationResult]:
        """Generate an investigation finding when agent uses investigate_further."""
        rc = self._ground_truth.get("root_cause_service", "unknown")
        ft = self._ground_truth.get("root_cause_type", "unknown")

        # Choose finding type based on fault type
        type_map = {
            "misconfiguration":    "config_diff",
            "memory_leak":         "metric_deep_dive",
            "network_partition":   "log_entry",
            "crash_loop":          "log_entry",
            "resource_exhaustion": "metric_deep_dive",
        }
        finding_type = type_map.get(ft, "log_entry")
        templates = INVESTIGATION_FINDINGS.get(finding_type, INVESTIGATION_FINDINGS["log_entry"])
        template = self._rng.choice(templates)

        # Fill template
        finding_text = template.format(
            service=rc,
            downstream=rc,
            upstream=rc,
            max=100,
            timestamp="2025-01-01T02:14:00Z",
        )

        # Investigation findings point at root cause with high probability
        reveals_root = self._rng.random() < 0.7

        return InvestigationResult(
            step=step,
            service=rc if reveals_root else "unknown",
            finding_type=finding_type,
            finding=finding_text,
            reveals_root=reveals_root,
        )
