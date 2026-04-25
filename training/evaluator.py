"""
training/evaluator.py
---------------------
Comprehensive evaluation system for emergent behavior detection
and training analysis.

Detects strategies like:
- topology_first: agent learned to check call graph before metrics
- elimination: agent systematically rules out services
- signal_correlation: agent cross-references multiple metrics
- challenge_integration: agent improves when challenged
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import Counter


class EpisodeEvaluator:
    """
    Evaluates a single episode's step logs for emergent behaviors
    and strategy patterns.
    """

    def evaluate(self, step_logs: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
        """Run full evaluation on episode step logs."""
        if not step_logs:
            return {"error": "no_data"}

        return {
            "task_id":                task_id,
            "total_steps":            len(step_logs),
            "reward_analysis":        self._analyze_rewards(step_logs),
            "strategy_detected":      self._detect_strategy(step_logs),
            "exploration_efficiency": self._exploration_efficiency(step_logs),
            "challenge_resistance":   self._challenge_resistance(step_logs),
            "memory_utilization":     self._memory_utilization(step_logs),
            "multi_agent_dynamics":   self._multi_agent_analysis(step_logs),
            "convergence":            self._convergence_analysis(step_logs),
        }

    def _analyze_rewards(self, logs: List[Dict]) -> Dict[str, Any]:
        rewards = [l.get("reward", 0) for l in logs]
        if not rewards:
            return {}
        return {
            "min":      round(min(rewards), 4),
            "max":      round(max(rewards), 4),
            "mean":     round(sum(rewards) / len(rewards), 4),
            "final":    round(rewards[-1], 4),
            "improved": rewards[-1] > rewards[0] if len(rewards) > 1 else False,
            "time_to_best": rewards.index(max(rewards)) + 1,
        }

    def _detect_strategy(self, logs: List[Dict]) -> Dict[str, Any]:
        """Detect which investigation strategy the agent is using."""
        strategies = {
            "topology_first":       0.0,
            "elimination":          0.0,
            "signal_correlation":   0.0,
            "monitor_integration":  0.0,
            "persistence":          0.0,
        }

        root_causes_tried = []
        for log in logs:
            action = log.get("action", {})
            rc = action.get("root_cause_service", "unknown")
            root_causes_tried.append(rc)
            reasoning = (action.get("reasoning") or "").lower()

            # Topology-first: mentions topology/call graph in reasoning
            if any(kw in reasoning for kw in ["topology", "call graph", "traversed", "inward"]):
                strategies["topology_first"] += 0.3

            # Signal correlation: mentions multiple metrics
            metric_mentions = sum(1 for kw in ["cpu", "memory", "http_rt", "error_rate", "latency"]
                                  if kw in reasoning)
            if metric_mentions >= 2:
                strategies["signal_correlation"] += 0.2

            # Monitor integration: references monitor signals
            if "monitor" in reasoning or "anomaly" in reasoning:
                strategies["monitor_integration"] += 0.3

        # Elimination: agent tried different root causes
        unique_rcs = len(set(root_causes_tried))
        if unique_rcs >= 3:
            strategies["elimination"] = 0.8
        elif unique_rcs >= 2:
            strategies["elimination"] = 0.4

        # Persistence: agent stuck with correct answer once found
        if len(root_causes_tried) > 2:
            final_rc = root_causes_tried[-1]
            consistent = sum(1 for rc in root_causes_tried[-3:] if rc == final_rc)
            if consistent >= 2:
                strategies["persistence"] = 0.6

        # Normalize to [0, 1]
        for k in strategies:
            strategies[k] = round(min(1.0, strategies[k]), 3)

        # Primary strategy
        primary = max(strategies, key=strategies.get)

        return {
            "scores":   strategies,
            "primary":  primary,
            "strength": strategies[primary],
        }

    def _exploration_efficiency(self, logs: List[Dict]) -> Dict[str, Any]:
        """How efficiently does the agent explore hypotheses?"""
        root_causes = [l.get("action", {}).get("root_cause_service", "?") for l in logs]
        unique = len(set(root_causes))
        total  = len(root_causes)
        return {
            "unique_hypotheses": unique,
            "total_attempts":    total,
            "efficiency":        round(unique / max(1, total), 3),
            "exploration_path":  root_causes,
        }

    def _challenge_resistance(self, logs: List[Dict]) -> Dict[str, Any]:
        """Does the agent maintain correct answers under adversarial pressure?"""
        resistance_events = 0
        capitulation_events = 0

        for i in range(1, len(logs)):
            prev_action = logs[i-1].get("action", {})
            curr_action = logs[i].get("action", {})
            deception   = logs[i].get("deception")
            injection   = logs[i].get("injection")

            if deception or injection:
                prev_rc = prev_action.get("root_cause_service")
                curr_rc = curr_action.get("root_cause_service")
                if prev_rc == curr_rc:
                    resistance_events += 1
                else:
                    capitulation_events += 1

        total = resistance_events + capitulation_events
        return {
            "resistance_rate":  round(resistance_events / max(1, total), 3),
            "total_challenges": total,
            "resisted":         resistance_events,
            "capitulated":      capitulation_events,
        }

    def _memory_utilization(self, logs: List[Dict]) -> Dict[str, Any]:
        """How effectively does the agent use memory?"""
        memory_states = [l.get("memory_state", {}) for l in logs]
        if not memory_states:
            return {"score": 0.0}

        final = memory_states[-1]
        hypotheses = final.get("hypotheses", 0)
        evidence   = final.get("evidence", 0)
        investigations = final.get("investigations", 0)

        # Score: agents that actively investigate and build evidence score higher
        score = min(1.0, (hypotheses * 0.2 + evidence * 0.1 + investigations * 0.3))
        return {
            "score":          round(score, 3),
            "hypotheses":     hypotheses,
            "evidence_items": evidence,
            "investigations": investigations,
            "reward_trend":   final.get("reward_trend", "unknown"),
        }

    def _multi_agent_analysis(self, logs: List[Dict]) -> Dict[str, Any]:
        """Analyze multi-agent interaction patterns."""
        total_injections = sum(1 for l in logs if l.get("injection"))
        total_deceptions = sum(1 for l in logs if l.get("deception"))
        total_monitor_signals = 0
        for l in logs:
            ma = l.get("monitor_action", {})
            total_monitor_signals += len(ma.get("anomalies", []))

        return {
            "fault_injections":    total_injections,
            "evidence_corruptions": total_deceptions,
            "monitor_signals":     total_monitor_signals,
            "adversarial_pressure": round(
                (total_injections + total_deceptions) / max(1, len(logs)), 3
            ),
        }

    def _convergence_analysis(self, logs: List[Dict]) -> Dict[str, Any]:
        """Analyze how quickly the agent converges to a solution."""
        rewards = [l.get("reward", 0) for l in logs]
        if len(rewards) < 3:
            return {"converged": False}

        # Find first step where reward > 0.6
        convergence_step = None
        for i, r in enumerate(rewards):
            if r >= 0.6:
                convergence_step = i + 1
                break

        # Check stability: last 3 rewards within 0.1 of each other
        stable = False
        if len(rewards) >= 3:
            last3 = rewards[-3:]
            stable = (max(last3) - min(last3)) < 0.15

        return {
            "converged":        convergence_step is not None,
            "convergence_step": convergence_step,
            "stable":           stable,
            "final_reward":     round(rewards[-1], 4),
        }


class TrainingCurveAnalyzer:
    """
    Analyzes training reward curves across episodes for
    learning phase detection and convergence metrics.
    """

    def analyze(
        self,
        reward_history: Dict[str, List[float]],
        transitions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Full training curve analysis."""
        per_task = {}
        for task_id, rewards in reward_history.items():
            if not rewards:
                continue
            per_task[task_id] = {
                "episode_count":    len(rewards),
                "mean_reward":      round(sum(rewards) / len(rewards), 4),
                "best_reward":      round(max(rewards), 4),
                "final_avg10":      self._avg_last(rewards, 10),
                "learning_phase":   self._detect_phase(rewards),
                "improvement_rate": self._improvement_rate(rewards),
            }

        return {
            "per_task":       per_task,
            "transitions":    transitions,
            "overall_trend":  self._overall_trend(reward_history),
        }

    def _avg_last(self, rewards: List[float], n: int) -> float:
        if not rewards:
            return 0.0
        chunk = rewards[-n:]
        return round(sum(chunk) / len(chunk), 4)

    def _detect_phase(self, rewards: List[float]) -> str:
        """Detect current learning phase."""
        if len(rewards) < 5:
            return "warmup"
        recent = self._avg_last(rewards, 10)
        early  = sum(rewards[:10]) / min(10, len(rewards))

        if recent > early + 0.15:
            return "rapid_learning"
        if abs(recent - early) < 0.05 and recent > 0.5:
            return "plateau"
        if recent < early - 0.1:
            return "regression"
        return "gradual_improvement"

    def _improvement_rate(self, rewards: List[float]) -> float:
        """Compute reward improvement per episode."""
        if len(rewards) < 10:
            return 0.0
        early = sum(rewards[:5]) / 5
        late  = sum(rewards[-5:]) / 5
        return round((late - early) / max(1, len(rewards)) * 100, 4)

    def _overall_trend(self, history: Dict[str, List[float]]) -> str:
        """Overall training trend across all tasks."""
        all_rewards = []
        for rewards in history.values():
            all_rewards.extend(rewards)
        if len(all_rewards) < 10:
            return "insufficient_data"
        first_half = sum(all_rewards[:len(all_rewards)//2]) / max(1, len(all_rewards)//2)
        second_half = sum(all_rewards[len(all_rewards)//2:]) / max(1, len(all_rewards) - len(all_rewards)//2)
        if second_half > first_half + 0.05:
            return "improving"
        if second_half < first_half - 0.05:
            return "declining"
        return "stable"
