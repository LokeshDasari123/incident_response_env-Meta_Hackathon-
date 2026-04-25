"""
agents/progressive_memory.py
-----------------------------
Progressive Learning Memory System — Two-tier STM + LTM gated by complexity.

SHORT-TERM memory (per-episode):
  Sliding window of step observations, hypothesis chain, action history.
  Always injected. Detail level = low/medium/high based on complexity.

LONG-TERM memory (cross-episode, persisted to disk):
  Fault patterns, remediation success rates, routing stats, red-herring registry.
  Consulted only when complexity >= 0.40. Auto-saves every 10 episodes.
  Exponential decay: half-life = 50 episodes.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

LTM_READ_THRESHOLD  = 0.40
LTM_WRITE_THRESHOLD = 0.30
DECAY_HALF_LIFE     = 50


# ══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════
class ShortTermMemory:
    MAX_STEPS      = 5
    MAX_HYPOTHESES = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.step_observations: List[Dict[str, Any]] = []
        self.hypotheses:        List[Dict[str, Any]] = []
        self.action_history:    List[Dict[str, Any]] = []
        self.monitor_signals:   List[Dict[str, Any]] = []
        self.debate_challenges: List[str]            = []
        self.current_step:      int                  = 0

    def observe(self, obs: Dict[str, Any], step: int) -> None:
        metrics  = obs.get("metrics", {})
        alerts   = obs.get("alerts",  [])
        topology = obs.get("topology", [])
        unhealthy = {
            svc: {
                "cpu":    round(m.get("cpu_utilization", 0), 2),
                "mem":    round(m.get("memory_utilization", 0), 2),
                "status": m.get("status", "unknown"),
            }
            for svc, m in metrics.items()
            if not m.get("is_healthy", True)
        }
        snapshot = {
            "step":           step,
            "unhealthy_svcs": unhealthy,
            "alert_count":    len(alerts),
            "topology_edges": len(topology),
            "time_pressure":  round(obs.get("time_pressure", 0.0), 2),
        }
        self.step_observations.append(snapshot)
        if len(self.step_observations) > self.MAX_STEPS:
            self.step_observations.pop(0)
        self.current_step = step

    def add_hypothesis(
        self,
        service: str,
        fault_type: str,
        confidence: float,
        step: int,
        source: str = "agent",
    ) -> None:
        for h in self.hypotheses:
            if h["service"] == service:
                if confidence > h["confidence"]:
                    h.update({"fault_type": fault_type, "confidence": confidence,
                               "step": step, "source": source})
                return
        self.hypotheses.append({
            "service": service, "fault_type": fault_type,
            "confidence": confidence, "step": step,
            "source": source, "refuted": False,
        })
        self.hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
        if len(self.hypotheses) > self.MAX_HYPOTHESES:
            self.hypotheses = self.hypotheses[:self.MAX_HYPOTHESES]

    def refute_hypothesis(self, service: str) -> None:
        for h in self.hypotheses:
            if h["service"] == service:
                h["refuted"] = True

    def add_action(
        self,
        action: Dict[str, Any],
        reward: float,
        step: int,
        complexity: float,
        model_used: str = "rule_based",
    ) -> None:
        self.action_history.append({
            "step": step, "action": action, "reward": reward,
            "complexity": round(complexity, 3), "model_used": model_used,
        })

    def add_monitor_signal(self, signal: Dict[str, Any]) -> None:
        self.monitor_signals.append(signal)
        if len(self.monitor_signals) > 5:
            self.monitor_signals.pop(0)

    def add_debate_challenge(self, challenge: str) -> None:
        self.debate_challenges.append(challenge)

    def get_best_hypothesis(self) -> Optional[Dict[str, Any]]:
        active = [h for h in self.hypotheses if not h.get("refuted")]
        return max(active, key=lambda h: h["confidence"]) if active else None

    def get_reward_trend(self) -> str:
        rewards = [a["reward"] for a in self.action_history]
        if len(rewards) < 2:
            return "no_data"
        return "improving" if rewards[-1] > rewards[0] else (
            "declining" if rewards[-1] < rewards[0] else "stable"
        )

    def summarize_short(self, detail_level: str = "medium") -> str:
        lines = []
        best = self.get_best_hypothesis()
        if best:
            lines.append(
                f"[STM] Best hypothesis: {best['service']} "
                f"({best['fault_type']}) conf={best['confidence']:.0%} [step {best['step']}]"
            )
        if self.action_history:
            last = self.action_history[-1]
            lines.append(
                f"[STM] Last action: {last['action'].get('remediation_action','?')} "
                f"on {last['action'].get('root_cause_service','?')} "
                f"→ reward={last['reward']:.2f} | model={last['model_used']}"
            )
            lines.append(f"[STM] Reward trend: {self.get_reward_trend()}")

        if detail_level in ("medium", "high"):
            if self.step_observations:
                snap = self.step_observations[-1]
                svcs = list(snap.get("unhealthy_svcs", {}).keys())[:4]
                if svcs:
                    lines.append(f"[STM] Unhealthy services: {', '.join(svcs)}")
            refuted = [h["service"] for h in self.hypotheses if h.get("refuted")]
            if refuted:
                lines.append(f"[STM] Refuted: {', '.join(refuted)} — do NOT repeat")

        if detail_level == "high":
            for sig in self.monitor_signals[-2:]:
                lines.append(
                    f"[STM] Monitor: {sig.get('top_service','?')} "
                    f"anomaly={sig.get('anomaly_score',0):.2f}"
                )
            if self.debate_challenges:
                lines.append(f"[STM] Challenger: {self.debate_challenges[-1][:100]}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY
# ══════════════════════════════════════════════════════════════════════════════
class LongTermMemory:
    DEFAULT_FILE = "data/memory/long_term_memory.json"

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self.persist_path = Path(persist_path or self.DEFAULT_FILE)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.fault_patterns:    Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {"hits": 0, "weight": 1.0, "best_action": "investigate_further"})
        )
        self.remediation_stats: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"attempts": 0, "successes": 0})
        )
        self.routing_stats:     Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: {"count": 0.0, "avg_reward": 0.0})
        )
        self.red_herring_seen:  Dict[str, int] = defaultdict(int)
        self.total_episodes:    int   = 0
        self._load()

    def _load(self) -> None:
        if not self.persist_path.exists():
            return
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            for k, v in data.get("fault_patterns", {}).items():
                for fk, fv in v.items():
                    self.fault_patterns[k][fk] = fv
            for k, v in data.get("remediation_stats", {}).items():
                for rk, rv in v.items():
                    self.remediation_stats[k][rk] = rv
            for k, v in data.get("routing_stats", {}).items():
                for mk, mv in v.items():
                    self.routing_stats[k][mk] = mv
            self.red_herring_seen = defaultdict(int, data.get("red_herring_seen", {}))
            self.total_episodes   = data.get("total_episodes", 0)
        except Exception:
            pass

    def save(self) -> None:
        try:
            with open(self.persist_path, "w") as f:
                json.dump({
                    "fault_patterns":    {k: dict(v) for k, v in self.fault_patterns.items()},
                    "remediation_stats": {k: dict(v) for k, v in self.remediation_stats.items()},
                    "routing_stats":     {k: dict(v) for k, v in self.routing_stats.items()},
                    "red_herring_seen":  dict(self.red_herring_seen),
                    "total_episodes":    self.total_episodes,
                    "saved_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, f, indent=2)
        except Exception:
            pass

    def consolidate(
        self,
        task_id: str,
        fault_type: str,
        remediation_action: str,
        reward: float,
        model_used: str,
        complexity: float,
        false_positives: Optional[List[str]] = None,
    ) -> None:
        self.total_episodes += 1
        ep = self.total_episodes
        pat = self.fault_patterns[task_id][fault_type]
        pat["hits"] += 1
        if reward > pat.get("best_reward", 0.0):
            pat["best_action"] = remediation_action
            pat["best_reward"] = round(reward, 4)
        pat["weight"] = round(pat.get("weight", 1.0) * math.exp(-math.log(2) / DECAY_HALF_LIFE), 4)
        rem = self.remediation_stats[fault_type][remediation_action]
        rem["attempts"] += 1
        if reward >= 0.60:
            rem["successes"] += 1
        band = self._band(complexity)
        rs   = self.routing_stats[band][model_used]
        n    = rs["count"]
        rs["avg_reward"] = round((rs["avg_reward"] * n + reward) / (n + 1), 4)
        rs["count"]      = n + 1
        if false_positives:
            for svc in false_positives:
                self.red_herring_seen[svc] += 1
        if ep % 10 == 0:
            self.save()

    @staticmethod
    def _band(c: float) -> str:
        return "low" if c < 0.35 else ("high" if c >= 0.70 else "medium")

    def get_best_remediation(self, fault_type: str) -> Optional[str]:
        actions = self.remediation_stats.get(fault_type, {})
        if not actions:
            return None
        best = max(actions.items(), key=lambda kv: kv[1]["successes"] / max(1, kv[1]["attempts"]))
        return best[0] if best[1]["successes"] / max(1, best[1]["attempts"]) > 0.4 else None

    def get_fault_pattern_hint(self, task_id: str) -> Optional[str]:
        patterns = self.fault_patterns.get(task_id, {})
        if not patterns:
            return None
        ranked = sorted(patterns.items(), key=lambda kv: kv[1]["hits"] * kv[1].get("weight", 1.0), reverse=True)
        if ranked:
            ft, d = ranked[0]
            return f"{ft} (seen {d['hits']}x, best_action={d['best_action']})"
        return None

    def get_routing_recommendation(self, complexity: float) -> Optional[str]:
        band    = self._band(complexity)
        options = self.routing_stats.get(band, {})
        if not options:
            return None
        best = max(options.items(), key=lambda kv: kv[1]["avg_reward"])
        return best[0] if best[1]["count"] >= 3 else None

    def is_likely_red_herring(self, service: str) -> bool:
        return self.red_herring_seen.get(service, 0) >= 2

    def summarize_long(self, task_id: str, complexity: float) -> str:
        lines = ["[LTM] Cross-episode knowledge:"]
        hint  = self.get_fault_pattern_hint(task_id)
        if hint:
            lines.append(f"[LTM] Most common fault in '{task_id}': {hint}")
        rr = self.get_routing_recommendation(complexity)
        if rr:
            lines.append(f"[LTM] Best model for {self._band(complexity)} complexity: {rr}")
        rh = [s for s, c in self.red_herring_seen.items() if c >= 2]
        if rh:
            lines.append(f"[LTM] Known red herrings (ignore): {', '.join(rh[:5])}")
        lines.append(f"[LTM] Episodes learned from: {self.total_episodes}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESSIVE MEMORY SYSTEM  (unified STM + LTM interface)
# ══════════════════════════════════════════════════════════════════════════════
class ProgressiveMemorySystem:
    """
    Complexity-gated two-tier memory.
      complexity < 0.35  → STM only (low detail)
      0.35 ≤ c < 0.70   → STM (medium) + LTM hints
      c ≥ 0.70          → STM (high)  + full LTM
    """

    def __init__(self, agent_id: str = "responder", persist_dir: Optional[str] = None) -> None:
        self.agent_id = agent_id
        self.stm = ShortTermMemory()
        self.ltm = LongTermMemory(
            persist_path=str(Path(persist_dir or "data/memory") / f"{agent_id}_ltm.json")
        )
        self._current_task_id:  str        = "easy"
        self._current_episode:  int        = 0
        self._step_complexity:  List[float] = []

    def episode_reset(self, task_id: str, episode_num: int) -> None:
        self.stm.reset()
        self._current_task_id = task_id
        self._current_episode = episode_num
        self._step_complexity = []

    def episode_end(self, best_reward: float, false_positives: Optional[List[str]] = None) -> None:
        if best_reward < LTM_WRITE_THRESHOLD:
            self.ltm.total_episodes += 1
            if self.ltm.total_episodes % 10 == 0:
                self.ltm.save()
            return
        best = None
        if self.stm.action_history:
            best = max(self.stm.action_history, key=lambda a: a["reward"])
        if best:
            action     = best["action"]
            avg_c      = (sum(self._step_complexity) / len(self._step_complexity)
                          if self._step_complexity else 0.5)
            self.ltm.consolidate(
                task_id            = self._current_task_id,
                fault_type         = action.get("root_cause_type", "unknown"),
                remediation_action = action.get("remediation_action", "investigate_further"),
                reward             = best_reward,
                model_used         = best.get("model_used", "rule_based"),
                complexity         = avg_c,
                false_positives    = false_positives,
            )

    def observe(self, obs: Dict[str, Any], step: int, complexity: float) -> None:
        self.stm.observe(obs, step)
        self._step_complexity.append(complexity)

    def add_hypothesis(self, service: str, fault_type: str, confidence: float,
                       step: int, source: str = "agent") -> None:
        self.stm.add_hypothesis(service, fault_type, confidence, step, source)

    def add_action(self, action: Dict[str, Any], reward: float, step: int,
                   complexity: float, model_used: str = "rule_based") -> None:
        self.stm.add_action(action, reward, step, complexity, model_used)

    def add_monitor_signal(self, signal: Dict[str, Any]) -> None:
        self.stm.add_monitor_signal(signal)

    def add_debate_challenge(self, challenge: str) -> None:
        self.stm.add_debate_challenge(challenge)

    def refute_hypothesis(self, service: str) -> None:
        self.stm.refute_hypothesis(service)
        self.ltm.red_herring_seen[service] += 1

    def build_context(self, complexity: float, task_id: Optional[str] = None) -> str:
        tid   = task_id or self._current_task_id
        parts = []
        if complexity < 0.35:
            t = self.stm.summarize_short("low")
            if t:
                parts.append(t)
        elif complexity < 0.70:
            t = self.stm.summarize_short("medium")
            if t:
                parts.append(t)
            if self.ltm.total_episodes >= 5:
                hint = self.ltm.get_fault_pattern_hint(tid)
                if hint:
                    parts.append(f"[LTM] Common fault in '{tid}': {hint}")
                rh = [s for s, c in self.ltm.red_herring_seen.items() if c >= 2]
                if rh:
                    parts.append(f"[LTM] Known red herrings: {', '.join(rh[:4])}")
        else:
            t = self.stm.summarize_short("high")
            if t:
                parts.append(t)
            if self.ltm.total_episodes >= 3:
                parts.append(self.ltm.summarize_long(tid, complexity))
                rec = self.ltm.get_routing_recommendation(complexity)
                if rec:
                    parts.append(f"[LTM] Historically best model for this complexity: {rec}")
        return "\n".join(parts)

    # ── Legacy AgentMemory compatibility ──────────────────────────────────────
    def summarize(self, max_lines: int = 8) -> str:
        return self.stm.summarize_short("medium")

    def add_evidence(self, source: str, content: str,
                     supports: Optional[str] = None,
                     refutes: Optional[str] = None, step: int = 0) -> None:
        svc = supports or refutes or "unknown"
        self.stm.add_hypothesis(svc, "unknown", 0.3, step, source)

    def get_best_hypothesis(self) -> Optional[Dict[str, Any]]:
        return self.stm.get_best_hypothesis()

    def get_learning_stats(self) -> Dict[str, Any]:
        return {
            "total_episodes_in_ltm":    self.ltm.total_episodes,
            "fault_patterns_learned":   sum(len(v) for v in self.ltm.fault_patterns.values()),
            "remediation_patterns":     sum(len(v) for v in self.ltm.remediation_stats.values()),
            "red_herrings_identified":  len([s for s, c in self.ltm.red_herring_seen.items() if c >= 2]),
            "routing_decisions_in_ltm": sum(
                sum(int(m["count"]) for m in band.values())
                for band in self.ltm.routing_stats.values()
            ),
        }
