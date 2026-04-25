"""
agents/monitor_agent.py
-----------------------
Cooperative Monitoring Agent — provides anomaly detection signals
to the Incident Responder.

Sees:  Full metric time series (richer data than responder).
Acts:  Produces anomaly signals (trend analysis, threshold breaches).
Comms: Sends anomaly reports to responder via message bus.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from agents.base_agent  import BaseAgent
from agents.message_bus import Message, MessageBus, MessageType


class MonitorAgent(BaseAgent):
    """
    Cooperative agent that performs statistical anomaly detection on
    service metrics and communicates findings to the Responder.

    Configurable reliability: sometimes produces noisy or delayed signals,
    scaled by the curriculum controller.
    """

    ROLE = "monitor"

    def __init__(
        self,
        reliability: float = 0.85,
        noise_level: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__("monitor")
        self.reliability = reliability   # probability of correct signal
        self.noise_level = noise_level   # false positive rate
        self._rng        = random.Random(seed)
        self._metric_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self._metric_history = []

    def perceive(
        self,
        observation: Dict[str, Any],
        messages:    List[Message],
    ) -> Dict[str, Any]:
        """
        Monitor sees full metrics and alerts but NOT topology or timeline.
        This forces the Responder to rely on the Monitor for metric trends.
        """
        metrics = observation.get("metrics", {})
        self._metric_history.append(metrics)

        return {
            "step":     observation.get("step", 0),
            "metrics":  metrics,
            "alerts":   observation.get("alerts", []),
            "metric_history_len": len(self._metric_history),
        }

    def act(self, filtered_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform anomaly detection on current metrics.
        Returns anomaly signals for each degraded service.
        """
        metrics   = filtered_obs.get("metrics", {})
        step      = filtered_obs.get("step", 0)
        anomalies = []

        for svc, m in metrics.items():
            score = self._anomaly_score(svc, m)
            if score > 0.3:
                # Reliability check: sometimes the monitor is wrong
                if self._rng.random() < self.reliability:
                    anomalies.append({
                        "service":      svc,
                        "anomaly_score": round(score, 3),
                        "reason":       self._explain_anomaly(m),
                        "trend":        self._compute_trend(svc),
                        "reliable":     True,
                    })
                elif self._rng.random() < self.noise_level:
                    # False positive noise signal
                    anomalies.append({
                        "service":      svc,
                        "anomaly_score": round(score * 0.5, 3),
                        "reason":       "Possible anomaly detected (low confidence)",
                        "trend":        "unknown",
                        "reliable":     False,
                    })

        # Sort by anomaly score descending
        anomalies.sort(key=lambda a: -a["anomaly_score"])

        return {
            "anomalies": anomalies,
            "step":      step,
            "services_monitored": len(metrics),
        }

    def communicate(
        self,
        action: Dict[str, Any],
        obs:    Dict[str, Any],
        bus:    MessageBus,
        step:   int,
    ) -> Optional[Message]:
        """Send anomaly report to all agents (broadcast)."""
        anomalies = action.get("anomalies", [])
        if not anomalies:
            return None

        top_anomaly = anomalies[0]
        content = {
            "top_service":    top_anomaly["service"],
            "anomaly_score":  top_anomaly["anomaly_score"],
            "reason":         top_anomaly["reason"],
            "trend":          top_anomaly["trend"],
            "total_anomalies": len(anomalies),
            "all_anomalies":  anomalies[:5],
        }

        return bus.broadcast(
            sender   = self.agent_id,
            msg_type = MessageType.ANOMALY_REPORT,
            content  = content,
            step     = step,
            priority = 2 if top_anomaly["anomaly_score"] > 0.7 else 1,
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _anomaly_score(self, svc: str, m: Dict[str, Any]) -> float:
        """Compute a 0-1 anomaly score for a service's current metrics."""
        score = 0.0
        cpu = m.get("cpu_utilization", 0)
        mem = m.get("memory_utilization", 0)
        rt  = m.get("http_rt") or 0
        err = m.get("error_rate", 0)
        status = m.get("status", "healthy")

        if status == "failing":    score += 0.4
        elif status == "critical": score += 0.3
        elif status == "degraded": score += 0.15

        if cpu > 0.85:  score += 0.15
        if mem > 0.85:  score += 0.2
        if rt > 1000:   score += 0.15
        if err > 0.3:   score += 0.2

        # Trend bonus from history
        trend_score = self._trend_score(svc)
        score += trend_score * 0.1

        return min(1.0, score)

    def _compute_trend(self, svc: str) -> str:
        """Compute metric trend direction from history."""
        if len(self._metric_history) < 2:
            return "insufficient_data"

        recent_cpus = []
        for snap in self._metric_history[-5:]:
            if svc in snap:
                recent_cpus.append(snap[svc].get("cpu_utilization", 0))

        if len(recent_cpus) < 2:
            return "stable"

        delta = recent_cpus[-1] - recent_cpus[0]
        if delta > 0.1:   return "rapidly_increasing"
        if delta > 0.03:  return "increasing"
        if delta < -0.1:  return "rapidly_decreasing"
        if delta < -0.03: return "decreasing"
        return "stable"

    def _trend_score(self, svc: str) -> float:
        """Numeric trend score for anomaly scoring."""
        trend = self._compute_trend(svc)
        return {
            "rapidly_increasing": 1.0,
            "increasing":         0.6,
            "stable":             0.0,
            "decreasing":         -0.3,
            "rapidly_decreasing": -0.5,
            "insufficient_data":  0.0,
        }.get(trend, 0.0)

    def _explain_anomaly(self, m: Dict[str, Any]) -> str:
        """Generate human-readable anomaly explanation."""
        reasons = []
        if m.get("status") in ("failing", "critical"):
            reasons.append(f"Service status: {m['status']}")
        if m.get("cpu_utilization", 0) > 0.85:
            reasons.append(f"CPU at {m['cpu_utilization']:.0%}")
        if m.get("memory_utilization", 0) > 0.85:
            reasons.append(f"Memory at {m['memory_utilization']:.0%}")
        if (m.get("http_rt") or 0) > 1000:
            reasons.append(f"HTTP RT at {m.get('http_rt', 0):.0f}ms")
        if m.get("error_rate", 0) > 0.3:
            reasons.append(f"Error rate at {m['error_rate']:.0%}")
        return "; ".join(reasons) if reasons else "Minor degradation detected"
