"""
models/memory.py
----------------
Persistent agent memory model for long-horizon task complexity.
Tracks hypotheses, evidence, and investigation history across steps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """A single diagnosis hypothesis tracked across steps."""
    service:    str
    fault_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence:   str
    step:       int
    refuted:    bool = False


class Evidence(BaseModel):
    """A piece of evidence that supports or refutes a hypothesis."""
    source:   str
    content:  str
    supports: Optional[str] = None  # service name it supports
    refutes:  Optional[str] = None  # service name it refutes
    step:     int = 0


class InvestigationResult(BaseModel):
    """Result from an 'investigate_further' action — reveals hidden info."""
    step:          int
    service:       str
    finding_type:  str   # "log_entry", "config_diff", "trace_sample", "metric_deep_dive"
    finding:       str
    reveals_root:  bool = False   # True if this finding points at root cause


class EpisodeMemory(BaseModel):
    """
    Full episode memory persisted across steps.
    Serializable for prompt injection and evaluation.
    """
    hypotheses:         List[Hypothesis]         = Field(default_factory=list)
    evidence_log:       List[Evidence]            = Field(default_factory=list)
    investigation_results: List[InvestigationResult] = Field(default_factory=list)
    actions_taken:      List[Dict[str, Any]]      = Field(default_factory=list)
    confidence_history: List[float]               = Field(default_factory=list)
    reward_history:     List[float]               = Field(default_factory=list)
    step:               int                       = 0

    def add_hypothesis(
        self,
        service: str,
        fault_type: str,
        confidence: float,
        evidence: str,
        step: int,
    ) -> None:
        self.hypotheses.append(Hypothesis(
            service=service, fault_type=fault_type,
            confidence=confidence, evidence=evidence, step=step,
        ))

    def add_evidence(
        self,
        source: str,
        content: str,
        supports: Optional[str] = None,
        refutes: Optional[str] = None,
        step: int = 0,
    ) -> None:
        self.evidence_log.append(Evidence(
            source=source, content=content,
            supports=supports, refutes=refutes, step=step,
        ))

    def add_investigation_result(self, result: InvestigationResult) -> None:
        self.investigation_results.append(result)

    def add_action(self, action: Dict[str, Any], reward: float, step: int) -> None:
        self.actions_taken.append({"action": action, "reward": reward, "step": step})
        self.confidence_history.append(action.get("confidence", 0.5))
        self.reward_history.append(reward)

    def get_best_hypothesis(self) -> Optional[Hypothesis]:
        active = [h for h in self.hypotheses if not h.refuted]
        if not active:
            return None
        return max(active, key=lambda h: h.confidence)

    def get_reward_trend(self) -> str:
        if len(self.reward_history) < 2:
            return "insufficient_data"
        recent = self.reward_history[-3:]
        avg_recent = sum(recent) / len(recent)
        avg_all = sum(self.reward_history) / len(self.reward_history)
        if avg_recent > avg_all + 0.05:
            return "improving"
        if avg_recent < avg_all - 0.05:
            return "declining"
        return "stable"

    def summarize(self, max_lines: int = 10) -> str:
        """Compact text summary for LLM prompt injection."""
        lines = []
        # Best hypothesis
        best = self.get_best_hypothesis()
        if best:
            lines.append(
                f"Best hypothesis: {best.service} ({best.fault_type}) "
                f"confidence={best.confidence:.0%} [step {best.step}]"
            )
        # Last action + reward
        if self.actions_taken:
            last = self.actions_taken[-1]
            lines.append(
                f"Last action: {last['action'].get('remediation_action', '?')} "
                f"on {last['action'].get('root_cause_service', '?')} "
                f"→ reward={last['reward']:.2f}"
            )
        # Reward trend
        trend = self.get_reward_trend()
        if trend != "insufficient_data":
            lines.append(f"Reward trend: {trend}")
        # Investigation results
        if self.investigation_results:
            for ir in self.investigation_results[-2:]:
                lines.append(
                    f"Investigation [{ir.finding_type}]: {ir.finding[:60]}"
                )
        # Recent evidence
        if self.evidence_log:
            for ev in self.evidence_log[-2:]:
                lines.append(f"Evidence ({ev.source}): {ev.content[:60]}")
        return "\n".join(lines[:max_lines])
