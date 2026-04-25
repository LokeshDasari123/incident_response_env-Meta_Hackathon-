"""
agents/base_agent.py
--------------------
Abstract base class for all agents in the multi-agent incident response system.
Each agent has a unique role, partial observability, persistent memory, and
communication capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agents.message_bus import Message, MessageBus, MessageType


class AgentMemory:
    """
    Persistent memory that survives across episode steps.
    Tracks hypotheses, evidence, and investigation history.
    """

    def __init__(self) -> None:
        self.hypotheses:        List[Dict[str, Any]] = []
        self.evidence_log:      List[Dict[str, Any]] = []
        self.actions_taken:     List[Dict[str, Any]] = []
        self.confidence_history: List[float]          = []
        self.observations_digest: List[str]           = []
        self.step:              int                   = 0

    def reset(self) -> None:
        self.__init__()

    def add_hypothesis(
        self,
        service: str,
        fault_type: str,
        confidence: float,
        evidence: str,
        step: int,
    ) -> None:
        self.hypotheses.append({
            "service":    service,
            "fault_type": fault_type,
            "confidence": confidence,
            "evidence":   evidence,
            "step":       step,
        })

    def add_evidence(
        self,
        source: str,
        content: str,
        supports: Optional[str] = None,
        refutes: Optional[str]  = None,
        step: int = 0,
    ) -> None:
        self.evidence_log.append({
            "source":   source,
            "content":  content,
            "supports": supports,
            "refutes":  refutes,
            "step":     step,
        })

    def add_action(self, action: Dict[str, Any], reward: float, step: int) -> None:
        self.actions_taken.append({
            "action": action,
            "reward": reward,
            "step":   step,
        })
        self.confidence_history.append(action.get("confidence", 0.5))

    def get_best_hypothesis(self) -> Optional[Dict[str, Any]]:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h["confidence"])

    def get_last_action(self) -> Optional[Dict[str, Any]]:
        return self.actions_taken[-1] if self.actions_taken else None

    def summarize(self, max_lines: int = 8) -> str:
        """Produce a compact text summary for LLM prompt injection."""
        lines = []
        if self.hypotheses:
            best = self.get_best_hypothesis()
            lines.append(
                f"Best hypothesis: {best['service']} ({best['fault_type']}) "
                f"confidence={best['confidence']:.0%}"
            )
        if self.actions_taken:
            last = self.actions_taken[-1]
            lines.append(
                f"Last action: {last['action'].get('remediation_action','?')} "
                f"on {last['action'].get('root_cause_service','?')} "
                f"→ reward={last['reward']:.2f}"
            )
        if self.evidence_log:
            recent = self.evidence_log[-3:]
            for ev in recent:
                lines.append(f"Evidence ({ev['source']}): {ev['content'][:80]}")
        if self.confidence_history:
            trend = "↑" if len(self.confidence_history) > 1 and \
                self.confidence_history[-1] > self.confidence_history[-2] else "↓"
            lines.append(
                f"Confidence trend: {trend} "
                f"(current={self.confidence_history[-1]:.0%})"
            )
        return "\n".join(lines[:max_lines])


class BaseAgent(ABC):
    """
    Abstract base class for all multi-agent participants.

    Each agent has:
    - A role name (for message addressing)
    - Partial observability (sees a filtered view)
    - Persistent memory across steps
    - Communication via the message bus
    """

    ROLE: str = "base"

    def __init__(self, agent_id: Optional[str] = None) -> None:
        self.agent_id = agent_id or self.ROLE
        self.memory   = AgentMemory()
        self._step    = 0

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.memory.reset()
        self._step = 0

    @abstractmethod
    def perceive(
        self,
        observation: Dict[str, Any],
        messages:    List[Message],
    ) -> Dict[str, Any]:
        """
        Filter the full observation to this agent's partial view.
        Each agent sees different data based on its role.
        """
        ...

    @abstractmethod
    def act(
        self,
        filtered_obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produce this agent's action given its filtered observation.
        """
        ...

    @abstractmethod
    def communicate(
        self,
        action: Dict[str, Any],
        obs:    Dict[str, Any],
        bus:    MessageBus,
        step:   int,
    ) -> Optional[Message]:
        """
        After acting, produce a message for other agents.
        Returns the message sent, or None.
        """
        ...

    def receive_messages(self, messages: List[Message]) -> None:
        """Process incoming messages and update memory."""
        for msg in messages:
            self.memory.add_evidence(
                source  = msg.sender,
                content = str(msg.content),
                step    = msg.step,
            )

    def get_state(self) -> Dict[str, Any]:
        """Return agent state for serialisation / logging."""
        return {
            "agent_id": self.agent_id,
            "role":     self.ROLE,
            "step":     self._step,
            "memory":   {
                "hypotheses":    len(self.memory.hypotheses),
                "evidence":      len(self.memory.evidence_log),
                "actions":       len(self.memory.actions_taken),
                "confidence":    self.memory.confidence_history[-1]
                                 if self.memory.confidence_history else 0.0,
            },
        }
