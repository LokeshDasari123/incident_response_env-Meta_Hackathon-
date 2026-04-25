"""
agents/message_bus.py
---------------------
Inter-agent communication infrastructure.
Supports typed messages, broadcast/targeted delivery, and message interception.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    ANOMALY_REPORT       = "anomaly_report"
    DIAGNOSIS_HYPOTHESIS = "diagnosis_hypothesis"
    CHALLENGE            = "challenge"
    EVIDENCE             = "evidence"
    FAULT_ALERT          = "fault_alert"
    SIGNAL               = "signal"
    DECEPTION            = "deception"
    STATUS_UPDATE        = "status_update"


@dataclass
class Message:
    """A single inter-agent message."""
    sender:     str
    recipient:  str               # "*" = broadcast
    msg_type:   MessageType
    content:    Dict[str, Any]
    step:       int
    priority:   int   = 1         # 1=low, 2=medium, 3=high
    msg_id:     str   = field(default_factory=lambda: uuid.uuid4().hex[:8])
    intercepted: bool = False     # True if adversary modified this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id":      self.msg_id,
            "sender":      self.sender,
            "recipient":   self.recipient,
            "msg_type":    self.msg_type.value,
            "content":     self.content,
            "step":        self.step,
            "priority":    self.priority,
            "intercepted": self.intercepted,
        }


class MessageBus:
    """
    Central message bus for multi-agent communication.

    Features:
    - Broadcast and targeted message delivery
    - Full message history for evaluation
    - Adversarial interception support (with budget)
    """

    def __init__(self) -> None:
        self._history: List[Message]     = []
        self._pending: List[Message]     = []
        self._intercept_budget: int      = 0
        self._intercept_count:  int      = 0

    def reset(self, intercept_budget: int = 0) -> None:
        self._history           = []
        self._pending           = []
        self._intercept_budget  = intercept_budget
        self._intercept_count   = 0

    def send(self, message: Message) -> None:
        """Queue a message for delivery."""
        self._pending.append(message)

    def broadcast(
        self,
        sender:   str,
        msg_type: MessageType,
        content:  Dict[str, Any],
        step:     int,
        priority: int = 1,
    ) -> Message:
        """Convenience: send a broadcast message."""
        msg = Message(
            sender    = sender,
            recipient = "*",
            msg_type  = msg_type,
            content   = content,
            step      = step,
            priority  = priority,
        )
        self.send(msg)
        return msg

    def send_to(
        self,
        sender:    str,
        recipient: str,
        msg_type:  MessageType,
        content:   Dict[str, Any],
        step:      int,
        priority:  int = 1,
    ) -> Message:
        """Convenience: send a targeted message."""
        msg = Message(
            sender    = sender,
            recipient = recipient,
            msg_type  = msg_type,
            content   = content,
            step      = step,
            priority  = priority,
        )
        self.send(msg)
        return msg

    def deliver(self, agent_id: str) -> List[Message]:
        """
        Deliver all pending messages for a specific agent.
        Returns messages addressed to agent_id or broadcast ("*").
        Delivered messages move to history.
        """
        delivered = []
        remaining = []
        for msg in self._pending:
            if msg.recipient == "*" or msg.recipient == agent_id:
                delivered.append(msg)
            if msg.recipient != agent_id:
                # Keep broadcast messages for other agents; remove targeted
                remaining.append(msg)
        # Move delivered to history
        self._history.extend(delivered)
        self._pending = remaining
        return sorted(delivered, key=lambda m: -m.priority)

    def flush(self) -> None:
        """Move all pending messages to history (end of step)."""
        self._history.extend(self._pending)
        self._pending = []

    def intercept(self, message: Message, new_content: Dict[str, Any]) -> bool:
        """
        Adversarial agent intercepts and modifies a pending message.
        Returns True if interception succeeded (within budget).
        """
        if self._intercept_count >= self._intercept_budget:
            return False
        # Find and modify in pending
        for i, msg in enumerate(self._pending):
            if msg.msg_id == message.msg_id:
                self._pending[i] = Message(
                    sender      = msg.sender,
                    recipient   = msg.recipient,
                    msg_type    = msg.msg_type,
                    content     = new_content,
                    step        = msg.step,
                    priority    = msg.priority,
                    msg_id      = msg.msg_id,
                    intercepted = True,
                )
                self._intercept_count += 1
                return True
        return False

    def get_history(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get message history, optionally filtered by agent."""
        msgs = self._history
        if agent_id:
            msgs = [
                m for m in msgs
                if m.sender == agent_id or m.recipient in (agent_id, "*")
            ]
        return [m.to_dict() for m in msgs]

    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics for evaluation."""
        by_type: Dict[str, int] = {}
        by_sender: Dict[str, int] = {}
        for m in self._history:
            by_type[m.msg_type.value] = by_type.get(m.msg_type.value, 0) + 1
            by_sender[m.sender] = by_sender.get(m.sender, 0) + 1
        return {
            "total_messages":     len(self._history),
            "intercepted":        self._intercept_count,
            "intercept_budget":   self._intercept_budget,
            "by_type":            by_type,
            "by_sender":          by_sender,
        }
