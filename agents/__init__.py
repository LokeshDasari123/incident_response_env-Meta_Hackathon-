"""
agents/
-------
Multi-agent system for Incident Response Environment.
Exports all agent classes, the message bus, and the hybrid routing stack.
"""

from agents.message_bus          import Message, MessageBus, MessageType
from agents.base_agent           import BaseAgent, AgentMemory
from agents.monitor_agent        import MonitorAgent
from agents.fault_injector_agent import FaultInjectorAgent
from agents.adversarial_agent    import AdversarialAgent
from agents.responder_agent      import ResponderAgent

__all__ = [
    # Infrastructure
    "Message",
    "MessageBus",
    "MessageType",
    "BaseAgent",
    "AgentMemory",
    # Core agents
    "MonitorAgent",
    "FaultInjectorAgent",
    "AdversarialAgent",
    "ResponderAgent",
    # Hybrid routing stack (lazy-imported to avoid circular deps)
    "ComplexityRouter",
    "ChainOfThought",
    "ProgressiveMemorySystem",
    "ShortTermMemory",
    "LongTermMemory",
]

# ── Lazy imports for hybrid stack ─────────────────────────────────────────────
# Imported on first access to avoid circular import at package init time.
def __getattr__(name: str):
    if name == "ComplexityRouter":
        from agents.hybrid_router import ComplexityRouter
        return ComplexityRouter
    if name == "ChainOfThought":
        from agents.chain_of_thought import ChainOfThought
        return ChainOfThought
    if name in ("ProgressiveMemorySystem", "ShortTermMemory", "LongTermMemory"):
        import agents.progressive_memory as pm
        return getattr(pm, name)
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
