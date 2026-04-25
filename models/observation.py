"""
models/observation.py
---------------------
Typed Observation model for the Incident Response Environment.
This is what the agent SEES each step — the incident state.
Modeled on Alibaba MSCallGraph + MSMetrics schema.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Alert(BaseModel):
    """
    A single alert fired by the monitoring system.
    Modeled on real PagerDuty/Datadog alert structure.
    """
    alert_id: str = Field(..., description="Unique alert identifier")
    service: str = Field(..., description="Service that fired the alert")
    metric: str = Field(
        ...,
        description="Metric name: HTTP_RT, consumerRPC_RT, cpu_utilization, etc.",
        examples=["HTTP_RT", "consumerRPC_RT", "cpu_utilization", "memory_utilization"]
    )
    current_value: float = Field(..., description="Current metric value")
    threshold: float = Field(..., description="Threshold that was breached")
    severity: str = Field(..., description="Alert severity: critical/warning/info")
    fired_at_step: int = Field(..., description="Episode step when alert fired")
    is_red_herring: bool = Field(
        default=False,
        description="Internal flag — NOT shown to agent. Used by grader only."
    )

    def to_agent_view(self) -> Dict[str, Any]:
        """Returns alert dict without internal grader fields."""
        return {
            "alert_id": self.alert_id,
            "service": self.service,
            "metric": self.metric,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "severity": self.severity,
            "fired_at_step": self.fired_at_step,
        }


class ServiceMetrics(BaseModel):
    """
    Real-time metrics for a single microservice.
    Directly maps to Alibaba MS_Resource_Table + MS_MCR_RT_Table.
    """
    service_name: str
    cpu_utilization: float = Field(..., ge=0.0, le=1.0)
    memory_utilization: float = Field(..., ge=0.0, le=1.0)
    http_rt: Optional[float] = Field(None, description="HTTP response time ms")
    consumer_rpc_rt: Optional[float] = Field(None, description="Consumer RPC RT ms")
    provider_rpc_rt: Optional[float] = Field(None, description="Provider RPC RT ms")
    http_mcr: Optional[float] = Field(None, description="HTTP call rate req/s")
    consumer_rpc_mcr: Optional[float] = Field(None, description="Consumer RPC call rate")
    provider_rpc_mcr: Optional[float] = Field(None, description="Provider RPC call rate")
    is_healthy: bool = Field(default=True)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class TopologyEdge(BaseModel):
    """
    A directed edge in the service call graph.
    Maps directly to Alibaba MS_CallGraph_Table (um → dm).
    """
    upstream_service: str = Field(..., description="UM: calling service")
    downstream_service: str = Field(..., description="DM: called service")
    rpc_type: str = Field(
        ...,
        description="Communication type: rpc, http, mq, db, mc"
    )
    avg_latency_ms: float = Field(..., description="Baseline average latency")
    current_latency_ms: float = Field(..., description="Current observed latency")


class TimelineEvent(BaseModel):
    """A single event in the incident timeline."""
    step: int
    event_type: str  # alert_fired, metric_spike, service_degraded, etc.
    service: str
    description: str
    metric: Optional[str] = None
    value: Optional[float] = None


class IncidentObservation(BaseModel):
    """
    Full observation returned to the agent each step.

    Contains everything an SRE would see on their monitoring dashboard:
    - Active alerts (what's firing)
    - Service metrics (current state)
    - Topology (who calls whom)
    - Timeline (what happened when)
    - Context (step count, SLA pressure)
    """

    # Episode context
    step: int = Field(..., description="Current episode step (0-indexed)")
    max_steps: int = Field(..., description="Maximum steps for this task")
    task_id: str = Field(..., description="Task identifier: easy/medium/hard")
    episode_id: str = Field(..., description="Unique episode identifier")

    # Core incident data
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts visible to agent (red herrings included)"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current metrics per service {service_name: ServiceMetrics}"
    )
    topology: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Service dependency edges [{upstream, downstream, rpc_type, latency}]"
    )
    timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological incident events"
    )

    # Pressure signal
    time_pressure: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="SLA breach urgency. 0.0=no pressure, 1.0=SLA already breached"
    )
    sla_breach_in_steps: Optional[int] = Field(
        default=None,
        description="Steps remaining before SLA breach (hard task only)"
    )

    # Agent history
    previous_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Actions taken in previous steps with their scores"
    )
    current_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Running reward score so far this episode"
    )

    # Multi-agent debate (Responder ↔ Challenger)
    debate_challenge: Optional[str] = Field(
        default=None,
        description="Adversarial challenge from the Challenger agent."
    )
    debate_phase: Optional[str] = Field(
        default=None,
        description="Current debate phase: initial / challenged / resolved"
    )
    debate_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Full debate transcript for this episode"
    )
    debate_strategy: Optional[str] = Field(
        default=None,
        description="Challenge strategy used: topology_challenge, fault_type_challenge, etc."
    )

    # Multi-agent context
    agent_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Messages from other agents (monitor signals, challenges, etc.)"
    )
    investigation_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Results from 'investigate_further' actions revealing hidden evidence"
    )
    memory_summary: Optional[str] = Field(
        default=None,
        description="Agent's investigation memory summary (for LLM prompt context)"
    )
    hidden_evidence_unlocked: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence revealed by investigation actions"
    )


    # Terminal state
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    info: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 1,
                "max_steps": 10,
                "task_id": "easy",
                "episode_id": "ep_001",
                "alerts": [
                    {
                        "alert_id": "ALT-001",
                        "service": "checkout-ui",
                        "metric": "HTTP_RT",
                        "current_value": 2400.0,
                        "threshold": 500.0,
                        "severity": "critical",
                        "fired_at_step": 0
                    }
                ],
                "metrics": {
                    "checkout-ui": {
                        "cpu_utilization": 0.45,
                        "memory_utilization": 0.52,
                        "http_rt": 2400.0,
                        "is_healthy": False
                    }
                },
                "topology": [
                    {
                        "upstream_service": "checkout-ui",
                        "downstream_service": "payments-api",
                        "rpc_type": "http",
                        "avg_latency_ms": 120.0,
                        "current_latency_ms": 1800.0
                    }
                ],
                "time_pressure": 0.3,
                "done": False,
                "reward": 0.0
            }
        }
    }