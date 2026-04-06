"""
models/action.py
----------------
Typed Action model for the Incident Response Environment.
The agent submits this to step() after analyzing the observation.
"""

from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class RootCauseType(str, Enum):
    """AIOpsLab-derived fault taxonomy."""
    MISCONFIGURATION    = "misconfiguration"
    MEMORY_LEAK         = "memory_leak"
    NETWORK_PARTITION   = "network_partition"
    CRASH_LOOP          = "crash_loop"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    AUTH_FAILURE        = "auth_failure"
    DEPENDENCY_FAILURE  = "dependency_failure"
    UNKNOWN             = "unknown"

    @classmethod
    def _missing_(cls, value):
        """Fall back to UNKNOWN for any unrecognised value from LLM."""
        return cls.UNKNOWN


class SeverityLevel(str, Enum):
    """Google SRE severity taxonomy."""
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"

    @classmethod
    def _missing_(cls, value):
        """Fall back to P2 for any unrecognised severity."""
        if isinstance(value, str):
            v = value.upper()
            for member in cls:
                if member.value == v:
                    return member
        return cls.P2


class RemediationAction(str, Enum):
    """Valid remediation actions the agent can prescribe."""
    ROLLBACK                 = "rollback"
    RESTART_SERVICE          = "restart_service"
    SCALE_UP                 = "scale_up"
    FIX_CONFIG               = "fix_config"
    INCREASE_CONNECTION_POOL = "increase_connection_pool"
    FLUSH_CACHE              = "flush_cache"
    REROUTE_TRAFFIC          = "reroute_traffic"
    ESCALATE                 = "escalate"
    INVESTIGATE_FURTHER      = "investigate_further"

    @classmethod
    def _missing_(cls, value):
        """Fall back to INVESTIGATE_FURTHER for unknown actions."""
        return cls.INVESTIGATE_FURTHER


class IncidentAction(BaseModel):
    """
    Structured action submitted by the agent each step.

    Modeled after Google SRE incident management process:
    - Identify root cause (Ch 12: Effective Troubleshooting)
    - Classify severity (Ch 14: Managing Incidents)
    - Prescribe remediation (Ch 13: Emergency Response)
    - Communicate to stakeholders (Ch 14: Managing Incidents)
    """

    # Core diagnosis
    root_cause_service: str = Field(
        ...,
        description="Name of the service identified as root cause",
        examples=["payments-db", "auth-service", "api-gateway"]
    )
    root_cause_type: RootCauseType = Field(
        ...,
        description="Type of fault causing the incident"
    )

    # Impact assessment
    severity: SeverityLevel = Field(
        ...,
        description="Incident severity: P0=revenue impact, P1=user-facing, P2=partial, P3=minor"
    )
    affected_services: List[str] = Field(
        default_factory=list,
        description="All services impacted by this incident",
        examples=[["payments-api", "checkout-ui", "order-service"]]
    )

    # Remediation
    remediation_action: RemediationAction = Field(
        ...,
        description="Prescribed remediation action"
    )

    # Communication (required for P0/P1)
    stakeholder_message: Optional[str] = Field(
        default=None,
        description="Stakeholder update message. Required for P0 and P1 incidents.",
        max_length=500
    )

    # Meta
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in this diagnosis (0.0-1.0)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning chain (used for partial credit scoring)",
        max_length=1000
    )

    @field_validator("affected_services")
    @classmethod
    def validate_affected_services(cls, v: List[str]) -> List[str]:
        """Ensure no duplicate services and all are non-empty strings."""
        cleaned = [s.strip() for s in v if s.strip()]
        return list(dict.fromkeys(cleaned))  # deduplicate preserving order

    @field_validator("stakeholder_message")
    @classmethod
    def validate_stakeholder_message(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.strip() if v.strip() else None
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "root_cause_service": "payments-db",
                "root_cause_type": "misconfiguration",
                "severity": "P0",
                "affected_services": [
                    "payments-db",
                    "payments-api",
                    "checkout-ui"
                ],
                "remediation_action": "fix_config",
                "stakeholder_message": (
                    "Investigating payment processing delays caused by "
                    "misconfigured connection pool. ETA to resolution: 8 minutes."
                ),
                "confidence": 0.87,
                "reasoning": (
                    "HTTP_RT spike at checkout-ui traced via call graph to "
                    "payments-api (consumerRPC_RT elevated), then to payments-db "
                    "where ConfigMap shows wrong replica count."
                )
            }
        }
    }