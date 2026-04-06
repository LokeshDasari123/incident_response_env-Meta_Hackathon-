"""
models/state.py
---------------
Typed State model for the Incident Response Environment.
Returned by state() endpoint — full episode metadata.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IncidentState(BaseModel):
    """
    Complete episode state returned by GET /state.
    Contains metadata about the current episode,
    scenario configuration, and scoring progress.
    """

    # Episode identity
    episode_id: str = Field(..., description="Unique episode identifier")
    task_id: str = Field(..., description="Task: easy/medium/hard")
    session_id: str = Field(..., description="WebSocket session identifier")

    # Episode progress
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(..., description="Maximum allowed steps")
    is_done: bool = Field(default=False, description="Whether episode is complete")
    started_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Episode start timestamp (UTC ISO format)"
    )

    # Scenario info (visible to agent)
    scenario_name: str = Field(..., description="Human-readable scenario name")
    scenario_description: str = Field(..., description="Scenario description")
    num_services: int = Field(..., description="Number of services in topology")
    num_alerts: int = Field(..., description="Number of active alerts")

    # Scoring progress
    current_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Best score achieved so far this episode"
    )
    best_action_step: Optional[int] = Field(
        default=None,
        description="Step number where best score was achieved"
    )
    actions_taken: int = Field(
        default=0,
        description="Number of actions submitted so far"
    )

    # Score history per step
    score_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-step score breakdown history"
    )

    # Internal grader state (not exposed to agent via API)
    # These are used server-side only
    _ground_truth: Dict[str, Any] = {}
    _red_herrings: List[str] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "episode_id": "ep_a1b2c3",
                "task_id": "easy",
                "session_id": "sess_x9y8z7",
                "step": 2,
                "max_steps": 10,
                "is_done": False,
                "started_at": "2025-01-01T02:00:00",
                "scenario_name": "Change-Induced Single Service Failure",
                "scenario_description": "Bad ConfigMap update causes payments cascade",
                "num_services": 3,
                "num_alerts": 5,
                "current_score": 0.35,
                "best_action_step": 1,
                "actions_taken": 2,
                "score_history": [
                    {"step": 0, "score": 0.0, "action": "investigate_further"},
                    {"step": 1, "score": 0.35, "action": "fix_config"}
                ]
            }
        }
    }