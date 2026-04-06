"""
models/reward.py
----------------
Typed Reward model for the Incident Response Environment.
Provides granular partial credit signal — not just binary win/lose.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class RewardBreakdown(BaseModel):
    """
    Granular reward breakdown so agents can learn WHAT to improve.
    Each component maps to a real SRE skill being evaluated.
    """

    # Core components (sum = raw score before penalties)
    root_cause_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Correct root cause identified. Weight: 0.35"
    )
    action_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Correct remediation action. Weight: 0.25"
    )
    severity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Correct P0/P1/P2/P3 severity. Weight: 0.20"
    )
    communication_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Stakeholder message quality. Weight: 0.10"
    )
    speed_bonus: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Resolved within time budget. Weight: 0.10"
    )

    # Penalties (subtracted from raw score)
    false_positive_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Penalty for flagging wrong service as root cause"
    )
    wrong_action_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Penalty for prescribing destructive/wrong action"
    )
    missed_escalation_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Penalty for not escalating a P0 incident"
    )

    # Computed totals
    raw_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted sum before penalties"
    )
    total_penalty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Total penalties applied"
    )
    final_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Final clipped score: max(0, raw_score - penalties)"
    )

    # Feedback for learning
    feedback: Optional[str] = Field(
        default=None,
        description="Human-readable feedback explaining the score"
    )
    partial_credits: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-component feedback {component: explanation}"
    )

    def compute(self) -> "RewardBreakdown":
        """
        Compute raw_score, total_penalty, and final_score
        from component scores. Called after all components are set.
        """
        # Weighted sum
        self.raw_score = round(
            self.root_cause_score  * 0.35 +
            self.action_score      * 0.25 +
            self.severity_score    * 0.20 +
            self.communication_score * 0.10 +
            self.speed_bonus       * 0.10,
            4
        )

        # Total penalties
        self.total_penalty = round(
            self.false_positive_penalty  * 0.15 +
            self.wrong_action_penalty    * 0.20 +
            self.missed_escalation_penalty * 0.25,
            4
        )

        # Final score clipped to [0, 1]
        self.final_score = round(
            max(0.0, min(1.0, self.raw_score - self.total_penalty)),
            4
        )

        return self


class IncidentReward(BaseModel):
    """
    Top-level reward object returned by the grader.
    Contains the scalar reward + full breakdown for transparency.
    """
    reward: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Scalar reward for this step (0.0-1.0)"
    )
    breakdown: RewardBreakdown = Field(
        ...,
        description="Granular component breakdown"
    )
    is_terminal: bool = Field(
        default=False,
        description="Whether this reward ends the episode"
    )
    step: int = Field(..., description="Episode step this reward is for")
    task_id: str = Field(..., description="Task: easy/medium/hard")

    model_config = {
        "json_schema_extra": {
            "example": {
                "reward": 0.72,
                "breakdown": {
                    "root_cause_score": 1.0,
                    "action_score": 0.8,
                    "severity_score": 1.0,
                    "communication_score": 0.6,
                    "speed_bonus": 0.5,
                    "false_positive_penalty": 0.0,
                    "wrong_action_penalty": 0.0,
                    "missed_escalation_penalty": 0.0,
                    "raw_score": 0.87,
                    "total_penalty": 0.0,
                    "final_score": 0.72,
                    "feedback": "Correct root cause and severity. Action appropriate. Stakeholder message lacks ETA.",
                    "partial_credits": {
                        "root_cause": "✅ Correctly identified payments-db misconfiguration",
                        "action": "✅ fix_config is correct. rollback would also score 0.6",
                        "severity": "✅ P0 correct — revenue impact confirmed",
                        "communication": "⚠️ Message present but missing resolution ETA"
                    }
                },
                "is_terminal": True,
                "step": 3,
                "task_id": "easy"
            }
        }
    }