"""
training/curriculum.py
----------------------
Performance-Adaptive Curriculum Controller.

Automatically adjusts difficulty based on agent performance:
- Promotes to harder difficulty when agent masters current level
- Scales noise, adversary budget, fault injector aggression
- Tracks curriculum state for visualization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CurriculumState:
    """Current state of the curriculum — serializable for UI."""
    difficulty:          str   = "easy"
    difficulty_index:    int   = 0
    noise_multiplier:    float = 1.0
    simultaneous_faults: int   = 1
    adversary_budget:    int   = 0
    adversary_cunning:   float = 0.2
    fault_budget:        int   = 1
    fault_aggression:    float = 0.3
    monitor_reliability: float = 0.90
    monitor_noise:       float = 0.05
    evidence_delay:      int   = 0
    red_herring_count:   int   = 0
    episode:             int   = 0
    total_promotions:    int   = 0
    total_demotions:     int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CurriculumController:
    """
    Manages adaptive difficulty progression.

    Promotion:  avg reward > promote_threshold for promote_window episodes
    Demotion:   avg reward < demote_threshold for demote_window episodes
    Intra-level: within each difficulty, gradually increase noise/adversary

    The controller exports its state for the frontend dashboard.
    """

    LEVELS = ["easy", "medium", "hard"]

    # Per-difficulty scaling ranges
    SCALING = {
        "easy": {
            "noise_range":        (1.0, 1.5),
            "adversary_budget":   (0, 1),
            "adversary_cunning":  (0.1, 0.3),
            "fault_budget":       (0, 2),
            "fault_aggression":   (0.1, 0.3),
            "monitor_reliability": (0.95, 0.85),
            "monitor_noise":      (0.02, 0.10),
            "red_herring_count":  (0, 1),
        },
        "medium": {
            "noise_range":        (1.2, 2.0),
            "adversary_budget":   (1, 3),
            "adversary_cunning":  (0.2, 0.5),
            "fault_budget":       (1, 3),
            "fault_aggression":   (0.3, 0.6),
            "monitor_reliability": (0.90, 0.75),
            "monitor_noise":      (0.05, 0.15),
            "red_herring_count":  (1, 2),
        },
        "hard": {
            "noise_range":        (1.5, 3.0),
            "adversary_budget":   (2, 5),
            "adversary_cunning":  (0.4, 0.8),
            "fault_budget":       (2, 5),
            "fault_aggression":   (0.5, 0.8),
            "monitor_reliability": (0.80, 0.60),
            "monitor_noise":      (0.10, 0.25),
            "red_herring_count":  (2, 4),
        },
    }

    def __init__(
        self,
        promote_threshold: float = 0.65,
        demote_threshold:  float = 0.25,
        promote_window:    int   = 10,
        demote_window:     int   = 8,
        intra_scaling_window: int = 20,
    ) -> None:
        self.promote_threshold    = promote_threshold
        self.demote_threshold     = demote_threshold
        self.promote_window       = promote_window
        self.demote_window        = demote_window
        self.intra_scaling_window = intra_scaling_window

        # State
        self._state = CurriculumState()
        self._reward_history: Dict[str, List[float]] = {
            "easy": [], "medium": [], "hard": [],
        }
        self._transition_log: List[Dict[str, Any]] = []
        self._intra_progress: float = 0.0   # 0-1 within current difficulty

    @property
    def state(self) -> CurriculumState:
        return self._state

    @property
    def current_difficulty(self) -> str:
        return self._state.difficulty

    def record_reward(self, reward: float, episode: int) -> None:
        """Record an episode reward and check for promotion/demotion."""
        diff = self._state.difficulty
        self._reward_history[diff].append(reward)
        self._state.episode = episode

        # Check promotion
        if self._should_promote():
            self._promote(episode)

        # Check demotion
        elif self._should_demote():
            self._demote(episode)

        # Update intra-level scaling
        self._update_intra_scaling()

    def get_env_params(self) -> Dict[str, Any]:
        """
        Get environment parameters for the current curriculum state.
        Used by the training loop to configure MultiAgentIncidentEnv.
        """
        s = self._state
        return {
            "task_id":             s.difficulty,
            "monitor_reliability": s.monitor_reliability,
            "monitor_noise":       s.monitor_noise,
            "fault_budget":        s.fault_budget,
            "fault_aggression":    s.fault_aggression,
            "adversary_budget":    s.adversary_budget,
            "adversary_cunning":   s.adversary_cunning,
        }

    def get_responder_skill(self, episode: int, total_episodes: int) -> float:
        """
        Compute responder skill level based on curriculum progress.
        Skill improves within each difficulty level.
        """
        import math
        # Base skill from episode progress
        progress = episode / max(1, total_episodes)
        raw = 1.0 / (1.0 + math.exp(-10 * (progress - 0.40)))

        # Difficulty ceiling
        ceilings = {"easy": 0.90, "medium": 0.78, "hard": 0.62}
        ceiling = ceilings.get(self._state.difficulty, 0.75)

        # Intra-level bonus
        intra_bonus = self._intra_progress * 0.1

        return min(0.97, raw * ceiling + intra_bonus)

    def get_transition_log(self) -> List[Dict[str, Any]]:
        return self._transition_log

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Full state for frontend dashboard."""
        return {
            "state":          self._state.to_dict(),
            "reward_history": {
                k: v[-50:] for k, v in self._reward_history.items()
            },
            "transitions":    self._transition_log[-20:],
            "intra_progress": round(self._intra_progress, 3),
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _should_promote(self) -> bool:
        diff = self._state.difficulty
        idx  = self.LEVELS.index(diff)
        if idx >= len(self.LEVELS) - 1:
            return False  # Already at max
        history = self._reward_history[diff]
        if len(history) < self.promote_window:
            return False
        avg = sum(history[-self.promote_window:]) / self.promote_window
        return avg > self.promote_threshold

    def _should_demote(self) -> bool:
        diff = self._state.difficulty
        idx  = self.LEVELS.index(diff)
        if idx <= 0:
            return False  # Already at min
        history = self._reward_history[diff]
        if len(history) < self.demote_window:
            return False
        avg = sum(history[-self.demote_window:]) / self.demote_window
        return avg < self.demote_threshold

    def _promote(self, episode: int) -> None:
        idx = self.LEVELS.index(self._state.difficulty)
        new_diff = self.LEVELS[idx + 1]
        self._transition_log.append({
            "type":     "promotion",
            "from":     self._state.difficulty,
            "to":       new_diff,
            "episode":  episode,
            "avg_reward": round(
                sum(self._reward_history[self._state.difficulty][-self.promote_window:])
                / self.promote_window, 3
            ),
        })
        self._state.difficulty       = new_diff
        self._state.difficulty_index = idx + 1
        self._state.total_promotions += 1
        self._intra_progress = 0.0
        self._update_intra_scaling()

    def _demote(self, episode: int) -> None:
        idx = self.LEVELS.index(self._state.difficulty)
        new_diff = self.LEVELS[idx - 1]
        self._transition_log.append({
            "type":     "demotion",
            "from":     self._state.difficulty,
            "to":       new_diff,
            "episode":  episode,
            "avg_reward": round(
                sum(self._reward_history[self._state.difficulty][-self.demote_window:])
                / self.demote_window, 3
            ),
        })
        self._state.difficulty       = new_diff
        self._state.difficulty_index = idx - 1
        self._state.total_demotions += 1
        self._intra_progress = 0.5  # Start mid-level after demotion
        self._update_intra_scaling()

    def _update_intra_scaling(self) -> None:
        """
        Gradually increase difficulty WITHIN the current level
        based on recent performance.
        """
        diff    = self._state.difficulty
        history = self._reward_history[diff]

        # Intra-progress: 0 (start of level) → 1 (ready to promote)
        if len(history) >= self.intra_scaling_window:
            recent_avg = sum(history[-self.intra_scaling_window:]) / self.intra_scaling_window
            # Map reward to progress (0.2 → 0.0, 0.65 → 1.0)
            self._intra_progress = max(0.0, min(1.0,
                (recent_avg - 0.2) / (self.promote_threshold - 0.2)
            ))
        elif history:
            self._intra_progress = max(0.0, min(0.5,
                sum(history) / len(history) / self.promote_threshold
            ))

        # Interpolate scaling parameters
        scaling = self.SCALING[diff]
        p = self._intra_progress

        def lerp(key):
            lo, hi = scaling[key]
            return round(lo + (hi - lo) * p, 3)

        self._state.noise_multiplier    = lerp("noise_range")
        self._state.adversary_budget    = int(lerp("adversary_budget"))
        self._state.adversary_cunning   = lerp("adversary_cunning")
        self._state.fault_budget        = int(lerp("fault_budget"))
        self._state.fault_aggression    = lerp("fault_aggression")
        self._state.monitor_reliability = lerp("monitor_reliability")
        self._state.monitor_noise       = lerp("monitor_noise")
        self._state.red_herring_count   = int(lerp("red_herring_count"))
