"""
training/experiment_logger.py
-----------------------------
Rich structured logging for training experiments.
Outputs JSONL step logs + summary JSON for frontend visualization.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """
    Structured logging for training experiments.

    Outputs:
    - JSONL step log: one line per step with full details
    - Summary JSON: live-updated dashboard data
    - Reward curves JSON: per-task reward history for charting
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.log_file     = self.log_dir / f"training_{ts}.jsonl"
        self.summary_file = self.log_dir / "latest_summary.json"
        self.curves_file  = self.log_dir / "reward_curves.json"

        self._file_handle = open(self.log_file, "w")

        # Accumulators
        self._all_rewards:    Dict[str, List[float]] = {}
        self._all_rc_scores:  Dict[str, List[float]] = {}
        self._all_strategies: Dict[str, List[str]]   = {}
        self._challenger_wins = 0
        self._start_time     = datetime.utcnow()

    def log_step(self, step_data: Dict[str, Any]) -> None:
        """Log a single training step."""
        step_data["timestamp"] = datetime.utcnow().isoformat()
        self._file_handle.write(json.dumps(step_data) + "\n")
        self._file_handle.flush()

    def log_episode(
        self,
        episode: int,
        task_id: str,
        summary: Dict[str, Any],
        step_logs: List[Dict[str, Any]],
        curriculum_state: Optional[Dict[str, Any]] = None,
        evaluation: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a complete episode and update summary files."""
        # Write step logs
        for sl in step_logs:
            sl["episode"] = episode
            self.log_step(sl)

        # Accumulate metrics
        self._all_rewards.setdefault(task_id, []).append(
            summary.get("best_reward", 0.0)
        )

        # Extract RC score from step logs
        rc_scores = [
            sl.get("action", {}).get("root_cause_service", "") == sl.get("ground_truth_rc", "")
            for sl in step_logs
        ]
        avg_rc = sum(rc_scores) / max(1, len(rc_scores))
        self._all_rc_scores.setdefault(task_id, []).append(round(avg_rc, 4))

        # Track strategy
        if evaluation:
            strategy = evaluation.get("strategy_detected", {}).get("primary", "unknown")
            self._all_strategies.setdefault(task_id, []).append(strategy)

        # Multi-agent stats
        self._challenger_wins += summary.get("challenger_wins", 0)

        # Update summary file
        self._write_summary(episode, curriculum_state, evaluation)

        # Update reward curves
        self._write_curves()

    def _write_summary(
        self,
        episode: int,
        curriculum_state: Optional[Dict] = None,
        evaluation: Optional[Dict] = None,
    ) -> None:
        elapsed = (datetime.utcnow() - self._start_time).total_seconds()

        summary = {
            "episode":       episode,
            "elapsed_s":     round(elapsed, 1),
            "per_task":      {},
            "curriculum":    curriculum_state,
            "evaluation":    evaluation,
            "challenger_wins_total": self._challenger_wins,
            "updated_at":    datetime.utcnow().isoformat(),
            "running":       True,
        }

        for task_id, rewards in self._all_rewards.items():
            summary["per_task"][task_id] = {
                "rewards":     rewards[-100:],
                "rc_scores":   self._all_rc_scores.get(task_id, [])[-100:],
                "avg_last10":  self._avg_last(rewards, 10),
                "avg_last50":  self._avg_last(rewards, 50),
                "best":        round(max(rewards), 4) if rewards else 0.0,
                "count":       len(rewards),
                "trend":       self._trend(rewards),
                "strategies":  self._strategy_distribution(task_id),
            }

        self.summary_file.write_text(json.dumps(summary, indent=2))

    def _write_curves(self) -> None:
        curves = {}
        for task_id, rewards in self._all_rewards.items():
            smoothed = self._rolling_avg(rewards, 10)
            curves[task_id] = {
                "raw":       rewards,
                "smoothed":  smoothed,
                "rc_scores": self._all_rc_scores.get(task_id, []),
                "episodes":  list(range(len(rewards))),
            }
        self.curves_file.write_text(json.dumps(curves, indent=2))

    def finalize(self) -> None:
        """Mark training as complete."""
        if self.summary_file.exists():
            summary = json.loads(self.summary_file.read_text())
            summary["running"] = False
            self.summary_file.write_text(json.dumps(summary, indent=2))
        self._file_handle.close()

    def _strategy_distribution(self, task_id: str) -> Dict[str, int]:
        strategies = self._all_strategies.get(task_id, [])
        dist: Dict[str, int] = {}
        for s in strategies:
            dist[s] = dist.get(s, 0) + 1
        return dist

    @staticmethod
    def _avg_last(lst: List[float], n: int) -> float:
        if not lst:
            return 0.0
        chunk = lst[-n:]
        return round(sum(chunk) / len(chunk), 4)

    @staticmethod
    def _rolling_avg(lst: List[float], window: int = 10) -> List[float]:
        result = []
        for i in range(len(lst)):
            chunk = lst[max(0, i - window + 1): i + 1]
            result.append(round(sum(chunk) / len(chunk), 4))
        return result

    @staticmethod
    def _trend(lst: List[float]) -> str:
        if len(lst) < 5:
            return "insufficient_data"
        recent = sum(lst[-5:]) / 5
        older  = sum(lst[-20:-5]) / max(1, len(lst[-20:-5]))
        diff   = recent - older
        if diff > 0.02:  return "improving"
        if diff < -0.02: return "declining"
        return "stable"
