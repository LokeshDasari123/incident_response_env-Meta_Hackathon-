"""
envs/incident_env.py
--------------------
Core Incident Response Environment.
Implements reset() / step() / state() — the full OpenEnv interface.
"""

import uuid
from typing import Any, Dict, Optional, Tuple

from envs.base_env      import BaseIncidentEnv
from envs.debate        import DebateEngine
from graders            import load_grader
from models.action      import IncidentAction
from models.observation import IncidentObservation
from models.state       import IncidentState
from scenarios          import load_scenario, BaseScenario
from scenarios.scenario_generator import generate_scenario_variant

# Per-task thresholds: episode ends when agent exceeds this score
# Hard task has NO early termination — must run all steps
DONE_THRESHOLDS = {
    "easy":   0.70,   # ends early on good answer
    "medium": 0.75,   # slightly harder to trigger early end
    "hard":   99.0,   # effectively never ends early — must run all steps
    "expert": 99.0,   # expert never ends early either
}


class IncidentResponseEnv(BaseIncidentEnv):
    """
    Simulates a production incident triage session.

    Episode flow:
        reset(task_id) -> observation
        step(action)   -> observation, reward, done, info
        state()        -> IncidentState
        close()        -> cleanup
    """

    VALID_TASKS = ("easy", "medium", "hard", "expert")

    def __init__(self) -> None:
        self._scenario:      Optional[BaseScenario]  = None
        self._state:         Optional[IncidentState] = None
        self._task_id:       str                     = "easy"
        self._step:          int                     = 0
        self._done:          bool                    = False
        self._session_id:    str                     = str(uuid.uuid4())[:8]
        self._score_history:   list           = []
        self._best_score:      float          = 0.0
        self._last_root_cause: str             = ""
        self._repeat_count:    int             = 0
        self._debate:          Optional[DebateEngine] = None
        self._last_action:     Optional[Dict]  = None
        self._debate_challenge: Optional[Dict] = None

    def reset(
        self,
        task_id: str = "easy",
        *,
        dynamic: bool = True,
        seed: Optional[int] = None,
    ) -> IncidentObservation:
        """
        Reset the environment to initial state.

        Args:
            task_id: Difficulty level ("easy", "medium", "hard")
            dynamic: If True (default), generates a randomized variant
                     with jittered metrics to prevent agent overfitting.
                     If False, uses the static base scenario template.
            seed:    Optional RNG seed for reproducible variants.
                     Only used when dynamic=True.
        """
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS}")

        self._task_id        = task_id
        self._step           = 0
        self._done           = False
        self._score_history   = []
        self._best_score      = 0.0
        self._last_root_cause = ""
        self._repeat_count    = 0
        self._debate          = DebateEngine(seed=seed)
        self._last_action     = None
        self._debate_challenge = None

        # Dynamic variant generation — each reset produces unique metrics
        if dynamic:
            self._scenario = generate_scenario_variant(task_id, seed=seed)
        else:
            self._scenario = load_scenario(task_id)

        episode_id = f"ep_{uuid.uuid4().hex[:6]}"
        self._state = IncidentState(
            episode_id           = episode_id,
            task_id              = task_id,
            session_id           = self._session_id,
            step                 = 0,
            max_steps            = self._scenario.max_steps,
            is_done              = False,
            scenario_name        = self._scenario.name,
            scenario_description = self._scenario.description,
            num_services         = len(self._scenario.services),
            num_alerts           = len(self._scenario.alerts),
            current_score        = 0.0,
            actions_taken        = 0,
        )
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: IncidentAction) -> Tuple[IncidentObservation, float, bool, Dict[str, Any]]:
        if self._scenario is None or self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        self._step            += 1
        self._state.step       = self._step
        self._state.actions_taken += 1

        grader          = load_grader(self._task_id)
        incident_reward = grader.grade(
            action    = action.model_dump(),
            step      = self._step,
            max_steps = self._scenario.max_steps,
        )
        reward    = incident_reward.reward
        breakdown = incident_reward.breakdown

        # ── Repetition penalty ───────────────────────────────────────────────
        # Penalise agent for repeatedly submitting the SAME root_cause_service.
        # Forces genuine exploration. Only applies from the 3rd repeat onward.
        current_root = action.root_cause_service
        if self._last_root_cause == current_root:
            self._repeat_count += 1
        else:
            self._repeat_count = 0
        self._last_root_cause = current_root

        if self._repeat_count >= 2:
            penalty = 0.05 * (self._repeat_count - 1)
            reward  = round(max(0.0, reward - penalty), 4)

        # Track best score across all steps
        if reward > self._best_score:
            self._best_score             = reward
            self._state.current_score    = reward
            self._state.best_action_step = self._step

        self._score_history.append({
            "step":       self._step,
            "score":      reward,
            "action":     action.remediation_action.value,
            "root_cause": action.root_cause_service,
            "feedback":   breakdown.feedback,
        })
        self._state.score_history = self._score_history

        # Episode termination — hard task never ends early
        threshold = DONE_THRESHOLDS.get(self._task_id, 0.70)
        done = (
            reward >= threshold
            or self._step >= self._scenario.max_steps
        )
        self._done          = done
        self._state.is_done = done

        # ── Multi-agent debate ────────────────────────────────────────────────
        # Generate adversarial challenge for next step's observation
        debate_bonus = 0.0
        debate_feedback_text = None
        if self._debate and self._last_action is not None:
            debate_bonus, debate_feedback_text = self._debate.score_debate_improvement(
                prev_action=self._last_action,
                curr_action=action.model_dump(),
                ground_truth=self._scenario.ground_truth,
            )
            reward = round(max(0.0, min(1.0, reward + debate_bonus)), 4)

        # Generate challenge for NEXT step
        if self._debate and not done:
            self._debate_challenge = self._debate.generate_challenge(
                action=action.model_dump(),
                metrics=self._scenario.get_metrics_at_step(self._step, self._scenario.max_steps),
                alerts=self._scenario.get_alerts_at_step(self._step, self._scenario.max_steps),
                topology=self._scenario.get_topology_at_step(self._step, self._scenario.max_steps),
                ground_truth=self._scenario.ground_truth,
                step=self._step,
                max_steps=self._scenario.max_steps,
            )
        else:
            self._debate_challenge = None

        self._last_action = action.model_dump()

        info = {
            "reward_breakdown": breakdown.model_dump(),
            "episode_id":       self._state.episode_id,
            "task_id":          self._task_id,
            "step":             self._step,
            "max_steps":        self._scenario.max_steps,
            "best_score":       self._best_score,
            "debate_bonus":     debate_bonus,
            "debate_feedback":  debate_feedback_text,
            "debate_summary":   self._debate.get_debate_summary() if self._debate else {},
        }

        obs = self._build_observation(
            reward          = reward,
            done            = done,
            previous_action = action.model_dump(),
            breakdown       = breakdown.model_dump(),
        )
        return obs, reward, done, info

    def state(self) -> IncidentState:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    def close(self) -> None:
        self._scenario = None
        self._state    = None
        self._done     = False

    def _build_observation(
        self,
        reward:          float = 0.0,
        done:            bool  = False,
        previous_action: Optional[Dict[str, Any]] = None,
        breakdown:       Optional[Dict[str, Any]] = None,
    ) -> IncidentObservation:
        assert self._scenario is not None
        assert self._state    is not None

        max_s         = self._scenario.max_steps
        sla_step      = self._scenario.sla_breach_step
        time_pressure = 0.0
        sla_breach_in = None
        if sla_step is not None:
            steps_until   = max(0, sla_step - self._step)
            sla_breach_in = steps_until
            time_pressure = min(1.0, self._step / sla_step)

        prev_actions = []
        if previous_action and breakdown:
            prev_actions = [{
                "step":     self._step,
                "action":   previous_action,
                "score":    reward,
                "feedback": breakdown.get("feedback", ""),
            }]
        elif self._score_history:
            prev_actions = self._score_history[-3:]

        # ── Progressive observations ─────────────────────────────────────────
        # Metrics, alerts, topology, and timeline all evolve per step.
        # The incident cascades outward from the root cause through the
        # service dependency graph. Early steps show partial degradation;
        # later steps show full cascade with all services affected.
        alerts   = self._scenario.get_alerts_at_step(self._step, max_s)
        metrics  = self._scenario.get_metrics_at_step(self._step, max_s)
        topology = self._scenario.get_topology_at_step(self._step, max_s)
        timeline = self._scenario.get_timeline_at_step(self._step, max_s)

        return IncidentObservation(
            step                = self._step,
            max_steps           = max_s,
            task_id             = self._task_id,
            episode_id          = self._state.episode_id,
            alerts              = alerts,
            metrics             = metrics,
            topology            = topology,
            timeline            = timeline,
            time_pressure       = time_pressure,
            sla_breach_in_steps = sla_breach_in,
            previous_actions    = prev_actions,
            current_score       = self._best_score,
            # Multi-agent debate fields
            debate_challenge    = self._debate_challenge.get("challenge_text") if self._debate_challenge else None,
            debate_phase        = (
                "challenged" if self._debate_challenge
                else ("initial" if self._step == 0 else "resolved")
            ),
            debate_history      = self._debate.history if self._debate else [],
            debate_strategy     = self._debate_challenge.get("strategy") if self._debate_challenge else None,
            done                = done,
            reward              = reward,
            info                = {
                "scenario_name":    self._scenario.name,
                "fault_type":       self._scenario.fault_type,
                "num_services":     len(self._scenario.services),
                "num_alerts":       len(alerts),
                "total_alerts":     len(self._scenario.alerts),
                "best_score":       self._best_score,
                "cascade_progress": round(self._step / max_s, 3) if max_s > 0 else 0.0,
            },
        )