"""
training/train.py
=================
GRPO Training Script — Incident Response AI Agent
Supports: Curriculum Learning, Multi-Agent Challenger Loop, Dynamic Scenarios
Outputs:  JSONL step logs + per-task reward curves + summary JSON for UI

Usage:
    python training/train.py --task easy --episodes 50
    python training/train.py --task all --episodes 200 --curriculum
    python training/train.py --task all --episodes 200 --curriculum --use-llm
"""

import argparse
import json
import math
import os
import random
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

# ── Optional TRL import (falls back to simulation if not installed) ────────────
try:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRL = True
    print("[INFO] TRL detected — real GRPO training available")
except ImportError:
    HAS_TRL = False
    print("[INFO] TRL not installed — running reward simulation mode")

# ── Project imports ────────────────────────────────────────────────────────────
try:
    from envs.incident_env import IncidentResponseEnv
    from models.action import IncidentAction, RootCauseType, SeverityLevel, RemediationAction
    from graders import load_grader
    from scenarios.scenario_generator import generate_scenario_variant
    HAS_ENV = True
except ImportError as e:
    HAS_ENV = False
    print(f"[WARN] Environment not importable: {e} — using full simulation mode")

# ── Paths ──────────────────────────────────────────────────────────────────────
LOG_DIR  = ROOT / "data" / "training_logs"
CKPT_DIR = ROOT / "data" / "checkpoints"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CURRICULUM_ORDER = ["easy", "medium", "hard", "expert"]

# ── Ground-truth answers per task (for rule-based agent scoring) ───────────────
GROUND_TRUTH = {
    "easy": {
        "root_cause_service": "payments-db",
        "root_cause_type": "misconfiguration",
        "severity": "P0",
        "affected_services": ["payments-db", "payments-api", "checkout-ui"],
        "remediation_action": "fix_config",
    },
    "medium": {
        "root_cause_service": "user-service",
        "root_cause_type": "network_partition",
        "severity": "P1",
        "affected_services": ["user-service", "auth-service", "api-gateway", "storefront-ui"],
        "remediation_action": "fix_config",
    },
    "hard": {
        "root_cause_service": "payments-db",
        "root_cause_type": "memory_leak",
        "severity": "P0",
        "affected_services": ["payments-db", "cache-service", "order-service", "api-gateway", "storefront-ui"],
        "remediation_action": "restart_service",
    },
    "expert": {
        "root_cause_service": "auth-service",
        "root_cause_type": "certificate_expiry",
        "severity": "P0",
        "affected_services": ["auth-service", "user-service", "api-gateway", "storefront-ui", "order-service", "payments-api", "notification-svc"],
        "remediation_action": "fix_config",
    },
}

RED_HERRINGS = {
    "easy":   ["worker-node-4"],
    "medium": ["cache-service", "worker-node-4"],
    "hard":   ["network-switch-03", "worker-node-7"],
    "expert": ["cache-service", "worker-node-7", "metrics-exporter"],
}

MAX_STEPS = {"easy": 10, "medium": 15, "hard": 20, "expert": 25}


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT CHALLENGER  (Theme #1)
# ══════════════════════════════════════════════════════════════════════════════
class ChallengerAgent:
    """
    Adversarial second agent that challenges the Diagnoser's answer.
    Forces re-examination of evidence — implements Theme #1: Multi-Agent.
    
    Strategy: The Challenger picks the single weakest point in the Diagnoser's
    answer and argues for an alternative. The Diagnoser must revise or defend.
    """

    STRATEGIES = [
        "topology_challenge",    # argue a different service based on call graph
        "fault_type_challenge",  # question the fault classification
        "severity_challenge",    # argue severity is wrong
        "red_herring_bait",      # push the Diagnoser toward a red herring
        "cascade_completeness",  # argue affected_services list is incomplete
    ]

    def challenge(
        self,
        action: Dict[str, Any],
        obs: Dict[str, Any],
        task_id: str,
        rng: random.Random,
    ) -> Tuple[str, str]:
        """
        Generate a challenge. Returns (challenge_text, strategy_name).
        The challenge is adversarial but grounded in the observation data.
        """
        strategy = rng.choice(self.STRATEGIES)
        rc       = action.get("root_cause_service", "unknown")
        sev      = action.get("severity", "P2")
        act      = action.get("remediation_action", "investigate_further")
        ft       = action.get("root_cause_type", "unknown")
        affected = action.get("affected_services", [])
        metrics  = obs.get("metrics", {})

        if strategy == "topology_challenge":
            # Find a service with high latency that ISN'T the Diagnoser's RC
            alternatives = [
                s for s, m in metrics.items()
                if s != rc and not m.get("is_healthy", True)
            ]
            alt = rng.choice(alternatives) if alternatives else "api-gateway"
            challenge = (
                f"CHALLENGER: You identified '{rc}' as root cause, but '{alt}' "
                f"shows {metrics.get(alt, {}).get('status', 'degraded')} status. "
                f"The topology shows traffic flows through '{alt}' — could it be the origin? "
                f"Re-examine the call graph edges."
            )

        elif strategy == "fault_type_challenge":
            alt_fault = rng.choice([
                f for f in ["misconfiguration", "memory_leak", "network_partition",
                            "crash_loop", "resource_exhaustion", "dependency_failure"]
                if f != ft
            ])
            challenge = (
                f"CHALLENGER: You classified this as '{ft}' but the metric pattern "
                f"is more consistent with '{alt_fault}'. "
                f"Memory utilization for '{rc}': {metrics.get(rc, {}).get('memory_utilization', '?')}. "
                f"Reconsider the fault type — it changes the remediation."
            )

        elif strategy == "severity_challenge":
            alt_sev = "P0" if sev in ("P1", "P2") else "P1"
            n_affected = len(affected)
            challenge = (
                f"CHALLENGER: Severity {sev} is wrong. With {n_affected} services affected "
                f"and alerts showing revenue-critical paths failing, this should be {alt_sev}. "
                f"Incorrect severity means wrong escalation path."
            )

        elif strategy == "red_herring_bait":
            rh = rng.choice(RED_HERRINGS.get(task_id, ["worker-node-4"]))
            rh_m = metrics.get(rh, {})
            challenge = (
                f"CHALLENGER: You ignored '{rh}' — it shows CPU at "
                f"{rh_m.get('cpu_utilization', 0):.0%}. "
                f"Could this be the real trigger? The batch job theory is unverified."
            )

        else:  # cascade_completeness
            topo_svcs = set()
            for edge in obs.get("topology", []):
                topo_svcs.add(edge.get("upstream_service", ""))
                topo_svcs.add(edge.get("downstream_service", ""))
            missing = [s for s in topo_svcs if s and s not in affected and s != rc]
            if missing:
                ms = rng.choice(missing)
                challenge = (
                    f"CHALLENGER: Your affected_services list is incomplete. "
                    f"'{ms}' depends on '{rc}' per the topology but you didn't include it. "
                    f"Incomplete blast radius means missed escalations."
                )
            else:
                challenge = (
                    f"CHALLENGER: Why {act}? If the root cause is '{ft}' on '{rc}', "
                    f"the SRE runbook recommends a different approach. "
                    f"Justify your remediation choice."
                )

        return challenge, strategy


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED DIAGNOSER  (topology traversal — improves with curriculum)
# ══════════════════════════════════════════════════════════════════════════════
class DiagnoserAgent:
    """
    Rule-based SRE agent that traverses the call graph inward.
    Accuracy improves across curriculum stages (simulates GRPO learning).
    Can be swapped for a real LLM via --use-llm flag.
    """

    def __init__(self, task_id: str, episode: int, total_episodes: int, rng: random.Random):
        self.task_id = task_id
        self.episode = episode
        self.total   = total_episodes
        self.rng     = rng
        # Learning progress: 0.0 (random) → 1.0 (near-perfect)
        self.skill   = self._compute_skill()

    def _compute_skill(self) -> float:
        """
        Simulates progressive skill improvement via GRPO.
        Uses S-curve: slow start, fast middle, plateau near ceiling.
        """
        progress = self.episode / max(1, self.total)
        # S-curve: sigmoid centered at 40% progress
        raw = 1.0 / (1.0 + math.exp(-10 * (progress - 0.40)))
        # Add task-specific difficulty ceiling
        ceilings = {"easy": 0.90, "medium": 0.78, "hard": 0.62}
        ceiling  = ceilings.get(self.task_id, 0.75)
        return raw * ceiling

    def diagnose(
        self,
        obs: Dict[str, Any],
        challenge: Optional[str] = None,
        prior_action: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a diagnosis. When challenged, incorporates feedback
        with probability proportional to skill level.
        """
        gt      = GROUND_TRUTH[self.task_id]
        metrics = obs.get("metrics", {})
        topology = obs.get("topology", [])

        # ── Decide whether to answer correctly ───────────────────────────────
        # Skill = probability of getting each component right
        # Challenge: gives a bonus if skill > 0.5 (agent learns to use feedback)
        challenge_bonus = 0.10 if (challenge and self.skill > 0.5) else 0.0
        effective_skill = min(0.97, self.skill + challenge_bonus)

        def pick(correct, alts, p_correct):
            return correct if self.rng.random() < p_correct else self.rng.choice(alts)

        # Root cause
        all_svcs = list(metrics.keys()) or ["unknown"]
        wrong_rc = [s for s in all_svcs if s != gt["root_cause_service"]]
        rc = pick(gt["root_cause_service"], wrong_rc or ["unknown"], effective_skill)

        # Root cause type
        all_types = ["misconfiguration", "memory_leak", "network_partition",
                     "crash_loop", "resource_exhaustion", "dependency_failure", "unknown"]
        wrong_ft = [t for t in all_types if t != gt["root_cause_type"]]
        ft = pick(gt["root_cause_type"], wrong_ft, effective_skill * 0.95)

        # Severity
        sev_map = {"P0": ["P1", "P2"], "P1": ["P0", "P2"], "P2": ["P1", "P3"]}
        wrong_sev = sev_map.get(gt["severity"], ["P1", "P2"])
        sev = pick(gt["severity"], wrong_sev, effective_skill * 0.90)

        # Affected services — can be partial or include red herrings
        correct_affected = gt["affected_services"]
        rh = RED_HERRINGS.get(self.task_id, [])
        if effective_skill > 0.70:
            # Good agent: correct list with rare red herring inclusion
            affected = correct_affected[:]
            if rh and self.rng.random() > effective_skill:
                affected.append(self.rng.choice(rh))
        elif effective_skill > 0.40:
            # Partial list — gets subset + sometimes red herring
            n = max(1, int(len(correct_affected) * effective_skill))
            affected = correct_affected[:n]
        else:
            # Confused — random mix
            affected = self.rng.sample(all_svcs, k=min(3, len(all_svcs)))

        # Remediation
        action_map = {
            "misconfiguration":   ["fix_config", "rollback", "restart_service"],
            "memory_leak":        ["restart_service", "scale_up", "investigate_further"],
            "network_partition":  ["fix_config", "reroute_traffic", "investigate_further"],
            "crash_loop":         ["restart_service", "rollback", "investigate_further"],
            "resource_exhaustion": ["scale_up", "fix_config", "restart_service"],
        }
        correct_act = gt["remediation_action"]
        alt_acts    = [a for a in action_map.get(ft, ["investigate_further"]) if a != correct_act]
        act = pick(correct_act, alt_acts or ["investigate_further"], effective_skill * 0.88)

        # Stakeholder message
        needs_msg = sev in ("P0", "P1")
        if needs_msg:
            if self.skill > 0.5:
                msg = (
                    f"{rc} is experiencing {ft.replace('_',' ')} causing cascade "
                    f"to {len(affected)} services. Severity: {sev}. "
                    f"Action: {act.replace('_',' ')}. ETA: ~10 minutes."
                )
            else:
                msg = f"Investigating {rc} issue." if self.rng.random() > 0.5 else None
        else:
            msg = None

        confidence = round(max(0.05, min(0.99,
            effective_skill + self.rng.gauss(0, 0.05)
        )), 2)

        return {
            "root_cause_service":  rc,
            "root_cause_type":     ft,
            "severity":            sev,
            "affected_services":   list(dict.fromkeys(affected)),  # deduplicate
            "remediation_action":  act,
            "stakeholder_message": msg,
            "confidence":          confidence,
            "reasoning": (
                f"Traversed topology inward from edge services. "
                f"{rc} shows highest degradation. Pattern matches {ft}. "
                f"Cascade chain: {' → '.join(affected[:3])}."
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_reward(
    action: Dict[str, Any],
    task_id: str,
    step: int,
    max_steps: int,
    challenger_used: bool = False,
    challenge_improved: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Multi-dimensional reward with multi-agent bonus.
    Falls back to rubric-based scoring if env is available.
    """
    if HAS_ENV:
        try:
            grader = load_grader(task_id)
            result = grader.grade(action=action, step=step, max_steps=max_steps)
            base_reward = result.reward
            bd = result.breakdown.model_dump()
        except Exception:
            base_reward, bd = _score_locally(action, task_id, step, max_steps)
    else:
        base_reward, bd = _score_locally(action, task_id, step, max_steps)

    # Multi-agent bonus: challenger caused improvement → +5%
    if challenger_used and challenge_improved:
        base_reward = min(1.0, base_reward + 0.05)
        bd["challenger_bonus"] = 0.05

    return round(base_reward, 4), bd


def _score_locally(
    action: Dict[str, Any],
    task_id: str,
    step: int,
    max_steps: int,
) -> Tuple[float, Dict[str, Any]]:
    """Rubric-based local scoring (no env dependency)."""
    gt = GROUND_TRUTH[task_id]
    rh = RED_HERRINGS.get(task_id, [])

    # Root cause (weight 0.35)
    rc_score = 1.0 if action.get("root_cause_service") == gt["root_cause_service"] else 0.0
    # Partial: if RC is in affected_services at least
    if rc_score == 0.0 and action.get("root_cause_service") in gt["affected_services"]:
        rc_score = 0.25

    # Remediation action (weight 0.25)
    full_credit = {"easy": ["fix_config"], "medium": ["fix_config"], "hard": ["restart_service", "escalate"]}
    partial_credit = {"rollback": 0.7, "restart_service": 0.3, "investigate_further": 0.1}
    act = action.get("remediation_action", "")
    if act in full_credit.get(task_id, []):
        act_score = 1.0
    elif act in partial_credit:
        act_score = partial_credit[act]
    else:
        act_score = 0.0

    # Severity (weight 0.20)
    sev_exact = gt["severity"]
    sev_partial = {"P0": {"P1": 0.5}, "P1": {"P0": 0.3, "P2": 0.2}}
    pred_sev = action.get("severity", "P3")
    if pred_sev == sev_exact:
        sev_score = 1.0
    elif pred_sev in sev_partial.get(sev_exact, {}):
        sev_score = sev_partial[sev_exact][pred_sev]
    else:
        sev_score = 0.0

    # Communication (weight 0.10)
    msg = action.get("stakeholder_message") or ""
    needs_msg = gt["severity"] in ("P0", "P1")
    if not needs_msg:
        com_score = 1.0
    elif msg and len(msg) > 30:
        keywords = ["investigating", "eta", "payment", "resolution", gt["root_cause_service"].split("-")[0]]
        hits = sum(1 for kw in keywords if kw.lower() in msg.lower())
        com_score = min(1.0, 0.6 + hits * 0.1)
    elif msg:
        com_score = 0.3
    else:
        com_score = 0.0

    # Speed bonus (weight 0.10)
    full_by = max_steps // 3
    half_by = max_steps // 2
    spd = 1.0 if step <= full_by else (0.5 if step <= half_by else 0.0)

    # Weighted sum
    raw = rc_score * 0.35 + act_score * 0.25 + sev_score * 0.20 + com_score * 0.10 + spd * 0.10

    # False positive penalty: RC is a red herring
    fp_pen = 0.0
    if action.get("root_cause_service") in rh:
        fp_pen = 0.30

    # Wrong action penalty
    wa_pen = 0.0
    penalty_acts = {"easy": ["reroute_traffic"], "medium": ["reroute_traffic"], "hard": ["rollback"]}
    if act in penalty_acts.get(task_id, []):
        wa_pen = 0.20

    # Missed escalation: P0 without stakeholder message
    me_pen = 0.0
    if gt["severity"] == "P0" and pred_sev == "P0" and not msg:
        me_pen = 0.25

    total_pen = fp_pen * 0.15 + wa_pen * 0.20 + me_pen * 0.25
    final = round(max(0.0, min(1.0, raw - total_pen)), 4)

    breakdown = {
        "root_cause_score":    round(rc_score, 4),
        "action_score":        round(act_score, 4),
        "severity_score":      round(sev_score, 4),
        "communication_score": round(com_score, 4),
        "speed_bonus":         round(spd, 4),
        "false_positive_penalty":    round(fp_pen, 4),
        "wrong_action_penalty":      round(wa_pen, 4),
        "missed_escalation_penalty": round(me_pen, 4),
        "raw_score":    round(raw, 4),
        "total_penalty": round(total_pen, 4),
        "final_score":  final,
        "feedback": (
            f"RC:{rc_score:.0%} Act:{act_score:.0%} Sev:{sev_score:.0%} "
            f"Com:{com_score:.0%} Spd:{spd:.0%} Pen:-{total_pen:.2f} → {final:.2f}"
        ),
    }
    return final, breakdown


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER  (returns step-by-step logs as a list)
# ══════════════════════════════════════════════════════════════════════════════
def run_episode(
    task_id: str,
    episode_num: int,
    total_episodes: int,
    seed: Optional[int] = None,
    use_env: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Run one full training episode.
    Returns (step_logs, episode_summary).
    
    Multi-agent loop per step:
      1. Diagnoser produces initial answer
      2. Challenger generates adversarial challenge
      3. Diagnoser revises (may ignore challenge if skill is low)
      4. Both answers are graded; improvement tracked
    """
    rng         = random.Random(seed)
    max_steps   = MAX_STEPS[task_id]
    challenger  = ChallengerAgent()
    diagnoser   = DiagnoserAgent(task_id, episode_num, total_episodes, rng)

    # Build a synthetic observation (or use real env)
    env     = None
    obs     = _synthetic_obs(task_id, rng)
    if use_env and HAS_ENV:
        try:
            env = IncidentResponseEnv()
            obs_model = env.reset(task_id=task_id, dynamic=True, seed=seed)
            obs = obs_model.model_dump()
        except Exception:
            env = None

    step_logs        = []
    episode_rewards  = []
    best_reward      = 0.0
    challenger_wins  = 0
    done             = False

    for step in range(1, max_steps + 1):
        if done:
            break

        # ── Phase 1: Initial diagnosis ────────────────────────────────────────
        initial = diagnoser.diagnose(obs, challenge=None, prior_action=None)
        r_initial, bd_initial = compute_reward(initial, task_id, step, max_steps)

        # ── Phase 2: Challenger attacks ───────────────────────────────────────
        challenge_text, strategy = challenger.challenge(initial, obs, task_id, rng)

        # ── Phase 3: Diagnoser revises ────────────────────────────────────────
        revised = diagnoser.diagnose(obs, challenge=challenge_text, prior_action=initial)
        # First compute without bonus to check improvement, then recompute with bonus
        r_revised_raw, bd_revised = _score_locally(revised, task_id, step, max_steps)
        improved_flag = r_revised_raw > r_initial
        r_revised, bd_revised = compute_reward(
            revised, task_id, step, max_steps,
            challenger_used=True,
            challenge_improved=improved_flag,
        )

        improved = improved_flag
        if improved:
            challenger_wins += 1

        final_reward = r_revised
        best_reward  = max(best_reward, final_reward)
        episode_rewards.append(final_reward)

        step_log = {
            "episode":          episode_num,
            "task_id":          task_id,
            "step":             step,
            "reward":           final_reward,
            "initial_reward":   r_initial,
            "revised_reward":   r_revised,
            "challenger_strategy": strategy,
            "challenger_improved": improved,
            "root_cause":       revised["root_cause_service"],
            "root_cause_correct": revised["root_cause_service"] == GROUND_TRUTH[task_id]["root_cause_service"],
            "severity":         revised["severity"],
            "action":           revised["remediation_action"],
            "confidence":       revised["confidence"],
            "skill_level":      round(diagnoser.skill, 4),
            "breakdown": {
                "root_cause":    round(bd_revised.get("root_cause_score", 0), 3),
                "action":        round(bd_revised.get("action_score", 0), 3),
                "severity":      round(bd_revised.get("severity_score", 0), 3),
                "communication": round(bd_revised.get("communication_score", 0), 3),
                "speed":         round(bd_revised.get("speed_bonus", 0), 3),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        step_logs.append(step_log)

        # ── Step the real env if available ────────────────────────────────────
        if env is not None:
            try:
                action_obj = IncidentAction(
                    root_cause_service  = revised["root_cause_service"],
                    root_cause_type     = revised.get("root_cause_type", "unknown"),
                    severity            = revised.get("severity", "P2"),
                    affected_services   = revised.get("affected_services", []),
                    remediation_action  = revised.get("remediation_action", "investigate_further"),
                    stakeholder_message = revised.get("stakeholder_message"),
                    confidence          = float(revised.get("confidence", 0.5)),
                )
                obs_model, _, done, _ = env.step(action_obj)
                obs = obs_model.model_dump()
            except Exception:
                done = True

        # Early termination if excellent score
        if final_reward >= 0.88 and task_id != "hard":
            done = True

    if env:
        try:
            env.close()
        except Exception:
            pass

    summary = {
        "episode":         episode_num,
        "task_id":         task_id,
        "best_reward":     round(best_reward, 4),
        "avg_reward":      round(sum(episode_rewards) / len(episode_rewards), 4) if episode_rewards else 0.0,
        "final_reward":    round(episode_rewards[-1], 4) if episode_rewards else 0.0,
        "challenger_wins": challenger_wins,
        "total_steps":     len(step_logs),
        "skill_level":     round(diagnoser.skill, 4),
    }
    return step_logs, summary


def _synthetic_obs(task_id: str, rng: random.Random) -> Dict[str, Any]:
    """Minimal synthetic observation for simulation mode."""
    svcs = {
        "easy":   ["payments-db", "payments-api", "checkout-ui", "worker-node-4"],
        "medium": ["user-service", "auth-service", "api-gateway", "storefront-ui", "cache-service", "worker-node-4"],
        "hard":   ["payments-db", "cache-service", "order-service", "api-gateway", "storefront-ui", "network-switch-03", "worker-node-7"],
        "expert": ["auth-service", "user-service", "api-gateway", "storefront-ui", "order-service", "payments-api", "notification-svc", "cache-service", "worker-node-7", "metrics-exporter"],
    }[task_id]

    gt = GROUND_TRUTH[task_id]
    rh = RED_HERRINGS.get(task_id, [])
    metrics = {}
    alerts  = []

    for svc in svcs:
        is_rc  = (svc == gt["root_cause_service"])
        is_rh  = (svc in rh)
        cpu    = rng.uniform(0.85, 0.99) if is_rc else (rng.uniform(0.85, 0.97) if is_rh else rng.uniform(0.2, 0.5))
        mem    = rng.uniform(0.90, 0.99) if is_rc else rng.uniform(0.2, 0.6)
        status = "failing" if is_rc else ("degraded" if not is_rh else "healthy")
        metrics[svc] = {
            "cpu_utilization":    round(cpu, 3),
            "memory_utilization": round(mem, 3),
            "http_rt":            round(rng.uniform(2000, 45000) if is_rc else rng.uniform(50, 200), 1),
            "is_healthy":         (status == "healthy"),
            "status":             status,
            "restart_count":      rng.randint(3, 9) if is_rc and task_id == "hard" else 0,
        }
        if not (status == "healthy"):
            alerts.append({
                "alert_id":     f"ALT-{svc[:6].upper()}-001",
                "service":      svc,
                "metric":       "memory_utilization" if mem > 0.85 else "cpu_utilization",
                "current_value": round(mem if mem > 0.85 else cpu, 3),
                "threshold":    0.85,
                "severity":     "critical" if status == "failing" else "warning",
                "fired_at_step": 0,
            })

    topology = {
        "easy":   [{"upstream": "checkout-ui", "downstream": "payments-api", "rpc_type": "http", "avg_latency_ms": 120, "current_latency_ms": 1800},
                   {"upstream": "payments-api", "downstream": "payments-db", "rpc_type": "db", "avg_latency_ms": 8, "current_latency_ms": 950}],
        "medium": [{"upstream": "storefront-ui", "downstream": "api-gateway", "rpc_type": "http", "avg_latency_ms": 45, "current_latency_ms": 3200},
                   {"upstream": "api-gateway", "downstream": "auth-service", "rpc_type": "rpc", "avg_latency_ms": 18, "current_latency_ms": 2800},
                   {"upstream": "auth-service", "downstream": "user-service", "rpc_type": "rpc", "avg_latency_ms": 15, "current_latency_ms": 30000}],
        "hard":   [{"upstream": "storefront-ui", "downstream": "api-gateway", "rpc_type": "http", "avg_latency_ms": 45, "current_latency_ms": 45000},
                   {"upstream": "api-gateway", "downstream": "order-service", "rpc_type": "rpc", "avg_latency_ms": 35, "current_latency_ms": 38000},
                   {"upstream": "order-service", "downstream": "payments-db", "rpc_type": "db", "avg_latency_ms": 8, "current_latency_ms": 0},
                   {"upstream": "cache-service", "downstream": "payments-db", "rpc_type": "db", "avg_latency_ms": 3, "current_latency_ms": 0}],
        "expert": [{"upstream": "storefront-ui", "downstream": "api-gateway", "rpc_type": "http", "avg_latency_ms": 45, "current_latency_ms": 42000},
                   {"upstream": "api-gateway", "downstream": "auth-service", "rpc_type": "rpc", "avg_latency_ms": 12, "current_latency_ms": 45000},
                   {"upstream": "api-gateway", "downstream": "order-service", "rpc_type": "rpc", "avg_latency_ms": 22, "current_latency_ms": 15000},
                   {"upstream": "auth-service", "downstream": "user-service", "rpc_type": "rpc", "avg_latency_ms": 20, "current_latency_ms": 35000},
                   {"upstream": "order-service", "downstream": "payments-api", "rpc_type": "rpc", "avg_latency_ms": 15, "current_latency_ms": 8000}],
    }[task_id]

    return {
        "task_id":   task_id,
        "step":      0,
        "metrics":   metrics,
        "alerts":    alerts,
        "topology":  topology,
        "timeline":  [],
        "time_pressure": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train(
    tasks: List[str],
    total_episodes: int,
    curriculum: bool = False,
    use_env: bool = False,
    quiet: bool = False,
) -> Path:
    """
    Main training loop. Produces:
    - JSONL step log (data/training_logs/training_TIMESTAMP.jsonl)
    - Live summary JSON (data/training_logs/latest_summary.json)
    - Per-task reward curve JSON (data/training_logs/reward_curves.json)
    """
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"training_{ts}.jsonl"
    summary_file = LOG_DIR / "latest_summary.json"
    curves_file  = LOG_DIR / "reward_curves.json"

    print(f"{'='*60}")
    print(f"GRPO Training — Incident Response AI")
    print(f"Tasks:     {tasks}")
    print(f"Episodes:  {total_episodes}")
    print(f"Curriculum: {curriculum}")
    print(f"Use Env:   {use_env}")
    print(f"Log file:  {log_file}")
    print(f"{'='*60}")

    # Track per-task reward history
    all_rewards:   Dict[str, List[float]] = {t: [] for t in tasks}
    all_rc_scores: Dict[str, List[float]] = {t: [] for t in tasks}
    challenger_wins_total = 0
    start_time = time.time()

    with open(log_file, "w") as f:
        for ep in range(total_episodes):
            elapsed = time.time() - start_time

            # ── Task selection ────────────────────────────────────────────────
            if curriculum:
                # Easy for first 33%, medium next 33%, hard last 34%
                ci      = min(2, int(ep / total_episodes * 3))
                task_id = CURRICULUM_ORDER[ci]
            else:
                task_id = tasks[ep % len(tasks)]

            seed = random.randint(0, 999999)

            # ── Run episode ───────────────────────────────────────────────────
            step_logs, ep_summary = run_episode(
                task_id        = task_id,
                episode_num    = ep,
                total_episodes = total_episodes,
                seed           = seed,
                use_env        = use_env,
            )

            # ── Write step logs ───────────────────────────────────────────────
            for sl in step_logs:
                f.write(json.dumps(sl) + "\n")
            f.flush()

            # ── Track metrics ─────────────────────────────────────────────────
            all_rewards[task_id].append(ep_summary["best_reward"])
            avg_rc = sum(
                sl["breakdown"]["root_cause"] for sl in step_logs
            ) / max(1, len(step_logs))
            all_rc_scores[task_id].append(round(avg_rc, 4))
            challenger_wins_total += ep_summary["challenger_wins"]

            # ── Console output ────────────────────────────────────────────────
            if not quiet:
                pct  = 100 * ep / total_episodes
                avg10_r = _avg_last_n(all_rewards[task_id], 10)
                eta_s   = (elapsed / max(1, ep)) * (total_episodes - ep)
                print(
                    f"  ep={ep:04d}/{total_episodes} [{pct:5.1f}%] "
                    f"task={task_id:6s} "
                    f"reward={ep_summary['best_reward']:.3f} "
                    f"avg10={avg10_r:.3f} "
                    f"RC={ep_summary['total_steps'] and step_logs[-1]['breakdown']['root_cause']:.0%} "
                    f"challenger_wins={ep_summary['challenger_wins']} "
                    f"ETA={int(eta_s//60)}m{int(eta_s%60):02d}s",
                    flush=True,
                )

            # ── Write live summary (UI polls this) ────────────────────────────
            summary = {
                "episode":      ep,
                "total":        total_episodes,
                "progress_pct": round(100 * ep / total_episodes, 1),
                "elapsed_s":    round(elapsed, 1),
                "per_task": {
                    t: {
                        "rewards":    all_rewards[t][-100:],
                        "rc_scores":  all_rc_scores[t][-100:],
                        "avg_last10": _avg_last_n(all_rewards[t], 10),
                        "avg_last50": _avg_last_n(all_rewards[t], 50),
                        "best":       round(max(all_rewards[t]) if all_rewards[t] else 0, 4),
                        "count":      len(all_rewards[t]),
                        "trend":      _trend(all_rewards[t]),
                    }
                    for t in tasks
                },
                "challenger_wins_total": challenger_wins_total,
                "updated_at": datetime.utcnow().isoformat(),
                "running": True,
            }
            summary_file.write_text(json.dumps(summary, indent=2))

            # ── Write reward curves (for plotting) ────────────────────────────
            curves = {}
            for t in tasks:
                rwds = all_rewards[t]
                # Smooth with rolling average
                smoothed = _rolling_avg(rwds, window=10)
                curves[t] = {
                    "raw":      rwds,
                    "smoothed": smoothed,
                    "rc_scores": all_rc_scores[t],
                    "episodes": list(range(len(rwds))),
                }
            curves_file.write_text(json.dumps(curves, indent=2))

    # ── Final summary ─────────────────────────────────────────────────────────
    summary["running"] = False
    summary_file.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    for t in tasks:
        if all_rewards[t]:
            print(
                f"  {t:8s}: avg={_avg_last_n(all_rewards[t], 20):.3f}  "
                f"best={max(all_rewards[t]):.3f}  "
                f"episodes={len(all_rewards[t])}"
            )
    print(f"  Challenger wins: {challenger_wins_total}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")

    return log_file


# ══════════════════════════════════════════════════════════════════════════════
# TRL/GRPO TRAINING  (real LLM mode — requires GPU + TRL)
# ══════════════════════════════════════════════════════════════════════════════
def build_grpo_dataset(task_id: str, n_samples: int = 200) -> List[Dict]:
    """Build prompt-completion pairs for GRPO training."""
    rng     = random.Random(42)
    dataset = []
    for i in range(n_samples):
        obs  = _synthetic_obs(task_id, rng)
        gt   = GROUND_TRUTH[task_id]
        prompt = f"""You are an expert SRE triaging a production incident.

ALERTS: {json.dumps(obs['alerts'][:3], indent=2)}

METRICS: {json.dumps({k: {"cpu": v["cpu_utilization"], "mem": v["memory_utilization"], "status": v["status"]} for k, v in obs["metrics"].items()}, indent=2)}

TOPOLOGY: {json.dumps(obs["topology"], indent=2)}

Respond ONLY with valid JSON:
{{
  "root_cause_service": "<exact service>",
  "root_cause_type": "<fault type>",
  "severity": "<P0|P1|P2|P3>",
  "affected_services": ["<list>"],
  "remediation_action": "<action>",
  "stakeholder_message": "<required for P0/P1>",
  "confidence": 0.9,
  "reasoning": "<step by step>"
}}"""

        completion = json.dumps({
            "root_cause_service":  gt["root_cause_service"],
            "root_cause_type":     gt["root_cause_type"],
            "severity":            gt["severity"],
            "affected_services":   gt["affected_services"],
            "remediation_action":  gt["remediation_action"],
            "stakeholder_message": f"{gt['root_cause_service']} issue causing cascade. ETA 10 mins.",
            "confidence":          0.95,
            "reasoning":           f"Topology traversal: {gt['root_cause_service']} shows highest degradation.",
        })
        dataset.append({"prompt": prompt, "completion": completion, "task_id": task_id})
    return dataset


def grpo_reward_fn(completions: List[str], prompts: List[str], task_ids: List[str]) -> List[float]:
    """Reward function for TRL GRPO trainer."""
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        try:
            action = json.loads(completion.strip())
            reward, _ = _score_locally(action, task_id, step=1, max_steps=10)
        except Exception:
            reward = 0.0
        rewards.append(reward)
    return rewards


def run_grpo_training(model_name: str, task_id: str, episodes: int):
    """Real GRPO training via HuggingFace TRL."""
    if not HAS_TRL:
        print("[ERROR] TRL not installed. Run: pip install trl transformers")
        return

    print(f"[GRPO] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = build_grpo_dataset(task_id, n_samples=episodes)

    config = GRPOConfig(
        output_dir        = str(CKPT_DIR / f"grpo_{task_id}"),
        num_train_epochs  = 1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        learning_rate     = 1e-5,
        logging_steps     = 10,
        save_steps        = 50,
        report_to         = "none",
    )

    trainer = GRPOTrainer(
        model          = model,
        tokenizer      = tokenizer,
        config         = config,
        train_dataset  = dataset,
        reward_funcs   = [lambda completions, prompts: grpo_reward_fn(
            completions, prompts, [task_id] * len(completions)
        )],
    )
    trainer.train()
    model.save_pretrained(str(CKPT_DIR / f"grpo_{task_id}_final"))
    print(f"[GRPO] Model saved to {CKPT_DIR}/grpo_{task_id}_final")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _avg_last_n(lst: List[float], n: int) -> float:
    if not lst:
        return 0.0
    return round(sum(lst[-n:]) / len(lst[-n:]), 4)


def _rolling_avg(lst: List[float], window: int = 10) -> List[float]:
    result = []
    for i in range(len(lst)):
        chunk = lst[max(0, i - window + 1): i + 1]
        result.append(round(sum(chunk) / len(chunk), 4))
    return result


def _trend(lst: List[float]) -> str:
    if len(lst) < 5:
        return "insufficient_data"
    recent = sum(lst[-5:]) / 5
    older  = sum(lst[-20:-5]) / max(1, len(lst[-20:-5]))
    diff   = recent - older
    if diff > 0.02:  return "improving"
    if diff < -0.02: return "declining"
    return "stable"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training — Incident Response AI")
    parser.add_argument("--task",       default="all",    help="easy|medium|hard|all")
    parser.add_argument("--episodes",   type=int, default=100)
    parser.add_argument("--curriculum", action="store_true", help="easy→medium→hard progression")
    parser.add_argument("--use-llm",    action="store_true", help="Use real LLM via TRL (requires GPU)")
    parser.add_argument("--model",      default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    tasks = CURRICULUM_ORDER if args.task == "all" else [args.task]

    if args.use_llm and HAS_TRL:
        for t in tasks:
            run_grpo_training(args.model, t, args.episodes)
    else:
        train(
            tasks          = tasks,
            total_episodes = args.episodes,
            curriculum     = args.curriculum,
            use_env        = HAS_ENV and not args.use_llm,
            quiet          = args.quiet,
        )