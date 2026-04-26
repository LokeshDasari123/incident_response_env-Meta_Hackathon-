"""
training/grpo_trainer.py
------------------------
Clean GRPO Training Pipeline using HuggingFace TRL.

Uses the multi-agent incident env as the reward source with
composite reward shaping for format, reasoning, and collaboration.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -- Load environment variables from .env file ----------------------------------
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from training.llm_env_wrapper import LLMEnvWrapper
from training.curriculum      import CurriculumController

# ── Optional TRL imports ──────────────────────────────────────────────────────
try:
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRL = True
except ImportError:
    HAS_TRL = False


# ── Ground truth for dataset generation ───────────────────────────────────────
GROUND_TRUTH = {
    "easy": {
        "root_cause_service": "payments-db",
        "root_cause_type":    "misconfiguration",
        "severity":           "P0",
        "affected_services":  ["payments-db", "payments-api", "checkout-ui"],
        "remediation_action": "fix_config",
    },
    "medium": {
        "root_cause_service": "user-service",
        "root_cause_type":    "network_partition",
        "severity":           "P1",
        "affected_services":  ["user-service", "auth-service", "api-gateway", "storefront-ui"],
        "remediation_action": "fix_config",
    },
    "hard": {
        "root_cause_service": "payments-db",
        "root_cause_type":    "memory_leak",
        "severity":           "P0",
        "affected_services":  ["payments-db", "cache-service", "order-service", "api-gateway", "storefront-ui"],
        "remediation_action": "restart_service",
    },
}


def build_training_dataset(
    tasks: List[str],
    n_per_task: int = 100,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Build prompt-completion pairs for GRPO training.
    Each sample includes multi-agent context (monitor signals, memory).
    """
    wrapper = LLMEnvWrapper()
    rng     = random.Random(seed)
    dataset = []

    for task_id in tasks:
        gt = GROUND_TRUTH[task_id]
        for i in range(n_per_task):
            # Build synthetic observation
            obs = _synthetic_obs(task_id, rng)

            # Add monitor signals
            monitor_signals = [{
                "top_service":   gt["root_cause_service"],
                "anomaly_score": round(rng.uniform(0.6, 0.95), 2),
                "reason":        f"Service status: critical; CPU at {rng.randint(85,99)}%",
                "trend":         "rapidly_increasing",
            }]

            # Build prompt
            prompt = wrapper.observation_to_prompt(
                obs             = obs,
                memory_summary  = f"Step {i%5}: investigating {gt['root_cause_service']}",
                monitor_signals = monitor_signals,
            )

            # Build completion
            completion = json.dumps({
                "root_cause_service":  gt["root_cause_service"],
                "root_cause_type":     gt["root_cause_type"],
                "severity":            gt["severity"],
                "affected_services":   gt["affected_services"],
                "remediation_action":  gt["remediation_action"],
                "stakeholder_message": (
                    f"{gt['root_cause_service']} {gt['root_cause_type'].replace('_',' ')} "
                    f"causing cascade to {len(gt['affected_services'])} services. "
                    f"Severity: {gt['severity']}. ETA: 10 minutes."
                ),
                "confidence":  0.95,
                "reasoning":   (
                    f"Topology traversal inward. {gt['root_cause_service']} shows "
                    f"highest degradation. Pattern matches {gt['root_cause_type']}. "
                    f"Monitor agent confirms with score 0.90."
                ),
            })

            dataset.append({
                "prompt":    prompt,
                "completion": completion,
                "task_id":   task_id,
            })

    rng.shuffle(dataset)
    return dataset


def grpo_reward_function(
    completions: List[str],
    prompts: List[str],
    task_ids: Optional[List[str]] = None,
) -> List[float]:
    """
    Reward function for TRL GRPO trainer.
    Parses completions as IncidentAction, scores via grader + shaping.
    """
    wrapper = LLMEnvWrapper()
    rewards = []

    for i, completion in enumerate(completions):
        task_id = task_ids[i] if task_ids else "easy"
        try:
            action = wrapper.parse_response(completion)

            # Score via rubric
            from training.train import _score_locally
            base_reward, _ = _score_locally(action, task_id, step=1, max_steps=10)

            # Shape reward
            shaped, _ = wrapper.shape_reward(
                env_reward=base_reward,
                action=action,
                raw_text=completion,
                memory_used=False,
                monitor_integrated="monitor" in (action.get("reasoning") or "").lower(),
            )
            rewards.append(shaped)
        except Exception:
            rewards.append(0.0)

    return rewards


def run_grpo_training(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    tasks: Optional[List[str]] = None,
    n_samples: int = 200,
    output_dir: Optional[str] = None,
) -> None:
    """
    Full GRPO training pipeline.

    1. Build curriculum-aware dataset
    2. Configure GRPO trainer
    3. Train with env-based reward function
    4. Save model
    """
    if not HAS_TRL:
        print("[ERROR] TRL not installed. Run: pip install trl transformers")
        return

    tasks = tasks or ["easy", "medium", "hard"]
    out   = output_dir or str(ROOT / "data" / "checkpoints" / "grpo_multiagent")

    print(f"[GRPO] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"[GRPO] Building dataset: {n_samples} samples across {tasks}")
    dataset = build_training_dataset(tasks, n_per_task=n_samples // len(tasks))

    config = GRPOConfig(
        output_dir                  = out,
        num_train_epochs            = 1,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        learning_rate               = 1e-5,
        logging_steps               = 10,
        save_steps                  = 50,
        report_to                   = "none",
    )

    # Build task_ids for reward function
    task_id_list = [d["task_id"] for d in dataset]

    trainer = GRPOTrainer(
        model         = model,
        tokenizer     = tokenizer,
        config        = config,
        train_dataset = dataset,
        reward_funcs  = [
            lambda completions, prompts: grpo_reward_function(
                completions, prompts, task_id_list[:len(completions)]
            )
        ],
    )

    print("[GRPO] Starting training...")
    trainer.train()

    model.save_pretrained(out + "_final")
    tokenizer.save_pretrained(out + "_final")
    print(f"[GRPO] Model saved to {out}_final")


# ── Synthetic observation helper ──────────────────────────────────────────────
def _synthetic_obs(task_id: str, rng: random.Random) -> Dict[str, Any]:
    """Minimal synthetic observation for dataset generation."""
    svcs_map = {
        "easy":   ["payments-db", "payments-api", "checkout-ui", "worker-node-4"],
        "medium": ["user-service", "auth-service", "api-gateway", "storefront-ui",
                    "cache-service", "worker-node-4"],
        "hard":   ["payments-db", "cache-service", "order-service", "api-gateway",
                    "storefront-ui", "network-switch-03", "worker-node-7"],
    }
    svcs = svcs_map.get(task_id, svcs_map["easy"])
    gt   = GROUND_TRUTH[task_id]

    metrics = {}
    alerts  = []
    for svc in svcs:
        is_rc = svc == gt["root_cause_service"]
        cpu   = rng.uniform(0.85, 0.99) if is_rc else rng.uniform(0.2, 0.5)
        mem   = rng.uniform(0.90, 0.99) if is_rc else rng.uniform(0.2, 0.6)
        status = "failing" if is_rc else "healthy"
        metrics[svc] = {
            "cpu_utilization":    round(cpu, 3),
            "memory_utilization": round(mem, 3),
            "http_rt":            round(rng.uniform(2000, 45000) if is_rc else rng.uniform(50, 200), 1),
            "is_healthy":         not is_rc,
            "status":             status,
            "error_rate":         round(rng.uniform(0.5, 0.9) if is_rc else 0.0, 3),
        }
        if not (status == "healthy"):
            alerts.append({
                "alert_id": f"ALT-{svc[:6].upper()}-001",
                "service":  svc,
                "metric":   "cpu_utilization",
                "current_value": round(cpu, 3),
                "threshold":  0.85,
                "severity":   "critical",
                "fired_at_step": 0,
            })

    return {
        "task_id":  task_id,
        "step":     0,
        "max_steps": {"easy": 10, "medium": 15, "hard": 20}[task_id],
        "metrics":  metrics,
        "alerts":   alerts,
        "topology": [],
        "timeline": [],
        "time_pressure": 0.0,
    }
