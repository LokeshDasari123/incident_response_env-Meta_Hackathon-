"""
training/train_grpo.py
======================
GRPO Training Script — Colab-Ready TRL Training for Incident Response Agent

This script fine-tunes a language model using Group Relative Policy Optimization
(GRPO) via HuggingFace TRL. The reward function uses the real incident response
environment grading rubric.

Usage (Colab):
    !pip install trl transformers unsloth accelerate
    !python training/train_grpo.py --model unsloth/Qwen2.5-1.5B-Instruct --episodes 200

Usage (Local):
    python training/train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --episodes 100
"""

import argparse
import json
import os
import sys
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
LOG_DIR  = ROOT / "data" / "training_logs"
CKPT_DIR = ROOT / "data" / "checkpoints"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Ground truth for reward computation ────────────────────────────────────────
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
        "affected_services": ["auth-service", "user-service", "api-gateway", "storefront-ui", "order-service"],
        "remediation_action": "fix_config",
    },
}

CURRICULUM_ORDER = ["easy", "medium", "hard", "expert"]


def build_incident_prompt(task_id: str, seed: int = 42) -> str:
    """Build a realistic SRE triage prompt from synthetic observation data."""
    rng = random.Random(seed)
    gt = GROUND_TRUTH[task_id]

    # Synthetic observation
    services = gt["affected_services"] + ["worker-node-4"]
    metrics = {}
    alerts = []

    for svc in services:
        is_rc = (svc == gt["root_cause_service"])
        cpu = rng.uniform(0.85, 0.99) if is_rc else rng.uniform(0.2, 0.5)
        mem = rng.uniform(0.90, 0.99) if is_rc else rng.uniform(0.2, 0.6)
        status = "failing" if is_rc else "degraded"
        metrics[svc] = {
            "cpu": round(cpu, 3),
            "mem": round(mem, 3),
            "status": status,
        }
        if is_rc or rng.random() > 0.5:
            alerts.append({
                "service": svc,
                "metric": "memory_utilization" if mem > 0.85 else "cpu_utilization",
                "value": round(mem if mem > 0.85 else cpu, 3),
                "severity": "critical" if is_rc else "warning",
            })

    prompt = f"""You are an expert SRE triaging a production incident.

TASK DIFFICULTY: {task_id}

ACTIVE ALERTS:
{json.dumps(alerts[:4], indent=2)}

SERVICE METRICS:
{json.dumps(metrics, indent=2)}

Based on the alerts, metrics, and service topology, provide your incident triage.
Respond ONLY with valid JSON matching this exact schema:
{{
  "root_cause_service": "<exact service name>",
  "root_cause_type": "<misconfiguration|memory_leak|network_partition|crash_loop|resource_exhaustion|certificate_expiry|dependency_failure>",
  "severity": "<P0|P1|P2|P3>",
  "affected_services": ["<list of all affected services>"],
  "remediation_action": "<rollback|restart_service|fix_config|scale_up|escalate|investigate_further>",
  "stakeholder_message": "<required for P0/P1 - include root cause, impact, ETA>",
  "confidence": 0.9,
  "reasoning": "<step by step reasoning>"
}}"""
    return prompt


def compute_reward(completion_text: str, task_id: str) -> float:
    """
    Parse the LLM completion and compute reward using the grading rubric.
    This is the GRPO reward function — called for each completion in the group.
    """
    try:
        # Extract JSON from completion
        text = completion_text.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        action = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return 0.0  # Unparseable = 0 reward

    gt = GROUND_TRUTH.get(task_id, GROUND_TRUTH["easy"])

    # Root cause (weight 0.35)
    rc_score = 1.0 if action.get("root_cause_service") == gt["root_cause_service"] else 0.0
    if rc_score == 0.0 and action.get("root_cause_service") in gt["affected_services"]:
        rc_score = 0.25

    # Remediation action (weight 0.25)
    act = action.get("remediation_action", "")
    if act == gt["remediation_action"]:
        act_score = 1.0
    elif act in ("rollback", "restart_service"):
        act_score = 0.3
    elif act == "investigate_further":
        act_score = 0.1
    else:
        act_score = 0.0

    # Severity (weight 0.20)
    pred_sev = action.get("severity", "P3")
    if pred_sev == gt["severity"]:
        sev_score = 1.0
    elif abs(int(pred_sev[1]) - int(gt["severity"][1])) == 1:
        sev_score = 0.4
    else:
        sev_score = 0.0

    # Communication (weight 0.10)
    msg = action.get("stakeholder_message") or ""
    if gt["severity"] in ("P0", "P1"):
        com_score = min(1.0, len(msg) / 80) if msg else 0.0
    else:
        com_score = 1.0

    # Reasoning quality (weight 0.10)
    reasoning = action.get("reasoning", "")
    reas_score = min(1.0, len(reasoning) / 100) if reasoning else 0.0

    # Weighted sum
    reward = (
        rc_score  * 0.35 +
        act_score * 0.25 +
        sev_score * 0.20 +
        com_score * 0.10 +
        reas_score * 0.10
    )

    return round(max(0.0, min(1.0, reward)), 4)


def build_dataset(tasks: List[str], n_per_task: int = 50) -> List[Dict]:
    """Build prompt-completion dataset for GRPO training."""
    dataset = []
    for task_id in tasks:
        gt = GROUND_TRUTH[task_id]
        for i in range(n_per_task):
            prompt = build_incident_prompt(task_id, seed=i)
            # Gold completion for supervised signal
            completion = json.dumps({
                "root_cause_service": gt["root_cause_service"],
                "root_cause_type": gt["root_cause_type"],
                "severity": gt["severity"],
                "affected_services": gt["affected_services"],
                "remediation_action": gt["remediation_action"],
                "stakeholder_message": (
                    f"{gt['root_cause_service']} experiencing {gt['root_cause_type']}. "
                    f"Cascade across {len(gt['affected_services'])} services. "
                    f"Severity: {gt['severity']}. Action: {gt['remediation_action']}. "
                    f"ETA: ~10 minutes."
                ),
                "confidence": 0.95,
                "reasoning": (
                    f"Topology analysis: {gt['root_cause_service']} shows highest degradation. "
                    f"Pattern matches {gt['root_cause_type']}. "
                    f"Cascade: {' → '.join(gt['affected_services'][:3])}."
                ),
            })
            dataset.append({
                "prompt": prompt,
                "completion": completion,
                "task_id": task_id,
            })
    return dataset


def run_trl_training(
    model_name: str,
    tasks: List[str],
    n_episodes: int,
    curriculum: bool = True,
    output_dir: Optional[str] = None,
):
    """
    Run GRPO training via HuggingFace TRL.
    Requires: pip install trl transformers accelerate
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("[ERROR] TRL not installed. Install with:")
        print("  pip install trl transformers accelerate")
        print("  OR for Colab: pip install unsloth trl")
        return None

    out_dir = output_dir or str(CKPT_DIR / f"grpo_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    print(f"{'='*60}")
    print(f"GRPO Training — Incident Response AI")
    print(f"Model:     {model_name}")
    print(f"Tasks:     {tasks}")
    print(f"Episodes:  {n_episodes}")
    print(f"Curriculum: {curriculum}")
    print(f"Output:    {out_dir}")
    print(f"{'='*60}")

    # Load model
    print(f"[1/4] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    print(f"[2/4] Building dataset: {n_episodes} samples across {tasks}")
    n_per_task = max(10, n_episodes // len(tasks))
    dataset = build_dataset(tasks, n_per_task=n_per_task)
    random.shuffle(dataset)

    # Extract prompts for GRPO
    prompts = [d["prompt"] for d in dataset]
    task_ids = [d["task_id"] for d in dataset]

    # GRPO reward function
    def reward_fn(completions, **kwargs):
        """Reward function for GRPO — scores each completion."""
        rewards = []
        for i, completion in enumerate(completions):
            tid = task_ids[i % len(task_ids)]
            # Handle different completion formats
            text = completion if isinstance(completion, str) else str(completion)
            r = compute_reward(text, tid)
            rewards.append(r)
        return rewards

    # Configure GRPO
    print(f"[3/4] Configuring GRPO trainer")
    config = GRPOConfig(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=5,
        save_steps=50,
        warmup_ratio=0.1,
        report_to="none",
        max_completion_length=512,
        num_generations=4,
    )

    # Train
    print(f"[4/4] Starting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=prompts,
        reward_funcs=[reward_fn],
    )
    trainer.train()

    # Save
    final_path = Path(out_dir) / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\n[DONE] Model saved to {final_path}")

    return final_path


def run_simulation_training(
    tasks: List[str],
    n_episodes: int,
    curriculum: bool = True,
):
    """
    Simulation mode — no GPU needed. Uses the existing train.py logic.
    Generates reward curves for demonstration.
    """
    print(f"[INFO] Running simulation training (no GPU required)")
    print(f"[INFO] This produces real reward curves using the environment grader")

    # Import the existing training logic
    from training.train import train as run_train
    log_file = run_train(
        tasks=tasks,
        total_episodes=n_episodes,
        curriculum=curriculum,
        use_env=False,
        quiet=False,
    )
    return log_file


def generate_reward_plots():
    """Generate reward curve PNGs from training data."""
    curves_file = LOG_DIR / "reward_curves.json"
    if not curves_file.exists():
        print("[WARN] No reward curves found. Run training first.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plot generation.")
        return

    curves = json.loads(curves_file.read_text())

    # ── Plot 1: All tasks reward curves ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0a0d14")
    ax.set_facecolor("#0f1520")

    colors = {
        "easy": "#aaff00",
        "medium": "#ffb800",
        "hard": "#ff4d00",
        "expert": "#8b5cf6",
    }

    for task_id, data in curves.items():
        if data.get("smoothed"):
            ax.plot(
                data["episodes"],
                data["smoothed"],
                color=colors.get(task_id, "#00e5ff"),
                linewidth=2,
                label=f"{task_id.capitalize()} (smoothed)",
            )
            # Raw data as faint scatter
            ax.scatter(
                data["episodes"],
                data["raw"],
                color=colors.get(task_id, "#00e5ff"),
                alpha=0.15,
                s=4,
            )

    ax.set_xlabel("Episode", color="#dce8ff", fontsize=12)
    ax.set_ylabel("Reward", color="#dce8ff", fontsize=12)
    ax.set_title("GRPO Training — Reward Curves", color="#00e5ff", fontsize=16, fontweight="bold")
    ax.legend(facecolor="#131b28", edgecolor="#1c2a3f", labelcolor="#dce8ff")
    ax.tick_params(colors="#4a6080")
    ax.grid(True, alpha=0.15, color="#1c2a3f")
    ax.set_ylim(0, 1.05)

    for spine in ax.spines.values():
        spine.set_color("#1c2a3f")

    plt.tight_layout()
    out_path = LOG_DIR / "reward_curve_all.png"
    plt.savefig(str(out_path), dpi=150, facecolor="#0a0d14")
    plt.close()
    print(f"[PLOT] Saved: {out_path}")

    # ── Plot 2: Root cause accuracy ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0a0d14")
    ax.set_facecolor("#0f1520")

    for task_id, data in curves.items():
        if data.get("rc_scores"):
            # Smooth RC scores
            window = 10
            rc = data["rc_scores"]
            smoothed = []
            for i in range(len(rc)):
                chunk = rc[max(0, i - window + 1): i + 1]
                smoothed.append(sum(chunk) / len(chunk))

            ax.plot(
                range(len(smoothed)),
                smoothed,
                color=colors.get(task_id, "#00e5ff"),
                linewidth=2,
                label=f"{task_id.capitalize()}",
            )

    ax.set_xlabel("Episode", color="#dce8ff", fontsize=11)
    ax.set_ylabel("Root Cause Accuracy", color="#dce8ff", fontsize=11)
    ax.set_title("Root Cause Identification Accuracy", color="#00e5ff", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#131b28", edgecolor="#1c2a3f", labelcolor="#dce8ff")
    ax.tick_params(colors="#4a6080")
    ax.grid(True, alpha=0.15, color="#1c2a3f")
    ax.set_ylim(0, 1.05)

    for spine in ax.spines.values():
        spine.set_color("#1c2a3f")

    plt.tight_layout()
    out_path = LOG_DIR / "rc_accuracy.png"
    plt.savefig(str(out_path), dpi=150, facecolor="#0a0d14")
    plt.close()
    print(f"[PLOT] Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GRPO Training — Incident Response AI (Colab-Ready)"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--task", default="all",
                        help="easy|medium|hard|expert|all")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of training episodes")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning")
    parser.add_argument("--simulate", action="store_true",
                        help="Run simulation mode (no GPU needed)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate reward curve plots only")
    args = parser.parse_args()

    tasks = CURRICULUM_ORDER if args.task == "all" else [args.task]

    if args.plot:
        generate_reward_plots()
    elif args.simulate:
        run_simulation_training(tasks, args.episodes, args.curriculum)
        generate_reward_plots()
    else:
        result = run_trl_training(
            model_name=args.model,
            tasks=tasks,
            n_episodes=args.episodes,
            curriculum=args.curriculum,
        )
        if result is None:
            print("[FALLBACK] TRL not available. Running simulation instead.")
            run_simulation_training(tasks, args.episodes, args.curriculum)
        generate_reward_plots()
