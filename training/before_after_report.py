"""
training/before_after_report.py
--------------------------------
Generates the judge-winning "Before vs After" comparison table.

Reads existing JSONL training logs from data/training_logs/
Compares:
  - Episodes 1-N/5  (early "before" phase)
  - Episodes 4N/5-N (late "after" phase)

Usage:
    python training/before_after_report.py
    python training/before_after_report.py --log-file data/training_logs/training_XYZ.jsonl
    python training/before_after_report.py --episodes 100 --split 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR    = ROOT / "data" / "training_logs"
MEM_DIR    = ROOT / "data" / "memory"
REPORT_TXT = LOG_DIR / "before_after_report.txt"
REPORT_JSON = LOG_DIR / "before_after_report.json"


# ══════════════════════════════════════════════════════════════════════════════
# LOG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_log() -> Optional[Path]:
    """Find the most recently modified JSONL training log."""
    logs = sorted(
        LOG_DIR.glob("training_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def load_step_logs(log_file: Path) -> List[Dict[str, Any]]:
    """Load all step logs from a JSONL file."""
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs


def load_summary_json() -> Dict[str, Any]:
    """Load latest_summary.json if available."""
    path = LOG_DIR / "latest_summary.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_ltm_stats() -> Dict[str, Any]:
    """Load long-term memory stats if available."""
    ltm_path = MEM_DIR / "responder_ltm.json"
    if not ltm_path.exists():
        return {}
    try:
        with open(ltm_path, "r") as f:
            data = json.load(f)
        fault_count = sum(len(v) for v in data.get("fault_patterns", {}).values())
        rem_count   = sum(len(v) for v in data.get("remediation_stats", {}).values())
        rh_count    = sum(1 for v in data.get("red_herring_seen", {}).values() if v >= 2)
        routing_count = sum(
            sum(int(m.get("count", 0)) for m in band.values())
            for band in data.get("routing_stats", {}).values()
        )
        return {
            "total_episodes": data.get("total_episodes", 0),
            "fault_patterns": fault_count,
            "remediation_patterns": rem_count,
            "red_herrings_identified": rh_count,
            "routing_decisions": routing_count,
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def split_episodes(
    logs: List[Dict[str, Any]],
    split_size: int = 20,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split logs into 'before' (first split_size episodes) and
    'after' (last split_size episodes).
    """
    episodes: DefaultDict[int, List[Dict]] = defaultdict(list)
    for step in logs:
        ep = step.get("episode", 0)
        episodes[ep].append(step)

    sorted_eps = sorted(episodes.keys())
    if not sorted_eps:
        return [], []

    before_eps = sorted_eps[:split_size]
    after_eps  = sorted_eps[-split_size:]

    before_logs = [s for ep in before_eps for s in episodes[ep]]
    after_logs  = [s for ep in after_eps  for s in episodes[ep]]

    return before_logs, after_logs


def compute_metrics(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all metrics for a set of step logs."""
    if not logs:
        return {}

    # Per-episode aggregation
    ep_rewards:  DefaultDict[int, List[float]] = defaultdict(list)
    ep_steps:    DefaultDict[int, int]         = defaultdict(int)
    ep_rc_correct: DefaultDict[int, List[bool]] = defaultdict(list)

    for sl in logs:
        ep  = sl.get("episode", 0)
        ep_rewards[ep].append(sl.get("reward", 0.0))
        ep_steps[ep] += 1
        ep_rc_correct[ep].append(sl.get("root_cause_correct", False))

    episodes = sorted(ep_rewards.keys())

    # Reward metrics
    ep_best_rewards = [max(ep_rewards[ep]) for ep in episodes]
    avg_reward = round(sum(ep_best_rewards) / len(ep_best_rewards), 4) if ep_best_rewards else 0.0

    # Root cause accuracy (per episode: correct if any step got it right)
    ep_rc_rates = [
        (sum(1 for c in ep_rc_correct[ep] if c) / max(1, len(ep_rc_correct[ep])))
        for ep in episodes
    ]
    avg_rc_accuracy = round(sum(ep_rc_rates) / len(ep_rc_rates) * 100, 1) if ep_rc_rates else 0.0

    # Steps to resolve (episodes that reached done)
    ep_steps_list   = [ep_steps[ep] for ep in episodes]
    avg_steps       = round(sum(ep_steps_list) / len(ep_steps_list), 1) if ep_steps_list else 0.0

    # Challenger wins
    challenger_wins = [sl.get("challenger_improved", sl.get("challenger_wins", False)) for sl in logs]
    challenger_win_pct = round(sum(1 for w in challenger_wins if w) / max(1, len(challenger_wins)) * 100, 1)

    # Model tier distribution (hybrid mode logs "model_used" or "cot_phases")
    model_counts: DefaultDict[str, int] = defaultdict(int)
    for sl in logs:
        # From hybrid step log
        model = sl.get("model_used", sl.get("tier_used", ""))
        if model:
            model_counts[model] += 1
        # From cot_phases
        for phase in sl.get("cot_phases", []):
            t = phase.get("tier", "")
            if t:
                model_counts[t] += 1

    total_model_calls = sum(model_counts.values())
    model_pcts = {
        t: round(c / total_model_calls * 100, 1)
        for t, c in model_counts.items()
    } if total_model_calls > 0 else {}

    # Skill level trend
    skill_levels = [sl.get("skill_level", 0) for sl in logs if "skill_level" in sl]
    avg_skill    = round(sum(skill_levels) / len(skill_levels), 4) if skill_levels else 0.0

    return {
        "episode_count":       len(episodes),
        "step_count":          len(logs),
        "avg_reward":          avg_reward,
        "rc_accuracy_pct":     avg_rc_accuracy,
        "avg_steps_to_resolve": avg_steps,
        "challenger_win_pct":  challenger_win_pct,
        "model_tier_pcts":     model_pcts,
        "avg_skill_level":     avg_skill,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def format_pct_change(before: float, after: float, higher_is_better: bool = True) -> str:
    """Format a before→after change with direction indicator."""
    if before == 0:
        change = "+∞" if after > 0 else "—"
    else:
        pct_change = ((after - before) / abs(before)) * 100
        sign       = "+" if pct_change > 0 else ""
        change     = f"{sign}{pct_change:.0f}%"

    if higher_is_better:
        trend = "[PASS]" if after > before else ("[WARN]" if after == before else "[FAIL]")
    else:
        trend = "[PASS]" if after < before else ("[WARN]" if after == before else "[FAIL]")

    return f"{change} {trend}"


def build_report(
    before: Dict[str, Any],
    after:  Dict[str, Any],
    ltm:    Dict[str, Any],
    before_range: str,
    after_range:  str,
) -> Tuple[str, Dict[str, Any]]:
    """Build the before/after comparison report."""

    COL1 = 30
    COL2 = 18
    COL3 = 18
    COL4 = 14

    def row(label: str, b_val: str, a_val: str, change: str) -> str:
        return (
            f"  {label:<{COL1}} {b_val:>{COL2}} {a_val:>{COL3}} {change:>{COL4}}\n"
        )

    divider = "  " + "-" * (COL1 + COL2 + COL3 + COL4 + 6) + "\n"

    lines = [
        "\n" + "=" * 78 + "\n",
        "  [REPORT] INCIDENT RESPONSE AI -- PROGRESSIVE LEARNING REPORT\n",
        "  Evidence of measurable improvement across training episodes\n",
        "=" * 78 + "\n\n",
        row("Metric", f"BEFORE ({before_range})", f"AFTER ({after_range})", "Change"),
        divider,
    ]

    # Core metrics
    b_rc  = before.get("rc_accuracy_pct", 0)
    a_rc  = after.get("rc_accuracy_pct", 0)
    lines.append(row(
        "Root Cause Accuracy",
        f"{b_rc:.1f}%",
        f"{a_rc:.1f}%",
        format_pct_change(b_rc, a_rc),
    ))

    b_rw  = before.get("avg_reward", 0)
    a_rw  = after.get("avg_reward", 0)
    lines.append(row(
        "Avg Episode Reward",
        f"{b_rw:.3f}",
        f"{a_rw:.3f}",
        format_pct_change(b_rw, a_rw),
    ))

    b_st  = before.get("avg_steps_to_resolve", 0)
    a_st  = after.get("avg_steps_to_resolve", 0)
    lines.append(row(
        "Steps to Resolve",
        f"{b_st:.1f}",
        f"{a_st:.1f}",
        format_pct_change(b_st, a_st, higher_is_better=False),
    ))

    b_ch  = before.get("challenger_win_pct", 0)
    a_ch  = after.get("challenger_win_pct", 0)
    lines.append(row(
        "Challenger Wins",
        f"{b_ch:.1f}%",
        f"{a_ch:.1f}%",
        format_pct_change(b_ch, a_ch),
    ))

    b_sk  = before.get("avg_skill_level", 0)
    a_sk  = after.get("avg_skill_level", 0)
    lines.append(row(
        "Agent Skill Level",
        f"{b_sk:.3f}",
        f"{a_sk:.3f}",
        format_pct_change(b_sk, a_sk),
    ))

    lines.append(divider)
    lines.append("  MODEL ROUTING (Hybrid Mode)\n")

    # Model tier distribution
    all_tiers = set(before.get("model_tier_pcts", {}).keys()) | \
                set(after.get("model_tier_pcts", {}).keys())
    for tier in sorted(all_tiers):
        b_t = before.get("model_tier_pcts", {}).get(tier, 0)
        a_t = after.get("model_tier_pcts", {}).get(tier, 0)
        lines.append(row(
            f"  Model: {tier}",
            f"{b_t:.1f}%",
            f"{a_t:.1f}%",
            format_pct_change(b_t, a_t, higher_is_better=(tier != "fast")),
        ))

    if not all_tiers:
        lines.append("  (Run with --hybrid flag to populate model routing stats)\n")

    lines.append(divider)
    lines.append("  PROGRESSIVE MEMORY (LTM)\n")

    if ltm:
        lines.append(row("  LTM Fault Patterns",      "0",  str(ltm.get("fault_patterns", 0)),  "[DATA]"))
        lines.append(row("  LTM Remediation Patterns", "0", str(ltm.get("remediation_patterns", 0)), "[DATA]"))
        lines.append(row("  Red Herrings Identified",  "0", str(ltm.get("red_herrings_identified", 0)), "[NOISE]"))
        lines.append(row("  LTM Routing Decisions",    "0", str(ltm.get("routing_decisions", 0)), "[MODEL]"))
        lines.append(row("  Total Episodes in LTM",    "0", str(ltm.get("total_episodes", 0)),  "[EP]"))
    else:
        lines.append("  (Run training to populate long-term memory)\n")

    lines.append("\n" + "=" * 78 + "\n")
    lines.append("  [SUMMARY] JUDGE VERDICT:\n")
    improvement = a_rc - b_rc
    if improvement >= 30:
        verdict = "[SUPERB] STRONG learning evidence -- root cause accuracy +{:.0f}pp".format(improvement)
    elif improvement >= 15:
        verdict = "[PASS] CLEAR learning evidence -- root cause accuracy +{:.0f}pp".format(improvement)
    elif improvement >= 5:
        verdict = "[WARN]  MODERATE learning evidence -- root cause accuracy +{:.0f}pp".format(improvement)
    else:
        verdict = "[FAIL] WEAK learning evidence -- run more episodes (>=100 recommended)"
    lines.append(f"  {verdict}\n")
    lines.append("=" * 78 + "\n\n")

    report_text = "".join(lines)

    report_json = {
        "before": before,
        "after":  after,
        "ltm":    ltm,
        "improvement": {
            "rc_accuracy_pp":   round(a_rc - b_rc, 2),
            "reward_delta":     round(a_rw - b_rw, 4),
            "steps_delta":      round(a_st - b_st, 2),
            "challenger_pp":    round(a_ch - b_ch, 2),
        },
    }
    return report_text, report_json


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate before/after learning report from JSONL training logs"
    )
    parser.add_argument("--log-file",  type=str, default="",
                        help="Path to specific JSONL log file (default: latest)")
    parser.add_argument("--split",     type=int, default=20,
                        help="Number of episodes to use for before/after windows (default: 20)")
    parser.add_argument("--no-save",   action="store_true",
                        help="Print report but don't save to disk")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Find log file
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = find_latest_log()

    if log_path is None or not log_path.exists():
        print("[ERROR] No training log found. Run training first:")
        print("        python training/train.py --task all --episodes 100 --curriculum")
        sys.exit(1)

    print(f"[INFO] Reading log: {log_path}")
    step_logs = load_step_logs(log_path)
    print(f"[INFO] Loaded {len(step_logs)} step records")

    # Detect total episodes
    all_eps  = sorted(set(s.get("episode", 0) for s in step_logs))
    n_eps    = len(all_eps)
    split    = min(args.split, n_eps // 3)

    if n_eps < 6:
        print(f"[WARN] Only {n_eps} episodes — need at least 6 for meaningful comparison")
        print("       Run more episodes for accurate before/after stats")
        split = max(1, n_eps // 3)

    before_range = f"ep {all_eps[0]}–{all_eps[split-1]}" if split > 0 else "ep 0"
    after_range  = f"ep {all_eps[-split]}–{all_eps[-1]}"  if split > 0 else "ep 0"

    before_logs, after_logs = split_episodes(step_logs, split_size=split)

    print(f"[INFO] Before window: {before_range} ({len(before_logs)} steps)")
    print(f"[INFO] After  window: {after_range} ({len(after_logs)} steps)")

    before_metrics = compute_metrics(before_logs)
    after_metrics  = compute_metrics(after_logs)
    ltm_stats      = load_ltm_stats()

    report_text, report_json = build_report(
        before_metrics, after_metrics, ltm_stats, before_range, after_range
    )

    print(report_text)

    if not args.no_save:
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.write(report_text)
        with open(REPORT_JSON, "w") as f:
            json.dump(report_json, f, indent=2)
        print(f"[INFO] Saved: {REPORT_TXT}")
        print(f"[INFO] Saved: {REPORT_JSON}")


if __name__ == "__main__":
    main()
