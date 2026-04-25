"""
agents/hybrid_router.py
-----------------------
Complexity-based Model Router.

Scores each incident observation on 5 dimensions → complexity_score ∈ [0,1]
Routes to the appropriate LLM tier:
  complexity < 0.35  →  FAST   (Groq/small — cheap, quick)
  0.35 ≤ c < 0.70   →  BALANCED (Gemini Flash — good reasoning)
  c ≥ 0.70          →  STRONG  (HF 70B — best quality, expensive)

Also tracks per-routing-decision outcomes so the before/after report
can show the router LEARNING which tier to use.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# ── Model tier configuration (from .env) ──────────────────────────────────────
FAST_BASE_URL     = os.getenv("GROQ_BASE_URL",   "https://api.groq.com/openai/v1")
FAST_MODEL        = os.getenv("GROQ_MODEL",      "llama-3.3-70b-versatile")
FAST_API_KEY      = os.getenv("GROQ_API_KEY",    "")

BALANCED_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
BALANCED_MODEL    = os.getenv("GEMINI_MODEL",    "gemini-2.0-flash")
BALANCED_API_KEY  = os.getenv("GEMINI_API_KEY",  "")

STRONG_BASE_URL   = os.getenv("API_BASE_URL",    "https://router.huggingface.co/v1")
STRONG_MODEL      = os.getenv("MODEL_NAME",      "meta-llama/Llama-3.3-70B-Instruct")
STRONG_API_KEY    = os.getenv("HF_TOKEN",        "") or os.getenv("API_KEY", "")

# Thresholds
FAST_THRESHOLD    = 0.35
STRONG_THRESHOLD  = 0.70


class ComplexityRouter:
    """
    5-dimension complexity scorer + model tier router.

    Scoring dimensions (weights sum to 1.0):
      alert_count      0.25  — number of active firing alerts
      cascade_depth    0.25  — longest path in topology call graph
      ambiguity_score  0.20  — number of concurrently unhealthy services
      time_pressure    0.15  — SLA breach imminence (0=no pressure, 1=breach now)
      prior_failures   0.15  — fraction of prior steps with wrong root cause

    Total complexity ∈ [0.0, 1.0].
    """

    WEIGHTS = {
        "alert_count":     0.25,
        "cascade_depth":   0.25,
        "ambiguity_score": 0.20,
        "time_pressure":   0.15,
        "prior_failures":  0.15,
    }

    # Normalisation constants
    MAX_ALERTS        = 12
    MAX_CASCADE_DEPTH = 6
    MAX_UNHEALTHY     = 8

    def __init__(self) -> None:
        # Track routing history for learning analytics
        self._routing_history: List[Dict[str, Any]] = []

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score(
        self,
        obs: Dict[str, Any],
        prior_step_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score an observation. Returns (complexity_score, dimension_breakdown).
        """
        metrics   = obs.get("metrics", {})
        alerts    = obs.get("alerts",  [])
        topology  = obs.get("topology", [])
        tp        = obs.get("time_pressure", 0.0)

        # 1. Alert count dimension
        alert_raw  = min(len(alerts), self.MAX_ALERTS) / self.MAX_ALERTS
        alert_dim  = alert_raw * self.WEIGHTS["alert_count"]

        # 2. Cascade depth — longest chain through topology
        depth      = self._cascade_depth(topology)
        depth_raw  = min(depth, self.MAX_CASCADE_DEPTH) / self.MAX_CASCADE_DEPTH
        depth_dim  = depth_raw * self.WEIGHTS["cascade_depth"]

        # 3. Ambiguity — how many services are simultaneously failing/degraded
        unhealthy  = sum(1 for m in metrics.values() if not m.get("is_healthy", True))
        amb_raw    = min(unhealthy, self.MAX_UNHEALTHY) / self.MAX_UNHEALTHY
        amb_dim    = amb_raw * self.WEIGHTS["ambiguity_score"]

        # 4. Time pressure (already normalized [0,1])
        tp_dim     = float(tp) * self.WEIGHTS["time_pressure"]

        # 5. Prior failures — fraction of past steps where root cause was wrong
        pf_raw     = 0.0
        if prior_step_logs:
            total     = len(prior_step_logs)
            wrong     = sum(1 for sl in prior_step_logs if not sl.get("root_cause_correct", True))
            pf_raw    = wrong / total if total > 0 else 0.0
        pf_dim     = pf_raw * self.WEIGHTS["prior_failures"]

        total      = round(alert_dim + depth_dim + amb_dim + tp_dim + pf_dim, 4)
        breakdown  = {
            "alert_count":     round(alert_dim,  4),
            "cascade_depth":   round(depth_dim,  4),
            "ambiguity_score": round(amb_dim,    4),
            "time_pressure":   round(tp_dim,     4),
            "prior_failures":  round(pf_dim,     4),
            "total":           total,
        }
        return total, breakdown

    @staticmethod
    def _cascade_depth(topology: List[Dict[str, Any]]) -> int:
        """BFS longest path in the service call graph."""
        if not topology:
            return 0
        # Build adjacency list (downstream direction = deeper)
        graph: Dict[str, List[str]] = {}
        for edge in topology:
            up = edge.get("upstream", edge.get("upstream_service", ""))
            dn = edge.get("downstream", edge.get("downstream_service", ""))
            if up and dn:
                graph.setdefault(up, []).append(dn)

        # Find nodes with no incoming edges (entry points)
        all_nodes   = set(graph.keys())
        downstream  = {v for vs in graph.values() for v in vs}
        roots       = all_nodes - downstream or all_nodes

        def dfs(node: str, visited: set) -> int:
            if node in visited:
                return 0
            visited = visited | {node}
            children = graph.get(node, [])
            if not children:
                return 1
            return 1 + max(dfs(c, visited) for c in children)

        return max((dfs(r, set()) for r in roots), default=0)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(
        self,
        complexity: float,
        ltm_recommendation: Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        """
        Map complexity score to model tier.
        Optionally incorporates LTM routing recommendation.

        Returns: (tier_name, base_url, model_name, api_key)
        """
        # LTM override: if LTM strongly recommends a tier (≥20 data points)
        if ltm_recommendation in ("fast", "balanced", "strong"):
            tier = ltm_recommendation
        elif complexity < FAST_THRESHOLD:
            tier = "fast"
        elif complexity < STRONG_THRESHOLD:
            tier = "balanced"
        else:
            tier = "strong"

        config = self._tier_config(tier)
        return tier, config["base_url"], config["model"], config["api_key"]

    def _tier_config(self, tier: str) -> Dict[str, str]:
        if tier == "fast":
            return {
                "base_url": FAST_BASE_URL,
                "model":    FAST_MODEL,
                "api_key":  FAST_API_KEY,
            }
        if tier == "strong":
            return {
                "base_url": STRONG_BASE_URL,
                "model":    STRONG_MODEL,
                "api_key":  STRONG_API_KEY,
            }
        # balanced (default)
        return {
            "base_url": BALANCED_BASE_URL,
            "model":    BALANCED_MODEL,
            "api_key":  BALANCED_API_KEY,
        }

    def has_api_key(self, tier: str) -> bool:
        """Check if an API key is configured for this tier."""
        cfg = self._tier_config(tier)
        return bool(cfg["api_key"] and cfg["api_key"] != "your_" + tier + "_api_key_here")

    # ── History tracking ──────────────────────────────────────────────────────

    def record_outcome(
        self,
        episode: int,
        step: int,
        complexity: float,
        tier_used: str,
        reward: float,
        root_cause_correct: bool,
    ) -> None:
        """Record a routing decision outcome for learning analytics."""
        self._routing_history.append({
            "episode":            episode,
            "step":               step,
            "complexity":         round(complexity, 4),
            "tier_used":          tier_used,
            "reward":             round(reward, 4),
            "root_cause_correct": root_cause_correct,
            "ts":                 time.time(),
        })

    def get_routing_history(self) -> List[Dict[str, Any]]:
        return self._routing_history

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Compute per-tier accuracy and reward stats.
        Used by the before/after report generator.
        """
        if not self._routing_history:
            return {}

        by_tier: Dict[str, List[Dict]] = {}
        for rec in self._routing_history:
            t = rec["tier_used"]
            by_tier.setdefault(t, []).append(rec)

        stats = {}
        for tier, recs in by_tier.items():
            rewards   = [r["reward"] for r in recs]
            correct   = [r["root_cause_correct"] for r in recs]
            complexities = [r["complexity"] for r in recs]
            stats[tier] = {
                "count":          len(recs),
                "pct":            round(len(recs) / len(self._routing_history) * 100, 1),
                "avg_reward":     round(sum(rewards) / len(rewards), 4),
                "accuracy":       round(sum(correct) / len(correct) * 100, 1),
                "avg_complexity": round(sum(complexities) / len(complexities), 3),
            }
        return stats
