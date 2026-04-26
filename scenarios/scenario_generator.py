"""
scenarios/scenario_generator.py
--------------------------------
Dynamic scenario variant generator for the Incident Response Environment.

Produces randomized scenario variants from base templates while preserving
ground truth integrity for grader compatibility. Ensures agents cannot
overfit to fixed metric values and must actually learn incident triage.

Design invariants:
    1. Ground truth (root cause, severity, correct action) is NEVER modified
    2. Grader rubric remains identical — scores are deterministic & comparable
    3. All randomization is seeded for reproducibility
    4. Variants are structurally valid and pass scenario validation

Key randomizations:
    - Service metric jitter (CPU, memory, latency ± configurable %)
    - Alert value jitter (current_value wobbles, threshold stays fixed)
    - Timeline event shuffle (same-step events reordered)
    - Red herring diversity (extra distractors injected from pool)
    - Topology latency jitter (baseline & current latencies vary)
    - Correlated noise (CPU↑ → memory↑ for realism)
"""

import copy
import random
from typing import Any, Dict, List, Optional, Set

from scenarios.base_scenario import BaseScenario, load_scenario


# ── Extra red herring service pool ────────────────────────────────────────────
# Plausible but unrelated services that might alert during a real incident.
# Each maps to a realistic reason why it would fire during an outage.
EXTRA_RED_HERRING_POOL: List[Dict[str, Any]] = [
    {
        "name": "monitoring-agent",
        "normal_cpu": 0.15, "normal_mem": 0.30,
        "incident_cpu": 0.91, "incident_mem": 0.32,
        "reason": "Monitoring agent CPU spike from 10x scrape frequency during incident.",
        "alert_metric": "cpu_utilization",
        "alert_severity": "warning",
    },
    {
        "name": "log-aggregator",
        "normal_cpu": 0.22, "normal_mem": 0.45,
        "incident_cpu": 0.25, "incident_mem": 0.89,
        "reason": "Log pipeline memory pressure from error flood (10x normal volume).",
        "alert_metric": "memory_utilization",
        "alert_severity": "warning",
    },
    {
        "name": "ci-runner-02",
        "normal_cpu": 0.60, "normal_mem": 0.55,
        "incident_cpu": 0.95, "incident_mem": 0.58,
        "reason": "CI pipeline integration tests scheduled. Unrelated to incident.",
        "alert_metric": "cpu_utilization",
        "alert_severity": "warning",
    },
    {
        "name": "backup-daemon",
        "normal_cpu": 0.10, "normal_mem": 0.35,
        "incident_cpu": 0.12, "incident_mem": 0.92,
        "reason": "Nightly backup job running on schedule. Unrelated to incident.",
        "alert_metric": "memory_utilization",
        "alert_severity": "info",
    },
    {
        "name": "metrics-exporter",
        "normal_cpu": 0.18, "normal_mem": 0.28,
        "incident_cpu": 0.88, "incident_mem": 0.35,
        "reason": "Prometheus exporter overwhelmed by cardinality explosion from incident alerts.",
        "alert_metric": "cpu_utilization",
        "alert_severity": "warning",
    },
    {
        "name": "etcd-node-3",
        "normal_cpu": 0.25, "normal_mem": 0.40,
        "incident_cpu": 0.27, "incident_mem": 0.87,
        "reason": "etcd memory growth from watch stream backlog. Self-resolving.",
        "alert_metric": "memory_utilization",
        "alert_severity": "warning",
    },
    {
        "name": "cron-scheduler",
        "normal_cpu": 0.08, "normal_mem": 0.20,
        "incident_cpu": 0.92, "incident_mem": 0.22,
        "reason": "Cron job executing batch data import. Pre-scheduled task.",
        "alert_metric": "cpu_utilization",
        "alert_severity": "info",
    },
    {
        "name": "image-processor",
        "normal_cpu": 0.70, "normal_mem": 0.60,
        "incident_cpu": 0.98, "incident_mem": 0.62,
        "reason": "Image resize queue backlog from marketing campaign. Unrelated.",
        "alert_metric": "cpu_utilization",
        "alert_severity": "warning",
    },
    {
        "name": "mail-relay",
        "normal_cpu": 0.12, "normal_mem": 0.25,
        "incident_cpu": 0.14, "incident_mem": 0.86,
        "reason": "Email retry queue growing from alert notification storm.",
        "alert_metric": "memory_utilization",
        "alert_severity": "info",
    },
]


# ── Noise configuration per difficulty ────────────────────────────────────────
NOISE_PROFILES: Dict[str, Dict[str, Any]] = {
    "easy": {
        "metric_jitter_pct":      0.10,   # ±10% on metric values
        "alert_jitter_pct":       0.08,   # ±8% on alert current_value
        "latency_jitter_pct":     0.12,   # ±12% on topology latencies
        "extra_red_herrings":     0,      # No extra noise for easy
        "shuffle_timeline":       True,   # Shuffle same-step events
        "randomize_rh_severity":  False,  # Keep red herring severities as-is
        "correlated_noise":       False,  # No correlated CPU/mem noise
    },
    "medium": {
        "metric_jitter_pct":      0.18,
        "alert_jitter_pct":       0.15,
        "latency_jitter_pct":     0.15,
        "extra_red_herrings":     1,      # 1 extra distractor
        "shuffle_timeline":       True,
        "randomize_rh_severity":  True,   # Sometimes make RH look critical
        "correlated_noise":       True,   # Realistic CPU↔mem correlation
    },
    "hard": {
        "metric_jitter_pct":      0.25,
        "alert_jitter_pct":       0.20,
        "latency_jitter_pct":     0.20,
        "extra_red_herrings":     2,      # 2 extra distractors
        "shuffle_timeline":       True,
        "randomize_rh_severity":  True,
        "correlated_noise":       True,
    },
    "expert": {
        "metric_jitter_pct":      0.30,   # Maximum jitter
        "alert_jitter_pct":       0.25,
        "latency_jitter_pct":     0.25,
        "extra_red_herrings":     3,      # 3 extra distractors
        "shuffle_timeline":       True,
        "randomize_rh_severity":  True,
        "correlated_noise":       True,
    },
    "positive_easy": {
        "metric_jitter_pct":      0.05,
        "alert_jitter_pct":       0.05,
        "latency_jitter_pct":     0.06,
        "extra_red_herrings":     0,
        "shuffle_timeline":       True,
        "randomize_rh_severity":  False,
        "correlated_noise":       False,
    },
    "positive_medium": {
        "metric_jitter_pct":      0.08,
        "alert_jitter_pct":       0.08,
        "latency_jitter_pct":     0.10,
        "extra_red_herrings":     1,
        "shuffle_timeline":       True,
        "randomize_rh_severity":  True,
        "correlated_noise":       False,
    },
}


class DynamicScenario(BaseScenario):
    """
    A scenario variant generated from a base template.

    Inherits all property accessors from BaseScenario but
    bypasses file-based loading — data is injected directly
    by the ScenarioVariantGenerator.

    Ground truth and grader rubric remain identical to the
    base template, ensuring score comparability across variants.
    """

    def __init__(
        self,
        difficulty: str,
        scenario_data: Dict[str, Any],
        metadata: Dict[str, Any],
        variant_id: str = "v0",
    ) -> None:
        # Bypass BaseScenario.__init__ — no file I/O
        self.difficulty = difficulty
        self._scenario  = scenario_data
        self._metadata  = metadata
        self.variant_id = variant_id

    def validate(self) -> bool:
        """Validate structural integrity of generated variant."""
        assert len(self.services) >= 2, \
            f"Need ≥2 services, got {len(self.services)}"
        assert self.ground_truth.get("root_cause_service"), \
            "Missing root cause in ground truth"
        assert any(s.get("is_root_cause") for s in self.services), \
            "No service marked as root cause"
        assert len(self.alerts) >= 1, \
            f"Need ≥1 alert, got {len(self.alerts)}"
        assert len(self.topology) >= 1, \
            f"Need ≥1 topology edge, got {len(self.topology)}"

        # Verify root cause service still exists in services list
        rc = self.ground_truth["root_cause_service"]
        svc_names = {s["name"] for s in self.services}
        assert rc in svc_names, \
            f"Root cause '{rc}' missing from services: {svc_names}"

        return True

    def __repr__(self) -> str:
        return (
            f"DynamicScenario(difficulty={self.difficulty!r}, "
            f"variant={self.variant_id!r}, "
            f"services={len(self.services)}, "
            f"alerts={len(self.alerts)})"
        )


class ScenarioVariantGenerator:
    """
    Generates randomized scenario variants from base templates.

    Each call to generate() produces a unique variant with
    jittered metrics, shuffled events, and optionally extra
    distractor services — while preserving ground truth for
    deterministic scoring.

    Usage::

        gen = ScenarioVariantGenerator("easy", seed=42)
        v1 = gen.generate()   # DynamicScenario with jittered values
        v2 = gen.generate()   # Different variant (RNG state advances)

        # Reproducible: same seed → same sequence
        gen2 = ScenarioVariantGenerator("easy", seed=42)
        v3 = gen2.generate()  # v3 has same values as v1
    """

    def __init__(self, difficulty: str, seed: Optional[int] = None) -> None:
        if difficulty not in NOISE_PROFILES:
            raise ValueError(
                f"Unknown difficulty '{difficulty}'. "
                f"Use: {list(NOISE_PROFILES)}"
            )

        self.difficulty       = difficulty
        self.profile          = NOISE_PROFILES[difficulty]
        self.rng              = random.Random(seed)
        self._variant_counter = 0

        # Load base template once — reused across variants
        self._base = load_scenario(difficulty)

    def generate(self) -> DynamicScenario:
        """
        Generate one randomized scenario variant.

        Returns a DynamicScenario with the same interface as
        BaseScenario subclasses. Safe to pass to graders, env,
        and observation builders.
        """
        self._variant_counter += 1
        variant_id = f"v{self._variant_counter}"

        # Deep copy to avoid mutating the template
        scenario_data = copy.deepcopy(self._base._scenario)
        metadata      = copy.deepcopy(self._base._metadata)

        # ── Apply randomizations in deterministic order ──────────────

        # 1. Jitter service metrics (CPU, memory, RT, MCR)
        self._jitter_service_metrics(scenario_data["services"])

        # 2. Apply correlated noise (CPU↑ → memory↑)
        if self.profile["correlated_noise"]:
            self._apply_correlated_noise(scenario_data["services"])

        # 3. Jitter alert values (threshold stays fixed)
        self._jitter_alert_values(scenario_data["alerts"])

        # 4. Jitter topology latencies
        self._jitter_topology_latencies(scenario_data["topology"])

        # 5. Shuffle same-step timeline events
        if self.profile["shuffle_timeline"]:
            self._shuffle_same_step_events(scenario_data["timeline"])

        # 6. Inject extra red herring services & alerts
        n_extra = self.profile["extra_red_herrings"]
        if n_extra > 0:
            self._inject_extra_red_herrings(scenario_data, n_extra)

        # 7. Randomize red herring alert severities
        if self.profile["randomize_rh_severity"]:
            self._randomize_red_herring_severity(scenario_data["alerts"])

        # Build variant and validate
        variant = DynamicScenario(
            difficulty    = self.difficulty,
            scenario_data = scenario_data,
            metadata      = metadata,
            variant_id    = variant_id,
        )
        variant.validate()
        return variant

    # ── Private randomization methods ─────────────────────────────────────

    def _jitter(
        self,
        value: float,
        pct: float,
        lower: float = 0.0,
        upper: float = float("inf"),
    ) -> float:
        """Apply bounded percentage jitter using triangular distribution."""
        # Triangular distribution clusters near 0 → smaller typical changes
        delta = value * pct * (self.rng.triangular(-1.0, 1.0, 0.0))
        return round(max(lower, min(upper, value + delta)), 4)

    def _jitter_service_metrics(self, services: List[Dict[str, Any]]) -> None:
        """
        Jitter CPU, memory, and latency metrics for all services.

        Root cause services get only 30% of the jitter range to
        keep the signal strong enough for agent detection.
        """
        pct = self.profile["metric_jitter_pct"]

        for svc in services:
            is_root = svc.get("is_root_cause", False)
            # Root cause: light jitter to preserve signal strength
            effective_pct = pct * 0.3 if is_root else pct

            # CPU metrics (clamped 0.0–1.0)
            for key in ("normal_cpu", "incident_cpu"):
                if key in svc:
                    svc[key] = self._jitter(svc[key], effective_pct, 0.01, 0.99)

            # Memory metrics (clamped 0.0–1.0)
            for key in ("normal_mem", "incident_mem"):
                if key in svc:
                    svc[key] = self._jitter(svc[key], effective_pct, 0.01, 0.99)

            # Latency / response time metrics (must stay positive)
            for key in (
                "normal_http_rt", "incident_http_rt",
                "normal_consumer_rpc_rt", "incident_consumer_rpc_rt",
                "normal_provider_rpc_rt", "incident_provider_rpc_rt",
            ):
                if key in svc and svc[key] is not None and svc[key] > 0:
                    svc[key] = self._jitter(svc[key], effective_pct, 0.1)

            # MCR / call rate (must stay ≥ 0)
            for key in ("normal_mcr", "incident_mcr"):
                if key in svc and svc[key] is not None:
                    svc[key] = self._jitter(svc[key], effective_pct, 0.0)

    def _apply_correlated_noise(self, services: List[Dict[str, Any]]) -> None:
        """
        Apply correlated noise: when CPU goes up, memory should
        increase proportionally. Makes generated data realistic.

        Correlation model:
            Δmem = Δcpu × 0.4 + independent_noise × 0.1
        """
        for svc in services:
            if svc.get("is_root_cause"):
                continue  # Don't corrupt root cause signal

            cpu_normal   = svc.get("normal_cpu")
            cpu_incident = svc.get("incident_cpu")
            mem_incident = svc.get("incident_mem")

            if cpu_normal is not None and cpu_incident is not None and mem_incident is not None:
                cpu_delta = cpu_incident - cpu_normal
                # Push memory in same direction as CPU with 40% correlation
                mem_push  = cpu_delta * 0.4 * self.rng.uniform(0.5, 1.5)
                new_mem   = mem_incident + mem_push
                svc["incident_mem"] = round(max(0.01, min(0.99, new_mem)), 4)

    def _jitter_alert_values(self, alerts: List[Dict[str, Any]]) -> None:
        """
        Jitter alert current_value while keeping threshold fixed.

        For real alerts: ensures value stays on the breaching side of
        the threshold (above for spikes, below for drops).

        For red herring alerts: applies 1.5× jitter for more variety.
        """
        pct = self.profile["alert_jitter_pct"]

        for alert in alerts:
            original  = alert["current_value"]
            threshold = alert.get("threshold", 0)

            if alert.get("is_red_herring", False):
                # Red herring: more aggressive jitter
                alert["current_value"] = round(
                    self._jitter(original, pct * 1.5, 0.0), 2
                )
            else:
                # Real alert: jitter but maintain threshold breach
                val = self._jitter(original, pct, 0.0)

                if original > threshold:
                    # "Above threshold" alert (e.g., latency spike)
                    val = max(val, threshold * 1.05)
                elif original < threshold:
                    # "Below threshold" alert (e.g., MCR dropped)
                    val = min(val, threshold * 0.95)

                alert["current_value"] = round(val, 2)

    def _jitter_topology_latencies(self, topology: List[Dict[str, Any]]) -> None:
        """Jitter avg_rt_ms for topology edges."""
        pct = self.profile["latency_jitter_pct"]
        for edge in topology:
            if "avg_rt_ms" in edge:
                edge["avg_rt_ms"] = self._jitter(
                    edge["avg_rt_ms"], pct, 0.1
                )

    def _shuffle_same_step_events(self, timeline: List[Dict[str, Any]]) -> None:
        """
        Shuffle events that occur at the same step.

        Events at different steps maintain chronological order;
        only concurrent (same-step) events are reordered. This
        tests whether agents rely on event ordering within a step.
        """
        if not timeline:
            return

        # Group indices by step
        step_groups: Dict[int, List[int]] = {}
        for i, event in enumerate(timeline):
            step = event.get("step", 0)
            step_groups.setdefault(step, []).append(i)

        # Shuffle within each step group
        for indices in step_groups.values():
            if len(indices) > 1:
                events = [timeline[i] for i in indices]
                self.rng.shuffle(events)
                for j, idx in enumerate(indices):
                    timeline[idx] = events[j]

    def _inject_extra_red_herrings(
        self,
        scenario_data: Dict[str, Any],
        count: int,
    ) -> None:
        """
        Inject extra red herring services, alerts, and timeline
        events from the distractor pool.

        Injected services are marked as is_red_herring=True and
        will never match the ground truth root cause. The grader
        rubric is NOT modified — naming these as root cause still
        gets a 0 score for root_cause_service, which is correct.
        """
        # Avoid duplicating existing service names
        existing_names: Set[str] = {s["name"] for s in scenario_data["services"]}
        available = [
            rh for rh in EXTRA_RED_HERRING_POOL
            if rh["name"] not in existing_names
        ]

        if not available:
            return

        selected = self.rng.sample(available, min(count, len(available)))
        existing_alert_ids: Set[str] = {
            a["alert_id"] for a in scenario_data["alerts"]
        }

        for rh in selected:
            # ── Service entry ────────────────────────────────────────
            svc = {
                "name":              rh["name"],
                "is_root_cause":     False,
                "is_red_herring":    True,
                "is_stateful":       False,
                "normal_cpu":        rh["normal_cpu"],
                "normal_mem":        rh["normal_mem"],
                "incident_cpu":      self._jitter(
                    rh["incident_cpu"], 0.1, 0.5, 0.99
                ),
                "incident_mem":      self._jitter(
                    rh["incident_mem"], 0.1, 0.1, 0.99
                ),
                "red_herring_reason": rh["reason"],
            }
            scenario_data["services"].append(svc)

            # ── Alert entry ──────────────────────────────────────────
            alert_id = f"ALT-RH-{self.rng.randint(100, 999)}"
            while alert_id in existing_alert_ids:
                alert_id = f"ALT-RH-{self.rng.randint(100, 999)}"
            existing_alert_ids.add(alert_id)

            metric = rh["alert_metric"]
            if metric == "cpu_utilization":
                val       = svc["incident_cpu"]
                threshold = 0.85
            else:  # memory_utilization
                val       = svc["incident_mem"]
                threshold = 0.80

            alert = {
                "alert_id":      alert_id,
                "service":       rh["name"],
                "metric":        metric,
                "current_value": round(val, 4),
                "threshold":     threshold,
                "severity":      rh["alert_severity"],
                "fired_at_step": 0,
                "is_red_herring": True,
            }
            scenario_data["alerts"].append(alert)

            # ── Timeline entry ───────────────────────────────────────
            scenario_data["timeline"].append({
                "step":        0,
                "event_type":  "unrelated",
                "service":     rh["name"],
                "description": rh["reason"],
            })

    def _randomize_red_herring_severity(
        self, alerts: List[Dict[str, Any]]
    ) -> None:
        """
        Randomly promote/demote red herring alert severities.

        This makes some runs trickier — a red herring might appear
        as "critical", tempting the agent to investigate it.
        """
        severity_options = ["info", "warning", "critical"]
        # Weighted toward warning/critical to be challenging
        weights = [0.2, 0.5, 0.3]

        for alert in alerts:
            if alert.get("is_red_herring", False):
                alert["severity"] = self.rng.choices(
                    severity_options, weights=weights
                )[0]


# ── Public convenience function ───────────────────────────────────────────────

def generate_scenario_variant(
    difficulty: str,
    seed: Optional[int] = None,
) -> DynamicScenario:
    """
    Generate a single dynamic scenario variant.

    This is the primary entry point for the environment.
    On each reset(), the env calls this to get a fresh variant
    with randomized metrics while preserving ground truth.

    Args:
        difficulty: "easy", "medium", or "hard"
        seed: Optional RNG seed for reproducible variants.
              If None, uses system entropy (non-deterministic).

    Returns:
        DynamicScenario: fully compatible with BaseScenario interface

    Example::

        scenario = generate_scenario_variant("hard", seed=42)
        print(scenario.name)                # Same base name
        print(scenario.services[0])         # Jittered metrics
        print(scenario.ground_truth)        # Unchanged
        print(scenario.grader_rubric)       # Unchanged
    """
    return ScenarioVariantGenerator(difficulty, seed=seed).generate()
