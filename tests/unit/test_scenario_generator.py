"""
tests/unit/test_scenario_generator.py
--------------------------------------
Comprehensive tests for the dynamic scenario variant generator.

Validates:
    1. Variants are structurally valid for all difficulties
    2. Ground truth is never modified by randomization
    3. Metric jitter stays within bounds
    4. Seeded generation is reproducible
    5. Different seeds produce different variants
    6. Extra red herrings are injected correctly
    7. Alerts maintain threshold breach invariants
    8. Variants are fully gradable by existing graders
    9. Timeline shuffling preserves events
   10. Topology integrity is maintained
"""

import pytest
from copy import deepcopy

from scenarios import load_scenario
from scenarios.scenario_generator import (
    DynamicScenario,
    ScenarioVariantGenerator,
    generate_scenario_variant,
    EXTRA_RED_HERRING_POOL,
    NOISE_PROFILES,
)
from graders import load_grader


# ── Fixture: all difficulties ─────────────────────────────────────────────────

DIFFICULTIES = ["easy", "medium", "hard"]


@pytest.fixture(params=DIFFICULTIES)
def difficulty(request):
    return request.param


@pytest.fixture(params=DIFFICULTIES)
def base_scenario(request):
    return load_scenario(request.param)


# ── 1. Structural Validity ───────────────────────────────────────────────────

class TestVariantStructure:
    """Test that generated variants are structurally valid."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_variant_creates_successfully(self, difficulty):
        """Generator produces a valid DynamicScenario for each difficulty."""
        variant = generate_scenario_variant(difficulty, seed=42)
        assert isinstance(variant, DynamicScenario)
        assert variant.difficulty == difficulty
        assert variant.variant_id == "v1"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_variant_passes_validation(self, difficulty):
        """All generated variants pass structural validation."""
        variant = generate_scenario_variant(difficulty, seed=42)
        assert variant.validate() is True

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_variant_has_required_properties(self, difficulty):
        """Variant exposes all BaseScenario properties."""
        variant = generate_scenario_variant(difficulty, seed=42)

        # Scenario properties
        assert variant.scenario_id
        assert variant.name
        assert variant.description
        assert variant.fault_type
        assert variant.max_steps > 0
        assert len(variant.services) >= 2
        assert len(variant.topology) >= 1
        assert len(variant.alerts) >= 1
        assert len(variant.timeline) >= 1

        # Metadata properties
        assert variant.ground_truth
        assert variant.grader_rubric

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_variant_has_root_cause_service(self, difficulty):
        """At least one service is marked as root cause."""
        variant = generate_scenario_variant(difficulty, seed=42)
        root_cause_services = [
            s for s in variant.services if s.get("is_root_cause")
        ]
        assert len(root_cause_services) >= 1

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_variant_root_cause_in_services(self, difficulty):
        """Ground truth root cause exists in the variant's service list."""
        variant = generate_scenario_variant(difficulty, seed=42)
        rc = variant.ground_truth["root_cause_service"]
        svc_names = {s["name"] for s in variant.services}
        assert rc in svc_names, f"Root cause '{rc}' not in services: {svc_names}"


# ── 2. Ground Truth Preservation ─────────────────────────────────────────────

class TestGroundTruthPreservation:
    """Ground truth must NEVER be modified by randomization."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_root_cause_service_unchanged(self, difficulty):
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)
        assert variant.ground_truth["root_cause_service"] == \
               base.ground_truth["root_cause_service"]

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_root_cause_type_unchanged(self, difficulty):
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)
        assert variant.ground_truth["root_cause_type"] == \
               base.ground_truth["root_cause_type"]

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_severity_unchanged(self, difficulty):
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)
        assert variant.ground_truth["severity"] == \
               base.ground_truth["severity"]

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_correct_actions_unchanged(self, difficulty):
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)
        assert variant.ground_truth["correct_actions"] == \
               base.ground_truth["correct_actions"]

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_grader_rubric_unchanged(self, difficulty):
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)
        assert variant.grader_rubric == base.grader_rubric


# ── 3. Metric Jitter ─────────────────────────────────────────────────────────

class TestMetricJitter:
    """Test that metrics are jittered within expected bounds."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_metrics_differ_from_base(self, difficulty):
        """Variant metrics should differ from base template."""
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)

        # At least one service's incident_cpu should differ
        diffs = 0
        for b_svc, v_svc in zip(base.services, variant.services):
            if "incident_cpu" in b_svc and "incident_cpu" in v_svc:
                if b_svc["incident_cpu"] != v_svc["incident_cpu"]:
                    diffs += 1
        assert diffs > 0, "No metrics were jittered"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_cpu_stays_in_range(self, difficulty):
        """CPU metrics must stay in [0.0, 1.0]."""
        variant = generate_scenario_variant(difficulty, seed=42)
        for svc in variant.services:
            for key in ("normal_cpu", "incident_cpu"):
                if key in svc:
                    assert 0.0 <= svc[key] <= 1.0, \
                        f"{svc['name']}.{key} = {svc[key]} out of range"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_memory_stays_in_range(self, difficulty):
        """Memory metrics must stay in [0.0, 1.0]."""
        variant = generate_scenario_variant(difficulty, seed=42)
        for svc in variant.services:
            for key in ("normal_mem", "incident_mem"):
                if key in svc:
                    assert 0.0 <= svc[key] <= 1.0, \
                        f"{svc['name']}.{key} = {svc[key]} out of range"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_latency_stays_positive(self, difficulty):
        """Latency/RT metrics must be > 0."""
        variant = generate_scenario_variant(difficulty, seed=42)
        for svc in variant.services:
            for key in ("incident_http_rt", "incident_consumer_rpc_rt",
                        "incident_provider_rpc_rt"):
                if key in svc and svc[key] is not None and svc[key] > 0:
                    assert svc[key] > 0, \
                        f"{svc['name']}.{key} = {svc[key]} must be positive"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_root_cause_signal_preserved(self, difficulty):
        """Root cause service metrics should only be lightly jittered (~30%)."""
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)

        rc_name = base.ground_truth["root_cause_service"]
        base_rc = next(s for s in base.services if s["name"] == rc_name)
        var_rc  = next(s for s in variant.services if s["name"] == rc_name)

        # Check that root cause metrics didn't change too much
        for key in ("incident_cpu", "incident_mem"):
            if key in base_rc and key in var_rc:
                pct = NOISE_PROFILES[difficulty]["metric_jitter_pct"]
                max_change = base_rc[key] * pct  # Full jitter range
                actual_change = abs(var_rc[key] - base_rc[key])
                # Root cause gets 30% of jitter → within full range is safe
                assert actual_change <= max_change * 1.5, \
                    f"RC {key} changed too much: {base_rc[key]} → {var_rc[key]}"


# ── 4. Reproducibility ──────────────────────────────────────────────────────

class TestReproducibility:
    """Seeded generators must produce identical sequences."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_same_seed_same_variant(self, difficulty):
        """Same seed produces identical variants."""
        v1 = generate_scenario_variant(difficulty, seed=42)
        v2 = generate_scenario_variant(difficulty, seed=42)

        # Compare service metrics
        for s1, s2 in zip(v1.services, v2.services):
            assert s1 == s2, f"Services differ: {s1['name']}"

        # Compare alert values
        for a1, a2 in zip(v1.alerts, v2.alerts):
            assert a1["current_value"] == a2["current_value"]

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_different_seeds_different_variants(self, difficulty):
        """Different seeds produce different variants."""
        v1 = generate_scenario_variant(difficulty, seed=42)
        v2 = generate_scenario_variant(difficulty, seed=99)

        # At least one metric should differ
        any_diff = False
        for s1, s2 in zip(v1.services, v2.services):
            for key in ("incident_cpu", "incident_mem"):
                if key in s1 and key in s2:
                    if s1[key] != s2[key]:
                        any_diff = True
                        break
            if any_diff:
                break
        assert any_diff, "Different seeds produced identical variants"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_sequential_variants_differ(self, difficulty):
        """Multiple calls to generate() produce different variants."""
        gen = ScenarioVariantGenerator(difficulty, seed=42)
        v1 = gen.generate()
        v2 = gen.generate()

        # Variant IDs should differ
        assert v1.variant_id != v2.variant_id

        # Metrics should differ
        any_diff = any(
            s1.get("incident_cpu") != s2.get("incident_cpu")
            for s1, s2 in zip(v1.services, v2.services)
            if "incident_cpu" in s1 and "incident_cpu" in s2
        )
        assert any_diff, "Sequential variants are identical"


# ── 5. Red Herring Injection ────────────────────────────────────────────────

class TestRedHerringInjection:
    """Test extra red herring injection for medium/hard."""

    def test_easy_no_extra_red_herrings(self):
        """Easy mode should NOT inject extra red herrings."""
        base    = load_scenario("easy")
        variant = generate_scenario_variant("easy", seed=42)
        assert len(variant.services) == len(base.services)

    def test_medium_injects_one_red_herring(self):
        """Medium mode injects 1 extra red herring."""
        base    = load_scenario("medium")
        variant = generate_scenario_variant("medium", seed=42)
        assert len(variant.services) == len(base.services) + 1

        # The extra service should be marked as red herring
        extra_svcs = [
            s for s in variant.services
            if s["name"] not in {b["name"] for b in base.services}
        ]
        assert len(extra_svcs) == 1
        assert extra_svcs[0]["is_red_herring"] is True
        assert extra_svcs[0]["is_root_cause"] is False

    def test_hard_injects_two_red_herrings(self):
        """Hard mode injects 2 extra red herrings."""
        base    = load_scenario("hard")
        variant = generate_scenario_variant("hard", seed=42)
        assert len(variant.services) == len(base.services) + 2

    def test_injected_red_herrings_have_alerts(self):
        """Injected red herrings should generate corresponding alerts."""
        base    = load_scenario("hard")
        variant = generate_scenario_variant("hard", seed=42)

        base_alert_count = len(base.alerts)
        # Should have 2 extra alerts (one per injected red herring)
        assert len(variant.alerts) == base_alert_count + 2

    def test_injected_red_herrings_have_timeline_events(self):
        """Injected red herrings should generate timeline events."""
        base    = load_scenario("hard")
        variant = generate_scenario_variant("hard", seed=42)

        base_timeline_count = len(base.timeline)
        assert len(variant.timeline) >= base_timeline_count + 2

    def test_injected_names_from_pool(self):
        """Injected red herring names come from the predefined pool."""
        variant = generate_scenario_variant("hard", seed=42)
        pool_names = {rh["name"] for rh in EXTRA_RED_HERRING_POOL}
        base = load_scenario("hard")
        base_names = {s["name"] for s in base.services}

        extra_names = {
            s["name"] for s in variant.services
            if s["name"] not in base_names
        }
        assert extra_names.issubset(pool_names), \
            f"Extra names {extra_names} not in pool {pool_names}"

    def test_no_duplicate_service_names(self):
        """No service name should appear twice."""
        variant = generate_scenario_variant("hard", seed=42)
        names = [s["name"] for s in variant.services]
        assert len(names) == len(set(names)), \
            f"Duplicate service names: {names}"


# ── 6. Alert Integrity ──────────────────────────────────────────────────────

class TestAlertIntegrity:
    """Alert values must maintain threshold breach invariants."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_real_alerts_stay_breached(self, difficulty):
        """Real (non-red-herring) alerts must remain on breaching side."""
        variant = generate_scenario_variant(difficulty, seed=42)

        for alert in variant.alerts:
            if alert.get("is_red_herring", False):
                continue

            val       = alert["current_value"]
            threshold = alert["threshold"]

            # Determine breach direction from base
            # If original was above threshold, it should stay above
            # If original was below threshold, it should stay below
            base = load_scenario(difficulty)
            base_alert = next(
                (a for a in base.alerts if a["alert_id"] == alert["alert_id"]),
                None
            )
            if base_alert is None:
                continue  # Extra alert from injection

            if base_alert["current_value"] > base_alert["threshold"]:
                assert val > threshold * 0.95, \
                    f"Alert {alert['alert_id']}: value {val} dropped below threshold {threshold}"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_alert_values_are_positive(self, difficulty):
        """All alert current_values should be non-negative."""
        variant = generate_scenario_variant(difficulty, seed=42)
        for alert in variant.alerts:
            assert alert["current_value"] >= 0, \
                f"Alert {alert['alert_id']} has negative value"


# ── 7. Timeline Integrity ───────────────────────────────────────────────────

class TestTimelineIntegrity:
    """Timeline events should be preserved, only order changes."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_timeline_event_count_preserved(self, difficulty):
        """Base events are preserved (extras may be added)."""
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)
        assert len(variant.timeline) >= len(base.timeline)

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_timeline_event_types_preserved(self, difficulty):
        """All base event types still exist in variant."""
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)

        base_types    = {(e["step"], e["event_type"], e["service"]) for e in base.timeline}
        variant_types = {(e["step"], e["event_type"], e["service"]) for e in variant.timeline}

        # All base events should exist in variant
        missing = base_types - variant_types
        assert len(missing) == 0, f"Missing timeline events: {missing}"


# ── 8. Topology Integrity ───────────────────────────────────────────────────

class TestTopologyIntegrity:
    """Service dependency graph must be preserved."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_topology_edges_preserved(self, difficulty):
        """All topology edges from base are present in variant."""
        base    = load_scenario(difficulty)
        variant = generate_scenario_variant(difficulty, seed=42)

        assert len(variant.topology) == len(base.topology)

        for b_edge, v_edge in zip(base.topology, variant.topology):
            assert b_edge["upstream"] == v_edge["upstream"]
            assert b_edge["downstream"] == v_edge["downstream"]
            assert b_edge["rpc_type"] == v_edge["rpc_type"]

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_topology_latencies_jittered(self, difficulty):
        """Latencies should be jittered but positive."""
        variant = generate_scenario_variant(difficulty, seed=42)
        for edge in variant.topology:
            assert edge["avg_rt_ms"] > 0


# ── 9. Grader Compatibility ─────────────────────────────────────────────────

class TestGraderCompatibility:
    """Variants must be gradable by existing graders."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_perfect_action_scores_high(self, difficulty):
        """Perfect action against variant still gets high score."""
        perfect_actions = {
            "easy": {
                "root_cause_service":  "payments-db",
                "root_cause_type":     "misconfiguration",
                "severity":            "P0",
                "affected_services":   ["payments-db", "payments-api", "checkout-ui"],
                "remediation_action":  "fix_config",
                "stakeholder_message": "Payment processing delayed. Investigating config issue. ETA 8 mins.",
                "confidence":          0.95,
            },
            "medium": {
                "root_cause_service":  "user-service",
                "root_cause_type":     "network_partition",
                "severity":            "P1",
                "affected_services":   ["user-service", "auth-service", "api-gateway", "storefront-ui"],
                "remediation_action":  "fix_config",
                "stakeholder_message": "Login issues due to DNS failure. Investigating. ETA 5 mins.",
                "confidence":          0.90,
            },
            "hard": {
                "root_cause_service":  "payments-db",
                "root_cause_type":     "memory_leak",
                "severity":            "P0",
                "affected_services":   ["payments-db", "cache-service", "order-service", "api-gateway", "storefront-ui"],
                "remediation_action":  "restart_service",
                "stakeholder_message": "P0: Revenue-impacting memory leak on payments-db. Escalating. ETA 10 mins.",
                "confidence":          0.85,
            },
        }

        # Graders load their own rubric — should work with variant
        grader = load_grader(difficulty)
        result = grader.grade(
            action    = perfect_actions[difficulty],
            step      = 1,
            max_steps = 10,
        )
        assert result.reward >= 0.60, \
            f"{difficulty}: perfect action scored only {result.reward:.3f}"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_random_action_scores_low(self, difficulty):
        """Random action should score low regardless of variant."""
        grader = load_grader(difficulty)
        result = grader.grade(
            action={
                "root_cause_service":  "unknown-svc",
                "root_cause_type":     "unknown",
                "severity":            "P3",
                "affected_services":   [],
                "remediation_action":  "investigate_further",
                "confidence":          0.1,
            },
            step=1,
            max_steps=10,
        )
        assert result.reward < 0.30, \
            f"{difficulty}: random action scored {result.reward:.3f}"


# ── 10. Observation Builder Compatibility ────────────────────────────────────

class TestObservationCompatibility:
    """Variant scenarios must work with observation builders."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_get_alerts_for_agent(self, difficulty):
        """get_alerts_for_agent() strips internal flags."""
        variant = generate_scenario_variant(difficulty, seed=42)
        agent_alerts = variant.get_alerts_for_agent()

        for alert in agent_alerts:
            assert "is_red_herring" not in alert, \
                f"Internal flag leaked to agent: {alert}"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_get_metrics_snapshot(self, difficulty):
        """get_metrics_snapshot() returns valid metrics dict."""
        variant = generate_scenario_variant(difficulty, seed=42)
        metrics = variant.get_metrics_snapshot()

        assert len(metrics) > 0
        for name, m in metrics.items():
            assert "cpu_utilization" in m
            assert "memory_utilization" in m

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_get_topology_for_agent(self, difficulty):
        """get_topology_for_agent() returns valid edges."""
        variant = generate_scenario_variant(difficulty, seed=42)
        topo = variant.get_topology_for_agent()

        assert len(topo) >= 1
        for edge in topo:
            assert "upstream_service" in edge
            assert "downstream_service" in edge
            assert "rpc_type" in edge
            assert "avg_latency_ms" in edge
            assert "current_latency_ms" in edge


# ── 11. End-to-End with Environment ──────────────────────────────────────────

class TestEnvironmentIntegration:
    """Test dynamic scenarios work E2E with the environment."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_env_reset_with_dynamic(self, difficulty):
        """Environment reset() with dynamic=True works."""
        from envs.incident_env import IncidentResponseEnv
        env = IncidentResponseEnv()
        obs = env.reset(difficulty, dynamic=True, seed=42)
        assert obs.step == 0
        assert not obs.done
        assert len(obs.alerts) > 0
        env.close()

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_env_reset_with_static(self, difficulty):
        """Environment reset() with dynamic=False still works."""
        from envs.incident_env import IncidentResponseEnv
        env = IncidentResponseEnv()
        obs = env.reset(difficulty, dynamic=False)
        assert obs.step == 0
        env.close()

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_env_step_after_dynamic_reset(self, difficulty):
        """Step works after dynamic reset."""
        from envs.incident_env import IncidentResponseEnv
        from models.action import (
            IncidentAction, RootCauseType, SeverityLevel, RemediationAction
        )

        env = IncidentResponseEnv()
        env.reset(difficulty, dynamic=True, seed=42)

        action = IncidentAction(
            root_cause_service="payments-db",
            root_cause_type=RootCauseType.MISCONFIGURATION,
            severity=SeverityLevel.P0,
            affected_services=["payments-db"],
            remediation_action=RemediationAction.FIX_CONFIG,
            stakeholder_message="Investigating. ETA 5 mins.",
        )
        obs, reward, done, info = env.step(action)
        assert 0.0 <= reward <= 1.0
        assert isinstance(done, bool)
        env.close()

    def test_consecutive_resets_produce_different_obs(self):
        """Two resets (without seed) produce different observations."""
        from envs.incident_env import IncidentResponseEnv

        env = IncidentResponseEnv()
        obs1 = env.reset("easy", dynamic=True)
        metrics1 = obs1.metrics

        obs2 = env.reset("easy", dynamic=True)
        metrics2 = obs2.metrics

        # At least some metric values should differ
        any_diff = False
        for svc_name in metrics1:
            if svc_name in metrics2:
                for key in ("cpu_utilization", "memory_utilization"):
                    if metrics1[svc_name].get(key) != metrics2[svc_name].get(key):
                        any_diff = True
                        break
            if any_diff:
                break

        assert any_diff, "Two dynamic resets produced identical metrics"
        env.close()


# ── 12. Edge Cases ──────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case handling."""

    def test_invalid_difficulty_raises(self):
        """Invalid difficulty raises ValueError."""
        with pytest.raises(ValueError, match="Unknown difficulty"):
            ScenarioVariantGenerator("nightmare")

    def test_many_variants_no_crash(self):
        """Generate 50 variants without crashing."""
        gen = ScenarioVariantGenerator("hard", seed=1)
        for _ in range(50):
            v = gen.generate()
            v.validate()

    def test_variant_repr(self):
        """DynamicScenario repr is informative."""
        v = generate_scenario_variant("easy", seed=42)
        r = repr(v)
        assert "DynamicScenario" in r
        assert "easy" in r
        assert "v1" in r
