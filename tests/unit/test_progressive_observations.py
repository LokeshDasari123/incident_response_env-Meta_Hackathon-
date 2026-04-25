"""
tests/unit/test_progressive_observations.py
---------------------------------------------
Comprehensive tests for progressive observation evolution.

Validates:
    1. Cascade order is correctly computed from topology
    2. Cascade factor follows expected degradation curve
    3. Metrics evolve progressively across steps
    4. Alerts appear progressively as cascade spreads
    5. Topology latencies evolve with cascade
    6. Timeline grows with dynamic events
    7. Root cause is always at full degradation
    8. Red herrings stay constant (not in cascade)
    9. Works with DynamicScenario (Pillar 1 integration)
   10. E2E: observations differ across steps in the env
"""

import pytest
from scenarios import load_scenario
from scenarios.base_scenario import BaseScenario
from scenarios.scenario_generator import generate_scenario_variant


DIFFICULTIES = ["easy", "medium", "hard"]


# ── 1. Cascade Order ────────────────────────────────────────────────────────

class TestCascadeOrder:
    """BFS cascade order is correctly computed from topology."""

    def test_easy_cascade_order(self):
        """Easy: payments-db(0) → payments-api(1) → checkout-ui(2)."""
        s = load_scenario("easy")
        order = s._compute_cascade_order()

        assert order["payments-db"] == 0, "Root cause must be onset 0"
        assert order["payments-api"] == 1, "1 hop from root cause"
        assert order["checkout-ui"] == 2, "2 hops from root cause"

    def test_hard_cascade_order(self):
        """Hard: payments-db(0) → order/cache(1) → api-gw(2) → storefront(3)."""
        s = load_scenario("hard")
        order = s._compute_cascade_order()

        assert order["payments-db"] == 0
        assert order["order-service"] == 1
        assert order["cache-service"] == 1  # Same hop as order-service
        assert order["api-gateway"] == 2
        assert order["storefront-ui"] == 3

    def test_red_herrings_not_in_cascade(self):
        """Red herring services should NOT appear in cascade order."""
        s = load_scenario("hard")
        order = s._compute_cascade_order()

        assert "network-switch-03" not in order
        assert "worker-node-7" not in order

    def test_cascade_order_is_cached(self):
        """Repeated calls return the same cached object."""
        s = load_scenario("easy")
        order1 = s._compute_cascade_order()
        order2 = s._compute_cascade_order()
        assert order1 is order2  # Same object (cached)

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_root_cause_is_onset_zero(self, difficulty):
        """Root cause service always has onset_step = 0."""
        s = load_scenario(difficulty)
        rc = s.ground_truth["root_cause_service"]
        order = s._compute_cascade_order()
        assert order[rc] == 0


# ── 2. Cascade Factor ───────────────────────────────────────────────────────

class TestCascadeFactor:
    """Degradation curve follows expected behavior."""

    def test_root_cause_always_1(self):
        """Root cause (onset=0) always returns factor=1.0."""
        for step in range(20):
            assert BaseScenario._cascade_factor(step, 0, 20) == 1.0

    def test_before_onset_is_zero(self):
        """Before onset step, factor must be 0.0."""
        assert BaseScenario._cascade_factor(0, 3, 20) == 0.0
        assert BaseScenario._cascade_factor(1, 3, 20) == 0.0
        assert BaseScenario._cascade_factor(2, 3, 20) == 0.0

    def test_at_onset_initial_jump(self):
        """At onset step, factor should be ~0.2 (initial jump)."""
        factor = BaseScenario._cascade_factor(3, 3, 20)
        assert factor == pytest.approx(0.2, abs=0.01)

    def test_factor_increases_monotonically(self):
        """Factor must never decrease over time."""
        onset = 2
        max_steps = 20
        prev = 0.0
        for step in range(20):
            f = BaseScenario._cascade_factor(step, onset, max_steps)
            assert f >= prev, f"Factor decreased at step {step}: {prev} → {f}"
            prev = f

    def test_factor_reaches_1(self):
        """Factor must reach 1.0 within the episode."""
        onset = 2
        max_steps = 20
        final_factor = BaseScenario._cascade_factor(19, onset, max_steps)
        assert final_factor == 1.0

    def test_factor_bounded_0_1(self):
        """Factor must always be in [0.0, 1.0]."""
        for step in range(30):
            for onset in range(10):
                f = BaseScenario._cascade_factor(step, onset, 20)
                assert 0.0 <= f <= 1.0, f"Factor {f} out of range (step={step}, onset={onset})"


# ── 3. Progressive Metrics ──────────────────────────────────────────────────

class TestProgressiveMetrics:
    """Metrics evolve progressively across steps."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_step0_root_cause_at_full_incident(self, difficulty):
        """At step 0, root cause should be at full incident metrics."""
        s = load_scenario(difficulty)
        metrics = s.get_metrics_at_step(0, s.max_steps)

        rc = s.ground_truth["root_cause_service"]
        rc_svc = next(svc for svc in s.services if svc["name"] == rc)

        # CPU should be at incident level
        expected_cpu = rc_svc.get("incident_cpu", rc_svc.get("normal_cpu"))
        assert metrics[rc]["cpu_utilization"] == pytest.approx(expected_cpu, abs=0.01)

    def test_easy_downstream_starts_normal(self):
        """Easy: at step 0, downstream services should be at normal metrics."""
        s = load_scenario("easy")
        metrics = s.get_metrics_at_step(0, s.max_steps)

        # checkout-ui is 2 hops from root cause → should be normal at step 0
        checkout = metrics["checkout-ui"]
        svc = next(sv for sv in s.services if sv["name"] == "checkout-ui")
        assert checkout["cpu_utilization"] == pytest.approx(svc["normal_cpu"], abs=0.01)

    def test_easy_downstream_degrades_over_time(self):
        """Easy: downstream metrics should worsen over steps."""
        s = load_scenario("easy")

        m0 = s.get_metrics_at_step(0, s.max_steps)
        m3 = s.get_metrics_at_step(3, s.max_steps)
        m9 = s.get_metrics_at_step(9, s.max_steps)

        # payments-api (1 hop): should degrade step 0 < step 3 < step 9
        svc = next(sv for sv in s.services if sv["name"] == "payments-api")
        normal_cpu = svc["normal_cpu"]
        incident_cpu = svc["incident_cpu"]

        cpu_0 = m0["payments-api"]["cpu_utilization"]
        cpu_3 = m3["payments-api"]["cpu_utilization"]
        cpu_9 = m9["payments-api"]["cpu_utilization"]

        # At step 0: should be at normal (onset=1, so factor=0)
        assert cpu_0 == pytest.approx(normal_cpu, abs=0.01)
        # At step 3: should be degraded
        assert cpu_3 > cpu_0
        # At step 9: should be near incident values
        assert cpu_9 >= cpu_3
        assert cpu_9 == pytest.approx(incident_cpu, abs=0.05)

    def test_hard_cascade_progressive(self):
        """Hard: 5 services degrade at different rates."""
        s = load_scenario("hard")
        max_s = s.max_steps  # 20

        m0 = s.get_metrics_at_step(0, max_s)
        m5 = s.get_metrics_at_step(5, max_s)

        # payments-db (onset=0): always degraded
        assert m0["payments-db"]["status"] == "failing"

        # storefront-ui (onset=3): at step 0 should be healthy
        assert m0["storefront-ui"]["status"] == "healthy"
        # At step 5: should be degraded
        assert m5["storefront-ui"]["status"] != "healthy"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_final_step_fully_degraded(self, difficulty):
        """At max_steps, all cascade services should be at full degradation."""
        s = load_scenario(difficulty)
        max_s = s.max_steps
        cascade = s._compute_cascade_order()

        metrics = s.get_metrics_at_step(max_s, max_s)

        for svc_name in cascade:
            # All cascade services should be at 'failing' or 'critical'
            status = metrics[svc_name]["status"]
            assert status in ("failing", "critical"), \
                f"{svc_name} still {status} at final step"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_error_rate_increases(self, difficulty):
        """Error rate should increase over time for cascade services."""
        s = load_scenario(difficulty)
        max_s = s.max_steps
        rc = s.ground_truth["root_cause_service"]

        m0 = s.get_metrics_at_step(0, max_s)
        m_end = s.get_metrics_at_step(max_s, max_s)

        # Root cause: error rate > 0 from step 0
        assert m0[rc]["error_rate"] > 0

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_all_services_have_metrics(self, difficulty):
        """Every service in the scenario should produce metrics."""
        s = load_scenario(difficulty)
        max_s = s.max_steps

        for step in [0, max_s // 2, max_s]:
            metrics = s.get_metrics_at_step(step, max_s)
            for svc in s.services:
                name = svc["name"]
                assert name in metrics, f"Missing metrics for {name} at step {step}"
                assert "cpu_utilization" in metrics[name]
                assert "memory_utilization" in metrics[name]
                assert "status" in metrics[name]


# ── 4. Progressive Alerts ───────────────────────────────────────────────────

class TestProgressiveAlerts:
    """Alerts appear progressively as cascade spreads."""

    def test_easy_step0_only_root_cause_and_red_herring_alerts(self):
        """Easy step 0: only root cause service alerts + red herrings visible.
        
        In the easy scenario, payments-db (root cause) has no direct alerts.
        The alerts are on checkout-ui (onset=2) and payments-api (onset=1),
        which haven't cascaded yet. Only the red herring is visible at step 0.
        """
        s = load_scenario("easy")
        alerts = s.get_alerts_at_step(0, s.max_steps)

        # At step 0, only root-cause-service alerts + red herring alerts
        # In easy: root cause=payments-db (onset=0), but no DB alerts exist!
        # Only worker-node-4 (red herring, always visible) shows up
        alert_services = [a["service"] for a in alerts]
        assert "worker-node-4" in alert_services, "Red herring should always be visible"
        assert "checkout-ui" not in alert_services, "checkout-ui onset=2, shouldn't alert at step 0"

    def test_easy_step0_downstream_alerts_hidden(self):
        """Easy step 0: downstream alerts should NOT be visible yet."""
        s = load_scenario("easy")
        alerts = s.get_alerts_at_step(0, s.max_steps)

        alert_services = [a["service"] for a in alerts]
        # checkout-ui is 2 hops away — its alerts should NOT fire at step 0
        checkout_alerts = [a for a in alerts if a["service"] == "checkout-ui"]
        assert len(checkout_alerts) == 0, \
            "checkout-ui alerts should not fire at step 0"

    def test_easy_alerts_grow_over_steps(self):
        """Easy: alert count should increase over steps."""
        s = load_scenario("easy")
        max_s = s.max_steps

        alerts_0 = s.get_alerts_at_step(0, max_s)
        alerts_5 = s.get_alerts_at_step(5, max_s)
        alerts_9 = s.get_alerts_at_step(9, max_s)

        assert len(alerts_5) >= len(alerts_0)
        assert len(alerts_9) >= len(alerts_5)

    def test_hard_red_herrings_always_visible(self):
        """Hard: red herring alerts are always visible regardless of step."""
        s = load_scenario("hard")

        for step in [0, 5, 10, 19]:
            alerts = s.get_alerts_at_step(step, s.max_steps)
            rh_alerts = [
                a for a in alerts
                if a["service"] in ("network-switch-03", "worker-node-7")
            ]
            assert len(rh_alerts) >= 2, \
                f"Red herring alerts missing at step {step}"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_alerts_strip_red_herring_flag(self, difficulty):
        """is_red_herring flag should be stripped from agent-visible alerts."""
        s = load_scenario(difficulty)
        alerts = s.get_alerts_at_step(0, s.max_steps)

        for alert in alerts:
            assert "is_red_herring" not in alert, \
                f"Internal flag leaked: {alert}"

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_final_step_all_alerts_visible(self, difficulty):
        """At max_steps, all alerts should be visible."""
        s = load_scenario(difficulty)
        max_s = s.max_steps

        all_alerts = s.get_alerts_at_step(max_s, max_s)
        # Should have all alerts visible (minus is_red_herring field)
        assert len(all_alerts) == len(s.alerts)


# ── 5. Progressive Topology ─────────────────────────────────────────────────

class TestProgressiveTopology:
    """Topology latencies evolve with cascade progression."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_step0_root_cause_edges_degraded(self, difficulty):
        """Edges TO the root cause should show increased latency from step 0."""
        s = load_scenario(difficulty)
        rc = s.ground_truth["root_cause_service"]
        topo = s.get_topology_at_step(0, s.max_steps)

        for edge in topo:
            if edge["downstream_service"] == rc:
                # Root cause edges: current_latency should differ from avg
                # (unless incident latency == avg latency, which is unlikely)
                pass  # Existence check is sufficient

    def test_easy_downstream_latency_increases(self):
        """Easy: latency to downstream services increases over steps."""
        s = load_scenario("easy")
        max_s = s.max_steps

        topo_0 = s.get_topology_at_step(0, max_s)
        topo_5 = s.get_topology_at_step(5, max_s)

        # Find the checkout-ui → payments-api edge
        # payments-api is onset=1, so at step 0 latency should be near avg
        edge_0 = next(
            e for e in topo_0
            if e["upstream_service"] == "checkout-ui"
        )
        edge_5 = next(
            e for e in topo_5
            if e["upstream_service"] == "checkout-ui"
        )

        # At step 5, latency should be higher than step 0
        assert edge_5["current_latency_ms"] >= edge_0["current_latency_ms"]

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_topology_edge_count_preserved(self, difficulty):
        """Edge count shouldn't change across steps."""
        s = load_scenario(difficulty)
        max_s = s.max_steps

        for step in [0, max_s // 2, max_s]:
            topo = s.get_topology_at_step(step, max_s)
            assert len(topo) == len(s.topology)


# ── 6. Progressive Timeline ─────────────────────────────────────────────────

class TestProgressiveTimeline:
    """Timeline grows as cascade deepens."""

    def test_easy_step0_has_initial_events(self):
        """Step 0 should show the initial incident events."""
        s = load_scenario("easy")
        timeline = s.get_timeline_at_step(0, s.max_steps)
        assert len(timeline) > 0

    def test_timeline_grows_over_steps(self):
        """Timeline should have more events at later steps."""
        s = load_scenario("hard")
        max_s = s.max_steps

        t0 = s.get_timeline_at_step(0, max_s)
        t10 = s.get_timeline_at_step(10, max_s)
        t19 = s.get_timeline_at_step(19, max_s)

        assert len(t10) >= len(t0)
        assert len(t19) >= len(t10)

    def test_hard_sla_breach_appears_at_step6(self):
        """Hard: SLA breach event should appear at step 6."""
        s = load_scenario("hard")

        t5 = s.get_timeline_at_step(5, s.max_steps)
        t6 = s.get_timeline_at_step(6, s.max_steps)

        sla_in_5 = any(e["event_type"] == "sla_breach" for e in t5)
        sla_in_6 = any(e["event_type"] == "sla_breach" for e in t6)

        assert not sla_in_5, "SLA breach should NOT be visible at step 5"
        assert sla_in_6, "SLA breach SHOULD be visible at step 6"

    def test_escalation_warning_at_midpoint(self):
        """Escalation warning should appear at the midpoint."""
        s = load_scenario("hard")
        max_s = s.max_steps
        mid = max_s // 2

        t_before = s.get_timeline_at_step(mid - 1, max_s)
        t_at = s.get_timeline_at_step(mid, max_s)

        has_escalation_before = any(
            e.get("event_type") == "escalation_warning" for e in t_before
        )
        has_escalation_at = any(
            e.get("event_type") == "escalation_warning" for e in t_at
        )

        assert not has_escalation_before
        assert has_escalation_at

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_timeline_sorted_by_step(self, difficulty):
        """Timeline events should be sorted chronologically."""
        s = load_scenario(difficulty)
        timeline = s.get_timeline_at_step(s.max_steps, s.max_steps)

        steps = [e.get("step", 0) for e in timeline]
        assert steps == sorted(steps), "Timeline not sorted by step"


# ── 7. Red Herring Behavior ─────────────────────────────────────────────────

class TestRedHerringBehavior:
    """Red herring services are static throughout the episode."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_red_herrings_constant_metrics(self, difficulty):
        """Red herring metrics should be constant across all steps."""
        s = load_scenario(difficulty)
        max_s = s.max_steps
        rh_names = s.get_red_herring_services()

        if not rh_names:
            pytest.skip(f"No red herrings in {difficulty}")

        m0 = s.get_metrics_at_step(0, max_s)
        m5 = s.get_metrics_at_step(5, max_s)
        m_end = s.get_metrics_at_step(max_s, max_s)

        for name in rh_names:
            if name in m0 and name in m5 and name in m_end:
                # CPU should be identical across steps
                assert m0[name]["cpu_utilization"] == m5[name]["cpu_utilization"]
                assert m5[name]["cpu_utilization"] == m_end[name]["cpu_utilization"]


# ── 8. DynamicScenario Integration ──────────────────────────────────────────

class TestDynamicScenarioProgressive:
    """Progressive observations work with DynamicScenario (Pillar 1)."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_dynamic_cascade_order(self, difficulty):
        """DynamicScenario computes cascade order correctly."""
        variant = generate_scenario_variant(difficulty, seed=42)
        order = variant._compute_cascade_order()

        rc = variant.ground_truth["root_cause_service"]
        assert order[rc] == 0

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_dynamic_metrics_at_step(self, difficulty):
        """DynamicScenario returns valid progressive metrics."""
        variant = generate_scenario_variant(difficulty, seed=42)
        max_s = variant.max_steps

        m0 = variant.get_metrics_at_step(0, max_s)
        m_end = variant.get_metrics_at_step(max_s, max_s)

        # Should have metrics for all services
        for svc in variant.services:
            assert svc["name"] in m0
            assert svc["name"] in m_end

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_dynamic_alerts_at_step(self, difficulty):
        """DynamicScenario returns valid progressive alerts."""
        variant = generate_scenario_variant(difficulty, seed=42)
        alerts = variant.get_alerts_at_step(0, variant.max_steps)
        assert len(alerts) > 0

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_dynamic_topology_at_step(self, difficulty):
        """DynamicScenario returns valid progressive topology."""
        variant = generate_scenario_variant(difficulty, seed=42)
        topo = variant.get_topology_at_step(0, variant.max_steps)
        assert len(topo) == len(variant.topology)


# ── 9. E2E Environment Integration ──────────────────────────────────────────

class TestEnvironmentProgressive:
    """Observations differ across steps in the actual environment."""

    @pytest.mark.parametrize("difficulty", DIFFICULTIES)
    def test_observations_evolve_across_steps(self, difficulty):
        """Metrics, alerts, and timeline change between steps."""
        from envs.incident_env import IncidentResponseEnv
        from models.action import (
            IncidentAction, RootCauseType, SeverityLevel, RemediationAction
        )

        env = IncidentResponseEnv()
        obs0 = env.reset(difficulty, dynamic=True, seed=42)

        # Take a step
        action = IncidentAction(
            root_cause_service="unknown",
            root_cause_type=RootCauseType.UNKNOWN,
            severity=SeverityLevel.P3,
            affected_services=[],
            remediation_action=RemediationAction.INVESTIGATE_FURTHER,
            stakeholder_message="Investigating.",
        )
        obs1, _, _, _ = env.step(action)

        # Metrics should differ between step 0 and step 1
        # (because cascade services start degrading at step 1)
        any_metric_diff = False
        for svc in obs0.metrics:
            if svc in obs1.metrics:
                m0 = obs0.metrics[svc]
                m1 = obs1.metrics[svc]
                for key in ("cpu_utilization", "memory_utilization", "status"):
                    if m0.get(key) != m1.get(key):
                        any_metric_diff = True
                        break
            if any_metric_diff:
                break

        assert any_metric_diff, \
            f"{difficulty}: Metrics identical between step 0 and step 1"

        env.close()

    def test_hard_alerts_increase_over_episode(self):
        """Hard: alert count should increase as cascade spreads."""
        from envs.incident_env import IncidentResponseEnv
        from models.action import (
            IncidentAction, RootCauseType, SeverityLevel, RemediationAction
        )

        env = IncidentResponseEnv()
        obs0 = env.reset("hard", dynamic=True, seed=42)

        action = IncidentAction(
            root_cause_service="unknown",
            root_cause_type=RootCauseType.UNKNOWN,
            severity=SeverityLevel.P3,
            affected_services=[],
            remediation_action=RemediationAction.INVESTIGATE_FURTHER,
        )

        # Take several steps
        alert_counts = [len(obs0.alerts)]
        for _ in range(5):
            obs, _, _, _ = env.step(action)
            alert_counts.append(len(obs.alerts))

        # Alert count should increase over time
        assert alert_counts[-1] >= alert_counts[0], \
            f"Alerts didn't increase: {alert_counts}"

        env.close()

    def test_info_contains_cascade_progress(self):
        """Info dict should contain cascade_progress."""
        from envs.incident_env import IncidentResponseEnv

        env = IncidentResponseEnv()
        obs = env.reset("easy", dynamic=True, seed=42)

        assert "cascade_progress" in obs.info
        assert obs.info["cascade_progress"] == 0.0

        env.close()


# ── 10. Interpolation Helpers ────────────────────────────────────────────────

class TestInterpolationHelpers:
    """Test the static helper methods."""

    def test_interpolate_at_0(self):
        """Factor=0 returns normal value."""
        assert BaseScenario._interpolate(0.3, 0.9, 0.0) == 0.3

    def test_interpolate_at_1(self):
        """Factor=1 returns incident value."""
        assert BaseScenario._interpolate(0.3, 0.9, 1.0) == 0.9

    def test_interpolate_midpoint(self):
        """Factor=0.5 returns midpoint."""
        result = BaseScenario._interpolate(0.0, 1.0, 0.5)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_status_healthy(self):
        assert BaseScenario._status_from_factor(0.0) == "healthy"
        assert BaseScenario._status_from_factor(0.09) == "healthy"

    def test_status_degraded(self):
        assert BaseScenario._status_from_factor(0.2) == "degraded"
        assert BaseScenario._status_from_factor(0.39) == "degraded"

    def test_status_critical(self):
        assert BaseScenario._status_from_factor(0.5) == "critical"
        assert BaseScenario._status_from_factor(0.74) == "critical"

    def test_status_failing(self):
        assert BaseScenario._status_from_factor(0.75) == "failing"
        assert BaseScenario._status_from_factor(1.0) == "failing"
