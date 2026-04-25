"""Quick E2E test for Pillar 1 (Dynamic Scenarios) + Pillar 2 (Progressive Obs)."""
import sys
sys.path.insert(0, ".")
import httpx

BASE = "http://localhost:7860"
client = httpx.Client(base_url=BASE, timeout=10)

# ── Test 1: Dynamic Reset (Pillar 1) ──────────────────────────────────────────
print("=" * 60)
print("TEST 1: Dynamic Scenario Generation (Pillar 1)")
print("=" * 60)

r1 = client.post("/reset", json={"task_id": "hard", "dynamic": True, "seed": 42})
assert r1.status_code == 200, f"Reset failed: {r1.status_code}"
obs1 = r1.json()["observation"]
info1 = r1.json()["info"]
print(f"  Task:     {obs1['task_id']}")
print(f"  Step:     {obs1['step']}")
print(f"  Alerts:   {len(obs1['alerts'])}")
print(f"  Services: {len(obs1['metrics'])}")
print(f"  Dynamic:  {info1.get('dynamic', 'N/A')}")
print(f"  PASS")

# ── Test 2: Seeded Reproducibility ────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 2: Seeded Reproducibility")
print("=" * 60)

r2a = client.post("/reset", json={"task_id": "easy", "dynamic": True, "seed": 123})
r2b = client.post("/reset", json={"task_id": "easy", "dynamic": True, "seed": 123})
m2a = r2a.json()["observation"]["metrics"]
m2b = r2b.json()["observation"]["metrics"]

match = True
for svc in m2a:
    if svc in m2b:
        if m2a[svc]["cpu_utilization"] != m2b[svc]["cpu_utilization"]:
            match = False
            break

print(f"  Seed 123 run 1: checkout-ui CPU = {m2a.get('checkout-ui', {}).get('cpu_utilization', 'N/A')}")
print(f"  Seed 123 run 2: checkout-ui CPU = {m2b.get('checkout-ui', {}).get('cpu_utilization', 'N/A')}")
print(f"  Identical: {match}")
assert match, "Seeded runs should produce identical results"
print(f"  PASS")

# ── Test 3: Different Seeds = Different Metrics ───────────────────────────────
print()
print("=" * 60)
print("TEST 3: Different Seeds = Different Metrics")
print("=" * 60)

r3a = client.post("/reset", json={"task_id": "easy", "dynamic": True, "seed": 42})
r3b = client.post("/reset", json={"task_id": "easy", "dynamic": True, "seed": 99})
m3a = r3a.json()["observation"]["metrics"]
m3b = r3b.json()["observation"]["metrics"]

any_diff = False
for svc in m3a:
    if svc in m3b:
        if m3a[svc]["cpu_utilization"] != m3b[svc]["cpu_utilization"]:
            any_diff = True
            break

print(f"  Seed 42: checkout-ui CPU = {m3a.get('checkout-ui', {}).get('cpu_utilization', 'N/A')}")
print(f"  Seed 99: checkout-ui CPU = {m3b.get('checkout-ui', {}).get('cpu_utilization', 'N/A')}")
print(f"  Different: {any_diff}")
assert any_diff, "Different seeds should produce different metrics"
print(f"  PASS")

# ── Test 4: Progressive Observations (Pillar 2) ──────────────────────────────
print()
print("=" * 60)
print("TEST 4: Progressive Cascade (Pillar 2)")
print("=" * 60)

client.post("/reset", json={"task_id": "hard", "dynamic": True, "seed": 42})
action = {
    "root_cause_service": "unknown",
    "root_cause_type": "unknown",
    "severity": "P3",
    "affected_services": [],
    "remediation_action": "investigate_further",
    "stakeholder_message": "Investigating...",
    "confidence": 0.1,
}

alert_counts = []
storefront_statuses = []

# Take 8 steps and track cascade progression
for i in range(8):
    r = client.post("/step", json={"action": action})
    obs = r.json()["observation"]
    n_alerts = len(obs["alerts"])
    sf_status = obs["metrics"]["storefront-ui"]["status"]
    sf_cpu = obs["metrics"]["storefront-ui"]["cpu_utilization"]
    db_status = obs["metrics"]["payments-db"]["status"]
    cascade = obs["info"]["cascade_progress"]

    alert_counts.append(n_alerts)
    storefront_statuses.append(sf_status)

    print(f"  Step {obs['step']:2d} | Alerts: {n_alerts:2d} | "
          f"storefront: {sf_status:10s} (CPU={sf_cpu:.3f}) | "
          f"payments-db: {db_status:10s} | cascade: {cascade:.3f}")

# Verify cascade progression
assert alert_counts[-1] >= alert_counts[0], "Alerts should increase"
assert storefront_statuses[0] in ("healthy", "degraded"), "storefront should start healthy/degraded"
assert storefront_statuses[-1] in ("critical", "failing"), "storefront should degrade"
print(f"  PASS - cascade propagating correctly!")

# ── Test 5: Red Herrings Visible Throughout ───────────────────────────────────
print()
print("=" * 60)
print("TEST 5: Red Herrings Always Visible")
print("=" * 60)

client.post("/reset", json={"task_id": "hard", "dynamic": True, "seed": 42})
obs0 = client.post("/step", json={"action": action}).json()["observation"]
for _ in range(5):
    client.post("/step", json={"action": action})
obs6 = client.post("/step", json={"action": action}).json()["observation"]

rh_step1 = [a["service"] for a in obs0["alerts"] if a["service"] in ("network-switch-03", "worker-node-7")]
rh_step7 = [a["service"] for a in obs6["alerts"] if a["service"] in ("network-switch-03", "worker-node-7")]

print(f"  Red herrings at step 1: {rh_step1}")
print(f"  Red herrings at step 7: {rh_step7}")
assert len(rh_step1) >= 2, "Red herrings should be visible from step 1"
assert len(rh_step7) >= 2, "Red herrings should remain visible at step 7"
print(f"  PASS")

# ── Test 6: Grading Still Works ───────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 6: Grading Works with Dynamic + Progressive")
print("=" * 60)

client.post("/reset", json={"task_id": "easy", "dynamic": True, "seed": 42})
perfect_action = {
    "root_cause_service": "payments-db",
    "root_cause_type": "misconfiguration",
    "severity": "P0",
    "affected_services": ["payments-db", "payments-api", "checkout-ui"],
    "remediation_action": "fix_config",
    "stakeholder_message": "Investigating payment delays caused by misconfiguration. ETA 8 mins.",
    "confidence": 0.95,
}
r_grade = client.post("/step", json={"action": perfect_action})
reward = r_grade.json()["reward"]
print(f"  Perfect action reward: {reward:.3f}")
assert reward >= 0.65, f"Perfect action should score >= 0.65, got {reward}"
print(f"  PASS")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ALL 6 E2E TESTS PASSED!")
print("Pillar 1 (Dynamic Scenarios) + Pillar 2 (Progressive Obs) VERIFIED")
print("=" * 60)
