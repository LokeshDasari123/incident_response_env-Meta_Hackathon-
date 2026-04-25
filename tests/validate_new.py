"""Quick validation of new components: expert scenario, grader, debate engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenarios import load_scenario
from graders import load_grader
from envs.debate import DebateEngine

# Test expert scenario loads
s = load_scenario("expert")
print(f"Expert scenario: {s.name}")
print(f"  Services: {len(s.services)}")
print(f"  Alerts: {len(s.alerts)}")
print(f"  Red herrings: {s.get_red_herring_services()}")
print(f"  Root cause: {s.ground_truth['root_cause_service']}")
print(f"  SLA breach: step {s.sla_breach_step}")
s.validate()
print("  Validation: PASS")

# Test expert grader
g = load_grader("expert")
result = g.grade(
    action={
        "root_cause_service": "auth-service",
        "root_cause_type": "certificate_expiry",
        "severity": "P0",
        "affected_services": ["auth-service", "user-service", "api-gateway", "storefront-ui", "order-service"],
        "remediation_action": "fix_config",
        "stakeholder_message": "auth-service cert expired causing cascade. Fix in progress. ETA 10 min.",
    },
    step=3,
    max_steps=25,
)
print(f"  Perfect score: {result.reward}")

# Test debate engine
de = DebateEngine(seed=42)
challenge = de.generate_challenge(
    action={
        "root_cause_service": "cache-service",
        "severity": "P1",
        "root_cause_type": "memory_leak",
        "affected_services": ["cache-service"],
        "remediation_action": "restart_service",
    },
    metrics={
        "auth-service": {"cpu_utilization": 0.24, "memory_utilization": 0.38, "is_healthy": False, "status": "failing"},
        "cache-service": {"cpu_utilization": 0.52, "memory_utilization": 0.94, "is_healthy": False, "status": "critical"},
    },
    alerts=[],
    topology=[],
    ground_truth=s.ground_truth,
    step=1,
    max_steps=25,
)
print(f"\nDebate challenge:")
print(f"  Strategy: {challenge['strategy']}")
print(f"  Hint quality: {challenge['hint_quality']}")
print(f"  Text: {challenge['challenge_text'][:150]}...")

# Test full env with debate
from envs.incident_env import IncidentResponseEnv
from models.action import IncidentAction

env = IncidentResponseEnv()
obs = env.reset("expert", dynamic=False)
print(f"\nEnv reset: step={obs.step}, max_steps={obs.max_steps}, task={obs.task_id}")
print(f"  Debate phase: {obs.debate_phase}")
print(f"  Debate challenge: {obs.debate_challenge}")
print(f"  Alerts at step 0: {len(obs.alerts)}")

action = IncidentAction(
    root_cause_service="cache-service",
    root_cause_type="memory_leak",
    severity="P1",
    confidence=0.6,
    affected_services=["cache-service"],
    remediation_action="restart_service",
    reasoning="cache-service has 94% memory",
    stakeholder_message="cache-service memory issue",
)
obs2, reward, done, info = env.step(action)
print(f"\nStep 1: reward={reward}, done={done}")
print(f"  Debate phase: {obs2.debate_phase}")
print(f"  Debate challenge present: {obs2.debate_challenge is not None}")
if obs2.debate_challenge:
    print(f"  Challenge: {obs2.debate_challenge[:120]}...")
print(f"  Debate strategy: {obs2.debate_strategy}")

# Step 2: correct answer after debate
action2 = IncidentAction(
    root_cause_service="auth-service",
    root_cause_type="certificate_expiry",
    severity="P0",
    confidence=0.9,
    affected_services=["auth-service", "user-service", "api-gateway"],
    remediation_action="fix_config",
    reasoning="After debate, traced topology to auth-service cert expiry",
    stakeholder_message="auth-service cert expired. Cascading. Fix in progress. ETA 10 min.",
)
obs3, reward2, done2, info2 = env.step(action2)
print(f"\nStep 2 (after debate): reward={reward2}, done={done2}")
print(f"  Debate bonus: {info2.get('debate_bonus', 0)}")
print(f"  Debate feedback: {info2.get('debate_feedback', '')}")
print(f"  Debate history: {len(obs3.debate_history)} rounds")

print("\n" + "=" * 60)
print("ALL SYSTEMS GO!")
print("=" * 60)
