"""scripts/validate_env.py — Pre-submission local validation."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

def check(label, fn):
    try:
        fn()
        print(f"  ✅ {label}")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False

def main():
    print("\n=== OpenEnv Local Validator ===\n")
    passed = 0
    total  = 0

    checks = [
        ("Models import",          lambda: __import__("models")),
        ("Scenarios load (easy)",  lambda: __import__("scenarios").load_scenario("easy")),
        ("Scenarios load (medium)",lambda: __import__("scenarios").load_scenario("medium")),
        ("Scenarios load (hard)",  lambda: __import__("scenarios").load_scenario("hard")),
        ("Graders load (easy)",    lambda: __import__("graders").load_grader("easy")),
        ("Graders load (medium)",  lambda: __import__("graders").load_grader("medium")),
        ("Graders load (hard)",    lambda: __import__("graders").load_grader("hard")),
        ("Env reset/step",         lambda: _test_env()),
        ("openenv.yaml exists",    lambda: assert_file("openenv.yaml")),
        ("Dockerfile exists",      lambda: assert_file("Dockerfile")),
        ("inference.py exists",    lambda: assert_file("inference.py")),
    ]

    for label, fn in checks:
        total += 1
        if check(label, fn):
            passed += 1

    print(f"\n{'='*40}")
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  ✅ Ready to submit!")
    else:
        print("  ❌ Fix failures before submitting.")
    print(f"{'='*40}\n")
    return passed == total

def assert_file(name):
    root = Path(__file__).parents[1]
    assert (root / name).exists(), f"{name} not found"

def _test_env():
    from envs.incident_env import IncidentResponseEnv
    from models.action import IncidentAction, RootCauseType, SeverityLevel, RemediationAction
    env = IncidentResponseEnv()
    obs = env.reset("easy")
    assert obs.step == 0
    action = IncidentAction(
        root_cause_service="payments-db",
        root_cause_type=RootCauseType.MISCONFIGURATION,
        severity=SeverityLevel.P0,
        affected_services=["payments-db"],
        remediation_action=RemediationAction.FIX_CONFIG,
        stakeholder_message="test",
    )
    obs2, reward, done, info = env.step(action)
    assert 0.0 <= reward <= 1.0
    env.close()

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)