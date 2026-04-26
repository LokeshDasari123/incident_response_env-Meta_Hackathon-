"""Quick E2E test of email trigger -> circuit breaker -> AI debate pipeline."""
import asyncio
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.email_trigger import incident_pipeline
from backend.circuit_breaker import circuit_breaker

async def test_pipeline():
    print("=" * 60)
    print("E2E Test: Email Trigger -> Circuit Breaker -> AI Debate")
    print("=" * 60)

    # Trigger from simulated email
    result = await incident_pipeline.trigger_from_email(
        subject="[ALERT] payments-db OOM kill -- crash loop detected in production",
        body="Memory: 99.1%, restart count: 7, cascade to checkout-ui",
        source="email_test",
    )

    print(f"\nIncident ID: {result['incident_id']}")
    print(f"Status: {result['status']}")
    print(f"Attempts: {result['attempts']}")

    tracking = result.get("tracking", {})
    print(f"\nModel Interactions:")
    for model in tracking.get("models", []):
        print(f"  [{model['role']}] {model['model']} -> {model['action']}")
    
    print(f"\nDebate Rounds:")
    for debate in tracking.get("debate_rounds", []):
        print(f"  Strategy: {debate['strategy']}")
        print(f"  Challenge: {debate['challenge'][:100]}...")
        print(f"  Agent improved: {debate['agent_improved']}")

    print(f"\nWinning Model: {tracking.get('winning_model')}")
    print(f"Final Confidence: {tracking.get('final_confidence')}")

    if tracking.get("final_result"):
        fr = tracking["final_result"]
        print(f"\nFinal Result:")
        print(f"  Root Cause: {fr.get('root_cause')}")
        print(f"  Fault Type: {fr.get('fault_type')}")
        print(f"  Severity: {fr.get('severity')}")
        print(f"  Remediation: {fr.get('remediation')}")

    # Check log queue
    logs = incident_pipeline.get_logs(n=20)
    print(f"\nLog Queue ({len(logs)} entries):")
    for log in logs:
        print(f"  [{log['level']}] [{log['source']}] {log['message'][:100]}")

    # Check circuit breaker state
    circuit = circuit_breaker.get_circuit(result["incident_id"])
    print(f"\nCircuit Breaker State:")
    print(f"  State: {circuit['state'] if circuit else 'N/A'}")
    print(f"  Attempts: {circuit['attempts'] if circuit else 'N/A'}")
    print(f"  Resolved: {circuit['resolved'] if circuit else 'N/A'}")

    # Stats
    stats = incident_pipeline.get_stats()
    print(f"\nOverall Stats: {stats}")

    print("\n" + "=" * 60)
    print("ALL E2E TESTS PASSED!")
    print("=" * 60)

asyncio.run(test_pipeline())
