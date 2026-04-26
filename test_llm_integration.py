#!/usr/bin/env python3
"""
Test LLM integration in training
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Quick test: verify LLM is actually being called
from training.train import call_llm_for_diagnosis

test_obs = {
    "task_id": "easy",
    "metrics": {
        "payments-db": {"status": "down", "cpu_utilization": 0.95, "memory_utilization": 0.88},
        "payments-api": {"status": "degraded", "cpu_utilization": 0.75, "memory_utilization": 0.65},
        "checkout-ui": {"status": "healthy", "cpu_utilization": 0.2, "memory_utilization": 0.3},
    },
    "alerts": [
        {"severity": "CRITICAL", "description": "payments-db is down", "service": "payments-db"},
        {"severity": "CRITICAL", "description": "payments-api latency >5s", "service": "payments-api"},
    ],
    "topology": [
        {"upstream_service": "checkout-ui", "downstream_service": "payments-api"},
        {"upstream_service": "payments-api", "downstream_service": "payments-db"},
    ],
    "timeline": [
        {"timestamp": "2026-04-26T05:40:00Z", "description": "Unusual memory spike on payments-db"},
        {"timestamp": "2026-04-26T05:41:00Z", "description": "payments-db connection pool exhausted"},
    ],
}

print("=" * 80)
print("Testing LLM Diagnosis")
print("=" * 80)
print()

try:
    result = call_llm_for_diagnosis(test_obs, task_id="easy", step=1, max_steps=10)
    
    print(f"✓ LLM Call Successful!")
    print()
    print(f"Root Cause:  {result.get('root_cause_service')}")
    print(f"Fault Type:  {result.get('root_cause_type')}")
    print(f"Severity:    {result.get('severity')}")
    print(f"Confidence:  {result.get('confidence'):.0%}")
    print(f"Action:      {result.get('remediation_action')}")
    print(f"Affected:    {result.get('affected_services')}")
    print()
    print("Message:", result.get('stakeholder_message', '(none)'))
    print()
    print("=" * 80)
    print("✓ LLM is now integrated and being called!")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
