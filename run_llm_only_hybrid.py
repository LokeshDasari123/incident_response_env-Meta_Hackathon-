#!/usr/bin/env python3
"""
LLM-ONLY HYBRID TRAINING
========================
Pure LLM mode with no fallback.
- 120-second timeouts per API call
- Retry logic with exponential backoff (2, 4, 8, 16, 32 seconds)
- 4-phase chain of thought (SCAN, ANALYZE, DECIDE, COMMUNICATE)
- No rule-based bypass at any point

This will wait as long as needed for the LLM to respond.
Monitor the output for:
  - [CoT][ANALYZE] confidence values varying (should NOT be stuck at 51%)
  - LTM context being injected to improve over episodes
  - Reward progression: should increase as model learns
"""

import subprocess
import sys

print("=" * 80)
print("LLM-ONLY HYBRID TRAINING")
print("=" * 80)
print()
print("Configuration:")
print("  - Mode: hybrid_cot (LLM only, no rule-based fallback)")
print("  - Timeouts: 120 seconds per LLM call")
print("  - Retries: 5 attempts with exponential backoff")
print("  - Episodes: 200 (all curriculum levels)")
print("  - Curriculum: Easy → Medium → Hard → Expert")
print("  - Memory: STM (per-episode) + LTM (cross-episode)")
print()
print("Expected behavior:")
print("  - Episodes 0-50: Easy (~0.9+ accuracy)")
print("  - Episodes 50-100: Medium (~0.85+ accuracy)")
print("  - Episodes 100-150: Hard (~0.7+ accuracy)")
print("  - Episodes 150-200: Expert (~0.6+ accuracy)")
print()
print("Confidence should VARY by complexity, not stay at 51%")
print()
print("Running training...")
print("=" * 80)
print()

cmd = [
    "python", "training/train.py",
    "--task", "all",
    "--episodes", "200",
    "--curriculum",
    "--hybrid",
]

result = subprocess.run(cmd)
sys.exit(result.returncode)
