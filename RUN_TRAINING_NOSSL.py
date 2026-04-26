#!/usr/bin/env python3
"""
QUICK FIX: Run training in RULE-BASED ONLY mode to avoid SSL hangs
Use this while we diagnose the LLM connectivity issues
"""

import subprocess
import sys

print("=" * 70)
print("🔧 QUICK FIX: Running training in RULE-BASED mode (no LLM API calls)")
print("=" * 70)
print()
print("This avoids SSL timeouts while generating training data.")
print("Command:")
print()

cmd = " ".join([
    "python training/train.py",
    "--task all",
    "--episodes 200",
    "--curriculum",
    "--positive-ratio 0.0",
    "--no-llm"
])

print(f"  {cmd}")
print()
print("Running...")
print()

result = subprocess.run(cmd, shell=True)
sys.exit(result.returncode)
