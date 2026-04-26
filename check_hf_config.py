#!/usr/bin/env python3
"""
HUGGINGFACE-ONLY LLM CONFIGURATION
===================================

Updated configuration to use ONLY HuggingFace models for all tiers:

FAST (Parsing/SCAN):
  - Model: Qwen/Qwen2.5-1.5B-Instruct
  - Purpose: Quick alert parsing and service candidate identification
  - Speed: Very fast, minimal tokens

BALANCED (Orchestrator/DECIDE/COMMUNICATE):
  - Model: Qwen/Qwen2.5-7B-Instruct
  - Purpose: Severity assessment, remediation action selection, messaging
  - Quality: Good reasoning, mid-size

STRONG (Analysis/ANALYZE):
  - Model: meta-llama/Llama-2-70b-chat-hf
  - Purpose: Deep root cause analysis, type inference
  - Quality: Best reasoning capability

ALL tiers use:
  - Base URL: https://api-inference.huggingface.co/v1
  - Auth: HF_TOKEN from .env
  - Format: OpenAI-compatible API

Configuration:
  - Timeout per call: 120 seconds (waits full time)
  - Retries: 5 attempts with exponential backoff
  - No fallback: LLM-only training (fail if API unreachable)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[0]
load_dotenv(ROOT / ".env")

def check_config():
    """Verify HF configuration is correct."""
    print("=" * 80)
    print("HUGGINGFACE CONFIGURATION CHECK")
    print("=" * 80)
    print()
    
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("❌ HF_TOKEN not set in .env")
        return False
    
    if hf_token.startswith("hf_"):
        print(f"✓ HF_TOKEN: {hf_token[:20]}...{hf_token[-10:]}")
    else:
        print(f"⚠ HF_TOKEN format unexpected: {hf_token[:30]}")
    
    models = {
        "HF_MODEL_FAST": "Qwen/Qwen2.5-1.5B-Instruct",
        "HF_MODEL_BALANCED": "Qwen/Qwen2.5-7B-Instruct",
        "HF_MODEL_STRONG": "meta-llama/Llama-2-70b-chat-hf",
    }
    
    print()
    print("Models:")
    for env_var, default in models.items():
        value = os.getenv(env_var, default)
        print(f"  {env_var:25} = {value}")
    
    print()
    print("API Configuration:")
    print(f"  Base URL: https://api-inference.huggingface.co/v1")
    print(f"  Timeout:  120 seconds")
    print(f"  Retries:  5 attempts (exponential backoff)")
    
    print()
    print("✓ Configuration looks good!")
    print()
    print("To run LLM-only hybrid training:")
    print("  python run_llm_only_hybrid.py")
    print()
    print("Or:")
    print("  python training/train.py --task all --episodes 200 --curriculum --hybrid")
    
    return True

if __name__ == "__main__":
    success = check_config()
    sys.exit(0 if success else 1)
