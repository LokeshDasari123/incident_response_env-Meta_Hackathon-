#!/usr/bin/env python3
"""
Test HuggingFace model accessibility directly (not API endpoints)
Check which models we can download and use locally
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

print("=" * 80)
print("HUGGINGFACE MODEL ACCESSIBILITY TEST")
print("=" * 80)
print(f"HF Token Available: {bool(HF_TOKEN)}")
print()

# Test 1: Check if we can import transformers
print("1️⃣  Checking transformers library...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("   ✅ transformers imported successfully")
except ImportError as e:
    print(f"   ❌ transformers NOT available: {e}")
    exit(1)

# Test 2: Check if we can import torch
print("\n2️⃣  Checking torch...")
try:
    import torch
    print(f"   ✅ torch available (device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')})")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("   ⚠️  torch not available")

# Test 3: Try to load small models locally
models_to_test = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Small - FAST"),
    ("Qwen/Qwen2.5-3B-Instruct", "Tiny - Alternative"),
    ("meta-llama/Llama-2-7b-hf", "Medium - Alternative"),
]

print("\n3️⃣  Testing model availability...")
print("   (This will attempt to download model info, not the full model)")
print()

for model_id, description in models_to_test:
    print(f"   Testing: {model_id}")
    print(f"   Desc: {description}")
    try:
        # Just try to get model config (lightweight test)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"   ✅ Config accessible! Model can be loaded")
        print(f"      - Hidden size: {config.hidden_size}")
        print(f"      - Num layers: {config.num_hidden_layers}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
    print()

# Test 4: Check Unsloth
print("\n4️⃣  Checking Unsloth (for efficient LoRA fine-tuning)...")
try:
    from unsloth import FastLanguageModel
    print("   ✅ Unsloth available!")
    print("   Unsloth enables:")
    print("      - 2x faster training")
    print("      - 60% less memory")
    print("      - Ready for QLoRA fine-tuning")
except ImportError as e:
    print(f"   ❌ Unsloth not installed: {e}")
    print("   Install: pip install unsloth")

# Test 5: Check if we can use HF TRL (reinforcement learning library)
print("\n5️⃣  Checking HuggingFace TRL (for RL training)...")
try:
    from trl import PPOTrainer, DPOTrainer
    print("   ✅ HF TRL available!")
    print("   Available trainers:")
    print("      - PPOTrainer (for GRPO)")
    print("      - DPOTrainer (for preference learning)")
except ImportError as e:
    print(f"   ❌ HF TRL not installed: {e}")
    print("   Install: pip install trl")

# Test 6: Check bitsandbytes for quantization
print("\n6️⃣  Checking bitsandbytes (for 8-bit/4-bit quantization)...")
try:
    import bitsandbytes as bnb
    print("   ✅ bitsandbytes available!")
    print("   Enables:")
    print("      - 8-bit quantization")
    print("      - QLoRA (4-bit + LoRA)")
except ImportError:
    print("   ⚠️  bitsandbytes not installed (optional)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If all checks pass:
✅ You can use Unsloth + HF TRL for RL training with actual models
✅ Fine-tune a small model (1.5B-3B) using QLoRA
✅ This is what judges want to see!

Next: Create RL training script with:
1. Model loading via Unsloth
2. PPO/GRPO training loop
3. Reward signals from incident resolution
4. Logging and evaluation
""")
