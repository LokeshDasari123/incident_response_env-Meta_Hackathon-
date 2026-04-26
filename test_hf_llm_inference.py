import os
import requests
from huggingface_hub import whoami

def test_hf_models():
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        # Try to read from .env directly
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.strip().split("=", 1)[1].strip("'\"")
                        break
        except Exception as e:
            pass
            
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set.")
        return

    print("🔑 Testing Hugging Face Token Validation...")
    try:
        user_info = whoami(token=hf_token)
        print(f"✅ Token is valid! Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"⚠️ Token validation skipped or failed: {e}")
        # Continue anyway, let the inference test try

    # List of models to test
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-1B-Instruct"
    ]

    from huggingface_hub import InferenceClient

    print("\n🤖 Testing LLM Inference API (Sending 'Hi' to models)...")
    
    for model in models:
        print(f"\n--- Model: {model} ---")
        client = InferenceClient(model=model, token=hf_token)
        try:
            # We use chat_completion for instruction models
            response = client.chat_completion(
                messages=[{"role": "user", "content": "Hi, who are you? Answer in 1 short sentence."}], 
                max_tokens=50
            )
            print(f"✅ SUCCESS! Model Replied:\n{response.choices[0].message.content.strip()}")
        except Exception as e:
            print(f"❌ Request Error: {e}")

if __name__ == "__main__":
    test_hf_models()
