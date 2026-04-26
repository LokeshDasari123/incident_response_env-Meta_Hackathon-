#!/usr/bin/env python3
"""Test HuggingFace API endpoint"""
import os
import requests
import json

hf_token = os.getenv("HF_TOKEN", "").strip()
print(f"HF Token: {hf_token[:20]}...")

# Test different endpoint formats
endpoints = [
    ("OpenAI-compatible v1", "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct/v1/chat/completions"),
    ("Native HF API", "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"),
]

payload = {
    "inputs": "<|im_start|>user\nHello, what is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "parameters": {
        "max_new_tokens": 50
    }
}

for name, url in endpoints:
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"{'='*70}")
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    
    try:
        print(f"POST {url}")
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"Status: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('content-type', 'N/A')}")
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"✅ SUCCESS!")
                print(f"Response keys: {list(data.keys())}")
                if 'choices' in data:
                    print(f"  Message: {data['choices'][0].get('message', {}).get('content', 'N/A')[:100]}")
            except:
                print(f"Response (first 500 chars): {resp.text[:500]}")
        else:
            print(f"Response (first 500 chars): {resp.text[:500]}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)[:200]}")
