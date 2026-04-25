"""Quick Groq + env smoke test — single easy task."""
from dotenv import load_dotenv
load_dotenv()

import os, json
from openai import OpenAI
from client.http_client import IncidentEnvClient

API_BASE = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY  = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
MODEL    = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

print(f"API: {API_BASE}")
print(f"Model: {MODEL}")
print(f"Key: {API_KEY[:10]}...")

# 1. Reset env
client = IncidentEnvClient("http://localhost:7860")
result = client.reset("easy")
obs = result["observation"]
print(f"\nAlerts: {len(obs['alerts'])}")
print(f"Services: {list(obs['metrics'].keys())}")
print(f"Topology edges: {len(obs['topology'])}")

# 2. Call Groq LLM
llm = OpenAI(base_url=API_BASE, api_key=API_KEY)

system = """You are an expert SRE triaging a production incident.
Respond ONLY with valid JSON (no markdown):
{
  "root_cause_service": "<exact service name>",
  "root_cause_type": "<misconfiguration|memory_leak|network_partition|crash_loop|resource_exhaustion|auth_failure|dependency_failure|unknown>",
  "severity": "<P0|P1|P2|P3>",
  "affected_services": ["<ALL services in cascade>"],
  "remediation_action": "<rollback|restart_service|scale_up|fix_config|increase_connection_pool|flush_cache|reroute_traffic|escalate|investigate_further>",
  "stakeholder_message": "<P0/P1 only: service + impact + ETA>",
  "confidence": 0.9,
  "reasoning": "<step by step>"
}"""

user_msg = f"""INCIDENT TRIAGE | Task: easy | Step: 1

=== ALERTS ===
{json.dumps(obs['alerts'], indent=2)}

=== METRICS ===
{json.dumps(obs['metrics'], indent=2)}

=== TOPOLOGY ===
{json.dumps(obs['topology'], indent=2)}

=== TIMELINE ===
{json.dumps(obs['timeline'], indent=2)}

JSON only. Include ALL affected_services."""

resp = llm.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ],
    temperature=0.2,
    max_tokens=512,
)

text = (resp.choices[0].message.content or "").strip()
print(f"\n=== LLM Response ===")
print(text[:800])

# Parse JSON
if "```" in text:
    parts = text.split("```")
    text = parts[1] if len(parts) > 1 else parts[0]
    if text.startswith("json"):
        text = text[4:]
    text = text.strip()

action = json.loads(text)
print(f"\n=== Parsed Action ===")
print(json.dumps(action, indent=2))

# 3. Step
step_result = client.step(action)
print(f"\n=== RESULT ===")
print(f"Reward: {step_result['reward']}")
print(f"Done:   {step_result['done']}")
if 'info' in step_result:
    info = step_result['info']
    if 'reward_breakdown' in info:
        bd = info['reward_breakdown']
        print(f"Breakdown: RC={bd.get('root_cause_score',0):.0%} "
              f"Act={bd.get('action_score',0):.0%} "
              f"Sev={bd.get('severity_score',0):.0%} "
              f"Com={bd.get('communication_score',0):.0%} "
              f"Speed={bd.get('speed_bonus',0):.0%}")
        print(f"Feedback: {bd.get('feedback','').encode('ascii','replace').decode()}")

client.close()
