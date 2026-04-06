# 🚨 Incident Response OpenEnv

> A real-world OpenEnv environment where AI agents triage production incidents —
> exactly like an SRE at 2am.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 What This Environment Simulates

Every engineering team running microservices faces **production incidents**:
cascading failures, noisy alert storms, and pressure to resolve issues before
SLA breaches. Today this is done by on-call SREs at 2am — manually, under stress.

This environment trains AI agents to:
1. **Analyze** cascading alert storms across microservice call graphs
2. **Filter noise** — identify red herring alerts unrelated to the incident
3. **Identify root cause** by traversing the dependency graph inward
4. **Classify severity** (P0/P1/P2/P3) and prescribe correct remediation
5. **Communicate** to stakeholders under SLA time pressure

**Data foundation:** Scenarios are modeled on real Alibaba microservices cluster
traces (v2021), Microsoft AIOpsLab fault taxonomy, and Google SRE Book incident
patterns (Ch 13–16).

---

## 🏗️ Action Space

The agent submits a structured `IncidentAction` each step:

| Field | Type | Description |
|---|---|---|
| `root_cause_service` | string | Service identified as root cause |
| `root_cause_type` | enum | misconfiguration / memory_leak / network_partition / crash_loop / ... |
| `severity` | enum | P0 (revenue) / P1 (user-facing) / P2 (partial) / P3 (minor) |
| `affected_services` | list[str] | All impacted services |
| `remediation_action` | enum | rollback / restart_service / fix_config / escalate / ... |
| `stakeholder_message` | string | Required for P0/P1 incidents |
| `confidence` | float | Agent confidence 0.0–1.0 |
| `reasoning` | string | Chain of thought (used for partial credit) |

---

## 👁️ Observation Space

Each step the agent receives:

| Field | Description |
|---|---|
| `alerts` | Active monitoring alerts (service, metric, value, threshold) |
| `metrics` | Current CPU/memory/RT per service |
| `topology` | Service call graph edges (upstream → downstream) |
| `timeline` | Chronological incident events |
| `time_pressure` | SLA breach urgency 0.0–1.0 |
| `sla_breach_in_steps` | Steps until SLA breach (hard task only) |

---

## 📋 Tasks

### Task 1: Easy — Change-Induced Single Service Failure
- **Fault:** Bad ConfigMap update to `payments-db`
- **Cascade:** `payments-db` → `payments-api` → `checkout-ui`
- **Red herring:** CPU spike on `worker-node-4` (unrelated batch job)
- **Expected GPT-4 score:** 0.75 | **Random:** 0.15

### Task 2: Medium — Test-Induced Hidden Dependency Cascade
- **Fault:** DNS resolution failure breaks `auth-service` → `user-service`
- **Cascade:** `user-service` → `auth-service` → `api-gateway` → `storefront-ui`
- **Red herrings:** CPU spike + memory warning (both unrelated)
- **Expected GPT-4 score:** 0.52 | **Random:** 0.10

### Task 3: Hard — Process-Induced Cascading Failure with SLA Pressure
- **Fault:** Memory leak + crash-loop on `payments-db`
- **Cascade:** Across 5 services. Misleading network latency alerts.
- **SLA breach at step 6** — escalation mandatory
- **Expected GPT-4 score:** 0.31 | **Random:** 0.05

---

## 🎁 Reward Function

```
score = root_cause × 0.35
      + action      × 0.25
      + severity    × 0.20
      + comms       × 0.10
      + speed       × 0.10
      − false_positive × 0.15
      − wrong_action   × 0.20
      − missed_escalation × 0.25
```

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Run local validation
python scripts/validate_env.py

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

## 🐳 Docker

```bash
docker build -t incident-response-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  incident-response-env
```

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📊 Baseline Scores

| Task | Llama-3.3-70B | Random |
|---|---|---|
| Easy | 0.60 | 0.15 |
| Medium | 0.40 | 0.10 |
| Hard | 0.22 | 0.05 |

---

## 📚 Data Attribution

- **Alibaba Cluster Trace v2021** — metric patterns and service topology
- **Microsoft AIOpsLab** — fault injection taxonomy
- **Google SRE Book (Ch 13–16)** — incident scenario narratives and grader rubrics