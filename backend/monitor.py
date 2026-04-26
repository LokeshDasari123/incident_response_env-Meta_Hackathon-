"""
backend/monitor.py
==================
Production Monitoring Daemon + Email Alert Analyzer

Features:
- Periodic polling of system metrics (configurable interval)
- Email alert ingestion + full log analysis
- Root cause analysis via the trained incident env
- Auto-remediation execution OR human-in-loop suggestion
- Generates structured reports
- Streams results to UI via SSE
"""

import asyncio
import email
import imaplib
import json
import os
import re
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

ROOT = Path(__file__).resolve().parents[1]

# ── Simulated metric store (replace with real Prometheus/Datadog in prod) ─────
class MetricStore:
    """Generates realistic live metrics with occasional fault injection."""

    SERVICES = [
        "storefront-ui", "api-gateway", "auth-service",
        "payments-api", "payments-db", "order-service",
        "cache-service", "user-service", "notification-svc",
    ]

    FAULT_TYPES = [
        "misconfiguration", "memory_leak", "network_partition",
        "crash_loop", "resource_exhaustion",
    ]

    def __init__(self):
        self._baselines  = self._init_baselines()
        self._active_fault: Optional[Dict] = None
        self._fault_step = 0
        self._history: List[Dict] = []

    def _init_baselines(self) -> Dict:
        return {
            svc: {
                "cpu":    round(random.uniform(0.15, 0.45), 3),
                "mem":    round(random.uniform(0.25, 0.55), 3),
                "rt":     round(random.uniform(20, 150), 1),
                "mcr":    round(random.uniform(200, 1200), 0),
                "errors": 0.0,
            }
            for svc in self.SERVICES
        }

    def inject_fault(self, fault_type: Optional[str] = None, target: Optional[str] = None):
        """Inject a fault into a random or specified service."""
        ft  = fault_type or random.choice(self.FAULT_TYPES)
        tgt = target or random.choice(self.SERVICES)
        self._active_fault = {
            "id":         str(uuid.uuid4())[:8],
            "type":       ft,
            "target":     tgt,
            "started_at": datetime.utcnow().isoformat(),
            "cascade":    [],
        }
        self._fault_step = 0
        return self._active_fault

    def clear_fault(self):
        self._active_fault = None
        self._fault_step   = 0

    def snapshot(self) -> Dict[str, Any]:
        """Return current metric snapshot, progressively degraded if fault active."""
        metrics = {}
        alerts  = []
        self._fault_step += 1

        for svc in self.SERVICES:
            b   = self._baselines[svc]
            cpu = b["cpu"] + random.gauss(0, 0.02)
            mem = b["mem"] + random.gauss(0, 0.015)
            rt  = b["rt"]  + random.gauss(0, b["rt"] * 0.05)
            mcr = b["mcr"] + random.gauss(0, b["mcr"] * 0.03)
            err = 0.0
            status = "healthy"

            # Apply fault degradation
            if self._active_fault:
                ft  = self._active_fault["type"]
                tgt = self._active_fault["target"]
                step = self._fault_step

                if svc == tgt:
                    factor = min(1.0, step * 0.15)
                    if ft == "memory_leak":
                        mem  = min(0.99, b["mem"] + factor * 0.45)
                        err  = factor * 0.6
                        status = "critical" if factor > 0.5 else "degraded"
                    elif ft == "misconfiguration":
                        rt   = b["rt"] * (1 + factor * 20)
                        mcr  = b["mcr"] * (1 - factor * 0.9)
                        err  = factor * 0.7
                        status = "failing" if factor > 0.7 else "critical"
                    elif ft == "crash_loop":
                        cpu  = min(0.99, b["cpu"] + factor * 0.5)
                        err  = factor * 0.8
                        status = "failing" if factor > 0.6 else "critical"
                    elif ft == "resource_exhaustion":
                        cpu  = min(0.99, b["cpu"] + factor * 0.6)
                        mem  = min(0.99, b["mem"] + factor * 0.3)
                        err  = factor * 0.5
                        status = "critical" if factor > 0.4 else "degraded"
                    elif ft == "network_partition":
                        rt   = b["rt"] * (1 + factor * 15)
                        mcr  = b["mcr"] * (1 - factor * 0.95)
                        err  = factor * 0.9
                        status = "failing" if factor > 0.5 else "critical"

                    # Track cascade for alert generation
                    if self._active_fault and svc not in self._active_fault["cascade"]:
                        self._active_fault["cascade"].append(svc)

            metrics[svc] = {
                "cpu_utilization":    round(max(0, min(1, cpu)), 3),
                "memory_utilization": round(max(0, min(1, mem)), 3),
                "http_rt":            round(max(1, rt), 1),
                "http_mcr":           round(max(0, mcr), 1),
                "error_rate":         round(max(0, min(1, err)), 3),
                "is_healthy":         status == "healthy",
                "status":             status,
            }

            # Generate alerts for unhealthy services
            if status != "healthy":
                alerts.append({
                    "alert_id":     f"ALT-{svc[:4].upper()}-{self._fault_step:03d}",
                    "service":      svc,
                    "metric":       "http_rt" if rt > b["rt"] * 3 else "cpu_utilization",
                    "current_value": round(rt if rt > b["rt"] * 3 else cpu, 2),
                    "threshold":    round(b["rt"] * 3 if rt > b["rt"] * 3 else 0.85, 2),
                    "severity":     "critical" if status == "failing" else "warning",
                    "fired_at_step": self._fault_step,
                })

        snap = {
            "timestamp":    datetime.utcnow().isoformat(),
            "metrics":      metrics,
            "alerts":       alerts,
            "active_fault": self._active_fault,
            "fault_step":   self._fault_step,
        }
        self._history.append(snap)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        return snap

    def get_history(self, n: int = 60) -> List[Dict]:
        return self._history[-n:]


# ── Root Cause Analyzer ───────────────────────────────────────────────────────
import os
import json
import time
import textwrap
from openai import OpenAI

class RootCauseAnalyzer:
    """
    Analyzes metrics + alerts to identify root cause using a real LLM.
    Supports email body as additional log context.
    """

    def analyze(
        self,
        metrics: Dict,
        alerts: List[Dict],
        topology: Optional[List] = None,
        email_body: Optional[str] = None,
        log_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full root cause analysis pipeline using LLM."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        api_key = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
        api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
        model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        
        fallback = {
            "root_cause_service": "unknown", "root_cause_type": "unknown",
            "severity": "P2", "affected_services": [],
            "remediation_action": "investigate_further",
            "stakeholder_message": "LLM analysis failed. Investigating manually.",
            "confidence": 0.0, "reasoning": "LLM failed or API key not configured."
        }

        if not api_key:
            print("[RootCauseAnalyzer] No API_KEY configured. Using fallback.", flush=True)
            return fallback

        client = OpenAI(api_key=api_key, base_url=api_base)

        system_prompt = textwrap.dedent("""
        You are an expert SRE triaging a production incident.
        
        APPROACH:
        1. Read the call graph topology — traverse INWARD from the edge service
        2. The root cause is the DEEPEST service showing abnormal metrics
        3. List ONLY services in the cascade chain in affected_services — NEVER include red herrings
        4. Red herrings = worker-node-*, network-switch-*, cache-service with only CPU spikes, unrelated batch jobs — EXCLUDE from affected_services
        
        FAULT TYPE RULES:
        - misconfiguration = config_change in timeline + RT spike immediately after
        - memory_leak = memory_utilization > 0.85 AND restart_count > 0
        - network_partition = providerRPC_MCR drops to 0 or near-zero
        - crash_loop = restart_count > 2 AND service cycling
        
        SEVERITY RULES (CRITICAL — follow strictly):
        - P0 = ANY cascading failure affecting 2+ services OR payment/checkout impacted OR config_change caused outage
        - P1 = Single service degraded with user-facing impact
        - P2 = Warning alerts, no user impact yet
        
        REMEDIATION RULES (CRITICAL):
        - memory_leak OR crash_loop -> restart_service
        - misconfiguration -> rollback
        - resource_exhaustion -> scale_up
        - network_partition -> fix_config
        
        OUTPUT FORMAT (Strict JSON only, no markdown blocks, no conversational text):
        {
          "root_cause_service": "string",
          "root_cause_type": "string (memory_leak|misconfiguration|network_partition|crash_loop|resource_exhaustion|dependency_failure)",
          "severity": "string (P0|P1|P2)",
          "affected_services": ["list", "of", "strings"],
          "remediation_action": "string",
          "stakeholder_message": "string",
          "reasoning": "string",
          "confidence": 0.0 to 1.0 (float)
        }
        """)

        user_prompt = textwrap.dedent(f"""
        === ALERTS ({len(alerts)}) ===
        {json.dumps(alerts, indent=2)}
        
        === METRICS ===
        {json.dumps(metrics, indent=2)}
        
        === CALL GRAPH TOPOLOGY ===
        {json.dumps(topology or [], indent=2)}
        
        === EMAIL/LOG BODY ===
        {email_body or log_text or "None"}
        
        Identify the root cause strictly following the rules. Return JSON only.
        """).strip()

        for attempt in range(1, 3):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                text = (resp.choices[0].message.content or "").strip()
                if "```" in text:
                    parts = text.split("```")
                    text = parts[1] if len(parts) > 1 else parts[0]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                
                result = json.loads(text)
                if "confidence" not in result:
                    result["confidence"] = 0.8  # Fallback if LLM forgets
                return result

            except Exception as e:
                print(f"[RootCauseAnalyzer] LLM error on attempt {attempt}: {e}", flush=True)
                time.sleep(2)

        return fallback

    def _score_services(self, metrics: Dict, alerts: List[Dict]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        alert_map: Dict[str, int] = {}
        for a in alerts:
            alert_map[a.get("service", "")] = alert_map.get(a.get("service", ""), 0) + 1

        for svc, m in metrics.items():
            s = 0.0
            if not m.get("is_healthy", True):            s += 0.5
            if m.get("status") == "failing":             s += 0.4
            elif m.get("status") == "critical":          s += 0.25
            elif m.get("status") == "degraded":          s += 0.1
            if m.get("memory_utilization", 0) > 0.85:   s += 0.3
            if m.get("cpu_utilization", 0) > 0.85:      s += 0.2
            if m.get("error_rate", 0) > 0.5:            s += 0.3
            if m.get("http_rt", 0) > 1000:              s += 0.2
            s += alert_map.get(svc, 0) * 0.1
            if s > 0:
                scores[svc] = s
        return scores

    def _parse_text_signals(self, text: str) -> Dict[str, float]:
        """Extract service mentions + error keywords from email/logs."""
        signals: Dict[str, float] = {}
        known_svcs = [
            "payments-db", "payments-api", "auth-service", "api-gateway",
            "storefront-ui", "order-service", "cache-service", "user-service",
            "checkout-ui", "notification-svc",
        ]
        error_kws = ["OOM", "crash", "timeout", "failed", "error", "exception",
                     "NXDOMAIN", "connection refused", "503", "502", "504"]

        text_lower = text.lower()
        for svc in known_svcs:
            if svc in text_lower:
                boost = 0.3
                # Extra boost if error keywords near service mention
                idx = text_lower.find(svc)
                context = text_lower[max(0, idx-100):idx+100]
                for kw in error_kws:
                    if kw.lower() in context:
                        boost += 0.15
                signals[svc] = boost

        return signals

    def _infer_fault_type(self, m: Dict, alerts: List, rc: str) -> str:
        if m.get("memory_utilization", 0) > 0.90:          return "memory_leak"
        if m.get("error_rate", 0) > 0.7:                   return "crash_loop"
        if m.get("http_rt", 0) > 5000 and m.get("http_mcr", 999) < 10: return "network_partition"
        if m.get("cpu_utilization", 0) > 0.90:             return "resource_exhaustion"
        return "misconfiguration"

    def _build_cascade(self, rc: str, topology: List, metrics: Dict) -> List[str]:
        affected = [rc]
        for edge in topology:
            if edge.get("downstream_service") == rc:
                us = edge.get("upstream_service", "")
                if us and us not in affected:
                    affected.append(us)
        # Also add any other unhealthy services
        for svc, m in metrics.items():
            if svc not in affected and not m.get("is_healthy", True):
                affected.append(svc)
        return affected[:6]

    def _infer_action(self, fault_type: str, m: Dict) -> str:
        mapping = {
            "memory_leak":        "restart_service",
            "crash_loop":         "restart_service",
            "network_partition":  "fix_config",
            "misconfiguration":   "fix_config",
            "resource_exhaustion": "scale_up",
        }
        return mapping.get(fault_type, "investigate_further")

    def _unknown_result(self) -> Dict:
        return {
            "root_cause_service":  "unknown",
            "root_cause_type":     "unknown",
            "severity":            "P2",
            "affected_services":   [],
            "remediation_action":  "investigate_further",
            "stakeholder_message": "Insufficient data to identify root cause. Manual investigation needed.",
            "confidence":          0.0,
            "scores":              {},
            "extra_signals":       {},
            "reasoning":           "No degraded services detected.",
        }


# ── Email Alert Ingestor ──────────────────────────────────────────────────────
class EmailAlertIngestor:
    """
    Connects to IMAP mailbox, fetches alert emails, extracts log content.
    Falls back to simulated emails if no real credentials provided.
    """

    def __init__(
        self,
        host:     str = "",
        username: str = "",
        password: str = "",
        folder:   str = "INBOX",
    ):
        self.host     = host     or os.getenv("IMAP_HOST",     "")
        self.username = username or os.getenv("IMAP_USER",     "")
        self.password = password or os.getenv("IMAP_PASS",     "")
        self.folder   = folder
        self._simulated_queue: List[Dict] = []

    def fetch_alerts(self, max_count: int = 10) -> List[Dict]:
        """Fetch real or simulated alert emails."""
        if self.host and self.username and self.password:
            return self._fetch_real(max_count)
        return self._fetch_simulated(max_count)

    def _fetch_real(self, max_count: int) -> List[Dict]:
        alerts = []
        try:
            mail = imaplib.IMAP4_SSL(self.host)
            mail.login(self.username, self.password)
            mail.select(self.folder)
            _, ids = mail.search(None, "UNSEEN")
            for eid in ids[0].split()[-max_count:]:
                _, data = mail.fetch(eid, "(RFC822)")
                msg     = email.message_from_bytes(data[0][1])
                subject = msg.get("Subject", "")
                body    = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                            break
                else:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
                alerts.append({
                    "id":        str(eid),
                    "subject":   subject,
                    "body":      body,
                    "received":  datetime.utcnow().isoformat(),
                    "source":    "email",
                })
            mail.logout()
        except Exception as exc:
            print(f"[EMAIL] IMAP error: {exc}")
        return alerts

    def _fetch_simulated(self, max_count: int) -> List[Dict]:
        """Generate realistic simulated alert emails for demo/training."""
        templates = [
            {
                "subject": "[ALERT] payments-db OOM kill — restart loop detected",
                "body": (
                    "Host: payments-db-pod-7f8d\n"
                    "Time: {ts}\n"
                    "Event: OOM kill #3 in last 5 minutes\n"
                    "Memory: 99.1% utilized\n"
                    "Logs:\n"
                    "  [ERROR] java.lang.OutOfMemoryError: Java heap space\n"
                    "  [ERROR] Connection pool exhausted: 0/100 available\n"
                    "  [WARN]  payments-api: connection timeout after 30000ms\n"
                    "  [ERROR] checkout-ui: HTTP 503 Service Unavailable\n"
                    "Action required: Immediate investigation of payments-db memory usage."
                ),
            },
            {
                "subject": "[PagerDuty] P0 — storefront-ui HTTP_RT > 45000ms",
                "body": (
                    "Alert: HTTP response time critical threshold breached\n"
                    "Service: storefront-ui → api-gateway → payments-db\n"
                    "Metric: HTTP_RT = 45234ms (threshold: 500ms)\n"
                    "Duration: 8 minutes\n"
                    "Recent deployments:\n"
                    "  2025-01-01 02:14 UTC — payments-db ConfigMap update (operator: jenkins)\n"
                    "  max_connections changed from 100 to 5\n"
                    "Cascade detected: payments-db → payments-api → checkout-ui → storefront-ui\n"
                    "Revenue impact: ~$12,000/min estimated"
                ),
            },
            {
                "subject": "[Datadog] WARN — auth-service DNS resolution failure",
                "body": (
                    "Monitor: auth-service.providerRPC_MCR dropped to 0\n"
                    "Reason: DNS lookup failure\n"
                    "  nslookup user-service.prod.svc.cluster.local → NXDOMAIN\n"
                    "  Last successful resolution: 2025-01-01T01:45:22Z\n"
                    "  Test environment cleanup removed DNS record at 02:01 UTC\n"
                    "Impact: All authentication requests failing\n"
                    "Affected: auth-service, api-gateway, storefront-ui\n"
                    "Suggested fix: Restore DNS record for user-service.prod"
                ),
            },
        ]

        if not self._simulated_queue:
            # Generate a batch
            for tmpl in templates[:max_count]:
                self._simulated_queue.append({
                    "id":       str(uuid.uuid4())[:8],
                    "subject":  tmpl["subject"],
                    "body":     tmpl["body"].format(ts=datetime.utcnow().isoformat()),
                    "received": datetime.utcnow().isoformat(),
                    "source":   "simulated",
                })

        result = self._simulated_queue[:max_count]
        self._simulated_queue = self._simulated_queue[max_count:]
        return result


# ── Auto-Remediation Engine ───────────────────────────────────────────────────
class RemediationEngine:
    """
    Executes remediation actions or drafts them for human approval.
    In real deployment: connects to Kubernetes, Terraform, AWS CLI etc.
    """

    AUTO_APPROVE_ACTIONS = {"fix_config", "flush_cache", "reroute_traffic"}
    REQUIRE_APPROVAL     = {"restart_service", "rollback", "scale_up", "escalate"}

    def execute(
        self,
        analysis: Dict,
        auto_remediate: bool = False,
    ) -> Dict[str, Any]:
        action   = analysis.get("remediation_action", "investigate_further")
        rc       = analysis.get("root_cause_service",  "unknown")
        severity = analysis.get("severity",             "P2")

        if auto_remediate and action in self.AUTO_APPROVE_ACTIONS:
            return self._execute_action(action, rc, severity)
        else:
            return self._draft_for_approval(action, rc, severity, analysis)

    def _execute_action(self, action: str, rc: str, severity: str) -> Dict:
        """Simulate execution — replace with real kubectl/API calls."""
        time.sleep(0.1)  # simulate API call
        return {
            "status":    "executed",
            "action":    action,
            "target":    rc,
            "severity":  severity,
            "message":   f"✅ Auto-executed: {action.replace('_',' ')} on {rc}",
            "executed_at": datetime.utcnow().isoformat(),
            "requires_approval": False,
        }

    def _draft_for_approval(self, action: str, rc: str, severity: str, analysis: Dict) -> Dict:
        steps = self._build_runbook(action, rc, analysis)
        return {
            "status":    "pending_approval",
            "action":    action,
            "target":    rc,
            "severity":  severity,
            "message":   f"⚠️ Approval required: {action.replace('_',' ')} on {rc}",
            "runbook":   steps,
            "executed_at": None,
            "requires_approval": True,
        }

    def _build_runbook(self, action: str, rc: str, analysis: Dict) -> List[str]:
        fault = analysis.get("root_cause_type", "unknown")
        runbooks = {
            "restart_service": [
                f"1. kubectl rollout restart deployment/{rc} -n production",
                f"2. Watch pod restart: kubectl get pods -l app={rc} -w",
                f"3. Verify metrics recover within 2 minutes",
                f"4. If no recovery, escalate to on-call lead",
            ],
            "rollback": [
                f"1. Check last known good deployment: kubectl rollout history deployment/{rc}",
                f"2. Rollback: kubectl rollout undo deployment/{rc}",
                f"3. Verify deployment: kubectl rollout status deployment/{rc}",
                f"4. Confirm metrics recovery",
            ],
            "scale_up": [
                f"1. kubectl scale deployment/{rc} --replicas=5",
                f"2. Monitor HPA: kubectl get hpa",
                f"3. Verify CPU/Memory distributes across replicas",
                f"4. Update HPA max replicas if recurring",
            ],
            "fix_config": [
                f"1. kubectl get configmap {rc}-config -o yaml > backup.yaml",
                f"2. Edit ConfigMap: kubectl edit configmap {rc}-config",
                f"3. Apply fix based on fault: {fault}",
                f"4. Restart pods to pick up new config: kubectl rollout restart deployment/{rc}",
            ],
            "escalate": [
                f"1. Page on-call engineer via PagerDuty",
                f"2. Open incident channel: #incident-{rc.replace('-', '_')}",
                f"3. Share this analysis in channel",
                f"4. Severity: {analysis.get('severity')} — {analysis.get('stakeholder_message','')[:100]}",
            ],
        }
        return runbooks.get(action, [f"Investigate {rc} manually. Fault: {fault}"])


# ── Report Generator ──────────────────────────────────────────────────────────
class ReportGenerator:
    """Generates JSON + Markdown incident reports."""

    REPORTS_DIR = ROOT / "data" / "reports"

    def __init__(self):
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        analysis:     Dict,
        remediation:  Dict,
        metrics_snap: Dict,
        email_alert:  Optional[Dict] = None,
    ) -> Dict[str, Any]:
        ts      = datetime.utcnow()
        rep_id  = f"INC-{ts.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:4].upper()}"

        report = {
            "report_id":     rep_id,
            "generated_at":  ts.isoformat(),
            "incident": {
                "id":               rep_id,
                "root_cause":       analysis.get("root_cause_service"),
                "fault_type":       analysis.get("root_cause_type"),
                "severity":         analysis.get("severity"),
                "affected":         analysis.get("affected_services", []),
                "confidence":       analysis.get("confidence"),
                "reasoning":        analysis.get("reasoning"),
                "stakeholder_msg":  analysis.get("stakeholder_message"),
            },
            "remediation": {
                "action":           remediation.get("action"),
                "status":           remediation.get("status"),
                "message":          remediation.get("message"),
                "runbook":          remediation.get("runbook", []),
                "executed_at":      remediation.get("executed_at"),
            },
            "metrics_at_detection": {
                svc: {
                    "cpu":    m.get("cpu_utilization"),
                    "mem":    m.get("memory_utilization"),
                    "rt":     m.get("http_rt"),
                    "status": m.get("status"),
                }
                for svc, m in metrics_snap.get("metrics", {}).items()
            },
            "email_alert": email_alert,
            "alert_count": len(metrics_snap.get("alerts", [])),
        }

        # Save JSON
        json_path = self.REPORTS_DIR / f"{rep_id}.json"
        json_path.write_text(json.dumps(report, indent=2))

        # Save Markdown
        md_path = self.REPORTS_DIR / f"{rep_id}.md"
        md_path.write_text(self._to_markdown(report))

        return report

    def _to_markdown(self, r: Dict) -> str:
        inc = r["incident"]
        rem = r["remediation"]
        ts  = r["generated_at"]
        lines = [
            f"# Incident Report — {r['report_id']}",
            f"**Generated:** {ts}",
            "",
            "## Summary",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Root Cause | `{inc['root_cause']}` |",
            f"| Fault Type | `{inc['fault_type']}` |",
            f"| Severity | **{inc['severity']}** |",
            f"| Confidence | {inc['confidence']*100:.0f}% |",
            f"| Affected Services | {', '.join(inc['affected'])} |",
            "",
            "## Stakeholder Message",
            f"> {inc['stakeholder_msg']}",
            "",
            "## Reasoning",
            inc['reasoning'],
            "",
            "## Remediation",
            f"**Action:** `{rem['action']}`  ",
            f"**Status:** {rem['status']}",
            "",
            "### Runbook",
        ]
        for step in rem.get("runbook", []):
            lines.append(f"- {step}")
        lines += [
            "",
            "## Service Metrics at Detection",
            "| Service | CPU | Memory | HTTP RT | Status |",
            "|---------|-----|--------|---------|--------|",
        ]
        for svc, m in r.get("metrics_at_detection", {}).items():
            lines.append(
                f"| {svc} | {m.get('cpu_utilization', 0)*100:.0f}% | {m.get('memory_utilization', 0)*100:.0f}% | "
                f"{m.get('http_rt', 0):.0f}ms | {m.get('status', 'unknown')} |"
            )
        return "\n".join(lines)

    def list_reports(self) -> List[Dict]:
        reports = []
        for p in sorted(self.REPORTS_DIR.glob("*.json"), reverse=True)[:20]:
            try:
                r = json.loads(p.read_text())
                reports.append({
                    "id":        r["report_id"],
                    "severity":  r["incident"]["severity"],
                    "root_cause": r["incident"]["root_cause"],
                    "fault_type": r["incident"]["fault_type"],
                    "generated": r["generated_at"],
                    "action":    r["remediation"]["action"],
                })
            except Exception:
                pass
        return reports


# ── Monitoring Daemon ─────────────────────────────────────────────────────────
class MonitoringDaemon:
    """
    Main daemon: polls metrics, ingests emails, runs analysis, generates reports.
    Streams events to connected UI clients via SSE.
    """

    def __init__(self, poll_interval: int = 30, auto_remediate: bool = False):
        self.poll_interval  = poll_interval
        self.auto_remediate = auto_remediate
        self.metric_store   = MetricStore()
        self.analyzer       = RootCauseAnalyzer()
        self.email_ingestor = EmailAlertIngestor()
        self.remediation    = RemediationEngine()
        self.reporter       = ReportGenerator()
        self._subscribers:  List[asyncio.Queue] = []
        self._running       = False
        self._incidents:    List[Dict] = []
        self._last_snapshot: Optional[Dict] = None

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = [s for s in self._subscribers if s is not q]

    async def _broadcast(self, event_type: str, data: Dict):
        msg = json.dumps({"type": event_type, "data": data, "ts": datetime.utcnow().isoformat()})
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    async def run(self):
        self._running = True
        print(f"[DAEMON] Started — poll_interval={self.poll_interval}s auto_remediate={self.auto_remediate}")

        # Inject a fault after 10 seconds for demo
        await asyncio.sleep(10)
        fault = self.metric_store.inject_fault()
        await self._broadcast("fault_injected", fault)
        print(f"[DAEMON] Fault injected: {fault['type']} on {fault['target']}")

        tick = 0
        while self._running:
            tick += 1
            snap = self.metric_store.snapshot()
            self._last_snapshot = snap
            await self._broadcast("metrics_update", snap)

            # Check for alerts
            if snap["alerts"]:
                await self._broadcast("alerts_fired", {"alerts": snap["alerts"], "count": len(snap["alerts"])})

                # Every 3 ticks with alerts: run full RCA
                if tick % 3 == 0:
                    await self._run_analysis(snap)

            # Check email every 5 ticks
            if tick % 5 == 0:
                emails = self.email_ingestor.fetch_alerts(max_count=2)
                for em in emails:
                    await self._broadcast("email_received", em)
                    await self._run_analysis(snap, email_alert=em)

            await asyncio.sleep(self.poll_interval)

    async def _run_analysis(self, snap: Dict, email_alert: Optional[Dict] = None):
        analysis = self.analyzer.analyze(
            metrics    = snap["metrics"],
            alerts     = snap["alerts"],
            email_body = email_alert.get("body") if email_alert else None,
        )
        remediation = self.remediation.execute(analysis, auto_remediate=self.auto_remediate)
        report      = self.reporter.generate(analysis, remediation, snap, email_alert)

        incident = {
            "id":          report["report_id"],
            "analysis":    analysis,
            "remediation": remediation,
            "report_id":   report["report_id"],
            "timestamp":   datetime.utcnow().isoformat(),
        }
        self._incidents.append(incident)

        await self._broadcast("incident_detected", incident)
        await self._broadcast("report_generated",  {"report_id": report["report_id"]})

    def stop(self):
        self._running = False

    def get_status(self) -> Dict:
        return {
            "running":        self._running,
            "poll_interval":  self.poll_interval,
            "auto_remediate": self.auto_remediate,
            "incident_count": len(self._incidents),
            "active_fault":   self._last_snapshot.get("active_fault") if self._last_snapshot else None,
            "last_poll":      self._last_snapshot.get("timestamp") if self._last_snapshot else None,
        }

    def get_incidents(self, n: int = 20) -> List[Dict]:
        return self._incidents[-n:]

    def inject_fault(self, fault_type: Optional[str] = None, target: Optional[str] = None) -> Dict:
        return self.metric_store.inject_fault(fault_type, target)

    def clear_fault(self):
        self.metric_store.clear_fault()

    def get_metrics_history(self, n: int = 60) -> List[Dict]:
        return self.metric_store.get_history(n)


# Singleton daemon instance
_daemon: Optional[MonitoringDaemon] = None


def get_daemon(poll_interval: int = 5, auto_remediate: bool = False) -> MonitoringDaemon:
    global _daemon
    if _daemon is None:
        _daemon = MonitoringDaemon(poll_interval=poll_interval, auto_remediate=auto_remediate)
    return _daemon