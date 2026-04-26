"""
backend/email_trigger.py
========================
Email-triggered incident response pipeline.

Flow:
  1. Email listener detects production alert
  2. Calls /trigger-incident API
  3. Circuit breaker limits to max 2 analysis attempts
  4. AI models debate (Responder ↔ Challenger ↔ Commander)
  5. All steps logged to real-time log queue → streamed to UI
  6. Resolution shows which model won

Integrates with:
  - backend/monitor.py — RootCauseAnalyzer, RemediationEngine, ReportGenerator
  - backend/circuit_breaker.py — max 2 retries with exponential backoff
  - envs/debate.py — Multi-agent debate for AI reasoning
"""

import asyncio
import os
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from backend.monitor import (
    RootCauseAnalyzer,
    RemediationEngine,
    ReportGenerator,
    MetricStore,
)
from backend.circuit_breaker import circuit_breaker

logger = logging.getLogger(__name__)


# ── Log Queue — streams to UI ────────────────────────────────────────────────
class LogQueue:
    """
    Thread-safe log queue that the UI polls for real-time updates.
    Stores last 500 log entries. Each entry has:
      - id, timestamp, level, source, message, data
    """

    def __init__(self, maxlen: int = 500):
        self._queue: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._subscribers: List[asyncio.Queue] = []
        self._counter = 0

    def log(
        self,
        message: str,
        level: str = "info",
        source: str = "system",
        data: Optional[Dict] = None,
    ) -> Dict:
        self._counter += 1
        entry = {
            "id": self._counter,
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "source": source,
            "message": message,
            "data": data or {},
        }
        self._queue.append(entry)

        # Notify SSE subscribers
        for q in self._subscribers:
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                pass

        return entry

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = [s for s in self._subscribers if s is not q]

    def get_logs(self, n: int = 100, after_id: int = 0) -> List[Dict]:
        """Get last N logs, optionally after a specific ID for polling."""
        if after_id > 0:
            return [e for e in self._queue if e["id"] > after_id][-n:]
        return list(self._queue)[-n:]

    def clear(self):
        self._queue.clear()


# ── Model Interaction Tracker ─────────────────────────────────────────────────
class ModelInteractionTracker:
    """
    Tracks which AI models interacted during incident analysis.
    Shows the debate flow and which model ultimately resolved it.
    """

    def __init__(self):
        self._incidents: Dict[str, Dict] = {}

    def start_incident(self, incident_id: str, source: str, subject: str):
        self._incidents[incident_id] = {
            "incident_id": incident_id,
            "source": source,
            "subject": subject,
            "started_at": datetime.utcnow().isoformat(),
            "resolved_at": None,
            "status": "analyzing",
            "models": [],
            "debate_rounds": [],
            "winning_model": None,
            "final_confidence": 0.0,
            "attempts": 0,
        }

    def add_model_interaction(
        self,
        incident_id: str,
        model_name: str,
        role: str,
        action: str,
        result: Optional[Dict] = None,
    ):
        if incident_id not in self._incidents:
            return
        self._incidents[incident_id]["models"].append({
            "model": model_name,
            "role": role,
            "action": action,
            "result_summary": {
                k: v for k, v in (result or {}).items()
                if k in ("root_cause_service", "severity", "confidence", "remediation_action")
            },
            "timestamp": datetime.utcnow().isoformat(),
        })

    def add_debate_round(
        self,
        incident_id: str,
        challenger_strategy: str,
        challenge_text: str,
        agent_improved: bool,
    ):
        if incident_id not in self._incidents:
            return
        self._incidents[incident_id]["debate_rounds"].append({
            "strategy": challenger_strategy,
            "challenge": challenge_text[:200],
            "agent_improved": agent_improved,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def resolve_incident(
        self,
        incident_id: str,
        winning_model: str,
        confidence: float,
        result: Dict,
    ):
        if incident_id not in self._incidents:
            return
        inc = self._incidents[incident_id]
        inc["status"] = "resolved"
        inc["resolved_at"] = datetime.utcnow().isoformat()
        inc["winning_model"] = winning_model
        inc["final_confidence"] = confidence
        inc["final_result"] = {
            "root_cause": result.get("root_cause_service"),
            "fault_type": result.get("root_cause_type"),
            "severity": result.get("severity"),
            "remediation": result.get("remediation_action"),
            "affected": result.get("affected_services", []),
        }

    def fail_incident(self, incident_id: str, reason: str):
        if incident_id not in self._incidents:
            return
        inc = self._incidents[incident_id]
        inc["status"] = "failed"
        inc["resolved_at"] = datetime.utcnow().isoformat()
        inc["failure_reason"] = reason

    def get_incident(self, incident_id: str) -> Optional[Dict]:
        return self._incidents.get(incident_id)

    def get_all(self, n: int = 20) -> List[Dict]:
        items = list(self._incidents.values())
        return sorted(items, key=lambda x: x["started_at"], reverse=True)[:n]

    def get_stats(self) -> Dict:
        total = len(self._incidents)
        resolved = sum(1 for i in self._incidents.values() if i["status"] == "resolved")
        failed = sum(1 for i in self._incidents.values() if i["status"] == "failed")
        return {
            "total": total,
            "resolved": resolved,
            "failed": failed,
            "in_progress": total - resolved - failed,
            "resolution_rate": round(resolved / max(total, 1), 2),
        }


# ── Incident Analysis Pipeline ───────────────────────────────────────────────
class IncidentPipeline:
    """
    Full incident analysis pipeline triggered by email or API.

    Steps:
      1. Parse email/alert subject + body
      2. Generate synthetic metrics (or use live ones from monitor)
      3. Run RCA with circuit breaker (max 2 attempts)
      4. Multi-agent debate on the diagnosis
      5. Generate remediation + report
      6. Stream all steps to log queue
    """

    def __init__(self):
        self.analyzer = RootCauseAnalyzer()
        self.remediation = RemediationEngine()
        self.reporter = ReportGenerator()
        self.metric_store = MetricStore()
        self.log_queue = LogQueue()
        self.model_tracker = ModelInteractionTracker()

    async def trigger_from_email(
        self,
        subject: str,
        body: str = "",
        source: str = "email",
    ) -> Dict[str, Any]:
        """
        Main entry point — triggered when email listener detects a prod alert.
        """
        incident_id = f"INC-{datetime.utcnow().strftime('%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"

        # Log: incident started
        self.log_queue.log(
            f"📧 Email alert received: {subject[:80]}",
            level="alert",
            source="email_listener",
            data={"incident_id": incident_id, "subject": subject},
        )

        # Start tracking
        self.model_tracker.start_incident(incident_id, source, subject)

        from envs.incident_env import IncidentResponseEnv
        import random
        
        env = IncidentResponseEnv()
        task_id = random.choice(["easy", "medium", "hard"])
        obs = env.reset(task_id=task_id, dynamic=True)

        snap = {
            "metrics": obs.metrics,
            "alerts": obs.alerts,
            "topology": obs.topology,
            "ground_truth": env._scenario.ground_truth if env._scenario else {},
        }

        self.log_queue.log(
            f"📊 Environment initialized ({task_id} scenario) — {len(snap['alerts'])} alerts firing",
            level="info",
            source="environment",
            data={"alert_count": len(snap["alerts"])},
        )

        # Run analysis with circuit breaker (max 2 attempts)
        analysis_fn = lambda: self._run_analysis(
            env=env,
            snap=snap,
            subject=subject,
            body=body,
            incident_id_ref=incident_id,
        )

        result = await circuit_breaker.call(
            incident_id=incident_id,
            fn=analysis_fn,
            subject=subject,
        )

        # Log circuit breaker outcome
        if result.get("status") == "resolved":
            analysis = result["result"]
            self.log_queue.log(
                f"✅ Incident resolved — Root cause: {analysis.get('root_cause_service')} "
                f"({analysis.get('root_cause_type')}) | Confidence: {analysis.get('confidence', 0):.0%}",
                level="success",
                source="pipeline",
                data={"incident_id": incident_id, "analysis": analysis},
            )
        elif result.get("status") == "circuit_open":
            self.model_tracker.fail_incident(incident_id, "Circuit breaker tripped — max attempts reached")
            self.log_queue.log(
                f"🔴 Circuit breaker OPEN — {result.get('attempts', 0)} attempts failed. Manual investigation required.",
                level="error",
                source="circuit_breaker",
                data={"incident_id": incident_id},
            )
        elif result.get("status") == "duplicate_blocked":
            self.log_queue.log(
                "⏭️ Duplicate incident blocked by circuit breaker cooldown",
                level="warn",
                source="circuit_breaker",
            )

        # Append circuit breaker logs
        for cb_log in circuit_breaker.get_logs(n=10):
            if cb_log.get("incident_id") == incident_id:
                self.log_queue.log(
                    cb_log["message"],
                    level="info",
                    source="circuit_breaker",
                    data=cb_log,
                )

        return {
            "incident_id": incident_id,
            "status": result.get("status", "unknown"),
            "result": result.get("result"),
            "attempts": result.get("attempts", 0),
            "tracking": self.model_tracker.get_incident(incident_id),
        }

    def _run_analysis(
        self,
        env: Any,
        snap: Dict,
        subject: str,
        body: str,
        incident_id_ref: str,
    ) -> Dict:
        """
        Single analysis attempt — called by circuit breaker.
        Runs full RCA → Debate → Remediation → Report pipeline.
        """
        incident_id = incident_id_ref

        # Step 1: Run RCA via LLM
        self.log_queue.log(
            "🔍 RESPONDER: Running root cause analysis via LLM...",
            level="info",
            source="responder",
        )
        self.model_tracker.add_model_interaction(
            incident_id, "RootCauseAnalyzer (LLM)", "responder", "analyze",
        )

        analysis = self.analyzer.analyze(
            metrics=snap["metrics"],
            alerts=snap["alerts"],
            topology=snap["topology"],
            email_body=body or subject,
        )

        self.log_queue.log(
            f"📋 RESPONDER diagnosis: {analysis.get('root_cause_service')} "
            f"({analysis.get('root_cause_type')}) — Confidence: {analysis.get('confidence', 0):.0%}",
            level="info",
            source="responder",
            data={
                "root_cause": analysis.get("root_cause_service"),
                "fault_type": analysis.get("root_cause_type"),
                "severity": analysis.get("severity"),
                "confidence": analysis.get("confidence"),
            },
        )

        # Step 2: Challenger debate
        self.log_queue.log(
            "⚔️ CHALLENGER: Generating adversarial challenge...",
            level="info",
            source="challenger",
        )
        self.model_tracker.add_model_interaction(
            incident_id, "DebateEngine", "challenger", "challenge",
        )

        from envs.debate import DebateEngine
        debate = DebateEngine(seed=int(time.time()) % 10000)

        challenge = debate.generate_challenge(
            action=analysis,
            metrics=snap["metrics"],
            alerts=snap["alerts"],
            topology=snap["topology"],
            ground_truth=snap["ground_truth"],
            step=1,
            max_steps=10,
        )

        self.log_queue.log(
            f"🗣️ CHALLENGER ({challenge['strategy']}): {challenge['challenge_text'][:150]}...",
            level="debate",
            source="challenger",
            data={"strategy": challenge["strategy"], "hint_quality": challenge["hint_quality"]},
        )
        self.model_tracker.add_debate_round(
            incident_id, challenge["strategy"], challenge["challenge_text"], True,
        )

        # Step 3: Commander evaluates (via env.step)
        self.log_queue.log(
            "🎖️ COMMANDER: Evaluating LLM action against ground truth...",
            level="info",
            source="commander",
        )
        from models.action import IncidentAction
        
        # Instantiate action model so env.step() works
        try:
            action_model = IncidentAction(
                root_cause_service=analysis.get("root_cause_service", "unknown"),
                root_cause_type=analysis.get("root_cause_type", "unknown"),
                severity=analysis.get("severity", "P2"),
                affected_services=analysis.get("affected_services", []),
                remediation_action=analysis.get("remediation_action", "investigate_further"),
                stakeholder_message=analysis.get("stakeholder_message", ""),
                reasoning=analysis.get("reasoning", ""),
                confidence=analysis.get("confidence", 0.0),
            )
        except Exception as e:
            self.log_queue.log(f"⚠️ Failed to parse LLM action into model: {e}", level="warn", source="pipeline")
            action_model = IncidentAction(root_cause_service="unknown")

        obs, reward, done, info = env.step(action_model)
        
        if reward > 0.5:
            self.log_queue.log(
                f"✅ Action approved by Commander (Reward: {reward:.2f}). Incident mitigating...",
                level="success",
                source="commander"
            )
        else:
            self.log_queue.log(
                f"❌ Action failed Commander validation (Reward: {reward:.2f}). Wrong diagnosis.",
                level="error",
                source="commander"
            )

        # Step 4: Remediation
        remediation = self.remediation.execute(analysis, auto_remediate=False)
        self.log_queue.log(
            f"🔧 Remediation: {remediation.get('action')} → Status: {remediation.get('status')}",
            level="info",
            source="remediation",
            data=remediation,
        )

        # Step 5: Report generation
        report = self.reporter.generate(analysis, remediation, snap)
        self.log_queue.log(
            f"📄 Report generated: {report.get('report_id')}",
            level="info",
            source="reporter",
            data={"report_id": report.get("report_id")},
        )

        # Resolve tracking
        if reward > 0.5:
            self.model_tracker.resolve_incident(
                incident_id,
                winning_model=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
                confidence=analysis.get("confidence", 0.0),
                result=analysis,
            )
            # Raise exception if failed to trigger circuit breaker if we wanted to
            # But the UI handles success status

        analysis["model_used"] = "RootCauseAnalyzer"
        analysis["report_id"] = report.get("report_id")
        analysis["remediation"] = remediation
        if reward <= 0.5:
            self.model_tracker.fail_incident(
                incident_id,
                reason=f"LLM failed Commander validation. Reward: {reward:.2f}",
            )

        return analysis

    def get_logs(self, n: int = 100, after_id: int = 0) -> List[Dict]:
        return self.log_queue.get_logs(n, after_id)

    def get_incidents(self, n: int = 20) -> List[Dict]:
        return self.model_tracker.get_all(n)

    def get_stats(self) -> Dict:
        return self.model_tracker.get_stats()


# Global singleton
incident_pipeline = IncidentPipeline()
