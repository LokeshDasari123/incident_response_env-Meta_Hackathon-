"""
backend/api.py
==============
FastAPI server — serves the full system:
- /metrics/stream  SSE: live metrics to UI
- /incidents/      incident list + detail
- /reports/        report list + download
- /train/start     kick off training
- /train/status    live training stats
- /fault/inject    inject fault for demo
- /remediate/approve  approve pending action
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.monitor import get_daemon, MonitoringDaemon

# ── Training log path ─────────────────────────────────────────────────────────
TRAINING_LOG_DIR = ROOT / "data" / "training_logs"
TRAINING_LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE     = TRAINING_LOG_DIR / "latest_summary.json"
REPORTS_DIR      = ROOT / "data" / "reports"

app = FastAPI(title="Incident Response AI — Full System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request models ────────────────────────────────────────────────────────────
class FaultRequest(BaseModel):
    fault_type: Optional[str] = None
    target:     Optional[str] = None

class TrainRequest(BaseModel):
    task:       str  = "all"
    episodes:   int  = 50
    curriculum: bool = True

class RemediateRequest(BaseModel):
    incident_id: str
    approved:    bool


# ── Startup: launch daemon ────────────────────────────────────────────────────
daemon: Optional[MonitoringDaemon] = None

@app.on_event("startup")
async def startup():
    global daemon
    poll = int(os.getenv("POLL_INTERVAL", "5"))
    auto = os.getenv("AUTO_REMEDIATE", "false").lower() == "true"
    daemon = get_daemon(poll_interval=poll, auto_remediate=auto)
    asyncio.create_task(daemon.run())


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0", "daemon": daemon.get_status() if daemon else None}


# ── SSE: live metrics stream ──────────────────────────────────────────────────
@app.get("/metrics/stream")
async def metrics_stream():
    """Server-Sent Events — pushes live metrics + alerts + incidents to UI."""
    if not daemon:
        raise HTTPException(503, "Daemon not started")

    q = daemon.subscribe()

    async def generate():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    # Keepalive
                    yield f"data: {json.dumps({'type':'ping','ts':datetime.utcnow().isoformat()})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            daemon.unsubscribe(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Metrics snapshot ──────────────────────────────────────────────────────────
@app.get("/metrics/snapshot")
async def metrics_snapshot():
    if not daemon:
        raise HTTPException(503)
    return daemon._last_snapshot or {}


@app.get("/metrics/history")
async def metrics_history(n: int = 60):
    if not daemon:
        raise HTTPException(503)
    return daemon.get_metrics_history(n)


# ── Incidents ─────────────────────────────────────────────────────────────────
@app.get("/incidents")
async def list_incidents(n: int = 20):
    if not daemon:
        raise HTTPException(503)
    return daemon.get_incidents(n)


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    if not daemon:
        raise HTTPException(503)
    for inc in daemon.get_incidents(100):
        if inc["id"] == incident_id:
            return inc
    raise HTTPException(404, "Incident not found")


# ── Reports ───────────────────────────────────────────────────────────────────
@app.get("/reports")
async def list_reports():
    from backend.monitor import ReportGenerator
    return ReportGenerator().list_reports()


@app.get("/reports/{report_id}")
async def get_report(report_id: str):
    path = REPORTS_DIR / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(404)
    return json.loads(path.read_text())


@app.get("/reports/{report_id}/markdown")
async def get_report_markdown(report_id: str):
    path = REPORTS_DIR / f"{report_id}.md"
    if not path.exists():
        raise HTTPException(404)
    return {"markdown": path.read_text()}


# ── Fault injection ───────────────────────────────────────────────────────────
@app.post("/fault/inject")
async def inject_fault(req: FaultRequest):
    if not daemon:
        raise HTTPException(503)
    fault = daemon.inject_fault(req.fault_type, req.target)
    return {"status": "injected", "fault": fault}


@app.post("/fault/clear")
async def clear_fault():
    if not daemon:
        raise HTTPException(503)
    daemon.clear_fault()
    return {"status": "cleared"}


# ── Remediation approval ──────────────────────────────────────────────────────
@app.post("/remediate/approve")
async def approve_remediation(req: RemediateRequest):
    if not daemon:
        raise HTTPException(503)
    for inc in daemon.get_incidents(50):
        if inc["id"] == req.incident_id:
            if req.approved:
                # Execute the action
                from backend.monitor import RemediationEngine
                engine    = RemediationEngine()
                action    = inc["remediation"]["action"]
                result    = engine._execute_action(
                    action, inc["analysis"]["root_cause_service"],
                    inc["analysis"]["severity"]
                )
                return {"status": "executed", "result": result}
            else:
                return {"status": "rejected", "incident_id": req.incident_id}
    raise HTTPException(404, "Incident not found")


# ── Training control ──────────────────────────────────────────────────────────
_training_task: Optional[asyncio.Task] = None
_training_running = False


@app.post("/train/start")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    global _training_task, _training_running
    if _training_running:
        return {"status": "already_running"}

    def run_training():
        global _training_running
        _training_running = True
        try:
            import subprocess
            tasks_arg = req.task
            cmd = [
                sys.executable,
                str(ROOT / "training" / "train.py"),
                "--task",     tasks_arg,
                "--episodes", str(req.episodes),
            ]
            if req.curriculum:
                cmd.append("--curriculum")
            subprocess.run(cmd, cwd=str(ROOT))
        finally:
            _training_running = False

    background_tasks.add_task(run_training)
    return {"status": "started", "task": req.task, "episodes": req.episodes}


@app.get("/train/status")
async def training_status():
    if not SUMMARY_FILE.exists():
        return {
            "running": _training_running,
            "progress": 0,
            "per_task": {},
        }
    try:
        summary = json.loads(SUMMARY_FILE.read_text())
        summary["running"] = _training_running
        return summary
    except Exception:
        return {"running": _training_running, "error": "failed to read summary"}


@app.get("/train/logs")
async def training_logs(n: int = 200):
    """Return last N training step logs."""
    logs_dir = TRAINING_LOG_DIR
    files    = sorted(logs_dir.glob("training_*.jsonl"), reverse=True)
    if not files:
        return {"logs": []}
    lines = []
    with open(files[0]) as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except Exception:
                pass
    return {"logs": lines[-n:], "file": files[0].name}


# ── Data generation (for training) ───────────────────────────────────────────
@app.post("/data/generate")
async def generate_training_data(n: int = 100):
    """Generate synthetic training scenarios and save to disk."""
    from scenarios.scenario_generator import generate_scenario_variant, NOISE_PROFILES
    import random

    out = []
    for i in range(n):
        diff = random.choice(["easy", "medium", "hard"])
        seed = random.randint(0, 99999)
        v    = generate_scenario_variant(diff, seed=seed)
        out.append({
            "difficulty": diff,
            "seed":       seed,
            "scenario_id": v.scenario_id,
            "root_cause":  v.ground_truth["root_cause_service"],
            "fault_type":  v.ground_truth["root_cause_type"],
        })

    out_path = TRAINING_LOG_DIR / "generated_scenarios.jsonl"
    with open(out_path, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")

    return {"generated": len(out), "file": str(out_path)}


# ── Daemon status ─────────────────────────────────────────────────────────────
@app.get("/daemon/status")
async def daemon_status():
    if not daemon:
        raise HTTPException(503)
    return daemon.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)