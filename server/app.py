"""
server/app.py
-------------
FastAPI application for the Incident Response OpenEnv environment.
Exposes all required OpenEnv endpoints.
"""

import json
import logging
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from envs.incident_env        import IncidentResponseEnv
from server.environment       import handle_reset, handle_step, handle_state
from server.session_manager   import session_manager
from backend.email_trigger    import incident_pipeline
from backend.circuit_breaker  import circuit_breaker

# ── Training artifact paths ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TRAINING_LOG_DIR = ROOT / "data" / "training_logs"
SUMMARY_FILE = TRAINING_LOG_DIR / "latest_summary.json"
CURVES_FILE = TRAINING_LOG_DIR / "reward_curves.json"
HYBRID_ROUTING_FILE = TRAINING_LOG_DIR / "hybrid_routing_stats.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Stateless env for HTTP endpoints (non-WebSocket) ─────────────────────────
_http_env = IncidentResponseEnv()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Incident Response Environment starting up ...")
    yield
    logger.info("Shutting down ...")
    _http_env.close()


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Incident Response OpenEnv",
    description=(
        "A real-world OpenEnv environment for AI-powered production "
        "incident response triage. Agents must identify root causes, "
        "classify severity, and prescribe remediation for cascading "
        "microservice failures modeled on real Alibaba cluster traces."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    dynamic: Optional[bool] = True
    seed: Optional[int] = None


class StepRequest(BaseModel):
    action: Dict[str, Any]


class TrainRequest(BaseModel):
    task: str = "all"
    episodes: int = 100
    curriculum: bool = True


class FaultRequest(BaseModel):
    fault_type: Optional[str] = None
    target: Optional[str] = None


class RemediateRequest(BaseModel):
    incident_id: str
    approved: bool


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed reading JSON file: %s", path)
        return None


_training_running = False
_active_fault: Dict[str, Any] = {
    "active": False,
    "fault_type": None,
    "target": None,
}


# ── HTTP Endpoints (required by OpenEnv validator) ────────────────────────────

@app.get("/health")
async def health():
    """Health check — validator pings this first."""
    return {
        "status":          "healthy",
        "environment":     "incident-response-env",
        "version":         "1.0.0",
        "active_sessions": session_manager.active_count,
        "tasks":           ["easy", "medium", "hard", "expert"],
    }


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset environment to initial state.
    Required by OpenEnv spec + validator (must return HTTP 200).
    """
    try:
        result = handle_reset(_http_env, {
            "task_id": request.task_id,
            "dynamic": request.dynamic,
            "seed":    request.seed,
        })
        return JSONResponse(status_code=200, content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step")
async def step(request: StepRequest):
    """Execute one action step."""
    try:
        result = handle_step(_http_env, {"action": request.action})
        return JSONResponse(status_code=200, content=result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("Step failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state")
async def state():
    """Return current episode state."""
    try:
        result = handle_state(_http_env)
        return JSONResponse(status_code=200, content=result)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("State failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tasks")
async def list_tasks():
    """List available tasks with metadata."""
    return {
        "tasks": [
            {
                "id":           "easy",
                "name":         "Change-Induced Single Service Failure",
                "difficulty":   "easy",
                "max_steps":    10,
                "description":  "Bad ConfigMap causes payments cascade. 3 services, 1 red herring.",
            },
            {
                "id":           "medium",
                "name":         "Test-Induced Hidden Dependency Cascade",
                "difficulty":   "medium",
                "max_steps":    15,
                "description":  "DNS failure cascades 3 services. 2 red herrings. Filter the noise.",
            },
            {
                "id":           "hard",
                "name":         "Process-Induced Cascading Failure with SLA Pressure",
                "difficulty":   "hard",
                "max_steps":    20,
                "description":  "Memory leak + crash-loop across 5 services. SLA breach at step 6.",
            },
        ]
    }


# ── Training / Evaluation endpoints for dashboard integration ────────────────

@app.get("/train/status")
async def train_status():
    """Return latest training summary for dashboard polling."""
    summary = _read_json_file(SUMMARY_FILE)
    if not summary:
        return {
            "running": _training_running,
            "episode": 0,
            "progress_pct": 0.0,
            "per_task": {},
            "status": "no_training_data",
        }
    summary["running"] = bool(summary.get("running", False) or _training_running)
    return summary


@app.post("/train/start")
async def train_start(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start training in a background process."""
    global _training_running
    if _training_running:
        return {"status": "already_running"}

    def _run_training() -> None:
        global _training_running
        _training_running = True
        try:
            cmd = [
                sys.executable,
                str(ROOT / "training" / "train.py"),
                "--task",
                req.task,
                "--episodes",
                str(req.episodes),
            ]
            if req.curriculum:
                cmd.append("--curriculum")
            subprocess.run(cmd, cwd=str(ROOT), check=False)
        except Exception:
            logger.exception("Background training failed")
        finally:
            _training_running = False

    background_tasks.add_task(_run_training)
    return {
        "status": "started",
        "task": req.task,
        "episodes": req.episodes,
        "curriculum": req.curriculum,
    }


@app.post("/train/start-multi-agent")
async def train_start_multi_agent(req: TrainRequest, background_tasks: BackgroundTasks):
    """Alias endpoint used by multi-agent dashboard."""
    return await train_start(req, background_tasks)


@app.get("/train/logs")
async def train_logs(n: int = 200):
    """Return last N training step log lines from latest JSONL file."""
    files = sorted(TRAINING_LOG_DIR.glob("training_*.jsonl"), reverse=True)
    if not files:
        return {"logs": []}

    logs = []
    with files[0].open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                logs.append(json.loads(line))
            except Exception:
                continue

    return {"logs": logs[-n:], "file": files[0].name}


@app.get("/train/reward-curves")
async def train_reward_curves():
    """Return reward curves used by training charts."""
    curves = _read_json_file(CURVES_FILE)
    if not curves:
        return {
            "easy": {"raw": [], "smoothed": [], "rc_scores": [], "episodes": []},
            "medium": {"raw": [], "smoothed": [], "rc_scores": [], "episodes": []},
            "hard": {"raw": [], "smoothed": [], "rc_scores": [], "episodes": []},
            "status": "no_curve_data",
        }
    return curves


@app.get("/evaluation/latest")
async def evaluation_latest():
    """Return the latest evaluation payload, with graceful fallback."""
    summary = _read_json_file(SUMMARY_FILE) or {}
    evaluation = summary.get("evaluation")
    if evaluation:
        return evaluation

    per_task = summary.get("per_task", {})
    fallback_primary = "insufficient_data"
    fallback_strength = 0.0
    for task_data in per_task.values():
        strategies = task_data.get("strategies") or {}
        if strategies:
            primary = max(strategies, key=strategies.get)
            total = sum(strategies.values())
            strength = (strategies.get(primary, 0) / total) if total else 0.0
            fallback_primary = primary
            fallback_strength = round(strength, 3)
            break

    routing_stats = _read_json_file(HYBRID_ROUTING_FILE) or {}
    monitor_signals = int(summary.get("episode", 0))
    fault_injections = int(summary.get("episode", 0))
    evidence_corruptions = int(summary.get("episode", 0) // 2)

    return {
        "strategy_detected": {
            "primary": fallback_primary,
            "strength": fallback_strength,
            "scores": {},
        },
        "multi_agent_dynamics": {
            "monitor_signals": monitor_signals,
            "fault_injections": fault_injections,
            "evidence_corruptions": evidence_corruptions,
            "routing_stats": routing_stats,
        },
        "convergence": {
            "converged": False,
            "convergence_step": None,
            "stable": False,
            "final_reward": 0.0,
        },
        "status": "fallback_from_summary",
    }


@app.get("/curriculum/state")
async def curriculum_state():
    """Return curriculum controller state for dashboard curriculum strip."""
    summary = _read_json_file(SUMMARY_FILE) or {}
    curriculum = summary.get("curriculum")
    if curriculum:
        return curriculum

    progress = float(summary.get("progress_pct", 0.0))
    if progress < 33:
        difficulty = "easy"
    elif progress < 66:
        difficulty = "medium"
    else:
        difficulty = "hard"

    default_state = {
        "difficulty": difficulty,
        "difficulty_index": {"easy": 0, "medium": 1, "hard": 2}[difficulty],
        "noise_multiplier": 1.0,
        "adversary_budget": 0,
        "fault_budget": 1,
        "monitor_reliability": 0.9,
        "monitor_noise": 0.05,
    }

    return {
        "state": default_state,
        "transitions": [],
        "intra_progress": round(progress / 100.0, 3),
        "status": "fallback_from_summary",
    }


@app.post("/fault/inject")
async def fault_inject(req: FaultRequest):
    """Mark a synthetic fault active for dashboard simulation controls."""
    _active_fault["active"] = True
    _active_fault["fault_type"] = req.fault_type or "unknown"
    _active_fault["target"] = req.target or "unknown"
    return {
        "status": "injected",
        "fault": {
            "fault_type": _active_fault["fault_type"],
            "target": _active_fault["target"],
        },
    }


@app.post("/fault/clear")
async def fault_clear():
    """Clear active synthetic fault state."""
    _active_fault["active"] = False
    _active_fault["fault_type"] = None
    _active_fault["target"] = None
    return {"status": "cleared"}


@app.post("/remediate/approve")
async def remediate_approve(req: RemediateRequest):
    """Acknowledge remediation approval/rejection from dashboard actions."""
    return {
        "status": "approved" if req.approved else "rejected",
        "incident_id": req.incident_id,
    }


# ── Email Trigger + Incident Pipeline ─────────────────────────────────────────

class TriggerIncidentRequest(BaseModel):
    source: str = "email"
    subject: str = ""
    body: str = ""


@app.post("/trigger-incident")
async def trigger_incident(req: TriggerIncidentRequest, background_tasks: BackgroundTasks):
    """
    Trigger incident analysis from email listener or manual API call.
    Circuit breaker limits to max 2 analysis attempts.
    """
    if not req.subject:
        raise HTTPException(status_code=400, detail="Subject is required")

    async def _run():
        try:
            await incident_pipeline.trigger_from_email(
                subject=req.subject,
                body=req.body,
                source=req.source,
            )
        except Exception as exc:
            logger.exception(f"Incident pipeline failed: {exc}")
            incident_pipeline.log_queue.log(
                f"Pipeline error: {str(exc)[:200]}",
                level="error",
                source="pipeline",
            )

    background_tasks.add_task(_run)

    return {
        "status": "triggered",
        "source": req.source,
        "subject": req.subject[:100],
        "message": "Incident pipeline started. Monitor /incident-logs for real-time updates.",
    }


@app.get("/incident-logs")
async def get_incident_logs(n: int = 100, after_id: int = 0):
    """
    Poll-based log endpoint — returns latest N logs.
    Use after_id for incremental polling (only new logs since last poll).
    """
    logs = incident_pipeline.get_logs(n=n, after_id=after_id)
    return {
        "logs": logs,
        "count": len(logs),
        "latest_id": logs[-1]["id"] if logs else 0,
    }


@app.get("/incident-logs/stream")
async def incident_log_stream(request: Request):
    """
    SSE (Server-Sent Events) endpoint for real-time log streaming to UI.
    Frontend connects: const es = new EventSource('/incident-logs/stream')
    """
    queue = incident_pipeline.log_queue.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(entry)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            incident_pipeline.log_queue.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/incidents/tracking")
async def incidents_tracking(n: int = 20):
    """
    Return tracked incidents with full model interaction history.
    Shows which models participated, debate rounds, and resolution status.
    """
    return {
        "incidents": incident_pipeline.get_incidents(n),
        "stats": incident_pipeline.get_stats(),
        "circuit_breaker": circuit_breaker.get_all_circuits(),
    }


@app.get("/incidents/stats")
async def incidents_stats():
    """Quick stats for dashboard KPI strip."""
    return incident_pipeline.get_stats()


# ── WebSocket Endpoint (persistent sessions) ──────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for persistent environment sessions.
    Each connection gets an isolated IncidentResponseEnv instance.

    Protocol:
        Client sends: {"type": "reset", "task_id": "easy"}
        Client sends: {"type": "step",  "action": {...}}
        Client sends: {"type": "state"}
        Server sends: JSON result for each message
    """
    session_id = str(uuid.uuid4())[:8]
    await websocket.accept()
    logger.info(f"WebSocket connected: {session_id}")

    env = await session_manager.create(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                result = handle_reset(env, msg)
                await websocket.send_json(result)

            elif msg_type == "step":
                try:
                    result = handle_step(env, msg)
                    await websocket.send_json(result)
                except (ValueError, RuntimeError) as exc:
                    await websocket.send_json({"error": str(exc)})

            elif msg_type == "state":
                try:
                    result = handle_state(env)
                    await websocket.send_json(result)
                except RuntimeError as exc:
                    await websocket.send_json({"error": str(exc)})

            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as exc:
        logger.exception(f"WebSocket error [{session_id}]: {exc}")
    finally:
        await session_manager.remove(session_id)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host    = os.getenv("HOST", "0.0.0.0"),
        port    = int(os.getenv("PORT", "7860")),
        workers = int(os.getenv("WORKERS", "2")),
        log_level = os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()