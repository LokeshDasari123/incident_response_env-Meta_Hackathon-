"""
server/app.py
-------------
FastAPI application for the Incident Response OpenEnv environment.
Exposes all required OpenEnv endpoints.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from envs.incident_env        import IncidentResponseEnv
from server.environment       import handle_reset, handle_step, handle_state
from server.session_manager   import session_manager

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


# ── HTTP Endpoints (required by OpenEnv validator) ────────────────────────────

@app.get("/health")
async def health():
    """Health check — validator pings this first."""
    return {
        "status":          "healthy",
        "environment":     "incident-response-env",
        "version":         "1.0.0",
        "active_sessions": session_manager.active_count,
        "tasks":           ["easy", "medium", "hard"],
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