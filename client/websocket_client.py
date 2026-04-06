"""
client/websocket_client.py
--------------------------
Async WebSocket client for persistent sessions.
"""

import asyncio
import json
import os
from typing import Any, Dict
import websockets

BASE_URL = os.getenv("ENV_BASE_URL", "ws://localhost:7860")


class IncidentEnvWSClient:
    """Async WebSocket client — one persistent connection per episode."""

    def __init__(self, base_url: str = BASE_URL):
        self.ws_url = base_url.replace("http", "ws").rstrip("/") + "/ws"
        self._ws    = None

    async def connect(self):
        self._ws = await websockets.connect(self.ws_url)

    async def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        await self._ws.send(json.dumps({"type": "reset", "task_id": task_id}))
        return json.loads(await self._ws.recv())

    async def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        await self._ws.send(json.dumps({"type": "step", "action": action}))
        return json.loads(await self._ws.recv())

    async def state(self) -> Dict[str, Any]:
        await self._ws.send(json.dumps({"type": "state"}))
        return json.loads(await self._ws.recv())

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()