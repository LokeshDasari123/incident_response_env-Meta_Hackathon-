"""
client/http_client.py
---------------------
HTTP client for the Incident Response Environment.
Used by inference.py for stateless episode runs.
"""

import os
from typing import Any, Dict, Optional
import httpx

BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")


class IncidentEnvClient:
    """Synchronous HTTP client wrapping the FastAPI server."""

    def __init__(self, base_url: str = BASE_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client  = httpx.Client(timeout=timeout)

    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        resp = self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._client.post(
            f"{self.base_url}/step",
            json={"action": action},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()