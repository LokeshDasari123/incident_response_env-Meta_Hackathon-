"""
server/session_manager.py
--------------------------
Manages per-WebSocket environment instances.
One session = one isolated IncidentResponseEnv.
"""

import asyncio
import logging
from typing import Dict, Optional
from envs.incident_env import IncidentResponseEnv

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Thread-safe registry of active environment sessions.
    Each WebSocket connection gets its own IncidentResponseEnv instance.
    """

    def __init__(self, max_sessions: int = 50) -> None:
        self._sessions:    Dict[str, IncidentResponseEnv] = {}
        self._lock:        asyncio.Lock = asyncio.Lock()
        self._max_sessions = max_sessions

    async def create(self, session_id: str) -> IncidentResponseEnv:
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(
                    f"Max concurrent sessions ({self._max_sessions}) reached."
                )
            env = IncidentResponseEnv()
            self._sessions[session_id] = env
            logger.info(f"Session created: {session_id} (total: {len(self._sessions)})")
            return env

    async def get(self, session_id: str) -> Optional[IncidentResponseEnv]:
        return self._sessions.get(session_id)

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            env = self._sessions.pop(session_id, None)
            if env:
                env.close()
                logger.info(f"Session removed: {session_id} (total: {len(self._sessions)})")

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# Global singleton
session_manager = SessionManager()