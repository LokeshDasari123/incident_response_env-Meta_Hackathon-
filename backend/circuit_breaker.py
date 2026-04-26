"""
backend/circuit_breaker.py
==========================
Circuit breaker for API calls to prevent infinite retry loops.

Rules:
  - Max 2 call attempts per incident
  - Exponential backoff between retries (1s, 2s)
  - If both fail, mark incident as FAILED and stop
  - Cool-down period prevents re-triggering same incident within 60s
"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal — calls go through
    OPEN = "open"           # Tripped — calls blocked
    HALF_OPEN = "half_open" # Testing — one call allowed


@dataclass
class CallAttempt:
    attempt: int
    timestamp: float
    success: bool
    error: Optional[str] = None
    result: Optional[Dict] = None
    duration_ms: float = 0.0


@dataclass
class IncidentCircuit:
    """Circuit breaker state for a single incident."""
    incident_id: str
    max_attempts: int = 2
    state: CircuitState = CircuitState.CLOSED
    attempts: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    final_result: Optional[Dict] = None
    winning_model: Optional[str] = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def can_retry(self) -> bool:
        return self.attempt_count < self.max_attempts and self.state != CircuitState.OPEN

    @property
    def backoff_seconds(self) -> float:
        """Exponential backoff: 1s, 2s"""
        if self.attempt_count == 0:
            return 0.0
        return min(2 ** self.attempt_count, 4.0)


class CircuitBreaker:
    """
    Global circuit breaker managing all incident analysis calls.

    Usage:
        breaker = CircuitBreaker(max_attempts=2)
        result = await breaker.call(incident_id, analyze_fn, **kwargs)
    """

    def __init__(self, max_attempts: int = 2, cooldown_seconds: float = 60.0):
        self.max_attempts = max_attempts
        self.cooldown_seconds = cooldown_seconds
        self._circuits: Dict[str, IncidentCircuit] = {}
        self._log: list = []

    def _get_circuit(self, incident_id: str) -> IncidentCircuit:
        if incident_id not in self._circuits:
            self._circuits[incident_id] = IncidentCircuit(
                incident_id=incident_id,
                max_attempts=self.max_attempts,
            )
        return self._circuits[incident_id]

    def _is_duplicate(self, subject: str) -> bool:
        """Check if this subject was already triggered recently."""
        now = time.time()
        for circuit in self._circuits.values():
            if now - circuit.created_at < self.cooldown_seconds:
                # Same incident within cooldown
                if any(
                    a.result and a.result.get("source_subject") == subject
                    for a in circuit.attempts
                ):
                    return True
        return False

    async def call(
        self,
        incident_id: str,
        fn: Callable,
        *args,
        subject: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute fn with circuit breaker protection.
        Max 2 attempts, exponential backoff between retries.
        """
        # Dedup check
        if subject and self._is_duplicate(subject):
            log_entry = {
                "incident_id": incident_id,
                "status": "duplicate_blocked",
                "message": f"Duplicate incident blocked (cooldown {self.cooldown_seconds}s)",
                "timestamp": time.time(),
            }
            self._log.append(log_entry)
            logger.info(f"[CIRCUIT] Duplicate blocked: {subject[:50]}")
            return {"status": "duplicate_blocked", "incident_id": incident_id}

        circuit = self._get_circuit(incident_id)

        while circuit.can_retry:
            attempt_num = circuit.attempt_count + 1
            backoff = circuit.backoff_seconds

            if backoff > 0:
                log_entry = {
                    "incident_id": incident_id,
                    "status": "backoff",
                    "attempt": attempt_num,
                    "backoff_seconds": backoff,
                    "message": f"Waiting {backoff:.1f}s before retry #{attempt_num}",
                    "timestamp": time.time(),
                }
                self._log.append(log_entry)
                logger.info(f"[CIRCUIT] Backoff {backoff:.1f}s for {incident_id}")
                await asyncio.sleep(backoff)

            start = time.time()
            try:
                result = await fn(*args, **kwargs) if asyncio.iscoroutinefunction(fn) else fn(*args, **kwargs)

                attempt = CallAttempt(
                    attempt=attempt_num,
                    timestamp=start,
                    success=True,
                    result={**result, "source_subject": subject} if isinstance(result, dict) else {"raw": result, "source_subject": subject},
                    duration_ms=round((time.time() - start) * 1000, 1),
                )
                circuit.attempts.append(attempt)

                # Check if resolution was successful
                confidence = 0.0
                if isinstance(result, dict):
                    confidence = result.get("confidence", 0.0)

                if confidence >= 0.5:
                    circuit.resolved = True
                    circuit.final_result = result
                    circuit.state = CircuitState.CLOSED
                    circuit.winning_model = result.get("model_used", f"attempt_{attempt_num}")

                    log_entry = {
                        "incident_id": incident_id,
                        "status": "resolved",
                        "attempt": attempt_num,
                        "confidence": confidence,
                        "winning_model": circuit.winning_model,
                        "message": f"✅ Resolved on attempt #{attempt_num} (confidence: {confidence:.0%})",
                        "duration_ms": attempt.duration_ms,
                        "timestamp": time.time(),
                    }
                    self._log.append(log_entry)
                    return {"status": "resolved", "result": result, "attempts": attempt_num}

                # Low confidence — will retry
                log_entry = {
                    "incident_id": incident_id,
                    "status": "low_confidence",
                    "attempt": attempt_num,
                    "confidence": confidence,
                    "message": f"⚠️ Attempt #{attempt_num}: confidence {confidence:.0%} < 50% threshold",
                    "duration_ms": attempt.duration_ms,
                    "timestamp": time.time(),
                }
                self._log.append(log_entry)

            except Exception as exc:
                attempt = CallAttempt(
                    attempt=attempt_num,
                    timestamp=start,
                    success=False,
                    error=str(exc),
                    duration_ms=round((time.time() - start) * 1000, 1),
                )
                circuit.attempts.append(attempt)

                log_entry = {
                    "incident_id": incident_id,
                    "status": "error",
                    "attempt": attempt_num,
                    "error": str(exc)[:200],
                    "message": f"❌ Attempt #{attempt_num} failed: {str(exc)[:100]}",
                    "duration_ms": attempt.duration_ms,
                    "timestamp": time.time(),
                }
                self._log.append(log_entry)
                logger.exception(f"[CIRCUIT] Attempt {attempt_num} failed for {incident_id}")

        # All attempts exhausted
        circuit.state = CircuitState.OPEN

        log_entry = {
            "incident_id": incident_id,
            "status": "failed",
            "message": f"🔴 Circuit OPEN — all {self.max_attempts} attempts exhausted",
            "attempts": circuit.attempt_count,
            "timestamp": time.time(),
        }
        self._log.append(log_entry)
        logger.warning(f"[CIRCUIT] OPEN for {incident_id} — {circuit.attempt_count} attempts failed")

        return {
            "status": "circuit_open",
            "incident_id": incident_id,
            "attempts": circuit.attempt_count,
            "last_error": circuit.attempts[-1].error if circuit.attempts else None,
        }

    def get_logs(self, n: int = 50) -> list:
        return self._log[-n:]

    def get_circuit(self, incident_id: str) -> Optional[Dict]:
        circuit = self._circuits.get(incident_id)
        if not circuit:
            return None
        return {
            "incident_id": circuit.incident_id,
            "state": circuit.state.value,
            "attempts": circuit.attempt_count,
            "max_attempts": circuit.max_attempts,
            "resolved": circuit.resolved,
            "winning_model": circuit.winning_model,
            "created_at": circuit.created_at,
        }

    def get_all_circuits(self) -> list:
        return [self.get_circuit(cid) for cid in self._circuits]

    def reset(self, incident_id: str):
        self._circuits.pop(incident_id, None)


# Global singleton
circuit_breaker = CircuitBreaker(max_attempts=2, cooldown_seconds=60.0)
