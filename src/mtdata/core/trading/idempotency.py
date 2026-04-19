"""Optional order idempotency layer.

Provides a lightweight in-memory dedupe store that trading paths can
use to prevent duplicate order submissions.  Keyed by a user-supplied
idempotency key; entries expire after a configurable TTL.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

_DEFAULT_TTL_SECONDS = 300.0  # 5 minutes


class _IdempotencyEntry:
    __slots__ = ("key", "outcome", "request_signature", "created_at", "ready_event")

    def __init__(
        self,
        key: str,
        outcome: Optional[Dict[str, Any]] = None,
        request_signature: Optional[str] = None,
        *,
        ready: bool = True,
    ) -> None:
        self.key = key
        self.outcome = outcome
        self.request_signature = request_signature
        self.created_at = time.monotonic()
        self.ready_event = threading.Event()
        if ready:
            self.ready_event.set()


def _entry_duplicate_payload(entry: _IdempotencyEntry) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "duplicate": True,
        "idempotency_key": entry.key,
        "request_signature": entry.request_signature,
    }
    if entry.ready_event.is_set():
        payload["original_outcome"] = entry.outcome
    else:
        payload["in_progress"] = True
    return payload


class IdempotencyStore:
    """Thread-safe in-memory dedupe store with TTL expiry."""

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._entries: Dict[str, _IdempotencyEntry] = {}
        self._lock = threading.Lock()

    def check(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return prior outcome if *key* exists and is not expired; else ``None``."""
        if key is None:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._entries[key]
                return None
            return _entry_duplicate_payload(entry)

    def reserve(
        self,
        key: Optional[str],
        *,
        request_signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Atomically reserve *key* or return the stored duplicate outcome."""
        if key is None:
            return None
        while True:
            with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    self._entries[key] = _IdempotencyEntry(
                        key,
                        request_signature=request_signature,
                        ready=False,
                    )
                    return None
                if self._is_expired(entry):
                    del self._entries[key]
                    self._entries[key] = _IdempotencyEntry(
                        key,
                        request_signature=request_signature,
                        ready=False,
                    )
                    return None
                stored_signature = entry.request_signature
                if (
                    stored_signature is not None
                    and request_signature is not None
                    and stored_signature != request_signature
                ):
                    return _entry_duplicate_payload(entry)
                if entry.ready_event.is_set():
                    return _entry_duplicate_payload(entry)
                wait_event = entry.ready_event
            wait_event.wait()

    def record(
        self,
        key: Optional[str],
        outcome: Dict[str, Any],
        *,
        request_signature: Optional[str] = None,
    ) -> None:
        """Record *outcome* for *key*.  No-op when key is ``None``."""
        if key is None:
            return
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _IdempotencyEntry(key, ready=False)
                self._entries[key] = entry
            entry.outcome = outcome
            if request_signature is not None or entry.request_signature is None:
                entry.request_signature = request_signature
            entry.created_at = time.monotonic()
            entry.ready_event.set()
            self._gc()

    def release(self, key: Optional[str], *, request_signature: Optional[str] = None) -> None:
        """Drop an in-progress reservation so another caller can retry it."""
        if key is None:
            return
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.ready_event.is_set():
                return
            if (
                request_signature is not None
                and entry.request_signature is not None
                and entry.request_signature != request_signature
            ):
                return
            del self._entries[key]
            entry.ready_event.set()

    def _is_expired(self, entry: _IdempotencyEntry) -> bool:
        if not entry.ready_event.is_set():
            return False
        return (time.monotonic() - entry.created_at) > self._ttl

    def _gc(self) -> None:
        """Remove expired entries.  Caller must hold ``_lock``."""
        expired = [
            k for k, v in self._entries.items()
            if self._is_expired(v)
        ]
        for k in expired:
            del self._entries[k]

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
