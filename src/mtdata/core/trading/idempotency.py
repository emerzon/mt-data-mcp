"""Optional order idempotency layer.

Provides a lightweight in-memory dedupe store that trading paths can
use to prevent duplicate order submissions.  Keyed by a user-supplied
idempotency key; entries expire after a configurable TTL.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple


_DEFAULT_TTL_SECONDS = 300.0  # 5 minutes


class _IdempotencyEntry:
    __slots__ = ("key", "outcome", "created_at")

    def __init__(self, key: str, outcome: Dict[str, Any]) -> None:
        self.key = key
        self.outcome = outcome
        self.created_at = time.monotonic()


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
            if (time.monotonic() - entry.created_at) > self._ttl:
                del self._entries[key]
                return None
            return {
                "duplicate": True,
                "idempotency_key": key,
                "original_outcome": entry.outcome,
            }

    def record(self, key: Optional[str], outcome: Dict[str, Any]) -> None:
        """Record *outcome* for *key*.  No-op when key is ``None``."""
        if key is None:
            return
        with self._lock:
            self._entries[key] = _IdempotencyEntry(key, outcome)
            self._gc()

    def _gc(self) -> None:
        """Remove expired entries.  Caller must hold ``_lock``."""
        now = time.monotonic()
        expired = [
            k for k, v in self._entries.items()
            if (now - v.created_at) > self._ttl
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
