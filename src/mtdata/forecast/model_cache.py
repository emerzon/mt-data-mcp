"""Shared pretrained-model runtime cache.

Provides a thread-safe, TTL-aware cache for heavy model objects so that
repeated forecast requests reuse already-loaded models instead of paying
cold-start costs each time.

Usage::

    from mtdata.forecast.model_cache import model_cache

    model = model_cache.get_or_load(
        key="chronos-t5-small/cpu",
        loader=lambda: load_chronos_model("small", device="cpu"),
    )
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 1800.0  # 30 minutes
_DEFAULT_MAX_ENTRIES = 8


class _CacheEntry:
    __slots__ = ("key", "model", "created_at", "last_used", "hit_count")

    def __init__(self, key: str, model: Any) -> None:
        self.key = key
        self.model = model
        now = time.monotonic()
        self.created_at = now
        self.last_used = now
        self.hit_count = 0


class ModelCache:
    """Thread-safe model cache with TTL expiry and size limits."""

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        # Per-key init locks prevent duplicate concurrent loads
        self._init_locks: Dict[str, threading.Lock] = {}
        self._init_locks_guard = threading.Lock()

    def get_or_load(
        self,
        key: str,
        loader: Callable[[], Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Return cached model or load via *loader*.

        Returns ``(model, metadata)`` where metadata reports cache
        hit/miss and timing.
        """
        # Fast path: cache hit
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if (time.monotonic() - entry.created_at) <= self._ttl:
                    entry.last_used = time.monotonic()
                    entry.hit_count += 1
                    return entry.model, {
                        "cache": "hit",
                        "key": key,
                        "hit_count": entry.hit_count,
                    }
                else:
                    del self._entries[key]

        # Slow path: acquire per-key init lock to avoid duplicate loads
        init_lock = self._get_init_lock(key)
        with init_lock:
            # Double-check after acquiring lock
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and (time.monotonic() - entry.created_at) <= self._ttl:
                    entry.last_used = time.monotonic()
                    entry.hit_count += 1
                    return entry.model, {
                        "cache": "hit",
                        "key": key,
                        "hit_count": entry.hit_count,
                    }

            t0 = time.monotonic()
            model = loader()
            load_time = time.monotonic() - t0

            with self._lock:
                self._evict_if_full()
                self._entries[key] = _CacheEntry(key, model)

            logger.debug("Model cache MISS for %s (loaded in %.2fs)", key, load_time)
            return model, {
                "cache": "miss",
                "key": key,
                "load_time_seconds": round(load_time, 3),
            }

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry. Returns True if found."""
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def keys(self) -> list[str]:
        """Return current cache keys."""
        with self._lock:
            return list(self._entries.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _get_init_lock(self, key: str) -> threading.Lock:
        with self._init_locks_guard:
            if key not in self._init_locks:
                self._init_locks[key] = threading.Lock()
            return self._init_locks[key]

    def _evict_if_full(self) -> None:
        """Evict LRU entry when at capacity.  Caller must hold ``_lock``."""
        while len(self._entries) >= self._max:
            lru_key = min(self._entries, key=lambda k: self._entries[k].last_used)
            del self._entries[lru_key]


# Module-level singleton
model_cache = ModelCache()
