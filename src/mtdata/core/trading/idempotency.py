"""Order idempotency stores for live trading operations."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_TTL_SECONDS = 86_400.0  # 24 hours
_DEFAULT_DATABASE_PATH = Path.home() / ".mtdata" / "trade_idempotency.sqlite3"


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
        self._gc_interval = min(max(float(ttl_seconds) / 4.0, 0.05), 30.0)
        self._last_gc_at = time.monotonic()
        self.scope = "process_memory"
        self.durable = False

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
            wait_timeout = None
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
                # Never expire a live reservation: doing so can admit a second
                # broker submission while the first call is still running.
                wait_timeout = 1.0
            wait_event.wait(timeout=wait_timeout)

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
        should_gc = False
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _IdempotencyEntry(key, ready=False)
                self._entries[key] = entry
            entry.outcome = outcome
            if request_signature is not None or entry.request_signature is None:
                entry.request_signature = request_signature
            entry.created_at = now
            entry.ready_event.set()
            should_gc = len(self._entries) > 1 and (now - self._last_gc_at) >= self._gc_interval
        if should_gc:
            self._gc(now=now)

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

    def _is_expired(self, entry: _IdempotencyEntry, *, now: Optional[float] = None) -> bool:
        observed_now = time.monotonic() if now is None else now
        return entry.ready_event.is_set() and (observed_now - entry.created_at) > self._ttl

    def _gc(self, *, now: Optional[float] = None) -> None:
        """Remove expired entries."""
        observed_now = time.monotonic() if now is None else now
        with self._lock:
            expired = [
                k for k, v in self._entries.items()
                if self._is_expired(v, now=observed_now)
            ]
            for k in expired:
                del self._entries[k]
            self._last_gc_at = observed_now

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
            self._last_gc_at = time.monotonic()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class SQLiteIdempotencyStore:
    """Process-safe durable idempotency store backed by SQLite.

    In-progress reservations are deliberately fail-closed after a process
    crash. Automatically recycling one could submit a second live trade when
    the first broker response was lost.
    """

    scope = "sqlite"
    durable = True

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._path = Path(database_path).expanduser().resolve()
        self._ttl = float(ttl_seconds)
        self._owner = uuid.uuid4().hex
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_idempotency (
                    key TEXT PRIMARY KEY,
                    request_signature TEXT,
                    outcome_json TEXT,
                    status TEXT NOT NULL,
                    owner TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(trade_idempotency)")
            }
            if "created_at" not in columns:
                connection.execute(
                    "ALTER TABLE trade_idempotency ADD COLUMN created_at REAL"
                )
                connection.execute(
                    "UPDATE trade_idempotency SET created_at = updated_at"
                )
            connection.execute(
                """CREATE INDEX IF NOT EXISTS trade_idempotency_expiry
                   ON trade_idempotency(status, updated_at)"""
            )

    @staticmethod
    def _duplicate_payload(row: sqlite3.Row) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "duplicate": True,
            "idempotency_key": row["key"],
            "request_signature": row["request_signature"],
        }
        if row["status"] == "complete":
            payload["original_outcome"] = json.loads(row["outcome_json"])
        else:
            payload["in_progress"] = True
        return payload

    def check(self, key: Optional[str]) -> Optional[Dict[str, Any]]:
        if key is None:
            return None
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM trade_idempotency
                   WHERE status = 'complete' AND updated_at < ?""",
                (now - self._ttl,),
            )
            row = connection.execute(
                "SELECT * FROM trade_idempotency WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return self._duplicate_payload(row)

    def reserve(
        self,
        key: Optional[str],
        *,
        request_signature: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if key is None:
            return None
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM trade_idempotency
                   WHERE status = 'complete' AND updated_at < ?""",
                (now - self._ttl,),
            )
            row = connection.execute(
                "SELECT * FROM trade_idempotency WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                connection.execute(
                    """INSERT INTO trade_idempotency
                       (key, request_signature, outcome_json, status, owner,
                        created_at, updated_at)
                       VALUES (?, ?, NULL, 'in_progress', ?, ?, ?)""",
                    (key, request_signature, self._owner, now, now),
                )
                return None
            return self._duplicate_payload(row)

    def record(
        self,
        key: Optional[str],
        outcome: Dict[str, Any],
        *,
        request_signature: Optional[str] = None,
    ) -> None:
        if key is None:
            return
        encoded = json.dumps(outcome, sort_keys=True, separators=(",", ":"), default=str)
        now = time.time()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, owner FROM trade_idempotency WHERE key = ?", (key,)
            ).fetchone()
            if row is not None and row["status"] == "in_progress" and row["owner"] != self._owner:
                return
            connection.execute(
                """INSERT INTO trade_idempotency
                   (key, request_signature, outcome_json, status, owner,
                    created_at, updated_at)
                   VALUES (?, ?, ?, 'complete', NULL, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       request_signature = excluded.request_signature,
                       outcome_json = excluded.outcome_json,
                       status = 'complete', owner = NULL,
                       updated_at = excluded.updated_at""",
                (key, request_signature, encoded, now, now),
            )

    def release(self, key: Optional[str], *, request_signature: Optional[str] = None) -> None:
        if key is None:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """DELETE FROM trade_idempotency
                   WHERE key = ? AND status = 'in_progress' AND owner = ?
                     AND (request_signature = ? OR request_signature IS NULL OR ? IS NULL)""",
                (key, self._owner, request_signature, request_signature),
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM trade_idempotency")

    def __len__(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM trade_idempotency").fetchone()
            return int(row[0])


def create_default_idempotency_store() -> SQLiteIdempotencyStore:
    """Create the durable store configured for all trading transports."""
    path = os.getenv("MTDATA_TRADE_IDEMPOTENCY_DB") or str(_DEFAULT_DATABASE_PATH)
    raw_ttl = os.getenv("MTDATA_TRADE_IDEMPOTENCY_TTL_SECONDS", "")
    try:
        ttl_seconds = float(raw_ttl) if raw_ttl else _DEFAULT_TTL_SECONDS
    except ValueError:
        ttl_seconds = _DEFAULT_TTL_SECONDS
    return SQLiteIdempotencyStore(path, ttl_seconds=max(ttl_seconds, 1.0))
