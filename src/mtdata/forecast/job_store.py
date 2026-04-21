"""Durable SQLite-backed registry for forecast training tasks."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

_DEFAULT_DB_PATH = os.path.join(
    os.path.expanduser("~"),
    ".mtdata",
    "forecast",
    "jobs.sqlite",
)


def _env_db_path() -> str:
    return os.environ.get("MTDATA_FORECAST_JOBS_DB", _DEFAULT_DB_PATH)


@dataclass(frozen=True)
class JobRecord:
    task_id: str
    method: str
    data_scope: str
    params_hash: str
    status: JobStatus
    created_at: float
    progress_payload: Optional[Dict[str, Any]] = None
    result_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    heartbeat_at: Optional[float] = None
    cancel_requested: bool = False
    model_id: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class JobStore:
    """Persist training task state across process restarts."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path) if path else Path(_env_db_path())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    task_id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    data_scope TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    pid INTEGER,
                    heartbeat_at REAL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    model_id TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_active
                ON jobs(method, data_scope, params_hash)
                WHERE status IN ('pending', 'running')
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_jobs_status_created_at
                ON jobs(status, created_at DESC)
                """
            )
            conn.commit()

    @staticmethod
    def _encode(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if payload is None:
            return None
        return json.dumps(payload, sort_keys=True, default=str)

    @staticmethod
    def _decode(payload: Optional[str]) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        data = json.loads(payload)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            task_id=str(row["task_id"]),
            method=str(row["method"]),
            data_scope=str(row["data_scope"]),
            params_hash=str(row["params_hash"]),
            status=str(row["status"]),
            progress_payload=JobStore._decode(row["progress_json"]),
            result_payload=JobStore._decode(row["result_json"]),
            error=row["error"],
            pid=row["pid"],
            heartbeat_at=row["heartbeat_at"],
            cancel_requested=bool(row["cancel_requested"]),
            model_id=row["model_id"],
            created_at=float(row["created_at"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    def upsert(self, record: JobRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    task_id, method, data_scope, params_hash, status,
                    progress_json, result_json, error, pid, heartbeat_at,
                    cancel_requested, model_id, created_at, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    method = excluded.method,
                    data_scope = excluded.data_scope,
                    params_hash = excluded.params_hash,
                    status = excluded.status,
                    progress_json = excluded.progress_json,
                    result_json = excluded.result_json,
                    error = excluded.error,
                    pid = excluded.pid,
                    heartbeat_at = excluded.heartbeat_at,
                    cancel_requested = excluded.cancel_requested,
                    model_id = excluded.model_id,
                    created_at = excluded.created_at,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at
                """,
                (
                    record.task_id,
                    record.method,
                    record.data_scope,
                    record.params_hash,
                    record.status,
                    self._encode(record.progress_payload),
                    self._encode(record.result_payload),
                    record.error,
                    record.pid,
                    record.heartbeat_at,
                    int(record.cancel_requested),
                    record.model_id,
                    record.created_at,
                    record.started_at,
                    record.completed_at,
                ),
            )
            conn.commit()

    def get(self, task_id: str) -> Optional[JobRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_jobs(self, *, status: Optional[str] = None) -> List[JobRecord]:
        query = "SELECT * FROM jobs"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_active(self, method: str, data_scope: str, params_hash: str) -> Optional[JobRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE method = ? AND data_scope = ? AND params_hash = ?
                  AND status IN ('pending', 'running')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (method, data_scope, params_hash),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def delete(self, task_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM jobs WHERE task_id = ?",
                (task_id,),
            )
            conn.commit()
        return cur.rowcount > 0

    def cleanup_completed(self, max_age_seconds: float) -> List[str]:
        cutoff = time.time() - max_age_seconds
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id FROM jobs
                WHERE status IN ('completed', 'failed', 'cancelled')
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (cutoff,),
            ).fetchall()
            task_ids = [str(row["task_id"]) for row in rows]
            if task_ids:
                conn.executemany(
                    "DELETE FROM jobs WHERE task_id = ?",
                    [(task_id,) for task_id in task_ids],
                )
                conn.commit()
        return task_ids

    def mark_active_jobs_failed(self, error: str) -> int:
        now = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    error = ?,
                    heartbeat_at = COALESCE(heartbeat_at, ?),
                    completed_at = COALESCE(completed_at, ?)
                WHERE status IN ('pending', 'running')
                """,
                (error, now, now),
            )
            conn.commit()
        return cur.rowcount
