"""Background training task manager for forecast methods.

Manages asynchronous training jobs via a bounded thread pool with
progress reporting, model persistence, and task lifecycle management.

Design:
- In-memory task registry (not persisted across restarts)
- Disk-based model persistence via ModelStore (survives restarts)
- GPU-aware scheduling: heavy methods limited to 1 concurrent via semaphore
- Deduplication: identical (method, data_scope, params_hash) shares one task
- Thread-safe: all mutations guarded by lock, status reads return snapshots
"""

from __future__ import annotations

import copy
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .interface import ForecastMethod, TrainedModelHandle, TrainingProgress, TrainResult
from .model_store import ModelStore
from .registry import ForecastRegistry

logger = logging.getLogger(__name__)

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

_MAX_WORKERS_DEFAULT = 4
_HEAVY_SEMAPHORE_DEFAULT = 1
_TASK_TTL_DEFAULT = 3600.0  # 1 hour for completed tasks

# Categories that count as "heavy" for GPU semaphore purposes
_HEAVY_CATEGORIES = frozenset({"heavy"})


@dataclass
class TrainingTask:
    """Snapshot of a background training job's state."""

    task_id: str
    method: str
    data_scope: str
    params_hash: str
    status: TaskStatus = "pending"
    progress: Optional[TrainingProgress] = None
    result: Optional[TrainedModelHandle] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


def _snapshot(task: TrainingTask) -> TrainingTask:
    """Return a shallow copy safe to hand to callers."""
    return copy.copy(task)


class TaskManager:
    """Manages background training tasks with bounded concurrency.

    Thread-safe.  Heavy methods (GPU neural nets) are serialised
    through a semaphore so only one trains at a time.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        heavy_limit: int | None = None,
        store: ModelStore | None = None,
    ) -> None:
        workers = max_workers or int(os.environ.get("MTDATA_TRAIN_WORKERS", _MAX_WORKERS_DEFAULT))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="train")
        self._heavy_sem = threading.Semaphore(
            heavy_limit or int(os.environ.get("MTDATA_HEAVY_LIMIT", _HEAVY_SEMAPHORE_DEFAULT))
        )

        self._tasks: Dict[str, TrainingTask] = {}
        self._futures: Dict[str, Future] = {}
        # Dedupe index: (method, data_scope, params_hash) → task_id
        self._active_keys: Dict[tuple, str] = {}
        self._lock = threading.Lock()
        self._store = store  # resolved lazily if None
        self._shutdown = False

    @property
    def store(self) -> ModelStore:
        if self._store is None:
            from .model_store import model_store as _default
            self._store = _default
        return self._store

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        method_name: str,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: dict,
        data_scope: str,
        params_hash: str,
        *,
        exog: Optional[np.ndarray] = None,
        timeframe: str = "",
        **kwargs: Any,
    ) -> tuple[str, bool]:
        """Submit a training task.

        Returns ``(task_id, is_new)`` where *is_new* is False when an
        identical in-flight task already exists (dedup hit).
        """
        if self._shutdown:
            raise RuntimeError("TaskManager is shut down")

        dedup_key = (method_name, data_scope, params_hash)

        with self._lock:
            existing_tid = self._active_keys.get(dedup_key)
            if existing_tid is not None:
                existing = self._tasks.get(existing_tid)
                if existing and existing.status in ("pending", "running"):
                    return existing_tid, False

            task_id = uuid.uuid4().hex[:12]
            task = TrainingTask(
                task_id=task_id,
                method=method_name,
                data_scope=data_scope,
                params_hash=params_hash,
            )
            self._tasks[task_id] = task
            self._active_keys[dedup_key] = task_id

        # Determine if heavy
        is_heavy = False
        try:
            info = ForecastRegistry.get_method_info(method_name)
            is_heavy = info.get("training_category") in _HEAVY_CATEGORIES
        except Exception:
            pass

        def _run() -> None:
            acquired_heavy = False
            try:
                if is_heavy:
                    self._heavy_sem.acquire()
                    acquired_heavy = True

                with self._lock:
                    task.status = "running"
                    task.started_at = time.time()

                method_obj = ForecastRegistry.get(method_name)

                def _on_progress(p: TrainingProgress) -> None:
                    with self._lock:
                        task.progress = p

                result: TrainResult = method_obj.train(
                    series,
                    horizon,
                    seasonality,
                    params,
                    progress_callback=_on_progress,
                    exog=exog,
                    timeframe=timeframe,
                    **kwargs,
                )

                handle = self.store.save(
                    method=method_name,
                    data_scope=data_scope,
                    params_hash=params_hash,
                    artifact_bytes=result.artifact_bytes,
                    metadata={**(result.metadata or {}), "params_used": result.params_used},
                )

                with self._lock:
                    task.result = handle
                    task.status = "completed"
                    task.completed_at = time.time()

                logger.info("Training completed: %s %s → %s", method_name, data_scope, handle.model_id)

            except Exception as exc:
                with self._lock:
                    task.error = str(exc)
                    task.status = "failed"
                    task.completed_at = time.time()
                logger.exception("Training failed: %s %s", method_name, data_scope)
            finally:
                if acquired_heavy:
                    self._heavy_sem.release()

        try:
            future = self._executor.submit(_run)
        except RuntimeError:
            with self._lock:
                task.status = "failed"
                task.error = "Executor rejected task (pool shut down?)"
                task.completed_at = time.time()
            return task_id, True

        with self._lock:
            self._futures[task_id] = future

        return task_id, True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_status(self, task_id: str) -> Optional[TrainingTask]:
        """Return a snapshot of the task, or None if unknown."""
        with self._lock:
            task = self._tasks.get(task_id)
            return _snapshot(task) if task else None

    def list_tasks(self, *, status: Optional[str] = None) -> List[TrainingTask]:
        """Return snapshots of all tasks, newest first."""
        with self._lock:
            tasks = [_snapshot(t) for t in self._tasks.values()]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, task_id: str) -> bool:
        """Attempt to cancel a pending task. Returns True on success."""
        with self._lock:
            task = self._tasks.get(task_id)
            future = self._futures.get(task_id)
            if task is None or task.status in ("completed", "failed", "cancelled"):
                return False

        # Future.cancel() only works if not yet started
        if future and future.cancel():
            with self._lock:
                task.status = "cancelled"
                task.completed_at = time.time()
            return True
        return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_completed(self, max_age_seconds: float = _TASK_TTL_DEFAULT) -> int:
        """Remove finished tasks older than *max_age_seconds*. Returns count removed."""
        cutoff = time.time() - max_age_seconds
        removed = 0
        with self._lock:
            to_remove = [
                tid
                for tid, t in self._tasks.items()
                if t.status in ("completed", "failed", "cancelled")
                and t.completed_at is not None
                and t.completed_at < cutoff
            ]
            for tid in to_remove:
                t = self._tasks.pop(tid)
                self._futures.pop(tid, None)
                key = (t.method, t.data_scope, t.params_hash)
                if self._active_keys.get(key) == tid:
                    del self._active_keys[key]
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the executor. No new tasks accepted after this."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)


# ------------------------------------------------------------------
# Module-level lazy singleton
# ------------------------------------------------------------------

_global_manager: Optional[TaskManager] = None
_global_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    """Return (or create) the process-wide TaskManager singleton."""
    global _global_manager
    if _global_manager is None:
        with _global_lock:
            if _global_manager is None:
                _global_manager = TaskManager()
    return _global_manager
