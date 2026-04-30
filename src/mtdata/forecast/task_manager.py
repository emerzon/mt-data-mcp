"""Background training runtime for forecast methods."""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .interface import (
    CancelToken,
    ForecastMethod,
    TrainingCancelledError,
    TrainingProgress,
    TrainedModelHandle,
)
from .job_store import JobRecord, JobStore
from .model_store import ModelStore
from .forecast_registry import ForecastRegistry

logger = logging.getLogger(__name__)

TaskStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
TaskKind = Literal["prepared", "forecast_request"]

_MAX_WORKERS_DEFAULT = 4
_HEAVY_WORKERS_DEFAULT = 1
_TASK_TTL_DEFAULT = 3600.0
_SWEEPER_INTERVAL_DEFAULT = 60.0
_HEARTBEAT_INTERVAL_DEFAULT = 2.0
_CANCEL_GRACE_SECONDS_DEFAULT = 3.0
_TIMEOUT_DEFAULTS = {
    "instant": 30.0,
    "fast": 120.0,
    "moderate": 600.0,
    "heavy": 1800.0,
}
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


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
    pid: Optional[int] = None
    heartbeat_at: Optional[float] = None
    cancel_requested: bool = False


@dataclass(frozen=True)
class _TrainingSpec:
    task_kind: TaskKind
    method_name: str
    data_scope: str
    params_hash: str
    horizon: int
    seasonality: int
    params: Dict[str, Any]
    timeframe: str = ""
    series: Optional[pd.Series] = None
    exog: Optional[np.ndarray] = None
    request_payload: Optional[Dict[str, Any]] = None


@dataclass
class _HeavyProcessControl:
    process: mp.Process
    event_queue: Any
    cancel_event: Any


def _snapshot(task: TrainingTask) -> TrainingTask:
    return copy.copy(task)


def _serialize_progress(progress: Optional[TrainingProgress]) -> Optional[Dict[str, Any]]:
    if progress is None:
        return None
    payload: Dict[str, Any] = {
        "step": int(progress.step),
        "total_steps": int(progress.total_steps),
    }
    if progress.loss is not None:
        payload["loss"] = float(progress.loss)
    if progress.metrics is not None:
        payload["metrics"] = dict(progress.metrics)
    if progress.eta_seconds is not None:
        payload["eta_seconds"] = float(progress.eta_seconds)
    if progress.message is not None:
        payload["message"] = str(progress.message)
    return payload


def _deserialize_progress(payload: Optional[Dict[str, Any]]) -> Optional[TrainingProgress]:
    if not payload:
        return None
    return TrainingProgress(
        step=int(payload.get("step", 0)),
        total_steps=int(payload.get("total_steps", 0)),
        loss=float(payload["loss"]) if payload.get("loss") is not None else None,
        metrics=dict(payload["metrics"]) if isinstance(payload.get("metrics"), dict) else None,
        eta_seconds=(
            float(payload["eta_seconds"])
            if payload.get("eta_seconds") is not None
            else None
        ),
        message=str(payload["message"]) if payload.get("message") is not None else None,
    )


def _serialize_result(handle: Optional[TrainedModelHandle]) -> Optional[Dict[str, Any]]:
    if handle is None:
        return None
    return {
        "model_id": handle.model_id,
        "method": handle.method,
        "data_scope": handle.data_scope,
        "params_hash": handle.params_hash,
        "created_at": float(handle.created_at),
        "metadata": dict(handle.metadata),
        "store_metadata": dict(handle.store_metadata),
    }


def _deserialize_result(
    payload: Optional[Dict[str, Any]],
    *,
    method: str,
    data_scope: str,
    params_hash: str,
    created_at: float,
) -> Optional[TrainedModelHandle]:
    if payload is None:
        return None
    model_id = payload.get("model_id")
    if not model_id:
        return None
    return TrainedModelHandle(
        model_id=str(model_id),
        method=str(payload.get("method") or method),
        data_scope=str(payload.get("data_scope") or data_scope),
        params_hash=str(payload.get("params_hash") or params_hash),
        created_at=float(payload.get("created_at", created_at)),
        metadata=dict(payload.get("metadata") or {}),
        store_metadata=dict(payload.get("store_metadata") or {}),
    )


def _task_to_job_record(task: TrainingTask) -> JobRecord:
    return JobRecord(
        task_id=task.task_id,
        method=task.method,
        data_scope=task.data_scope,
        params_hash=task.params_hash,
        status=task.status,
        progress_payload=_serialize_progress(task.progress),
        result_payload=_serialize_result(task.result),
        error=task.error,
        pid=task.pid,
        heartbeat_at=task.heartbeat_at,
        cancel_requested=task.cancel_requested,
        model_id=task.result.model_id if task.result is not None else None,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
    )


def _job_record_to_task(record: JobRecord) -> TrainingTask:
    return TrainingTask(
        task_id=record.task_id,
        method=record.method,
        data_scope=record.data_scope,
        params_hash=record.params_hash,
        status=record.status,
        progress=_deserialize_progress(record.progress_payload),
        result=_deserialize_result(
            record.result_payload,
            method=record.method,
            data_scope=record.data_scope,
            params_hash=record.params_hash,
            created_at=record.completed_at or record.created_at,
        ),
        error=record.error,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        pid=record.pid,
        heartbeat_at=record.heartbeat_at,
        cancel_requested=record.cancel_requested,
    )


def _ensure_training_methods_registered() -> None:
    from .methods import ets_arima  # noqa: F401
    from .methods import mlforecast  # noqa: F401
    from .methods import neural  # noqa: F401
    from .methods import sktime  # noqa: F401
    from .methods import statsforecast  # noqa: F401


def _prepare_spec_inputs(spec: _TrainingSpec) -> Dict[str, Any]:
    if spec.task_kind == "prepared":
        if spec.series is None:
            raise ValueError("Prepared training task requires a series")
        return {
            "method_name": spec.method_name,
            "data_scope": spec.data_scope,
            "params_hash": spec.params_hash,
            "series": spec.series,
            "horizon": spec.horizon,
            "seasonality": spec.seasonality,
            "params": dict(spec.params),
            "exog": spec.exog,
            "timeframe": spec.timeframe,
        }

    from .forecast_engine import build_training_context

    payload = dict(spec.request_payload or {})
    context = build_training_context(**payload)
    return {
        "method_name": context.method_l,
        "data_scope": context.data_scope,
        "params_hash": spec.params_hash,
        "series": context.target_series,
        "horizon": context.horizon,
        "seasonality": context.seasonality,
        "params": dict(context.method_params),
        "exog": context.exog_used,
        "timeframe": context.timeframe,
    }


def _execute_training_spec(
    spec: _TrainingSpec,
    *,
    store_root: str,
    progress_callback,
    cancel_token: CancelToken,
) -> TrainedModelHandle:
    _ensure_training_methods_registered()
    prepared = _prepare_spec_inputs(spec)
    cancel_token.raise_if_cancelled()
    method_obj = ForecastRegistry.get(prepared["method_name"])
    result = method_obj.train(
        prepared["series"],
        int(prepared["horizon"]),
        int(prepared["seasonality"]),
        dict(prepared["params"]),
        progress_callback=progress_callback,
        cancel_token=cancel_token,
        exog=prepared["exog"],
        timeframe=prepared["timeframe"],
    )
    cancel_token.raise_if_cancelled()
    store = ModelStore(root=store_root)
    return store.save(
        method=prepared["method_name"],
        data_scope=prepared["data_scope"],
        params_hash=prepared["params_hash"],
        artifact_bytes=result.artifact_bytes,
        metadata={**(result.metadata or {}), "params_used": result.params_used},
    )


def _process_heartbeat_loop(event_queue: Any, stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        event_queue.put({
            "type": "heartbeat",
            "heartbeat_at": time.time(),
        })


def _process_training_entry(
    spec: _TrainingSpec,
    store_root: str,
    event_queue: Any,
    cancel_event: Any,
    heartbeat_interval: float,
) -> None:
    token = CancelToken(cancel_event.is_set)
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_process_heartbeat_loop,
        args=(event_queue, heartbeat_stop, heartbeat_interval),
        daemon=True,
    )
    heartbeat_thread.start()

    def _on_progress(progress: TrainingProgress) -> None:
        event_queue.put({
            "type": "progress",
            "heartbeat_at": time.time(),
            "progress": _serialize_progress(progress),
        })

    try:
        handle = _execute_training_spec(
            spec,
            store_root=store_root,
            progress_callback=_on_progress,
            cancel_token=token,
        )
        event_queue.put({
            "type": "completed",
            "heartbeat_at": time.time(),
            "completed_at": time.time(),
            "result": _serialize_result(handle),
        })
    except TrainingCancelledError as exc:
        event_queue.put({
            "type": "cancelled",
            "heartbeat_at": time.time(),
            "completed_at": time.time(),
            "error": str(exc),
        })
    except BaseException as exc:
        event_queue.put({
            "type": "failed",
            "heartbeat_at": time.time(),
            "completed_at": time.time(),
            "error": str(exc),
        })
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


class TaskManager:
    """Durable training runtime with light thread workers and heavy processes."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        heavy_limit: int | None = None,
        store: ModelStore | None = None,
        job_store: JobStore | None = None,
    ) -> None:
        workers = max_workers or int(os.environ.get("MTDATA_TRAIN_WORKERS", _MAX_WORKERS_DEFAULT))
        heavy_workers = heavy_limit or int(os.environ.get("MTDATA_HEAVY_LIMIT", _HEAVY_WORKERS_DEFAULT))

        self._light_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="train-light")
        self._heavy_executor = ThreadPoolExecutor(max_workers=heavy_workers, thread_name_prefix="train-heavy")
        self._mp_context = mp.get_context("spawn")

        self._tasks: Dict[str, TrainingTask] = {}
        self._futures: Dict[str, Future] = {}
        self._thread_cancel_events: Dict[str, threading.Event] = {}
        self._process_controls: Dict[str, _HeavyProcessControl] = {}
        self._active_keys: Dict[tuple[str, str, str], str] = {}
        self._lock = threading.Lock()
        self._store = store
        self._job_store = job_store if job_store is not None else JobStore()
        self._shutdown = False
        self._sweeper_stop = threading.Event()
        self._sweeper = threading.Thread(target=self._sweeper_loop, name="forecast-task-sweeper", daemon=True)

        self._recover_persisted_tasks()
        self._sweeper.start()

    @property
    def store(self) -> ModelStore:
        if self._store is None:
            from .model_store import model_store as _default

            self._store = _default
        return self._store

    def _timeout_for_category(self, category: str) -> float:
        normalized = str(category or "moderate").lower()
        env_key = f"MTDATA_TRAIN_TIMEOUT_{normalized.upper()}_SECONDS"
        fallback = _TIMEOUT_DEFAULTS.get(normalized, _TIMEOUT_DEFAULTS["moderate"])
        try:
            return max(1.0, float(os.environ.get(env_key, fallback)))
        except (TypeError, ValueError):
            return fallback

    def _heartbeat_interval(self) -> float:
        try:
            return max(0.5, float(os.environ.get("MTDATA_FORECAST_HEARTBEAT_SECONDS", _HEARTBEAT_INTERVAL_DEFAULT)))
        except (TypeError, ValueError):
            return _HEARTBEAT_INTERVAL_DEFAULT

    def _cancel_grace_seconds(self) -> float:
        try:
            return max(0.1, float(os.environ.get("MTDATA_FORECAST_CANCEL_GRACE_SECONDS", _CANCEL_GRACE_SECONDS_DEFAULT)))
        except (TypeError, ValueError):
            return _CANCEL_GRACE_SECONDS_DEFAULT

    def _cache_task(self, task: TrainingTask) -> None:
        self._tasks[task.task_id] = task
        key = (task.method, task.data_scope, task.params_hash)
        if task.status in ("pending", "running"):
            self._active_keys[key] = task.task_id
        elif self._active_keys.get(key) == task.task_id:
            del self._active_keys[key]

    def _persist_task(self, task: TrainingTask) -> None:
        self._job_store.upsert(_task_to_job_record(task))

    def _mutate_task(self, task_id: str, **changes: Any) -> Optional[TrainingTask]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for key, value in changes.items():
                setattr(task, key, value)
            self._cache_task(task)
            snapshot = _snapshot(task)
        self._persist_task(snapshot)
        return snapshot

    def _recover_persisted_tasks(self) -> None:
        stale = self._job_store.mark_active_jobs_failed(
            "Task registry recovered after process restart; in-flight task was orphaned.",
        )
        persisted = self._job_store.list_jobs()
        with self._lock:
            for task in (_job_record_to_task(record) for record in persisted):
                self._cache_task(task)
        if stale:
            logger.warning("Marked %d persisted forecast task(s) as failed during recovery", stale)

    def _compute_params_hash(
        self,
        method_name: str,
        *,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        timeframe: str,
        has_exog: bool,
    ) -> str:
        _ensure_training_methods_registered()
        forecaster = ForecastRegistry.get(method_name)
        fingerprint = forecaster.training_fingerprint(
            horizon=horizon,
            seasonality=seasonality,
            params=dict(params or {}),
            timeframe=timeframe,
            has_exog=has_exog,
        )
        return ForecastMethod.hash_fingerprint(fingerprint)

    def _get_existing_task(self, method_name: str, data_scope: str, params_hash: str) -> Optional[TrainingTask]:
        dedup_key = (method_name, data_scope, params_hash)
        with self._lock:
            existing_tid = self._active_keys.get(dedup_key)
            if existing_tid is not None:
                existing = self._tasks.get(existing_tid)
                if existing and existing.status in ("pending", "running"):
                    return _snapshot(existing)
        persisted = self._job_store.find_active(method_name, data_scope, params_hash)
        if persisted is None:
            return None
        task = _job_record_to_task(persisted)
        with self._lock:
            self._cache_task(task)
        return task

    def _create_task(self, method_name: str, data_scope: str, params_hash: str) -> TrainingTask:
        task = TrainingTask(
            task_id=uuid.uuid4().hex[:12],
            method=method_name,
            data_scope=data_scope,
            params_hash=params_hash,
        )
        with self._lock:
            self._cache_task(task)
            snapshot = _snapshot(task)
        self._persist_task(snapshot)
        return task

    def _submit_spec(
        self,
        spec: _TrainingSpec,
        *,
        training_category: str,
    ) -> tuple[str, bool]:
        if self._shutdown:
            raise RuntimeError("TaskManager is shut down")

        existing = self._get_existing_task(spec.method_name, spec.data_scope, spec.params_hash)
        if existing is not None:
            return existing.task_id, False

        task = self._create_task(spec.method_name, spec.data_scope, spec.params_hash)
        timeout_seconds = self._timeout_for_category(training_category)

        try:
            if str(training_category).lower() == "heavy":
                future = self._heavy_executor.submit(self._run_heavy_task, task.task_id, spec, timeout_seconds)
            else:
                cancel_event = threading.Event()
                with self._lock:
                    self._thread_cancel_events[task.task_id] = cancel_event
                future = self._light_executor.submit(self._run_light_task, task.task_id, spec, cancel_event)
        except RuntimeError:
            self._mutate_task(
                task.task_id,
                status="failed",
                error="Executor rejected task (pool shut down?)",
                completed_at=time.time(),
                heartbeat_at=time.time(),
            )
            return task.task_id, True

        with self._lock:
            self._futures[task.task_id] = future
        return task.task_id, True

    def submit(
        self,
        method_name: str,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: dict,
        data_scope: str,
        params_hash: Optional[str] = None,
        *,
        exog: Optional[np.ndarray] = None,
        timeframe: str = "",
        **kwargs: Any,
    ) -> tuple[str, bool]:
        if self._shutdown:
            raise RuntimeError("TaskManager is shut down")
        info = ForecastRegistry.get_method_info(method_name)
        params_for_hash = dict(params or {})
        canonical_hash = self._compute_params_hash(
            method_name,
            horizon=int(horizon),
            seasonality=int(seasonality),
            params=params_for_hash,
            timeframe=str(timeframe or ""),
            has_exog=exog is not None,
        )
        spec = _TrainingSpec(
            task_kind="prepared",
            method_name=method_name,
            data_scope=data_scope,
            params_hash=canonical_hash,
            horizon=int(horizon),
            seasonality=int(seasonality),
            params=dict(params or {}),
            timeframe=str(timeframe or ""),
            series=series.copy(),
            exog=np.array(exog, copy=True) if isinstance(exog, np.ndarray) else exog,
        )
        return self._submit_spec(spec, training_category=str(info.get("training_category", "moderate")))

    def submit_forecast_request(
        self,
        *,
        symbol: str,
        timeframe: str,
        method_name: str,
        horizon: int,
        lookback: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        quantity: str = "price",
    ) -> tuple[str, bool]:
        if self._shutdown:
            raise RuntimeError("TaskManager is shut down")
        from .common import default_seasonality

        info = ForecastRegistry.get_method_info(method_name)
        if not info.get("supports_training"):
            raise ValueError(f"Method '{method_name}' does not support separate training.")

        data_scope = f"{symbol}_{timeframe}"
        request_params = dict(params or {})
        if request_params.get("seasonality") is not None:
            seasonality = int(request_params["seasonality"])
        else:
            seasonality = int(default_seasonality(timeframe))
        params_for_hash = dict(request_params)
        params_for_hash["quantity"] = str(quantity)
        canonical_hash = self._compute_params_hash(
            method_name,
            horizon=int(horizon),
            seasonality=seasonality,
            params=params_for_hash,
            timeframe=str(timeframe),
            has_exog=False,
        )
        spec = _TrainingSpec(
            task_kind="forecast_request",
            method_name=method_name,
            data_scope=data_scope,
            params_hash=canonical_hash,
            horizon=int(horizon),
            seasonality=seasonality,
            params=dict(request_params),
            timeframe=str(timeframe),
            request_payload={
                "symbol": symbol,
                "timeframe": timeframe,
                "method": method_name,
                "horizon": int(horizon),
                "lookback": lookback,
                "params": dict(request_params),
                "quantity": quantity,
            },
        )
        return self._submit_spec(spec, training_category=str(info.get("training_category", "moderate")))

    def _run_light_task(self, task_id: str, spec: _TrainingSpec, cancel_event: threading.Event) -> None:
        self._mutate_task(
            task_id,
            status="running",
            started_at=time.time(),
            pid=os.getpid(),
            heartbeat_at=time.time(),
        )
        token = CancelToken(cancel_event.is_set)

        def _on_progress(progress: TrainingProgress) -> None:
            token.raise_if_cancelled()
            self._mutate_task(
                task_id,
                progress=progress,
                heartbeat_at=time.time(),
            )

        try:
            handle = _execute_training_spec(
                spec,
                store_root=str(self.store.root),
                progress_callback=_on_progress,
                cancel_token=token,
            )
        except TrainingCancelledError as exc:
            self._mutate_task(
                task_id,
                status="cancelled",
                error=str(exc),
                completed_at=time.time(),
                heartbeat_at=time.time(),
                cancel_requested=True,
            )
        except Exception as exc:
            logger.exception("Training failed: %s %s", spec.method_name, spec.data_scope)
            self._mutate_task(
                task_id,
                status="failed",
                error=str(exc),
                completed_at=time.time(),
                heartbeat_at=time.time(),
            )
        else:
            self._mutate_task(
                task_id,
                result=handle,
                status="completed",
                completed_at=time.time(),
                heartbeat_at=time.time(),
            )
            logger.info("Training completed: %s %s → %s", spec.method_name, spec.data_scope, handle.model_id)
        finally:
            with self._lock:
                self._thread_cancel_events.pop(task_id, None)

    def _handle_process_event(self, task_id: str, spec: _TrainingSpec, event: Dict[str, Any]) -> bool:
        event_type = str(event.get("type") or "").lower()
        heartbeat_at = float(event.get("heartbeat_at", time.time()))
        if event_type == "heartbeat":
            self._mutate_task(task_id, heartbeat_at=heartbeat_at)
            return False
        if event_type == "progress":
            self._mutate_task(
                task_id,
                progress=_deserialize_progress(event.get("progress")),
                heartbeat_at=heartbeat_at,
            )
            return False
        if event_type == "completed":
            handle = _deserialize_result(
                event.get("result"),
                method=spec.method_name,
                data_scope=spec.data_scope,
                params_hash=spec.params_hash,
                created_at=float(event.get("completed_at", time.time())),
            )
            self._mutate_task(
                task_id,
                result=handle,
                status="completed",
                completed_at=float(event.get("completed_at", time.time())),
                heartbeat_at=heartbeat_at,
            )
            return True
        if event_type == "cancelled":
            self._mutate_task(
                task_id,
                status="cancelled",
                error=str(event.get("error") or "Training cancelled"),
                completed_at=float(event.get("completed_at", time.time())),
                heartbeat_at=heartbeat_at,
                cancel_requested=True,
            )
            return True
        if event_type == "failed":
            self._mutate_task(
                task_id,
                status="failed",
                error=str(event.get("error") or "Training failed"),
                completed_at=float(event.get("completed_at", time.time())),
                heartbeat_at=heartbeat_at,
            )
            return True
        return False

    def _run_heavy_task(self, task_id: str, spec: _TrainingSpec, timeout_seconds: float) -> None:
        event_queue = self._mp_context.Queue()
        cancel_event = self._mp_context.Event()
        process = self._mp_context.Process(
            target=_process_training_entry,
            args=(spec, str(self.store.root), event_queue, cancel_event, self._heartbeat_interval()),
            daemon=True,
        )
        process.start()
        with self._lock:
            self._process_controls[task_id] = _HeavyProcessControl(
                process=process,
                event_queue=event_queue,
                cancel_event=cancel_event,
            )
        self._mutate_task(
            task_id,
            status="running",
            started_at=time.time(),
            pid=process.pid,
            heartbeat_at=time.time(),
        )

        terminal = False
        started_at = time.time()
        cancel_grace = self._cancel_grace_seconds()
        try:
            while True:
                try:
                    event = event_queue.get(timeout=1.0)
                except queue.Empty:
                    event = None
                if isinstance(event, dict):
                    terminal = self._handle_process_event(task_id, spec, event) or terminal

                if (
                    not terminal
                    and timeout_seconds > 0
                    and process.is_alive()
                    and (time.time() - started_at) > timeout_seconds
                ):
                    cancel_event.set()
                    time.sleep(cancel_grace)
                    if process.is_alive():
                        process.terminate()
                    self._mutate_task(
                        task_id,
                        status="failed",
                        error=f"Training timed out after {int(timeout_seconds)} seconds.",
                        completed_at=time.time(),
                        heartbeat_at=time.time(),
                        cancel_requested=True,
                    )
                    terminal = True

                if not process.is_alive():
                    while True:
                        try:
                            pending_event = event_queue.get_nowait()
                        except queue.Empty:
                            break
                        if isinstance(pending_event, dict):
                            terminal = self._handle_process_event(task_id, spec, pending_event) or terminal
                    if not terminal:
                        status = self.get_status(task_id)
                        if status is None or status.status not in _TERMINAL_STATUSES:
                            error_text = "Training worker exited unexpectedly."
                            if process.exitcode not in (0, None):
                                error_text = f"Training worker exited with code {process.exitcode}."
                            self._mutate_task(
                                task_id,
                                status="failed",
                                error=error_text,
                                completed_at=time.time(),
                                heartbeat_at=time.time(),
                            )
                    break
        finally:
            process.join(timeout=1.0)
            with self._lock:
                self._process_controls.pop(task_id, None)
            try:
                event_queue.close()
            except Exception:
                pass

    def get_status(self, task_id: str) -> Optional[TrainingTask]:
        persisted = self._job_store.get(task_id)
        if persisted is not None:
            task = _job_record_to_task(persisted)
            with self._lock:
                self._cache_task(task)
            return _snapshot(task)
        with self._lock:
            task = self._tasks.get(task_id)
            return _snapshot(task) if task else None

    def list_tasks(self, *, status: Optional[str] = None) -> List[TrainingTask]:
        persisted = self._job_store.list_jobs(status=status)
        if persisted:
            tasks = [_job_record_to_task(record) for record in persisted]
            with self._lock:
                for task in tasks:
                    self._cache_task(task)
            return tasks
        with self._lock:
            tasks = [_snapshot(t) for t in self._tasks.values()]
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def wait_for_status(self, task_id: str, timeout_seconds: float = 30.0, poll_interval: float = 0.25) -> Optional[TrainingTask]:
        deadline = time.time() + max(0.0, float(timeout_seconds))
        while True:
            task = self.get_status(task_id)
            if task is None:
                return None
            if task.status in _TERMINAL_STATUSES or time.time() >= deadline:
                return task
            time.sleep(max(0.05, float(poll_interval)))

    def cancel(self, task_id: str) -> Dict[str, Any]:
        task = self.get_status(task_id)
        if task is None:
            return {
                "task_id": task_id,
                "cancel_requested": False,
                "terminated": False,
                "status": "not_found",
            }
        if task.status in _TERMINAL_STATUSES:
            return {
                "task_id": task_id,
                "cancel_requested": False,
                "terminated": False,
                "status": task.status,
            }

        terminated = False
        cancel_requested = False
        future: Optional[Future] = None
        control: Optional[_HeavyProcessControl] = None
        cancel_event: Optional[threading.Event] = None
        with self._lock:
            task_ref = self._tasks.get(task_id)
            if task_ref is not None:
                task_ref.cancel_requested = True
                snapshot = _snapshot(task_ref)
            else:
                snapshot = None
            future = self._futures.get(task_id)
            control = self._process_controls.get(task_id)
            cancel_event = self._thread_cancel_events.get(task_id)
        if snapshot is not None:
            self._persist_task(snapshot)

        if future and future.cancel():
            self._mutate_task(
                task_id,
                status="cancelled",
                completed_at=time.time(),
                heartbeat_at=time.time(),
                cancel_requested=True,
                error="Task cancelled before execution started.",
            )
            return {
                "task_id": task_id,
                "cancel_requested": True,
                "terminated": False,
                "status": "cancelled",
            }

        if cancel_event is not None:
            cancel_event.set()
            cancel_requested = True

        if control is not None:
            control.cancel_event.set()
            cancel_requested = True
            time.sleep(self._cancel_grace_seconds())
            if control.process.is_alive():
                control.process.terminate()
                terminated = True

        return {
            "task_id": task_id,
            "cancel_requested": cancel_requested,
            "terminated": terminated,
            "status": "cancelling" if cancel_requested else "not_cancelled",
        }

    def cleanup_completed(self, max_age_seconds: float = _TASK_TTL_DEFAULT) -> int:
        removed_ids = set(self._job_store.cleanup_completed(max_age_seconds))
        cutoff = time.time() - max_age_seconds
        with self._lock:
            to_remove = [
                tid
                for tid, task in self._tasks.items()
                if task.status in _TERMINAL_STATUSES
                and task.completed_at is not None
                and task.completed_at < cutoff
            ]
            for tid in to_remove:
                task = self._tasks.pop(tid)
                self._futures.pop(tid, None)
                self._thread_cancel_events.pop(tid, None)
                self._process_controls.pop(tid, None)
                key = (task.method, task.data_scope, task.params_hash)
                if self._active_keys.get(key) == tid:
                    del self._active_keys[key]
                removed_ids.add(tid)
        return len(removed_ids)

    def _sweeper_loop(self) -> None:
        try:
            interval = max(
                5.0,
                float(os.environ.get("MTDATA_FORECAST_SWEEPER_SECONDS", _SWEEPER_INTERVAL_DEFAULT)),
            )
        except (TypeError, ValueError):
            interval = _SWEEPER_INTERVAL_DEFAULT

        while not self._sweeper_stop.wait(interval):
            try:
                self.cleanup_completed()
                self.store.cleanup_expired()
            except Exception:
                logger.exception("Forecast training sweeper failed")

    def shutdown(self, wait: bool = False) -> None:
        self._shutdown = True
        self._sweeper_stop.set()
        for control in list(self._process_controls.values()):
            try:
                control.cancel_event.set()
                if control.process.is_alive():
                    control.process.terminate()
            except Exception:
                logger.debug("Failed to stop heavy training worker during shutdown", exc_info=True)
        self._light_executor.shutdown(wait=wait)
        self._heavy_executor.shutdown(wait=wait)
        self._sweeper.join(timeout=1.0)


_global_manager: Optional[TaskManager] = None
_global_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    global _global_manager
    if _global_manager is None:
        with _global_lock:
            if _global_manager is None:
                _global_manager = TaskManager()
    return _global_manager
