"""MCP tools for forecast training task management and model management.

Provides tools for:
- Starting explicit training jobs (``forecast_train``)
- Polling task progress (``forecast_task_status``)
- Waiting on task completion (``forecast_task_wait``)
- Cancelling running tasks (``forecast_task_cancel``)
- Listing active tasks (``forecast_task_list``)
- Listing stored trained models (``forecast_models_list``)
- Deleting stored models (``forecast_models_delete``)
- Cleaning up stale stored models (``forecast_models_cleanup``)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..shared.schema import CompactFullDetailLiteral, TimeframeLiteral
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)

DetailLevel = CompactFullDetailLiteral


def _format_epoch_utc(value: Any) -> Optional[str]:
    try:
        timestamp = float(value)
    except Exception:
        return None
    try:
        return (
            datetime.fromtimestamp(timestamp, timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except Exception:
        return None


def _days(value: Any) -> Optional[float]:
    try:
        seconds = float(value)
    except Exception:
        return None
    return round(max(0.0, seconds) / 86400.0, 3)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ForecastTrainRequest(BaseModel):
    """Request to start an explicit training job."""

    symbol: str
    timeframe: TimeframeLiteral = "H1"
    method: str = Field(..., description="Forecast method name (e.g. nhits, tft, mlf_rf).")
    horizon: int = Field(12, ge=1)
    lookback: Optional[int] = Field(None, ge=1)
    params: Optional[Dict[str, Any]] = None
    quantity: Literal["price", "return"] = "price"


class ForecastTaskStatusRequest(BaseModel):
    task_id: str = Field(..., description="Task ID returned by forecast_train or auto-training.")
    detail: DetailLevel = Field(
        "compact",
        description="Response detail level: 'compact' for summary fields or 'full' for expanded task details.",
    )


class ForecastTaskCancelRequest(BaseModel):
    task_id: str = Field(..., description="Task ID to cancel.")


class ForecastTaskCancelAllRequest(BaseModel):
    status_filter: Optional[str] = Field(
        "running",
        description="Task status to cancel. Defaults to running; use pending to cancel queued tasks.",
    )
    method: Optional[str] = Field(None, description="Optional method filter.")
    data_scope: Optional[str] = Field(None, description="Optional data_scope filter such as EURUSD_H1.")
    since_minutes: Optional[float] = Field(None, ge=0.0, description="Only cancel tasks created within this many minutes.")
    dry_run: bool = Field(True, description="Preview matching tasks without cancelling them.")


class ForecastTaskWaitRequest(BaseModel):
    task_id: str = Field(..., description="Task ID returned by forecast_train or forecast_generate async mode.")
    timeout_seconds: float = Field(30.0, ge=0.0, le=300.0)
    detail: DetailLevel = Field(
        "compact",
        description="Response detail level: 'compact' for summary fields or 'full' for expanded task details.",
    )


class ForecastModelsDeleteRequest(BaseModel):
    model_id: str = Field(..., description="Model ID in format method/data_scope/params_hash.")


class ForecastModelsCleanupRequest(BaseModel):
    older_than_days: Optional[float] = Field(
        None,
        ge=0.0,
        description="Delete models idle for at least this many days. Omit to use the store TTL.",
    )
    method: Optional[str] = Field(None, description="Optional method filter.")
    dry_run: bool = Field(True, description="Preview matching models without deleting them.")
    detail: DetailLevel = Field(
        "compact",
        description="Response detail level: compact returns model IDs; full includes age and size fields.",
    )


# ---------------------------------------------------------------------------
# Payload shaping helpers
# ---------------------------------------------------------------------------

def _detail_mode(value: Any) -> DetailLevel:
    return "full" if str(value or "compact").strip().lower() == "full" else "compact"


def _task_matches_filters(
    task: Any,
    *,
    method: Optional[str] = None,
    data_scope: Optional[str] = None,
    since_minutes: Optional[float] = None,
) -> bool:
    if method and str(getattr(task, "method", "")) != str(method):
        return False
    if data_scope and str(getattr(task, "data_scope", "")) != str(data_scope):
        return False
    if since_minutes is not None:
        try:
            since_seconds = max(0.0, float(since_minutes)) * 60.0
        except Exception:
            return False
        created_at = getattr(task, "created_at", None)
        try:
            if time.time() - float(created_at) > since_seconds:
                return False
        except Exception:
            return False
    return True


def _serialize_progress(progress: Any, *, detail: DetailLevel) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": progress.step,
        "total_steps": progress.total_steps,
        "fraction": progress.fraction,
    }
    if progress.loss is not None:
        payload["loss"] = progress.loss
    if detail == "full":
        if progress.eta_seconds is not None:
            payload["eta_seconds"] = progress.eta_seconds
        if progress.message:
            payload["message"] = progress.message
        metrics = getattr(progress, "metrics", None)
        if metrics is not None:
            payload["metrics"] = metrics
    return payload


def _serialize_model_handle(
    handle: Any,
    *,
    detail: DetailLevel,
    store: Any = None,
) -> Dict[str, Any]:
    created_at_epoch = getattr(handle, "created_at", None)
    store_info: Dict[str, Any] = {}
    describe = getattr(store, "describe_model", None)
    if callable(describe):
        try:
            raw_info = describe(handle)
            if isinstance(raw_info, dict):
                store_info = raw_info
        except Exception:
            store_info = {}
    created_at_epoch = store_info.get("created_at", created_at_epoch)
    last_used_epoch = store_info.get("last_used")
    payload: Dict[str, Any] = {
        "model_id": handle.model_id,
        "method": handle.method,
        "data_scope": handle.data_scope,
        "params_hash": handle.params_hash,
        "created_at": _format_epoch_utc(created_at_epoch) or created_at_epoch,
    }
    model_metadata = dict(getattr(handle, "metadata", {}) or {})
    source_task_id = model_metadata.get("source_task_id")
    if source_task_id not in (None, ""):
        payload["source_task_id"] = str(source_task_id)
    if last_used_epoch is not None:
        payload["last_used"] = _format_epoch_utc(last_used_epoch) or last_used_epoch
    age_days = _days(store_info.get("age_seconds"))
    idle_days = _days(store_info.get("idle_seconds"))
    expires_in_days = _days(store_info.get("expires_in_seconds"))
    if age_days is not None:
        payload["age_days"] = age_days
    if idle_days is not None:
        payload["idle_days"] = idle_days
    if store_info.get("size_bytes") is not None:
        payload["size_bytes"] = int(store_info.get("size_bytes") or 0)
    if expires_in_days is not None:
        payload["expires_in_days"] = expires_in_days
    if store_info.get("expired") is not None:
        payload["expired"] = bool(store_info.get("expired"))
    store_metadata = dict(getattr(handle, "store_metadata", {}) or {})
    try:
        from ..forecast.model_store import describe_store_metadata_compatibility

        compatibility = describe_store_metadata_compatibility(store_metadata)
    except Exception:
        compatibility = {}
    compatibility_status = compatibility.get("status") if isinstance(compatibility, dict) else None
    if compatibility_status:
        payload["compatibility_status"] = compatibility_status
    if detail == "full":
        from ..forecast.model_store import (
            sanitize_store_metadata,
        )

        payload["created_at_epoch"] = created_at_epoch
        if last_used_epoch is not None:
            payload["last_used_epoch"] = last_used_epoch
        if store_info:
            payload["ttl_days"] = _days(store_info.get("ttl_seconds"))
            payload["file_count"] = int(store_info.get("file_count") or 0)
            payload["model_dir"] = store_info.get("model_dir")
        payload["metadata"] = model_metadata
        payload["store_metadata"] = sanitize_store_metadata(store_metadata)
        payload["compatibility"] = compatibility
    return payload


def _model_store_state_payload(handle: Any) -> Dict[str, Any]:
    try:
        store = _get_model_store()
        info = store.describe_model(handle)
    except Exception:
        return {"model_store_status": "unknown", "model_stored": None}

    file_count = int(info.get("file_count") or 0)
    expired = bool(info.get("expired")) if info.get("expired") is not None else False
    if file_count <= 0:
        status = "missing"
    elif expired:
        status = "expired"
    else:
        status = "present"

    payload: Dict[str, Any] = {
        "model_stored": file_count > 0,
        "model_store_status": status,
        "model_store_path": info.get("model_dir"),
        "model_store_file_count": file_count,
    }
    if info.get("ttl_seconds") is not None:
        payload["model_store_ttl_days"] = _days(info.get("ttl_seconds"))
    return payload


def _recent_completed_model_tasks(
    *,
    method: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    try:
        tasks = _get_task_manager().list_tasks(status="completed")
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for task in tasks:
        handle = getattr(task, "result", None)
        if handle is None:
            continue
        if method and getattr(handle, "method", None) != method:
            continue
        row = {
            "task_id": task.task_id,
            "model_id": handle.model_id,
            "completed_at": getattr(task, "completed_at", None),
        }
        row.update(_model_store_state_payload(handle))
        rows.append(row)
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def _task_runtime_payload(task: Any, runtime: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(runtime, dict):
        return {}
    task_id = str(getattr(task, "task_id", "") or "")
    out: Dict[str, Any] = {}
    training_category = None
    try:
        info = _get_registry().get_method_info(str(getattr(task, "method", "") or ""))
        if isinstance(info, dict):
            training_category = str(info.get("training_category") or "").strip().lower() or None
    except Exception:
        training_category = None
    if training_category:
        out["training_category"] = training_category
        out["worker_type"] = "heavy" if training_category == "heavy" else "light"
    queue_positions = runtime.get("queue_positions")
    if isinstance(queue_positions, dict) and task_id in queue_positions:
        out["queue_position"] = queue_positions.get(task_id)
    heavy_task_ids = runtime.get("heavy_task_ids")
    if isinstance(heavy_task_ids, list) and task_id in heavy_task_ids:
        out["worker_pool"] = "heavy"
        out["worker_type"] = "heavy"
    elif getattr(task, "status", None) == "running":
        out["worker_pool"] = "light"
        out.setdefault("worker_type", "light")
    return out


def _task_status_payload(
    task: Any,
    *,
    detail: DetailLevel,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": True,
        "detail": detail,
        "task_id": task.task_id,
        "method": task.method,
        "data_scope": task.data_scope,
        "status": task.status,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "heartbeat_at": getattr(task, "heartbeat_at", None),
        "cancel_requested": bool(getattr(task, "cancel_requested", False)),
    }
    pid = getattr(task, "pid", None)
    if pid is not None:
        payload["pid"] = pid

    if task.progress is not None:
        payload["progress_fraction"] = task.progress.fraction
        payload["progress"] = _serialize_progress(task.progress, detail=detail)
    payload.update(_task_runtime_payload(task, runtime))

    if task.status == "completed" and task.result is not None:
        payload["model_id"] = task.result.model_id
        payload["produced_model_ids"] = [task.result.model_id]
        payload["message"] = (
            f"Training complete. Model stored as '{task.result.model_id}'. "
            "Subsequent forecast_generate calls will use this model automatically."
        )
        if detail == "full":
            payload["result"] = _serialize_model_handle(task.result, detail="full")

    if task.status == "failed" and task.error:
        payload["error"] = task.error

    if detail == "full":
        if runtime:
            payload["runtime"] = {
                "workers": runtime.get("workers"),
                "queue": runtime.get("queue"),
            }
        params_hash = getattr(task, "params_hash", None)
        if params_hash:
            payload["params_hash"] = params_hash

    return payload


def _task_list_item_payload(
    task: Any,
    *,
    detail: DetailLevel,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    started_at = getattr(task, "started_at", None)
    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "method": task.method,
        "data_scope": task.data_scope,
        "status": task.status,
        "created_at": task.created_at,
        "started_at": started_at,
        "heartbeat_at": getattr(task, "heartbeat_at", None),
        "cancel_requested": bool(getattr(task, "cancel_requested", False)),
    }
    completed_at = getattr(task, "completed_at", None)
    elapsed_start = started_at or getattr(task, "created_at", None)
    elapsed_end = completed_at or time.time()
    if elapsed_start is not None:
        payload["elapsed_seconds"] = round(
            max(0.0, float(elapsed_end) - float(elapsed_start)),
            3,
        )
    pid = getattr(task, "pid", None)
    if pid is not None:
        payload["pid"] = pid
    if task.progress is not None:
        payload["progress_fraction"] = task.progress.fraction
    payload.update(_task_runtime_payload(task, runtime))
    if task.result is not None:
        payload["model_id"] = task.result.model_id
        payload["produced_model_ids"] = [task.result.model_id]
        payload.update(_model_store_state_payload(task.result))
    if task.error:
        payload["error"] = task.error

    if detail == "full":
        payload["completed_at"] = task.completed_at
        params_hash = getattr(task, "params_hash", None)
        if params_hash:
            payload["params_hash"] = params_hash
        if task.progress is not None:
            payload["progress"] = _serialize_progress(task.progress, detail="full")
        if task.result is not None:
            payload["result"] = _serialize_model_handle(task.result, detail="full")

    return payload


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_task_manager():
    from ..forecast.task_manager import get_task_manager
    return get_task_manager()


def _get_model_store():
    from ..forecast.model_store import model_store
    return model_store


def _get_registry():
    from ..forecast.forecast_registry import ForecastRegistry
    return ForecastRegistry


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def forecast_train(request: ForecastTrainRequest) -> Dict[str, Any]:
    """Start an explicit background training job for a trainable forecast method."""
    def _execute() -> Dict[str, Any]:
        from ..utils.mt5 import ensure_mt5_connection_or_raise

        ensure_mt5_connection_or_raise()

        tm = _get_task_manager()
        task_id, _ = tm.submit_forecast_request(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method_name=request.method,
            horizon=request.horizon,
            lookback=request.lookback,
            params=request.params,
            quantity=request.quantity,
        )
        task = tm.get_status(task_id)
        if task is None:
            return {
                "success": True,
                "task_id": task_id,
                "status": "pending",
                "method": request.method,
                "data_scope": f"{request.symbol}_{request.timeframe}",
            }
        payload = _task_status_payload(
            task,
            detail="compact",
            runtime=tm.runtime_snapshot(),
        )
        payload["message"] = (
            "Training task queued. Poll forecast_task_status or use forecast_task_wait "
            "to observe completion."
        )
        return payload

    return run_logged_operation(
        logger,
        operation="forecast_train",
        func=_execute,
        symbol=request.symbol,
        timeframe=request.timeframe,
        method=request.method,
    )


@mcp.tool()
def forecast_task_status(request: ForecastTaskStatusRequest) -> Dict[str, Any]:
    """Get the current status and progress of a forecast training task.

    Returns task status, progress, and completion info.
    Use ``extras='metadata'`` for expanded task/result metadata.
    """
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(request.detail)
        tm = _get_task_manager()
        task = tm.get_status(request.task_id)
        if task is None:
            out = build_error_payload(
                f"Task '{request.task_id}' not found.",
                code="forecast_task_not_found",
                operation="forecast_task_status",
            )
            out["detail"] = detail_mode
            out["task_id"] = request.task_id
            return out
        return _task_status_payload(
            task,
            detail=detail_mode,
            runtime=tm.runtime_snapshot(),
        )

    return run_logged_operation(
        logger,
        operation="forecast_task_status",
        func=_execute,
        task_id=request.task_id,
        detail=request.detail,
    )


@mcp.tool()
def forecast_task_cancel(request: ForecastTaskCancelRequest) -> Dict[str, Any]:
    """Cancel a running forecast training task."""
    def _execute() -> Dict[str, Any]:
        tm = _get_task_manager()
        result = tm.cancel(request.task_id)
        result["success"] = bool(result["cancel_requested"])
        if result["cancel_requested"]:
            result["message"] = (
                "Task cancellation requested."
                if not result["terminated"]
                else "Task cancellation requested and worker terminated."
            )
        else:
            out = build_error_payload(
                "Task could not be cancelled.",
                code="forecast_task_cancel_failed",
                operation="forecast_task_cancel",
            )
            for key in ("task_id", "cancel_requested", "terminated", "status"):
                if key in result:
                    out[key] = result[key]
            return out
        return result

    return run_logged_operation(
        logger,
        operation="forecast_task_cancel",
        func=_execute,
        task_id=request.task_id,
    )


@mcp.tool()
def forecast_task_cancel_all(request: ForecastTaskCancelAllRequest) -> Dict[str, Any]:
    """Preview or cancel matching non-terminal forecast training tasks."""
    def _execute() -> Dict[str, Any]:
        status_value = str(request.status_filter or "running").strip().lower()
        if status_value not in {"pending", "running"}:
            return build_error_payload(
                "forecast_task_cancel_all only supports status_filter=pending or running.",
                code="forecast_task_cancel_all_invalid_status",
                operation="forecast_task_cancel_all",
            )
        tm = _get_task_manager()
        tasks = [
            task
            for task in tm.list_tasks(status=status_value)
            if _task_matches_filters(
                task,
                method=request.method,
                data_scope=request.data_scope,
                since_minutes=request.since_minutes,
            )
        ]
        matches = [
            {
                "task_id": task.task_id,
                "method": task.method,
                "data_scope": task.data_scope,
                "status": task.status,
                "created_at": task.created_at,
            }
            for task in tasks
        ]
        cancelled = []
        if not request.dry_run:
            for task in tasks:
                result = tm.cancel(task.task_id)
                if result.get("cancel_requested"):
                    cancelled.append(result)
        return {
            "success": True,
            "dry_run": bool(request.dry_run),
            "status_filter": status_value,
            "method": request.method,
            "data_scope": request.data_scope,
            "since_minutes": request.since_minutes,
            "matched": len(matches),
            "cancelled": len(cancelled),
            "tasks": matches,
            "results": cancelled,
        }

    return run_logged_operation(
        logger,
        operation="forecast_task_cancel_all",
        func=_execute,
        status_filter=request.status_filter,
        method=request.method,
        dry_run=request.dry_run,
    )


@mcp.tool()
def forecast_task_wait(request: ForecastTaskWaitRequest) -> Dict[str, Any]:
    """Wait for a forecast training task to complete or timeout."""
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(request.detail)
        tm = _get_task_manager()
        task = tm.wait_for_status(request.task_id, timeout_seconds=request.timeout_seconds)
        if task is None:
            out = build_error_payload(
                f"Task '{request.task_id}' not found.",
                code="forecast_task_not_found",
                operation="forecast_task_wait",
            )
            out["detail"] = detail_mode
            out["task_id"] = request.task_id
            return out
        payload = _task_status_payload(
            task,
            detail=detail_mode,
            runtime=tm.runtime_snapshot(),
        )
        payload["wait_timeout_seconds"] = request.timeout_seconds
        return payload

    return run_logged_operation(
        logger,
        operation="forecast_task_wait",
        func=_execute,
        task_id=request.task_id,
        timeout_seconds=request.timeout_seconds,
        detail=request.detail,
    )


@mcp.tool()
def forecast_task_list(
    status_filter: Optional[str] = None,
    since_minutes: Optional[float] = None,
    method: Optional[str] = None,
    data_scope: Optional[str] = None,
    detail: DetailLevel = "compact",
) -> Dict[str, Any]:
    """List active and recent forecast training tasks.

    Optionally filter by status, method, data_scope, or recent creation window.
    Use ``extras='metadata'`` for expanded progress and result payloads.
    """
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(detail)
        if since_minutes is not None and float(since_minutes) < 0:
            return build_error_payload(
                "since_minutes must be >= 0.",
                code="forecast_task_list_invalid_since",
                operation="forecast_task_list",
            )
        tm = _get_task_manager()
        runtime = tm.runtime_snapshot()
        tasks = [
            task
            for task in tm.list_tasks(status=status_filter)
            if _task_matches_filters(
                task,
                method=method,
                data_scope=data_scope,
                since_minutes=since_minutes,
            )
        ]
        items = [
            _task_list_item_payload(task, detail=detail_mode, runtime=runtime)
            for task in tasks
        ]
        summary: Dict[str, int] = {}
        for item in items:
            status = str(item.get("status") or "unknown")
            summary[status] = summary.get(status, 0) + 1
        out = {
            "success": True,
            "detail": detail_mode,
            "count": len(items),
            "summary": summary,
            "filters": {
                "status_filter": status_filter,
                "since_minutes": since_minutes,
                "method": method,
                "data_scope": data_scope,
            },
            "runtime": {
                "workers": runtime.get("workers"),
                "queue": runtime.get("queue"),
            },
            "tasks": items,
        }
        if not items:
            if status_filter:
                out["message"] = f"No forecast tasks matched status_filter={status_filter!r}."
            else:
                out["message"] = "No forecast tasks found."
            out["hint"] = (
                "Create tasks with forecast_train or forecast_backtest_run; "
                "status_filter values: pending,running,completed,failed,cancelled."
            )
        return out

    return run_logged_operation(
        logger,
        operation="forecast_task_list",
        func=_execute,
        status_filter=status_filter,
        since_minutes=since_minutes,
        method=method,
        data_scope=data_scope,
        detail=detail,
    )


@mcp.tool()
def forecast_models_list(
    method: Optional[str] = None,
    detail: DetailLevel = "compact",
) -> Dict[str, Any]:
    """List all stored trained forecast models.

    Optionally filter by method name (e.g. nhits, tft, mlforecast).
    Use ``extras='metadata'`` to include stored model metadata.
    """
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(detail)
        store = _get_model_store()
        handles = store.list_models(method=method)
        items = [
            _serialize_model_handle(h, detail=detail_mode, store=store)
            for h in handles
        ]
        out = {
            "success": True,
            "detail": detail_mode,
            "count": len(items),
            "models": items,
        }
        if not items:
            out["model_store"] = {
                "path": str(getattr(store, "root", "")),
                "ttl_days": _days(getattr(store, "ttl_seconds", 0.0)),
                "models_cached": 0,
            }
            if method:
                out["message"] = f"No stored forecast models matched method={method!r}."
            else:
                out["message"] = (
                    "No stored forecast models found. Trainable methods persist "
                    "artifacts here after forecast_train or async forecast_generate."
                )
            out["hint"] = (
                "Use forecast_train to create a model, or run forecast_list_methods "
                "with profile=all and supports_training=true to inspect trainable methods."
            )
            out["actions"] = [
                "mtdata-cli forecast_list_methods --profile all --supports_training true",
                "mtdata-cli forecast_train --help",
            ]
            out["related_tools"] = [
                "forecast_train",
                "forecast_list_methods",
                "forecast_models_cleanup",
            ]
            recent_tasks = _recent_completed_model_tasks(method=method)
            if recent_tasks:
                out["recent_completed_tasks"] = recent_tasks
        return out

    return run_logged_operation(
        logger,
        operation="forecast_models_list",
        func=_execute,
        method=method,
        detail=detail,
    )


@mcp.tool()
def forecast_models_delete(request: ForecastModelsDeleteRequest) -> Dict[str, Any]:
    """Delete a stored trained forecast model by model_id."""
    def _execute() -> Dict[str, Any]:
        store = _get_model_store()
        deleted = store.delete(request.model_id)
        if deleted:
            return {
                "success": True,
                "model_id": request.model_id,
                "deleted": True,
                "message": f"Model '{request.model_id}' deleted.",
            }
        out = build_error_payload(
            f"Model '{request.model_id}' not found.",
            code="forecast_model_not_found",
            operation="forecast_models_delete",
        )
        out["model_id"] = request.model_id
        out["deleted"] = False
        return out

    return run_logged_operation(
        logger,
        operation="forecast_models_delete",
        func=_execute,
        model_id=request.model_id,
    )


@mcp.tool()
def forecast_models_cleanup(request: ForecastModelsCleanupRequest) -> Dict[str, Any]:
    """Preview or delete stale stored forecast models."""
    def _execute() -> Dict[str, Any]:
        detail_mode = _detail_mode(request.detail)
        store = _get_model_store()
        handles = store.list_models(method=request.method)
        generated_at = time.time()
        matches = []
        for handle in handles:
            info = store.describe_model(handle)
            if request.older_than_days is None:
                matched = bool(info.get("expired"))
                reason = "expired_by_ttl"
            else:
                idle_seconds = float(info.get("idle_seconds") or 0.0)
                matched = idle_seconds >= float(request.older_than_days) * 86400.0
                reason = "idle_age"
            if not matched:
                continue
            row: Dict[str, Any] = {
                "model_id": handle.model_id,
                "method": handle.method,
                "reason": reason,
            }
            if detail_mode == "full":
                row.update(
                    {
                        "data_scope": handle.data_scope,
                        "created_at": _format_epoch_utc(info.get("created_at")),
                        "last_used": _format_epoch_utc(info.get("last_used")),
                        "age_days": _days(info.get("age_seconds")),
                        "idle_days": _days(info.get("idle_seconds")),
                        "expires_in_days": _days(info.get("expires_in_seconds")),
                        "size_bytes": int(info.get("size_bytes") or 0),
                    }
                )
            matches.append(row)

        deleted = 0
        if not request.dry_run:
            for row in matches:
                if store.delete(str(row.get("model_id") or "")):
                    deleted += 1

        return {
            "success": True,
            "detail": detail_mode,
            "dry_run": bool(request.dry_run),
            "method": request.method,
            "older_than_days": request.older_than_days,
            "ttl_days": _days(getattr(store, "ttl_seconds", 0.0)),
            "matched": len(matches),
            "deleted": deleted,
            "models": matches,
            "generated_at": _format_epoch_utc(generated_at),
        }

    return run_logged_operation(
        logger,
        operation="forecast_models_cleanup",
        func=_execute,
        method=request.method,
        dry_run=request.dry_run,
    )
