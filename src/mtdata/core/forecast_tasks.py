"""MCP tools for forecast training task management and model management.

Provides tools for:
- Starting explicit training jobs (``forecast_train``)
- Polling task progress (``forecast_task_status``)
- Cancelling running tasks (``forecast_task_cancel``)
- Listing active tasks (``forecast_task_list``)
- Listing stored trained models (``forecast_models_list``)
- Deleting stored models (``forecast_models_delete``)
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..shared.schema import TimeframeLiteral
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation

logger = logging.getLogger(__name__)


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


class ForecastTaskCancelRequest(BaseModel):
    task_id: str = Field(..., description="Task ID to cancel.")


class ForecastModelsDeleteRequest(BaseModel):
    model_id: str = Field(..., description="Model ID in format method/data_scope/params_hash.")


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
    from ..forecast.registry import ForecastRegistry
    return ForecastRegistry


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def forecast_train(request: ForecastTrainRequest) -> Dict[str, Any]:
    """Start an explicit background training job for a forecast method.

    Returns a task_id that can be polled with ``forecast_task_status``.
    Once training completes, the trained model is stored and used automatically
    by subsequent ``forecast_generate`` calls with the same method+symbol+timeframe+params.
    """
    def _execute() -> Dict[str, Any]:
        from ..forecast.forecast_engine import forecast_engine
        from ..utils.mt5 import ensure_mt5_connection_or_raise

        ensure_mt5_connection_or_raise()

        result = forecast_engine(
            symbol=request.symbol,
            timeframe=request.timeframe,
            method=request.method,
            horizon=request.horizon,
            lookback=request.lookback,
            params=request.params,
            quantity=request.quantity,
            async_mode=True,
        )

        # If the method supports async training, the engine returns a task dict
        if isinstance(result, dict) and result.get("status") in (
            "training_started",
            "training_in_progress",
        ):
            return result

        # Otherwise the method ran synchronously (fast method) — return success
        return {
            "status": "completed_sync",
            "method": request.method,
            "message": (
                f"Method '{request.method}' ran synchronously (fast method). "
                "No background task needed."
            ),
            "result": result,
        }

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

    Returns task status, progress (step/total/loss), and result on completion.
    """
    def _execute() -> Dict[str, Any]:
        tm = _get_task_manager()
        task = tm.get_status(request.task_id)
        if task is None:
            return {"error": f"Task '{request.task_id}' not found."}

        resp: Dict[str, Any] = {
            "task_id": task.task_id,
            "method": task.method,
            "data_scope": task.data_scope,
            "status": task.status,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
        }

        if task.progress is not None:
            resp["progress"] = {
                "step": task.progress.step,
                "total_steps": task.progress.total_steps,
                "fraction": task.progress.fraction,
                "loss": task.progress.loss,
                "eta_seconds": task.progress.eta_seconds,
                "message": task.progress.message,
            }

        if task.status == "completed" and task.result is not None:
            resp["model_id"] = task.result.model_id
            resp["message"] = (
                f"Training complete. Model stored as '{task.result.model_id}'. "
                "Subsequent forecast_generate calls will use this model automatically."
            )

        if task.status == "failed" and task.error:
            resp["error"] = task.error

        return resp

    return run_logged_operation(
        logger,
        operation="forecast_task_status",
        func=_execute,
        task_id=request.task_id,
    )


@mcp.tool()
def forecast_task_cancel(request: ForecastTaskCancelRequest) -> Dict[str, Any]:
    """Cancel a running forecast training task."""
    def _execute() -> Dict[str, Any]:
        tm = _get_task_manager()
        cancelled = tm.cancel(request.task_id)
        if cancelled:
            return {
                "task_id": request.task_id,
                "status": "cancelled",
                "message": "Task cancellation requested.",
            }
        return {
            "task_id": request.task_id,
            "status": "not_cancelled",
            "message": "Task could not be cancelled (may have already completed or not found).",
        }

    return run_logged_operation(
        logger,
        operation="forecast_task_cancel",
        func=_execute,
        task_id=request.task_id,
    )


@mcp.tool()
def forecast_task_list(
    status_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """List active and recent forecast training tasks.

    Optionally filter by status: pending, running, completed, failed, cancelled.
    """
    def _execute() -> Dict[str, Any]:
        tm = _get_task_manager()
        tasks = tm.list_tasks(status_filter=status_filter)
        items: List[Dict[str, Any]] = []
        for t in tasks:
            item: Dict[str, Any] = {
                "task_id": t.task_id,
                "method": t.method,
                "data_scope": t.data_scope,
                "status": t.status,
                "created_at": t.created_at,
            }
            if t.progress is not None:
                item["progress_fraction"] = t.progress.fraction
            if t.result is not None:
                item["model_id"] = t.result.model_id
            if t.error:
                item["error"] = t.error
            items.append(item)
        return {
            "count": len(items),
            "tasks": items,
        }

    return run_logged_operation(
        logger,
        operation="forecast_task_list",
        func=_execute,
        status_filter=status_filter,
    )


@mcp.tool()
def forecast_models_list(
    method: Optional[str] = None,
) -> Dict[str, Any]:
    """List all stored trained forecast models.

    Optionally filter by method name (e.g. nhits, tft, mlforecast).
    """
    def _execute() -> Dict[str, Any]:
        store = _get_model_store()
        handles = store.list_models(method=method)
        items = [
            {
                "model_id": h.model_id,
                "method": h.method,
                "data_scope": h.data_scope,
                "params_hash": h.params_hash,
                "created_at": h.created_at,
                "metadata": h.metadata,
            }
            for h in handles
        ]
        return {
            "count": len(items),
            "models": items,
        }

    return run_logged_operation(
        logger,
        operation="forecast_models_list",
        func=_execute,
        method=method,
    )


@mcp.tool()
def forecast_models_delete(request: ForecastModelsDeleteRequest) -> Dict[str, Any]:
    """Delete a stored trained forecast model by model_id."""
    def _execute() -> Dict[str, Any]:
        store = _get_model_store()
        deleted = store.delete(request.model_id)
        if deleted:
            return {
                "model_id": request.model_id,
                "deleted": True,
                "message": f"Model '{request.model_id}' deleted.",
            }
        return {
            "model_id": request.model_id,
            "deleted": False,
            "message": f"Model '{request.model_id}' not found.",
        }

    return run_logged_operation(
        logger,
        operation="forecast_models_delete",
        func=_execute,
        model_id=request.model_id,
    )
