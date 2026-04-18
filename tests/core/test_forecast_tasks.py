"""Tests for Phase 3 MCP tools: forecast_tasks module.

Tests the new MCP tool handlers (forecast_train, forecast_task_status,
forecast_task_cancel, forecast_task_list, forecast_models_list,
forecast_models_delete) and the async_mode/model_id passthrough from
forecast_generate.

Because these functions are ``@mcp.tool()``-decorated, calling them
directly triggers the FastMCP dispatch / parameter-coercion layer.
We call ``fn.__wrapped__`` (set by ``functools.wraps``) to bypass the
MCP wrapper and test the business logic directly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.mtdata.core.forecast_tasks import (
    ForecastModelsDeleteRequest,
    ForecastTaskCancelRequest,
    ForecastTaskStatusRequest,
    ForecastTrainRequest,
)
from src.mtdata.forecast.interface import (
    TrainedModelHandle,
    TrainingProgress,
)

# ---------------------------------------------------------------------------
# Patch targets — module-level lazy-import helpers in forecast_tasks.py
# ---------------------------------------------------------------------------

_PATCH_TM = "src.mtdata.core.forecast_tasks._get_task_manager"
_PATCH_STORE = "src.mtdata.core.forecast_tasks._get_model_store"
_PATCH_MT5 = "src.mtdata.core.forecast_tasks.ensure_mt5_connection_or_raise"
_PATCH_ENGINE = "src.mtdata.core.forecast_tasks.forecast_engine"


def _unwrap(fn):
    """Return the original function, skipping the MCP tool wrapper."""
    return getattr(fn, "__wrapped__", fn)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_id: str = "task-abc",
    method: str = "nhits",
    data_scope: str = "EURUSD_H1",
    params_hash: str = "hash-123",
    status: str = "running",
    progress: Optional[TrainingProgress] = None,
    result: Optional[TrainedModelHandle] = None,
    error: Optional[str] = None,
) -> SimpleNamespace:
    """Create a task-like object matching TaskManager.get_status() return."""
    return SimpleNamespace(
        task_id=task_id,
        method=method,
        data_scope=data_scope,
        params_hash=params_hash,
        status=status,
        progress=progress,
        result=result,
        error=error,
        created_at=1000.0,
        started_at=1001.0,
        completed_at=None if status != "completed" else 1060.0,
    )


# ---------------------------------------------------------------------------
# Tests: forecast_task_status
# ---------------------------------------------------------------------------

class TestForecastTaskStatus:

    def test_returns_task_info(self):
        from src.mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="running",
            progress=TrainingProgress(step=50, total_steps=100, loss=0.05),
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            req = ForecastTaskStatusRequest(task_id="task-abc")
            result = _unwrap(forecast_task_status)(req)

        assert result["detail"] == "compact"
        assert result["task_id"] == "task-abc"
        assert result["status"] == "running"
        assert result["progress"]["step"] == 50
        assert result["progress"]["fraction"] == 0.5
        assert result["progress"]["loss"] == 0.05
        assert "metrics" not in result["progress"]
        assert "params_hash" not in result

    def test_returns_error_when_not_found(self):
        from src.mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = None

        with patch(_PATCH_TM, return_value=mock_tm):
            req = ForecastTaskStatusRequest(task_id="nonexistent")
            result = _unwrap(forecast_task_status)(req)

        assert result["detail"] == "compact"
        assert "error" in result

    def test_completed_task_includes_model_id(self):
        from src.mtdata.core.forecast_tasks import forecast_task_status

        handle = TrainedModelHandle(
            model_id="nhits/EURUSD_H1/abc",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1060.0,
        )
        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="completed",
            result=handle,
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            req = ForecastTaskStatusRequest(task_id="task-abc")
            result = _unwrap(forecast_task_status)(req)

        assert result["status"] == "completed"
        assert result["model_id"] == "nhits/EURUSD_H1/abc"
        assert "this model automatically" in result["message"]
        assert "result" not in result

    def test_failed_task_includes_error(self):
        from src.mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="failed",
            error="CUDA out of memory",
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            req = ForecastTaskStatusRequest(task_id="task-abc")
            result = _unwrap(forecast_task_status)(req)

        assert result["status"] == "failed"
        assert result["error"] == "CUDA out of memory"

    def test_full_detail_includes_result_metadata(self):
        from src.mtdata.core.forecast_tasks import forecast_task_status

        handle = TrainedModelHandle(
            model_id="nhits/EURUSD_H1/abc",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1060.0,
            metadata={"epochs": 12},
            store_metadata={
                "metadata_version": 1,
                "compatibility_version": 1,
                "last_used": 1065.0,
            },
        )
        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="completed",
            progress=TrainingProgress(
                step=50,
                total_steps=100,
                loss=0.05,
                metrics={"rmse": 0.1},
                eta_seconds=30.0,
                message="Halfway there",
            ),
            result=handle,
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            req = ForecastTaskStatusRequest(task_id="task-abc", detail="full")
            result = _unwrap(forecast_task_status)(req)

        assert result["detail"] == "full"
        assert result["params_hash"] == "hash-123"
        assert result["progress"]["metrics"] == {"rmse": 0.1}
        assert result["progress"]["eta_seconds"] == 30.0
        assert result["progress"]["message"] == "Halfway there"
        assert result["result"]["metadata"] == {"epochs": 12}
        assert result["result"]["store_metadata"] == {
            "metadata_version": 1,
            "compatibility_version": 1,
            "last_used": 1065.0,
        }
        assert result["result"]["compatibility"]["status"] == "ok"
        assert result["result"]["compatibility"]["warnings"] == []
        assert result["result"]["compatibility"]["expected"] == {
            "metadata_version": 1,
            "compatibility_version": 1,
        }
        assert result["result"]["compatibility"]["actual"] == {
            "metadata_version": 1,
            "compatibility_version": 1,
        }
        assert result["result"]["params_hash"] == "abc"


# ---------------------------------------------------------------------------
# Tests: forecast_task_cancel
# ---------------------------------------------------------------------------

class TestForecastTaskCancel:

    def test_successful_cancel(self):
        from src.mtdata.core.forecast_tasks import forecast_task_cancel

        mock_tm = MagicMock()
        mock_tm.cancel.return_value = True

        with patch(_PATCH_TM, return_value=mock_tm):
            req = ForecastTaskCancelRequest(task_id="task-abc")
            result = _unwrap(forecast_task_cancel)(req)

        assert result["status"] == "cancelled"

    def test_cancel_nonexistent(self):
        from src.mtdata.core.forecast_tasks import forecast_task_cancel

        mock_tm = MagicMock()
        mock_tm.cancel.return_value = False

        with patch(_PATCH_TM, return_value=mock_tm):
            req = ForecastTaskCancelRequest(task_id="nope")
            result = _unwrap(forecast_task_cancel)(req)

        assert result["status"] == "not_cancelled"


# ---------------------------------------------------------------------------
# Tests: forecast_task_list
# ---------------------------------------------------------------------------

class TestForecastTaskList:

    def test_lists_tasks(self):
        from src.mtdata.core.forecast_tasks import forecast_task_list

        tasks = [
            _make_task("t1", status="running", progress=TrainingProgress(step=10, total_steps=100)),
            _make_task("t2", status="completed", result=TrainedModelHandle(
                model_id="nhits/EURUSD_H1/x", method="nhits",
                data_scope="EURUSD_H1", params_hash="x", created_at=1000.0,
            )),
        ]
        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = tasks

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_list)()

        assert result["detail"] == "compact"
        assert result["count"] == 2
        assert result["tasks"][0]["task_id"] == "t1"
        assert result["tasks"][0]["progress_fraction"] == 0.1
        assert result["tasks"][1]["model_id"] == "nhits/EURUSD_H1/x"
        assert "progress" not in result["tasks"][0]
        assert "result" not in result["tasks"][1]
        mock_tm.list_tasks.assert_called_once_with(status=None)

    def test_empty_list(self):
        from src.mtdata.core.forecast_tasks import forecast_task_list

        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = []

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_list)()

        assert result["count"] == 0
        assert result["tasks"] == []

    def test_passes_status_filter_through(self):
        from src.mtdata.core.forecast_tasks import forecast_task_list

        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = []

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_list)(status_filter="running")

        assert result["count"] == 0
        mock_tm.list_tasks.assert_called_once_with(status="running")

    def test_full_detail_expands_progress_and_result_fields(self):
        from src.mtdata.core.forecast_tasks import forecast_task_list

        tasks = [
            _make_task(
                "t1",
                status="running",
                progress=TrainingProgress(
                    step=10,
                    total_steps=100,
                    loss=0.2,
                    metrics={"rmse": 0.3},
                    eta_seconds=12.0,
                    message="training",
                ),
            ),
            _make_task(
                "t2",
                status="completed",
                result=TrainedModelHandle(
                    model_id="nhits/EURUSD_H1/x",
                    method="nhits",
                    data_scope="EURUSD_H1",
                    params_hash="x",
                    created_at=1000.0,
                    metadata={"epochs": 20},
                    store_metadata={
                        "metadata_version": 1,
                        "compatibility_version": 1,
                        "last_used": 1010.0,
                    },
                ),
            ),
        ]
        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = tasks

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_list)(detail="full")

        assert result["detail"] == "full"
        assert result["tasks"][0]["params_hash"] == "hash-123"
        assert result["tasks"][0]["progress"]["metrics"] == {"rmse": 0.3}
        assert result["tasks"][1]["result"]["metadata"] == {"epochs": 20}
        assert result["tasks"][1]["result"]["store_metadata"] == {
            "metadata_version": 1,
            "compatibility_version": 1,
            "last_used": 1010.0,
        }
        assert result["tasks"][1]["result"]["compatibility"]["status"] == "ok"
        assert result["tasks"][1]["result"]["model_id"] == "nhits/EURUSD_H1/x"


# ---------------------------------------------------------------------------
# Tests: forecast_models_list
# ---------------------------------------------------------------------------

class TestForecastModelsList:

    def test_lists_models(self):
        from src.mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle("nhits/EURUSD_H1/a", "nhits", "EURUSD_H1", "a", 1000.0),
            TrainedModelHandle("tft/GBPUSD_H4/b", "tft", "GBPUSD_H4", "b", 2000.0),
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)()

        assert result["success"] is True
        assert result["detail"] == "compact"
        assert result["count"] == 2
        assert result["models"][0]["model_id"] == "nhits/EURUSD_H1/a"
        assert result["models"][1]["method"] == "tft"
        assert "metadata" not in result["models"][0]
        assert "compatibility" not in result["models"][0]

    def test_filter_by_method(self):
        from src.mtdata.core.forecast_tasks import forecast_models_list

        mock_store = MagicMock()
        mock_store.list_models.return_value = []

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)(method="nhits")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["models"] == []
        mock_store.list_models.assert_called_once_with(method="nhits")

    def test_full_detail_includes_model_metadata(self):
        from src.mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle(
                "nhits/EURUSD_H1/a",
                "nhits",
                "EURUSD_H1",
                "a",
                1000.0,
                metadata={"epochs": 42},
                store_metadata={
                    "metadata_version": 1,
                    "compatibility_version": 1,
                    "last_used": 1012.0,
                },
            ),
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)(detail="full")

        assert result["detail"] == "full"
        assert result["models"][0]["metadata"] == {"epochs": 42}
        assert result["models"][0]["store_metadata"] == {
            "metadata_version": 1,
            "compatibility_version": 1,
            "last_used": 1012.0,
        }
        assert result["models"][0]["compatibility"]["status"] == "ok"
        assert result["models"][0]["compatibility"]["warnings"] == []
        assert result["models"][0]["compatibility"]["actual"] == {
            "metadata_version": 1,
            "compatibility_version": 1,
        }

    def test_full_detail_includes_legacy_compatibility_warning(self):
        from src.mtdata.core.forecast_tasks import forecast_models_list

        handles = [
            TrainedModelHandle(
                "nhits/EURUSD_H1/a",
                "nhits",
                "EURUSD_H1",
                "a",
                1000.0,
                metadata={"epochs": 42},
                store_metadata={
                    "metadata_version": 1,
                    "compatibility_version": 1,
                    "last_used": 1012.0,
                    "_store_metadata_source": "legacy_backfill",
                    "_actual_metadata_version": None,
                    "_actual_compatibility_version": None,
                },
            ),
        ]
        mock_store = MagicMock()
        mock_store.list_models.return_value = handles

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_list)(detail="full")

        assert result["models"][0]["store_metadata"] == {
            "metadata_version": 1,
            "compatibility_version": 1,
            "last_used": 1012.0,
        }
        assert result["models"][0]["compatibility"]["status"] == "warning"
        assert result["models"][0]["compatibility"]["actual"] == {
            "metadata_version": None,
            "compatibility_version": None,
        }
        assert "predates persisted store_metadata" in result["models"][0]["compatibility"]["warnings"][0]


# ---------------------------------------------------------------------------
# Tests: forecast_models_delete
# ---------------------------------------------------------------------------

class TestForecastModelsDelete:

    def test_delete_existing(self):
        from src.mtdata.core.forecast_tasks import forecast_models_delete

        mock_store = MagicMock()
        mock_store.delete.return_value = True

        with patch(_PATCH_STORE, return_value=mock_store):
            req = ForecastModelsDeleteRequest(model_id="nhits/EURUSD_H1/abc")
            result = _unwrap(forecast_models_delete)(req)

        assert result["deleted"] is True

    def test_delete_nonexistent(self):
        from src.mtdata.core.forecast_tasks import forecast_models_delete

        mock_store = MagicMock()
        mock_store.delete.return_value = False

        with patch(_PATCH_STORE, return_value=mock_store):
            req = ForecastModelsDeleteRequest(model_id="no/such/model")
            result = _unwrap(forecast_models_delete)(req)

        assert result["deleted"] is False


# ---------------------------------------------------------------------------
# Tests: forecast_train
# ---------------------------------------------------------------------------

class TestForecastTrain:

    def test_async_training_returns_task_info(self):
        from src.mtdata.core.forecast_tasks import forecast_train

        async_response = {
            "status": "training_started",
            "task_id": "task-train-1",
            "method": "nhits",
            "data_scope": "EURUSD_H1",
        }

        with (
            patch("src.mtdata.utils.mt5.ensure_mt5_connection_or_raise"),
            patch("src.mtdata.forecast.forecast_engine.forecast_engine", return_value=async_response) as mock_fe,
        ):
            req = ForecastTrainRequest(symbol="EURUSD", timeframe="H1", method="nhits", horizon=24)
            result = _unwrap(forecast_train)(req)

        assert result["status"] == "training_started"
        assert result["task_id"] == "task-train-1"
        mock_fe.assert_called_once()
        call_kwargs = mock_fe.call_args[1]
        assert call_kwargs["async_mode"] is True

    def test_fast_method_returns_sync_result(self):
        from src.mtdata.core.forecast_tasks import forecast_train

        sync_response = {"success": True, "forecast_price": [1.0, 2.0]}

        with (
            patch("src.mtdata.utils.mt5.ensure_mt5_connection_or_raise"),
            patch("src.mtdata.forecast.forecast_engine.forecast_engine", return_value=sync_response),
        ):
            req = ForecastTrainRequest(symbol="EURUSD", timeframe="H1", method="theta", horizon=12)
            result = _unwrap(forecast_train)(req)

        assert result["status"] == "completed_sync"
        assert "synchronously" in result["message"]


# ---------------------------------------------------------------------------
# Tests: ForecastGenerateRequest async fields
# ---------------------------------------------------------------------------

class TestForecastGenerateRequestAsync:

    def test_async_mode_field_in_schema(self):
        from src.mtdata.forecast.requests import ForecastGenerateRequest
        schema = ForecastGenerateRequest.model_json_schema()
        props = schema["properties"]
        assert "async_mode" in props
        assert "model_id" in props

    def test_defaults(self):
        from src.mtdata.forecast.requests import ForecastGenerateRequest
        req = ForecastGenerateRequest(symbol="X", timeframe="H1", method="theta")
        assert req.async_mode is False
        assert req.model_id is None


class TestForecastTaskStatusRequestSchema:

    def test_detail_field_in_schema(self):
        schema = ForecastTaskStatusRequest.model_json_schema()
        props = schema["properties"]
        assert "detail" in props
        assert props["detail"]["default"] == "compact"


# ---------------------------------------------------------------------------
# Tests: Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:

    def test_new_tools_registered(self):
        from src.mtdata.bootstrap.tools import bootstrap_tools
        from src.mtdata.core._mcp_instance import mcp

        bootstrap_tools()
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}

        expected_new = {
            "forecast_train",
            "forecast_task_status",
            "forecast_task_cancel",
            "forecast_task_list",
            "forecast_models_list",
            "forecast_models_delete",
        }
        missing = expected_new - tool_names
        assert not missing, f"Missing tools: {sorted(missing)}"
