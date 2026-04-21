"""Tests for forecast task MCP tool handlers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.mtdata.core.forecast_tasks import (
    ForecastModelsDeleteRequest,
    ForecastTaskCancelRequest,
    ForecastTaskStatusRequest,
    ForecastTaskWaitRequest,
    ForecastTrainRequest,
)
from src.mtdata.forecast.interface import TrainedModelHandle, TrainingProgress

_PATCH_TM = "src.mtdata.core.forecast_tasks._get_task_manager"
_PATCH_STORE = "src.mtdata.core.forecast_tasks._get_model_store"


def _unwrap(fn):
    return getattr(fn, "__wrapped__", fn)


def _make_task(
    task_id: str = "task-abc",
    method: str = "nhits",
    data_scope: str = "EURUSD_H1",
    params_hash: str = "hash-123",
    status: str = "running",
    progress: Optional[TrainingProgress] = None,
    result: Optional[TrainedModelHandle] = None,
    error: Optional[str] = None,
    cancel_requested: bool = False,
):
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
        heartbeat_at=1002.0,
        pid=4321,
        cancel_requested=cancel_requested,
    )


class TestForecastTaskStatus:
    def test_returns_task_info(self):
        from src.mtdata.core.forecast_tasks import forecast_task_status

        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(
            status="running",
            progress=TrainingProgress(step=50, total_steps=100, loss=0.05),
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(ForecastTaskStatusRequest(task_id="task-abc"))

        assert result["detail"] == "compact"
        assert result["task_id"] == "task-abc"
        assert result["status"] == "running"
        assert result["progress"]["fraction"] == 0.5
        assert result["pid"] == 4321
        assert result["cancel_requested"] is False

    def test_completed_task_includes_model(self):
        from src.mtdata.core.forecast_tasks import forecast_task_status

        handle = TrainedModelHandle(
            model_id="nhits/EURUSD_H1/abc",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1060.0,
        )
        mock_tm = MagicMock()
        mock_tm.get_status.return_value = _make_task(status="completed", result=handle)

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(ForecastTaskStatusRequest(task_id="task-abc"))

        assert result["status"] == "completed"
        assert result["model_id"] == "nhits/EURUSD_H1/abc"
        assert "result" not in result

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
            cancel_requested=True,
        )

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_status)(ForecastTaskStatusRequest(task_id="task-abc", detail="full"))

        assert result["detail"] == "full"
        assert result["params_hash"] == "hash-123"
        assert result["progress"]["metrics"] == {"rmse": 0.1}
        assert result["result"]["metadata"] == {"epochs": 12}
        assert result["cancel_requested"] is True


class TestForecastTaskCancel:
    def test_successful_cancel(self):
        from src.mtdata.core.forecast_tasks import forecast_task_cancel

        mock_tm = MagicMock()
        mock_tm.cancel.return_value = {
            "task_id": "task-abc",
            "cancel_requested": True,
            "terminated": False,
            "status": "cancelling",
        }

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_cancel)(ForecastTaskCancelRequest(task_id="task-abc"))

        assert result["cancel_requested"] is True
        assert result["status"] == "cancelling"

    def test_cancel_nonexistent(self):
        from src.mtdata.core.forecast_tasks import forecast_task_cancel

        mock_tm = MagicMock()
        mock_tm.cancel.return_value = {
            "task_id": "nope",
            "cancel_requested": False,
            "terminated": False,
            "status": "not_found",
        }

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_cancel)(ForecastTaskCancelRequest(task_id="nope"))

        assert result["status"] == "not_found"


class TestForecastTaskWait:
    def test_wait_returns_latest_status(self):
        from src.mtdata.core.forecast_tasks import forecast_task_wait

        mock_tm = MagicMock()
        mock_tm.wait_for_status.return_value = _make_task(status="completed")

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_wait)(ForecastTaskWaitRequest(task_id="task-abc", timeout_seconds=10.0))

        assert result["status"] == "completed"
        assert result["wait_timeout_seconds"] == 10.0


class TestForecastTaskList:
    def test_lists_tasks(self):
        from src.mtdata.core.forecast_tasks import forecast_task_list

        tasks = [
            _make_task("t1", status="running", progress=TrainingProgress(step=10, total_steps=100)),
            _make_task(
                "t2",
                status="completed",
                result=TrainedModelHandle(
                    model_id="nhits/EURUSD_H1/x",
                    method="nhits",
                    data_scope="EURUSD_H1",
                    params_hash="x",
                    created_at=1000.0,
                ),
            ),
        ]
        mock_tm = MagicMock()
        mock_tm.list_tasks.return_value = tasks

        with patch(_PATCH_TM, return_value=mock_tm):
            result = _unwrap(forecast_task_list)()

        assert result["count"] == 2
        assert result["tasks"][0]["progress_fraction"] == 0.1
        assert result["tasks"][0]["pid"] == 4321
        assert result["tasks"][1]["model_id"] == "nhits/EURUSD_H1/x"


class TestForecastModels:
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
        assert result["count"] == 2
        assert result["models"][0]["model_id"] == "nhits/EURUSD_H1/a"

    def test_delete_existing(self):
        from src.mtdata.core.forecast_tasks import forecast_models_delete

        mock_store = MagicMock()
        mock_store.delete.return_value = True

        with patch(_PATCH_STORE, return_value=mock_store):
            result = _unwrap(forecast_models_delete)(ForecastModelsDeleteRequest(model_id="nhits/EURUSD_H1/abc"))

        assert result["deleted"] is True


class TestForecastTrain:
    def test_training_returns_task_snapshot(self):
        from src.mtdata.core.forecast_tasks import forecast_train

        task = _make_task(status="pending")
        mock_tm = MagicMock()
        mock_tm.submit_forecast_request.return_value = ("task-train-1", True)
        mock_tm.get_status.return_value = task

        with (
            patch(_PATCH_TM, return_value=mock_tm),
            patch("src.mtdata.utils.mt5.ensure_mt5_connection_or_raise"),
        ):
            result = _unwrap(forecast_train)(ForecastTrainRequest(symbol="EURUSD", timeframe="H1", method="nhits", horizon=24))

        assert result["status"] == "pending"
        assert result["task_id"] == "task-abc"
        mock_tm.submit_forecast_request.assert_called_once()


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


class TestToolRegistration:
    def test_new_tools_registered(self):
        from src.mtdata.bootstrap.tools import bootstrap_tools
        from src.mtdata.core._mcp_instance import mcp

        bootstrap_tools()
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}

        expected = {
            "forecast_train",
            "forecast_task_status",
            "forecast_task_cancel",
            "forecast_task_wait",
            "forecast_task_list",
            "forecast_models_list",
            "forecast_models_delete",
        }
        missing = expected - tool_names
        assert not missing, f"Missing tools: {sorted(missing)}"
