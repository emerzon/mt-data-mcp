"""Tests for Phase 2 engine integration: model store lookup, async routing,
_compute_model_key, _try_predict_with_stored_model, and _submit_async_training.

These test the wiring added to forecast_engine.py without needing real ML
libraries — all heavy deps are mocked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import src.mtdata.forecast.forecast_engine as fe
from src.mtdata.forecast.interface import (
    ArtifactCompatibilityError,
    ForecastMethod,
    ForecastResult,
    TrainedModelHandle,
    TrainResult,
)

# Canonical patch targets for lazy imports inside forecast_engine helpers.
_PATCH_MODEL_STORE = "src.mtdata.forecast.model_store.model_store"
_PATCH_GET_TM = "src.mtdata.forecast.task_manager.get_task_manager"


# ---------------------------------------------------------------------------
# Helpers — minimal ForecastMethod subclass for testing
# ---------------------------------------------------------------------------

class _StubTrainable(ForecastMethod):
    """Trainable method stub that returns canned results."""

    name = "stub_trainable"

    def __init__(
        self,
        *,
        category: str = "heavy",
        predict_result: Optional[ForecastResult] = None,
        live_model_update: bool = False,
    ):
        self._category = category
        self._live_model_update = live_model_update
        self._predict_result = predict_result or ForecastResult(
            forecast=np.array([10.0, 11.0, 12.0]),
            ci_values=(np.array([9.0, 10.0, 11.0]), np.array([11.0, 12.0, 13.0])),
            params_used={"method": "stub_trainable"},
            metadata={"src": "predict_with_model"},
        )

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def supports_live_model_update(self) -> bool:
        return self._live_model_update

    @property
    def training_category(self) -> str:
        return self._category

    def forecast(self, series, horizon, seasonality, params, **kw) -> ForecastResult:
        return ForecastResult(
            forecast=np.array([1.0] * horizon),
            ci_values=None,
            params_used=params,
            metadata={"src": "forecast"},
        )

    def predict_with_model(self, model, series, horizon, seasonality, params, **kw) -> ForecastResult:
        return self._predict_result

    def train(self, series, horizon, seasonality, params, **kw):
        return TrainResult(
            artifact_bytes=b"fake_artifact",
            params_used=dict(params),
            metadata={"src": "train"},
        )

    def serialize_artifact(self, artifact):
        return artifact if isinstance(artifact, bytes) else b"serialized"

    def deserialize_artifact(self, raw: bytes):
        return raw


class _StubNonTrainable:
    """Bare class (not ForecastMethod subclass) — simulates old-style mock."""

    def forecast(self, series, horizon, seasonality, params, **kw):
        return ForecastResult(
            forecast=np.array([1.0] * horizon),
            ci_values=None,
            params_used=params,
            metadata={"src": "non_trainable"},
        )


def _sample_series(n: int = 100) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.standard_normal(n), name="close")


def _sample_df(n: int = 100) -> pd.DataFrame:
    t0 = 1_700_100_000
    times = np.arange(t0, t0 + n * 3600, 3600, dtype=float)
    rng = np.random.default_rng(42)
    close = np.linspace(100.0, 105.0, n, dtype=float)
    return pd.DataFrame({
        "time": times,
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.4,
        "close": close,
        "volume": np.linspace(1000.0, 1200.0, n),
    })


# ---------------------------------------------------------------------------
# Tests: _compute_model_key
# ---------------------------------------------------------------------------

class TestComputeModelKey:

    def test_returns_stable_hash(self):
        stub = _StubTrainable()
        key1 = fe._compute_model_key(stub, "stub_trainable", 10, 24, {"lr": 0.01}, "H1", False)
        key2 = fe._compute_model_key(stub, "stub_trainable", 10, 24, {"lr": 0.01}, "H1", False)
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 8

    def test_different_params_different_hash(self):
        stub = _StubTrainable()
        key1 = fe._compute_model_key(stub, "stub_trainable", 10, 24, {"lr": 0.01}, "H1", False)
        key2 = fe._compute_model_key(stub, "stub_trainable", 10, 24, {"lr": 0.1}, "H1", False)
        assert key1 != key2

    def test_different_horizon_different_hash(self):
        stub = _StubTrainable()
        key1 = fe._compute_model_key(stub, "stub_trainable", 10, 24, {}, "H1", False)
        key2 = fe._compute_model_key(stub, "stub_trainable", 20, 24, {}, "H1", False)
        assert key1 != key2

    def test_exog_flag_affects_hash(self):
        stub = _StubTrainable()
        key_no = fe._compute_model_key(stub, "stub_trainable", 10, 24, {}, "H1", False)
        key_yes = fe._compute_model_key(stub, "stub_trainable", 10, 24, {}, "H1", True)
        assert key_no != key_yes

    def test_pipeline_context_affects_hash(self):
        stub = _StubTrainable()
        first = fe._compute_model_key(
            stub,
            "stub_trainable",
            10,
            24,
            {"_training_context": {"target_points": 500, "denoise": {"method": "ema"}}},
            "H1",
            False,
        )
        second = fe._compute_model_key(
            stub,
            "stub_trainable",
            10,
            24,
            {"_training_context": {"target_points": 2000, "denoise": None}},
            "H1",
            False,
        )

        assert first != second

    def test_training_window_does_not_affect_hash(self):
        stub = _StubTrainable()
        context = {
            "base_col": "close",
            "denoise": None,
            "features": None,
            "target_spec": None,
            "exog_columns": 0,
        }
        first = fe._compute_model_key(
            stub,
            "stub_trainable",
            10,
            24,
            {
                "_training_context": {
                    **context,
                    "target_points": 500,
                    "history_start_epoch": 1000.0,
                    "training_end_epoch": 2000.0,
                }
            },
            "H1",
            False,
        )
        second = fe._compute_model_key(
            stub,
            "stub_trainable",
            10,
            24,
            {
                "_training_context": {
                    **context,
                    "target_points": 501,
                    "history_start_epoch": 1001.0,
                    "training_end_epoch": 5600.0,
                }
            },
            "H1",
            False,
        )

        assert first == second


# ---------------------------------------------------------------------------
# Tests: _try_predict_with_stored_model
# ---------------------------------------------------------------------------

class TestTryPredictWithStoredModel:

    def test_historical_prediction_rejects_different_training_anchor(self):
        stub = _StubTrainable()
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
            metadata={"training_context": {"training_end_epoch": 1000.0}},
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub,
                "stub_trainable",
                "EURUSD_H1",
                "abc123",
                _sample_series(),
                3,
                24,
                {},
                None,
                {},
                current_anchor_epoch=2000.0,
                require_exact_anchor=True,
            )

        assert result is None
        mock_store.load_bytes.assert_not_called()

    def test_live_prediction_reuses_older_model_and_reports_staleness(self):
        stub = _StubTrainable()
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
            metadata={"training_context": {"training_end_epoch": 1000.0}},
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"fake_artifact"

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub,
                "stub_trainable",
                "EURUSD_H1",
                "abc123",
                _sample_series(),
                3,
                24,
                {},
                None,
                {},
                current_anchor_epoch=8200.0,
                timeframe_seconds=3600,
            )

        assert result is not None
        assert result[2]["model_info"]["model_staleness_bars"] == 2.0
        assert result[2]["model_info"]["reuse_policy"] == "live_latest_artifact"

    def test_live_prediction_rejects_model_past_staleness_limit(self):
        stub = _StubTrainable()
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
            metadata={"training_context": {"training_end_epoch": 1000.0}},
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        rejection = {}

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub,
                "stub_trainable",
                "EURUSD_H1",
                "abc123",
                _sample_series(),
                3,
                24,
                {},
                None,
                {},
                current_anchor_epoch=1000.0 + 25 * 3600,
                timeframe_seconds=3600,
                max_staleness_bars=24,
                rejection=rejection,
            )

        assert result is None
        assert rejection["reason"] == "model_staleness_limit_exceeded"
        mock_store.load_bytes.assert_not_called()

    def test_returns_none_when_no_model(self):
        stub = _StubTrainable()
        mock_store = MagicMock()
        mock_store.find.return_value = None

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub, "stub_trainable", "EURUSD_H1", "abc123",
                _sample_series(), 3, 24, {}, None, {},
            )
        assert result is None

    def test_returns_prediction_when_model_exists(self):
        stub = _StubTrainable()
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"fake_artifact"

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub, "stub_trainable", "EURUSD_H1", "abc123",
                _sample_series(), 3, 24, {}, None, {},
            )

        assert result is not None
        forecast_arr, ci, metadata = result
        np.testing.assert_array_equal(forecast_arr, np.array([10.0, 11.0, 12.0]))
        assert ci is not None
        assert metadata["model_info"]["model_id"] == "stub_trainable/EURUSD_H1/abc123"
        assert metadata["model_info"]["source"] == "model_store"
        mock_store.mark_used.assert_called_once_with(handle.model_id)

    def test_failed_deserialization_does_not_renew_model_ttl(self):
        stub = _StubTrainable()
        stub.deserialize_artifact = MagicMock(side_effect=ValueError("corrupt artifact"))
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"corrupt"

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub, "stub_trainable", "EURUSD_H1", "abc123",
                _sample_series(), 3, 24, {}, None, {},
            )

        assert result is None
        mock_store.mark_used.assert_not_called()

    def test_incompatible_artifact_is_visible_and_rejected(self, caplog):
        stub = _StubTrainable()
        stub.deserialize_artifact = MagicMock(
            side_effect=ArtifactCompatibilityError("numpy version changed")
        )
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"incompatible"
        rejection = {}

        with (
            patch(_PATCH_MODEL_STORE, mock_store),
            caplog.at_level("WARNING", logger=fe.__name__),
        ):
            result = fe._try_predict_with_stored_model(
                stub,
                "stub_trainable",
                "EURUSD_H1",
                "abc123",
                _sample_series(),
                3,
                24,
                {},
                None,
                {},
                rejection=rejection,
            )

        assert result is None
        assert rejection["reason"] == "artifact_runtime_incompatible"
        assert "numpy version changed" in rejection["message"]
        assert "Stored model rejected" in caplog.text
        mock_store.mark_used.assert_not_called()

    def test_surfaces_legacy_compatibility_warning_when_model_exists(self):
        stub = _StubTrainable()
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
            store_metadata={
                "metadata_version": 1,
                "compatibility_version": 1,
                "last_used": 1000.0,
                "_store_metadata_source": "legacy_backfill",
                "_actual_metadata_version": None,
                "_actual_compatibility_version": None,
            },
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"fake_artifact"

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub, "stub_trainable", "EURUSD_H1", "abc123",
                _sample_series(), 3, 24, {}, None, {},
            )

        assert result is not None
        forecast_arr, ci, metadata = result
        np.testing.assert_array_equal(forecast_arr, np.array([10.0, 11.0, 12.0]))
        assert ci is not None
        assert metadata["model_info"]["compatibility"]["status"] == "warning"
        assert "predates persisted store_metadata" in metadata["warnings"][0]

    def test_returns_none_on_exception(self):
        stub = _StubTrainable()
        mock_store = MagicMock()
        mock_store.find.side_effect = RuntimeError("disk error")

        with patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._try_predict_with_stored_model(
                stub, "stub_trainable", "EURUSD_H1", "abc123",
                _sample_series(), 3, 24, {}, None, {},
            )
        assert result is None

    def test_load_bytes_called_with_model_id(self):
        """Regression test: load_bytes should receive handle.model_id, not 3 separate args."""
        stub = _StubTrainable()
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc123",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc123",
            created_at=1000.0,
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"fake"

        with patch(_PATCH_MODEL_STORE, mock_store):
            fe._try_predict_with_stored_model(
                stub, "stub_trainable", "EURUSD_H1", "abc123",
                _sample_series(), 3, 24, {}, None, {},
            )

        mock_store.load_bytes.assert_called_once_with("stub_trainable/EURUSD_H1/abc123")


# ---------------------------------------------------------------------------
# Tests: _submit_async_training
# ---------------------------------------------------------------------------

class TestSubmitAsyncTraining:

    def test_returns_async_response(self):
        stub = _StubTrainable(category="heavy")
        mock_tm = MagicMock()
        mock_tm.submit.return_value = ("task-uuid-1", True)

        with patch(_PATCH_GET_TM, return_value=mock_tm):
            resp = fe._submit_async_training(
                stub, "nhits", _sample_series(), 10, 24, {"lr": 0.01},
                "EURUSD_H1", "hash123", "H1", None,
            )

        assert resp["status"] == "pending"
        assert resp["task_id"] == "task-uuid-1"
        assert resp["method"] == "nhits"
        assert resp["data_scope"] == "EURUSD_H1"
        assert "1-10 minutes" in resp["estimated_duration"]

    def test_returns_in_progress_when_not_new(self):
        stub = _StubTrainable(category="moderate")
        mock_tm = MagicMock()
        mock_tm.submit.return_value = ("task-uuid-2", False)

        with patch(_PATCH_GET_TM, return_value=mock_tm):
            resp = fe._submit_async_training(
                stub, "mlforecast", _sample_series(), 5, 12, {},
                "GBPUSD_M15", "hash456", "M15", None,
            )

        assert resp["status"] == "running"


# ---------------------------------------------------------------------------
# Tests: _AsyncTrainingStarted exception
# ---------------------------------------------------------------------------

class TestAsyncTrainingStarted:

    def test_carries_response_dict(self):
        resp = {"task_id": "abc", "status": "pending"}
        exc = fe._AsyncTrainingStarted(resp)
        assert exc.response is resp
        assert "async training started" in str(exc)


# ---------------------------------------------------------------------------
# Tests: _run_registered_forecast_method integration
# ---------------------------------------------------------------------------

def _common_call_kwargs(
    method_l: str = "stub_trainable",
    horizon: int = 3,
    async_mode: bool = False,
    model_id: Optional[str] = None,
    model_cache: str = "reuse",
) -> dict:
    """Build kwargs for _run_registered_forecast_method."""
    df = _sample_df(100)
    return dict(
        method_l=method_l,
        method=method_l,
        df=df,
        target_series=pd.Series(df["close"].values, name="close"),
        horizon=horizon,
        seasonality=24,
        params={},
        ci_alpha=None,
        as_of=None,
        quantity_l="price",
        symbol="EURUSD",
        timeframe="H1",
        base_col="close",
        denoise_spec_used=None,
        X=None,
        future_exog=None,
        async_mode=async_mode,
        model_id=model_id,
        model_cache=model_cache,
    )


class TestRunRegisteredForecastMethodIntegration:

    def test_non_trainable_mock_skips_model_store(self):
        """Non-ForecastMethod objects (old mocks) bypass model store entirely."""
        non_trainable = _StubNonTrainable()

        class FakeReg:
            @staticmethod
            def get(name):
                return non_trainable

        with patch.object(fe, "ForecastRegistry", FakeReg):
            result = fe._run_registered_forecast_method(**_common_call_kwargs())

        forecast_arr, ci, metadata = result
        np.testing.assert_array_equal(forecast_arr, np.array([1.0, 1.0, 1.0]))
        assert metadata.get("src") == "non_trainable"

    def test_model_id_rejects_non_trainable_method(self):
        non_trainable = _StubNonTrainable()

        class FakeReg:
            @staticmethod
            def get(name):
                return non_trainable

        with patch.object(fe, "ForecastRegistry", FakeReg):
            with pytest.raises(ValueError, match="does not support stored model prediction"):
                fe._run_registered_forecast_method(
                    **_common_call_kwargs(
                        model_id="stub_trainable/EURUSD_H1/missing_model"
                    ),
                )

    def test_async_mode_rejects_non_trainable_method(self):
        non_trainable = _StubNonTrainable()

        class FakeReg:
            @staticmethod
            def get(name):
                return non_trainable

        with patch.object(fe, "ForecastRegistry", FakeReg):
            with pytest.raises(ValueError, match="async_mode is not supported"):
                fe._run_registered_forecast_method(
                    **_common_call_kwargs(async_mode=True)
                )

    def test_trainable_no_stored_model_trains_and_persists_synchronously(self):
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None
        mock_store.save.return_value = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/new",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="new",
            created_at=1000.0,
        )

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._run_registered_forecast_method(**_common_call_kwargs())

        forecast_arr, ci, metadata = result
        assert metadata.get("src") == "predict_with_model"
        assert metadata["model_info"]["source"] == "synchronous_training"
        mock_store.save.assert_called_once()

    def test_ephemeral_trainable_forecast_skips_model_store(self):
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._run_registered_forecast_method(
                **_common_call_kwargs(model_cache="ephemeral")
            )

        _forecast_arr, _ci, metadata = result
        assert metadata["model_info"] == {
            "model_id": None,
            "data_scope": "EURUSD_H1",
            "source": "ephemeral_training",
            "reuse_policy": "not_persisted",
            "training_end_epoch": float(_sample_df(100)["time"].iloc[-1]),
            "model_staleness_seconds": 0.0,
            "model_staleness_bars": 0.0,
        }
        mock_store.find.assert_not_called()
        mock_store.save.assert_not_called()

    def test_require_existing_rejects_cache_miss_without_training(self):
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None
        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store), \
             patch.object(stub, "train", wraps=stub.train) as train_mock, \
             pytest.raises(ValueError, match="require_existing.*no compatible"):
            fe._run_registered_forecast_method(
                **_common_call_kwargs(model_cache="require_existing")
            )

        mock_store.find.assert_called_once()
        mock_store.save.assert_not_called()
        train_mock.assert_not_called()

    def test_explicit_model_id_missing_raises(self):
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch.object(fe, "_compute_model_key", return_value="missing_model"), \
             patch(_PATCH_MODEL_STORE, mock_store):
            with pytest.raises(ValueError, match="was not found in the model store"):
                fe._run_registered_forecast_method(
                    **_common_call_kwargs(
                        model_id="stub_trainable/EURUSD_H1/missing_model"
                    ),
                )

        mock_store.find.assert_called_once_with(
            "stub_trainable",
            "EURUSD_H1",
            "missing_model",
        )

    def test_explicit_model_id_reports_historical_anchor_rejection(self):
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1000.0,
            metadata={"training_context": {"training_end_epoch": 1000.0}},
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        kwargs = _common_call_kwargs(
            model_id="stub_trainable/EURUSD_H1/abc"
        )
        kwargs["as_of"] = "2026-01-01T00:00:00Z"

        with patch.object(fe, "ForecastRegistry", FakeReg), patch.object(
            fe, "_compute_model_key", return_value="abc"
        ), patch(_PATCH_MODEL_STORE, mock_store):
            with pytest.raises(ValueError, match="historical_anchor_mismatch"):
                fe._run_registered_forecast_method(**kwargs)

    def test_trainable_with_stored_model_uses_predict(self):
        """Trainable method with stored model → uses predict_with_model."""
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1000.0,
            metadata={
                "training_context": {
                    "training_end_epoch": float(_sample_df(100)["time"].iloc[-1])
                }
            },
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"artifact"

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._run_registered_forecast_method(**_common_call_kwargs())

        forecast_arr, ci, metadata = result
        np.testing.assert_array_equal(forecast_arr, np.array([10.0, 11.0, 12.0]))
        assert metadata["model_info"]["source"] == "model_store"

    def test_live_refreshable_method_reuses_recent_stored_artifact(self):
        stub = _StubTrainable(live_model_update=True)

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        kwargs = _common_call_kwargs()
        current_anchor = float(kwargs["df"]["time"].iloc[-1])
        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1000.0,
            metadata={
                "training_context": {
                    "training_end_epoch": current_anchor - 3600.0,
                }
            },
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"artifact"

        with patch.object(fe, "ForecastRegistry", FakeReg), patch(
            _PATCH_MODEL_STORE, mock_store
        ):
            _forecast, _ci, metadata = fe._run_registered_forecast_method(**kwargs)

        assert metadata["model_info"]["source"] == "model_store"
        assert metadata["model_info"]["reuse_policy"] == "live_latest_artifact"
        assert metadata["model_info"]["model_staleness_bars"] == 1.0

    def test_async_mode_heavy_method_raises_async_started(self):
        """Heavy method with async_mode=True and no stored model → _AsyncTrainingStarted."""
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None

        mock_tm = MagicMock()
        mock_tm.submit.return_value = ("task-999", True)

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store), \
             patch(_PATCH_GET_TM, return_value=mock_tm):
            with pytest.raises(fe._AsyncTrainingStarted) as exc_info:
                fe._run_registered_forecast_method(
                    **_common_call_kwargs(async_mode=True),
                )

        assert exc_info.value.response["task_id"] == "task-999"
        assert exc_info.value.response["status"] == "pending"

    def test_async_mode_fast_method_also_submits_background_training(self):
        """Fast method with async_mode=True now also uses background training."""
        stub = _StubTrainable(category="fast")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None

        mock_tm = MagicMock()
        mock_tm.submit.return_value = ("task-fast", True)

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store), \
             patch(_PATCH_GET_TM, return_value=mock_tm):
            with pytest.raises(fe._AsyncTrainingStarted) as exc_info:
                fe._run_registered_forecast_method(
                    **_common_call_kwargs(async_mode=True),
                )

        assert exc_info.value.response["task_id"] == "task-fast"
        assert exc_info.value.response["status"] == "pending"

    def test_canonical_model_id_cannot_override_computed_hash(self):
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/custom_id",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="custom_id",
            created_at=2000.0,
            metadata={
                "training_context": {
                    "training_end_epoch": float(_sample_df(100)["time"].iloc[-1])
                }
            },
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"artifact"

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch.object(fe, "_compute_model_key", return_value="expected_id"), \
             patch(_PATCH_MODEL_STORE, mock_store):
            with pytest.raises(ValueError, match="incompatible"):
                fe._run_registered_forecast_method(
                    **_common_call_kwargs(
                        model_id="stub_trainable/EURUSD_H1/custom_id"
                    ),
                )

        mock_store.find.assert_not_called()

    def test_model_id_rejects_mismatched_scope(self):
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        with patch.object(fe, "ForecastRegistry", FakeReg):
            with pytest.raises(ValueError, match="does not match requested method"):
                fe._run_registered_forecast_method(
                    **_common_call_kwargs(
                        model_id="stub_trainable/GBPUSD_H1/custom_id"
                    ),
                )


# ---------------------------------------------------------------------------
# Tests: forecast_engine() top-level async_mode / model_id passthrough
# ---------------------------------------------------------------------------

class TestForecastEngineAsyncRouting:

    def _setup_monkeypatch(self, monkeypatch):
        """Common monkeypatches to make forecast_engine() callable in isolation."""
        monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
        monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
        monkeypatch.setattr(fe, "_get_available_methods", lambda: ("stub_trainable",))
        monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
        monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: SimpleNamespace(digits=5))

    def test_async_training_returns_task_response(self, monkeypatch):
        """forecast_engine() catches _AsyncTrainingStarted and returns task dict."""
        self._setup_monkeypatch(monkeypatch)

        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None

        mock_tm = MagicMock()
        mock_tm.submit.return_value = ("task-top-level", True)

        monkeypatch.setattr(fe, "ForecastRegistry", FakeReg)

        with patch(_PATCH_MODEL_STORE, mock_store), \
             patch(_PATCH_GET_TM, return_value=mock_tm):
            out = fe.forecast_engine(
                symbol="EURUSD",
                timeframe="H1",
                method="stub_trainable",
                horizon=3,
                async_mode=True,
                prefetched_df=_sample_df(50),
            )

        assert out["status"] == "pending"
        assert out["task_id"] == "task-top-level"

    def test_sync_fallback_when_no_model(self, monkeypatch):
        """forecast_engine() in sync mode with trainable method falls through to forecast()."""
        self._setup_monkeypatch(monkeypatch)

        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None

        monkeypatch.setattr(fe, "ForecastRegistry", FakeReg)

        with patch(_PATCH_MODEL_STORE, mock_store):
            out = fe.forecast_engine(
                symbol="EURUSD",
                timeframe="H1",
                method="stub_trainable",
                horizon=3,
                async_mode=False,
                prefetched_df=_sample_df(50),
            )

        assert out["success"] is True
        assert "forecast_price" in out

    def test_reused_model_surfaces_drifted_compatibility_warning(self, monkeypatch):
        self._setup_monkeypatch(monkeypatch)

        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        handle = TrainedModelHandle(
            model_id="stub_trainable/EURUSD_H1/abc",
            method="stub_trainable",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1000.0,
            metadata={
                "training_context": {
                    "training_end_epoch": float(_sample_df(50)["time"].iloc[-1])
                }
            },
            store_metadata={
                "metadata_version": 2,
                "compatibility_version": 3,
                "last_used": 1000.0,
            },
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"artifact"

        monkeypatch.setattr(fe, "ForecastRegistry", FakeReg)

        with patch(_PATCH_MODEL_STORE, mock_store):
            out = fe.forecast_engine(
                symbol="EURUSD",
                timeframe="H1",
                method="stub_trainable",
                horizon=3,
                async_mode=False,
                prefetched_df=_sample_df(50),
            )

        assert out["success"] is True
        assert out["model_info"]["source"] == "model_store"
        assert out["model_info"]["compatibility"]["status"] == "warning"
        assert any("metadata_version=2" in warning for warning in out["warnings"])
        assert any("compatibility_version=3" in warning for warning in out["warnings"])


# ---------------------------------------------------------------------------
# Tests: ForecastGenerateRequest new fields
# ---------------------------------------------------------------------------

class TestForecastGenerateRequestFields:

    def test_async_mode_defaults_false(self):
        from src.mtdata.forecast.requests import ForecastGenerateRequest
        req = ForecastGenerateRequest(symbol="EURUSD", timeframe="H1", method="nhits")
        assert req.async_mode is False
        assert req.model_id is None
        assert req.model_cache == "reuse"

    def test_async_mode_can_be_set(self):
        from src.mtdata.forecast.requests import ForecastGenerateRequest
        req = ForecastGenerateRequest(
            symbol="EURUSD", timeframe="H1", method="nhits",
            async_mode=True, model_id="nhits/EURUSD_H1/xyz",
        )
        assert req.async_mode is True
        assert req.model_id == "nhits/EURUSD_H1/xyz"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"model_cache": "ephemeral", "model_id": "x/y/z"}, "model_id"),
            ({"model_cache": "ephemeral", "async_mode": True}, "async_mode"),
            ({"model_cache": "require_existing", "async_mode": True}, "async_mode"),
        ],
    )
    def test_incompatible_cache_options_are_rejected(self, kwargs, message):
        from src.mtdata.forecast.requests import ForecastGenerateRequest

        with pytest.raises(ValidationError, match=message):
            ForecastGenerateRequest(symbol="EURUSD", **kwargs)
