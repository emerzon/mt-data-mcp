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

import src.mtdata.forecast.forecast_engine as fe
from src.mtdata.forecast.interface import (
    ForecastMethod,
    ForecastResult,
    TrainedModelHandle,
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
    ):
        self._category = category
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
        return b"fake_artifact"

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


# ---------------------------------------------------------------------------
# Tests: _try_predict_with_stored_model
# ---------------------------------------------------------------------------

class TestTryPredictWithStoredModel:

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

    def test_trainable_no_stored_model_falls_to_sync(self):
        """Trainable method with no stored model, sync mode → falls through to forecast()."""
        stub = _StubTrainable(category="heavy")

        class FakeReg:
            @staticmethod
            def get(name):
                return stub

        mock_store = MagicMock()
        mock_store.find.return_value = None

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._run_registered_forecast_method(**_common_call_kwargs())

        forecast_arr, ci, metadata = result
        # Falls through to forecast() because async_mode=False
        assert metadata.get("src") == "forecast"

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

    def test_model_id_overrides_computed_hash(self):
        """When model_id is given, it is used as params_hash for store lookup."""
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
        )
        mock_store = MagicMock()
        mock_store.find.return_value = handle
        mock_store.load_bytes.return_value = b"artifact"

        with patch.object(fe, "ForecastRegistry", FakeReg), \
             patch(_PATCH_MODEL_STORE, mock_store):
            result = fe._run_registered_forecast_method(
                **_common_call_kwargs(model_id="custom_id"),
            )

        # Should have used custom_id for lookup
        mock_store.find.assert_called_once_with("stub_trainable", "EURUSD_H1", "custom_id")
        forecast_arr, ci, metadata = result
        assert metadata["model_info"]["model_id"] == "stub_trainable/EURUSD_H1/custom_id"


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

    def test_async_mode_can_be_set(self):
        from src.mtdata.forecast.requests import ForecastGenerateRequest
        req = ForecastGenerateRequest(
            symbol="EURUSD", timeframe="H1", method="nhits",
            async_mode=True, model_id="nhits/EURUSD_H1/xyz",
        )
        assert req.async_mode is True
        assert req.model_id == "nhits/EURUSD_H1/xyz"
