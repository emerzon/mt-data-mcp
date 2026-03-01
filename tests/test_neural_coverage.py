"""Coverage tests for mtdata.forecast.methods.neural – targeting uncovered lines."""

import sys
import types
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Provide fake neuralforecast before any project imports ────────────────────
_nf_models = types.ModuleType("neuralforecast.models")
_nf_models.NHITS = MagicMock(name="NHITS")
_nf_models.NBEATSx = MagicMock(name="NBEATSx")
_nf_models.TFT = MagicMock(name="TFT")
_nf_models.PatchTST = MagicMock(name="PatchTST")

_nf_pkg = types.ModuleType("neuralforecast")
_nf_pkg.models = _nf_models

_orig_mt5 = sys.modules.get("MetaTrader5")
_orig_nf = sys.modules.get("neuralforecast")
_orig_nf_models = sys.modules.get("neuralforecast.models")

sys.modules.setdefault("neuralforecast", _nf_pkg)
sys.modules.setdefault("neuralforecast.models", _nf_models)

# Ensure MetaTrader5 mock exists for transitive imports
_mt5_mock = MagicMock()
_mt5_mock.TIMEFRAME_M1 = 1; _mt5_mock.TIMEFRAME_M2 = 2; _mt5_mock.TIMEFRAME_M3 = 3
_mt5_mock.TIMEFRAME_M4 = 4; _mt5_mock.TIMEFRAME_M5 = 5; _mt5_mock.TIMEFRAME_M6 = 6
_mt5_mock.TIMEFRAME_M10 = 10; _mt5_mock.TIMEFRAME_M12 = 12; _mt5_mock.TIMEFRAME_M15 = 15
_mt5_mock.TIMEFRAME_M20 = 20; _mt5_mock.TIMEFRAME_M30 = 30
_mt5_mock.TIMEFRAME_H1 = 16385; _mt5_mock.TIMEFRAME_H2 = 16386; _mt5_mock.TIMEFRAME_H3 = 16387
_mt5_mock.TIMEFRAME_H4 = 16388; _mt5_mock.TIMEFRAME_H6 = 16390; _mt5_mock.TIMEFRAME_H8 = 16392
_mt5_mock.TIMEFRAME_H12 = 16396; _mt5_mock.TIMEFRAME_D1 = 16408
_mt5_mock.TIMEFRAME_W1 = 32769; _mt5_mock.TIMEFRAME_MN1 = 49153
sys.modules["MetaTrader5"] = _mt5_mock

from mtdata.forecast.methods.neural import (
    forecast_neural,
    NeuralForecastMethod,
    NHITSMethod,
    NBEATSXMethod,
    TFTMethod,
    PatchTSTMethod,
)
from mtdata.forecast.interface import ForecastResult


@pytest.fixture(autouse=True, scope="module")
def _restore_sys_modules():
    yield
    for name, orig in [("MetaTrader5", _orig_mt5), ("neuralforecast", _orig_nf), ("neuralforecast.models", _orig_nf_models)]:
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_Yf(fh: int):
    """Create a fake NeuralForecast output DataFrame."""
    return pd.DataFrame({
        "unique_id": ["ts"] * fh,
        "ds": list(range(fh)),
        "y_hat": np.linspace(100, 110, fh).tolist(),
    })


def _make_series(n: int = 100):
    return pd.Series(np.linspace(100, 110, n), dtype=float)


# ── forecast_neural  (lines 31-87) ──────────────────────────────────────────

class TestForecastNeural:
    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_nhits_basic(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(12)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 100, "ds": list(range(100)), "y": np.linspace(100, 110, 100)})
        vals, params_used = forecast_neural(
            method="nhits", series=np.linspace(100, 110, 100),
            fh=12, timeframe="H1", n=100, m=24, params={}, Y_df=Y_df,
        )
        assert len(vals) == 12
        assert "max_epochs" in params_used

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_nbeatsx(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(8)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)})
        vals, _ = forecast_neural(
            method="nbeatsx", series=np.linspace(100, 110, 50),
            fh=8, timeframe="H1", n=50, m=12, params={}, Y_df=Y_df,
        )
        assert len(vals) == 8

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_tft(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(6)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)})
        vals, _ = forecast_neural(
            method="tft", series=np.linspace(100, 110, 50),
            fh=6, timeframe="H1", n=50, m=12, params={}, Y_df=Y_df,
        )
        assert len(vals) == 6

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_patchtst(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(6)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)})
        vals, _ = forecast_neural(
            method="patchtst", series=np.linspace(100, 110, 50),
            fh=6, timeframe="H1", n=50, m=12, params={}, Y_df=Y_df,
        )
        assert len(vals) == 6

    def test_unknown_method(self):
        Y_df = pd.DataFrame({"unique_id": ["ts"], "ds": [0], "y": [100.0]})
        with pytest.raises(RuntimeError, match="not available"):
            forecast_neural(
                method="unknown_model", series=np.array([100.0]),
                fh=1, timeframe="H1", n=1, m=1, params={}, Y_df=Y_df,
            )

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_custom_input_size(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(12)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 100, "ds": list(range(100)), "y": np.linspace(100, 110, 100)})
        vals, pu = forecast_neural(
            method="nhits", series=np.linspace(100, 110, 100),
            fh=12, timeframe="H1", n=100, m=24,
            params={"input_size": 32}, Y_df=Y_df,
        )
        assert pu["input_size"] == 32

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_max_steps_param(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(12)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 100, "ds": list(range(100)), "y": np.linspace(100, 110, 100)})
        _, pu = forecast_neural(
            method="nhits", series=np.linspace(100, 110, 100),
            fh=12, timeframe="H1", n=100, m=24,
            params={"max_steps": 100}, Y_df=Y_df,
        )
        assert pu["max_epochs"] == 100

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_learning_rate_param(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(12)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 100, "ds": list(range(100)), "y": np.linspace(100, 110, 100)})
        forecast_neural(
            method="nhits", series=np.linspace(100, 110, 100),
            fh=12, timeframe="H1", n=100, m=24,
            params={"learning_rate": 0.001}, Y_df=Y_df,
        )
        call_kw = predict_mock.call_args
        assert call_kw is not None

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    def test_pred_col_not_found(self, predict_mock):
        # Return df with only standard columns -> no pred column
        predict_mock.return_value = pd.DataFrame({
            "unique_id": ["ts"] * 5,
            "ds": list(range(5)),
            "y": [1.0] * 5,
        })
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)})
        with pytest.raises(RuntimeError, match="prediction columns not found"):
            forecast_neural(
                method="nhits", series=np.linspace(100, 110, 50),
                fh=5, timeframe="H1", n=50, m=12, params={}, Y_df=Y_df,
            )

    @patch("mtdata.forecast.methods.neural._nf_setup_and_predict")
    @patch("mtdata.forecast.methods.neural._edge_pad_to_length", side_effect=lambda v, l: v[:l] if len(v) >= l else np.pad(v, (0, l - len(v)), mode="edge"))
    def test_zero_seasonality(self, pad_mock, predict_mock):
        predict_mock.return_value = _make_Yf(12)
        Y_df = pd.DataFrame({"unique_id": ["ts"] * 100, "ds": list(range(100)), "y": np.linspace(100, 110, 100)})
        vals, pu = forecast_neural(
            method="nhits", series=np.linspace(100, 110, 100),
            fh=12, timeframe="H1", n=100, m=0, params={}, Y_df=Y_df,
        )
        assert pu["input_size"] == 96  # fallback when m=0


# ── NeuralForecastMethod and subclasses  (lines 90-177) ─────────────────────

class TestNeuralForecastMethodProperties:
    def test_nhits_name(self):
        m = NHITSMethod()
        assert m.name == "nhits"
        assert m.category == "neural"
        assert "neuralforecast" in m.required_packages

    def test_nbeatsx_name(self):
        m = NBEATSXMethod()
        assert m.name == "nbeatsx"

    def test_tft_name(self):
        m = TFTMethod()
        assert m.name == "tft"

    def test_patchtst_name(self):
        m = PatchTSTMethod()
        assert m.name == "patchtst"

    def test_supports_features(self):
        m = NHITSMethod()
        sf = m.supports_features
        assert sf["price"] is True
        assert sf["return"] is True
        assert sf["volatility"] is False
        assert sf["ci"] is False


class TestNeuralForecastMethodForecast:
    @patch("mtdata.forecast.methods.neural.forecast_neural")
    @patch("mtdata.forecast.common._create_training_dataframes")
    def test_forecast_nhits(self, ctd, fn):
        ctd.return_value = (pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)}), None, None)
        fn.return_value = (np.linspace(100, 110, 12), {"max_epochs": 50, "input_size": 64, "batch_size": 32})
        m = NHITSMethod()
        series = _make_series(50)
        result = m.forecast(series, horizon=12, seasonality=24, params={})
        assert isinstance(result, ForecastResult)
        assert len(result.forecast) == 12

    @patch("mtdata.forecast.methods.neural.forecast_neural")
    @patch("mtdata.forecast.common._create_training_dataframes")
    def test_forecast_with_exog_kwargs(self, ctd, fn):
        ctd.return_value = (pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)}), None, None)
        fn.return_value = (np.linspace(100, 110, 12), {"max_epochs": 50, "input_size": 64, "batch_size": 32})
        m = NBEATSXMethod()
        series = _make_series(50)
        exog = np.random.randn(50, 2)
        result = m.forecast(series, horizon=12, seasonality=24, params={}, exog_used=exog)
        assert isinstance(result, ForecastResult)

    @patch("mtdata.forecast.methods.neural.forecast_neural")
    @patch("mtdata.forecast.common._create_training_dataframes")
    def test_forecast_with_exog_in_params(self, ctd, fn):
        ctd.return_value = (pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)}), None, None)
        fn.return_value = (np.linspace(100, 110, 12), {"max_epochs": 50, "input_size": 64, "batch_size": 32})
        m = TFTMethod()
        series = _make_series(50)
        result = m.forecast(series, horizon=12, seasonality=24, params={"exog_used": np.ones((50, 1))})
        assert isinstance(result, ForecastResult)

    def test_forecast_too_few_observations(self):
        m = PatchTSTMethod()
        series = _make_series(3)
        with pytest.raises(ValueError, match="at least 5"):
            m.forecast(series, horizon=12, seasonality=24, params={})

    @patch("mtdata.forecast.methods.neural.forecast_neural")
    @patch("mtdata.forecast.common._create_training_dataframes")
    def test_forecast_exog_future_via_param(self, ctd, fn):
        ctd.return_value = (pd.DataFrame({"unique_id": ["ts"] * 50, "ds": list(range(50)), "y": np.linspace(100, 110, 50)}), None, None)
        fn.return_value = (np.linspace(100, 110, 12), {"max_epochs": 50, "input_size": 64, "batch_size": 32})
        m = NHITSMethod()
        series = _make_series(50)
        result = m.forecast(
            series, horizon=12, seasonality=24,
            params={"exog_future": np.ones((12, 1)), "timeframe": "D1"},
        )
        assert isinstance(result, ForecastResult)
