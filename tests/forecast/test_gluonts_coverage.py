"""Tests for mtdata.forecast.methods.gluonts_extra – coverage for lines 30-130, 141-414, 425-540, 567-593."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub gluonts and related packages
# ---------------------------------------------------------------------------


def _make_module(name, attrs=None):
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


class _FakeListDataset(list):
    def __init__(self, entries, freq="H"):
        super().__init__(entries)
        self.freq = freq


class _FakeForecastObj:
    """Minimal forecast object returned by GluonTS predictors."""
# ruff: noqa: E402

    def __init__(self, h=10):
        self.mean = np.ones(h) * 42.0
        self.samples = np.ones((20, h)) * 42.0

    def quantile(self, q):
        return np.ones(len(self.mean)) * q


class _FakeEstimator:
    def __init__(self, **kw):
        self._kw = kw
        self._h = kw.get("prediction_length", 10)

    def train(self, training_data):
        return self  # acts as predictor too

    def predict(self, ds):
        return iter([_FakeForecastObj(self._h)])


class _FakePredictor:
    def __init__(self, **kw):
        self._h = kw.get("prediction_length", 10)

    def predict(self, ds):
        return iter([_FakeForecastObj(self._h)])


# --- Build mock module tree -----------------------------------------------
_gluonts = _make_module("gluonts")
_gluonts_dataset = _make_module("gluonts.dataset")
_gluonts_dataset_common = _make_module(
    "gluonts.dataset.common", {"ListDataset": _FakeListDataset}
)
_gluonts_torch = _make_module(
    "gluonts.torch",
    {"DeepAREstimator": _FakeEstimator, "SimpleFeedForwardEstimator": _FakeEstimator},
)
_gluonts_torch_model = _make_module("gluonts.torch.model")
_gluonts_torch_model_deepar = _make_module(
    "gluonts.torch.model.deepar", {"DeepAREstimator": _FakeEstimator}
)
_gluonts_torch_model_sff = _make_module(
    "gluonts.torch.model.simple_feedforward",
    {"SimpleFeedForwardEstimator": _FakeEstimator},
)
_gluonts_torch_model_tft = _make_module(
    "gluonts.torch.model.tft", {"TemporalFusionTransformerEstimator": _FakeEstimator}
)
_gluonts_torch_model_wavenet = _make_module(
    "gluonts.torch.model.wavenet", {"WaveNetEstimator": _FakeEstimator}
)
_gluonts_torch_model_deep_npts = _make_module(
    "gluonts.torch.model.deep_npts", {"DeepNPTSEstimator": _FakeEstimator}
)
_gluonts_torch_model_mqf2 = _make_module("gluonts.torch.model.mqf2")
_gluonts_torch_model_mqf2_est = _make_module(
    "gluonts.torch.model.mqf2.estimator",
    {"MQF2Estimator": _FakeEstimator, "MQF2MultiHorizonEstimator": _FakeEstimator},
)
_gluonts_model = _make_module("gluonts.model")
_gluonts_model_prophet = _make_module(
    "gluonts.model.prophet", {"ProphetPredictor": _FakePredictor}
)
_gluonts_model_npts = _make_module(
    "gluonts.model.npts", {"NPTSPredictor": _FakePredictor}
)

_STUBS = {
    "gluonts": _gluonts,
    "gluonts.dataset": _gluonts_dataset,
    "gluonts.dataset.common": _gluonts_dataset_common,
    "gluonts.torch": _gluonts_torch,
    "gluonts.torch.model": _gluonts_torch_model,
    "gluonts.torch.model.deepar": _gluonts_torch_model_deepar,
    "gluonts.torch.model.simple_feedforward": _gluonts_torch_model_sff,
    "gluonts.torch.model.tft": _gluonts_torch_model_tft,
    "gluonts.torch.model.wavenet": _gluonts_torch_model_wavenet,
    "gluonts.torch.model.deep_npts": _gluonts_torch_model_deep_npts,
    "gluonts.torch.model.mqf2": _gluonts_torch_model_mqf2,
    "gluonts.torch.model.mqf2.estimator": _gluonts_torch_model_mqf2_est,
    "gluonts.model": _gluonts_model,
    "gluonts.model.prophet": _gluonts_model_prophet,
    "gluonts.model.npts": _gluonts_model_npts,
}

_originals = {}
for name, mod in _STUBS.items():
    _originals[name] = sys.modules.get(name)
    sys.modules[name] = mod

from mtdata.forecast.methods.gluonts_extra import (
    _build_list_dataset,
    _extract_forecast_arrays,
    forecast_gt_deepar,
    forecast_gt_sfeedforward,
    forecast_gt_prophet,
    forecast_gt_tft,
    forecast_gt_wavenet,
    forecast_gt_deepnpts,
    forecast_gt_mqf2,
    forecast_gt_npts,
    GluonTSExtraMethod,
    GTDeepARMethod,
    GTSimpleFeedForwardMethod,
    GTProphetMethod,
    GTTFTMethod,
    GTWaveNetMethod,
    GTDeepNPTSMethod,
    GTMQF2Method,
    GTNPTSMethod,
)


@pytest.fixture(autouse=True, scope="module")
def _restore_sys_modules():
    yield
    for name, orig in _originals.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig


from mtdata.forecast.interface import ForecastResult

# ===========================================================================
# Helpers
# ===========================================================================


def _arr(n=100):
    return np.random.rand(n) * 100 + 50


def _series(n=100):
    return pd.Series(_arr(n), name="price")


# ===========================================================================
# _build_list_dataset
# ===========================================================================


class TestBuildListDataset:
    def test_basic(self):
        ds = _build_list_dataset(np.ones(50), freq="H")
        assert ds is not None

    def test_different_freq(self):
        ds = _build_list_dataset(np.arange(20, dtype=float), freq="D")
        assert ds is not None

    def test_single_element(self):
        ds = _build_list_dataset(np.array([1.0]), freq="H")
        assert ds is not None


# ===========================================================================
# _extract_forecast_arrays
# ===========================================================================


class TestExtractForecastArrays:
    def test_with_mean(self):
        f = _FakeForecastObj(10)
        vals, fq = _extract_forecast_arrays(f, 10, None)
        assert vals is not None
        assert len(vals) == 10

    def test_with_quantiles(self):
        f = _FakeForecastObj(10)
        vals, fq = _extract_forecast_arrays(f, 10, [0.1, 0.5, 0.9])
        assert fq is not None
        assert "0.5" in fq

    def test_mean_none_fallback_quantile(self):
        f = _FakeForecastObj(10)
        f.mean = None
        vals, fq = _extract_forecast_arrays(f, 10, None)
        assert vals is not None

    def test_mean_empty_fallback_quantile(self):
        f = _FakeForecastObj(10)
        f.mean = np.array([])
        vals, fq = _extract_forecast_arrays(f, 10, None)
        assert vals is not None

    def test_samples_fallback(self):
        f = _FakeForecastObj(10)
        f.mean = None

        def _bad_quantile(q):
            raise RuntimeError("no quantile")

        f.quantile = _bad_quantile
        vals, fq = _extract_forecast_arrays(f, 10, None)
        assert vals is not None

    def test_none_when_all_fail(self):
        class _NoAttrs:
            """Object with no mean, no quantile, no samples."""

            def quantile(self, q):
                raise RuntimeError

        f = _NoAttrs()
        vals, fq = _extract_forecast_arrays(f, 10, None)
        assert vals is None

    def test_truncate_to_fh(self):
        f = _FakeForecastObj(20)
        vals, _ = _extract_forecast_arrays(f, 5, None)
        assert len(vals) == 5

    def test_pad_when_short(self):
        f = _FakeForecastObj(3)
        vals, _ = _extract_forecast_arrays(f, 10, None)
        assert len(vals) == 10

    def test_quantile_bad_value_skipped(self):
        f = _FakeForecastObj(10)
        vals, fq = _extract_forecast_arrays(f, 10, ["bad", 0.5])
        assert fq is not None
        assert "0.5" in fq


# ===========================================================================
# forecast_gt_deepar
# ===========================================================================


class TestForecastGtDeepAR:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_deepar(series=_arr(), fh=10, params={}, n=100)
        assert err is None
        assert f is not None
        assert len(f) == 10

    def test_with_quantiles(self):
        f, fq, pu, err = forecast_gt_deepar(
            series=_arr(), fh=10, params={"quantiles": [0.1, 0.9]}, n=100
        )
        assert fq is not None

    def test_custom_params(self):
        p = {
            "context_length": 32,
            "freq": "D",
            "train_epochs": 2,
            "batch_size": 16,
            "learning_rate": 0.01,
            "hidden_size": 20,
            "num_layers": 1,
            "dropout": 0.2,
        }
        f, fq, pu, err = forecast_gt_deepar(series=_arr(), fh=5, params=p, n=100)
        assert err is None
        assert pu["context_length"] == 32

    def test_no_forecasts(self):
        orig = _FakeEstimator.predict
        _FakeEstimator.predict = lambda self, ds: iter([])
        try:
            f, fq, pu, err = forecast_gt_deepar(series=_arr(), fh=5, params={}, n=100)
            assert err is not None
            assert "no forecasts" in err
        finally:
            _FakeEstimator.predict = orig

    def test_estimator_error(self):
        orig = _FakeEstimator.train
        _FakeEstimator.train = MagicMock(side_effect=RuntimeError("train fail"))
        try:
            f, fq, pu, err = forecast_gt_deepar(series=_arr(), fh=5, params={}, n=100)
            assert err is not None
        finally:
            _FakeEstimator.train = orig


# ===========================================================================
# forecast_gt_sfeedforward
# ===========================================================================


class TestForecastGtSFF:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_sfeedforward(
            series=_arr(), fh=10, params={}, n=100
        )
        assert err is None
        assert f is not None

    def test_custom_params(self):
        p = {"hidden_dim": 128, "num_hidden_layers": 3, "context_length": 48}
        f, fq, pu, err = forecast_gt_sfeedforward(series=_arr(), fh=5, params=p, n=100)
        assert err is None
        assert pu["hidden_dim"] == 128

    def test_no_forecasts(self):
        orig = _FakeEstimator.predict
        _FakeEstimator.predict = lambda self, ds: iter([])
        try:
            f, fq, pu, err = forecast_gt_sfeedforward(
                series=_arr(), fh=5, params={}, n=100
            )
            assert "no forecasts" in err
        finally:
            _FakeEstimator.predict = orig


# ===========================================================================
# forecast_gt_prophet
# ===========================================================================


class TestForecastGtProphet:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_prophet(series=_arr(), fh=10, params={}, n=100)
        assert err is None
        assert f is not None

    def test_with_prophet_params(self):
        p = {
            "prophet_params": {"growth": "linear", "seasonality_mode": "multiplicative"}
        }
        f, fq, pu, err = forecast_gt_prophet(series=_arr(), fh=5, params=p, n=100)
        assert err is None
        assert pu["prophet_params"]["growth"] == "linear"

    def test_no_forecasts(self):
        orig = _FakePredictor.predict
        _FakePredictor.predict = lambda self, ds: iter([])
        try:
            f, fq, pu, err = forecast_gt_prophet(series=_arr(), fh=5, params={}, n=100)
            assert "no forecasts" in err
        finally:
            _FakePredictor.predict = orig

    def test_invalid_prophet_params(self):
        p = {"prophet_params": "not_a_dict"}
        f, fq, pu, err = forecast_gt_prophet(series=_arr(), fh=5, params=p, n=100)
        assert err is None  # should default to {}


# ===========================================================================
# forecast_gt_tft
# ===========================================================================


class TestForecastGtTFT:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_tft(series=_arr(), fh=10, params={}, n=100)
        assert err is None
        assert f is not None

    def test_custom_params(self):
        p = {"context_length": 64, "hidden_size": 32, "dropout": 0.05}
        f, fq, pu, err = forecast_gt_tft(series=_arr(), fh=5, params=p, n=100)
        assert pu["hidden_size"] == 32


# ===========================================================================
# forecast_gt_wavenet
# ===========================================================================


class TestForecastGtWaveNet:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_wavenet(series=_arr(), fh=10, params={}, n=100)
        assert err is None

    def test_custom_params(self):
        p = {"dilation_depth": 3, "num_stacks": 2}
        f, fq, pu, err = forecast_gt_wavenet(series=_arr(), fh=5, params=p, n=100)
        assert pu["dilation_depth"] == 3
        assert pu["num_stacks"] == 2

    def test_num_blocks_alias(self):
        p = {"num_blocks": 4}
        f, fq, pu, err = forecast_gt_wavenet(series=_arr(), fh=5, params=p, n=100)
        assert pu["num_stacks"] == 4


# ===========================================================================
# forecast_gt_deepnpts
# ===========================================================================


class TestForecastGtDeepNPTS:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_deepnpts(series=_arr(), fh=10, params={}, n=100)
        assert err is None

    def test_custom_params(self):
        p = {"context_length": 64, "train_epochs": 3}
        f, fq, pu, err = forecast_gt_deepnpts(series=_arr(), fh=5, params=p, n=100)
        assert pu["context_length"] == 64


# ===========================================================================
# forecast_gt_mqf2
# ===========================================================================


class TestForecastGtMQF2:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_mqf2(series=_arr(), fh=10, params={}, n=100)
        assert err is None

    def test_with_quantiles(self):
        f, fq, pu, err = forecast_gt_mqf2(
            series=_arr(), fh=10, params={"quantiles": [0.25, 0.75]}, n=100
        )
        assert err is None

    def test_custom_params(self):
        p = {"context_length": 32, "batch_size": 16}
        f, fq, pu, err = forecast_gt_mqf2(series=_arr(), fh=5, params=p, n=100)
        assert pu["batch_size"] == 16


# ===========================================================================
# forecast_gt_npts
# ===========================================================================


class TestForecastGtNPTS:
    def test_basic(self):
        f, fq, pu, err = forecast_gt_npts(series=_arr(), fh=10, params={}, n=100)
        assert err is None

    def test_uniform_kernel(self):
        p = {"kernel": "uniform"}
        f, fq, pu, err = forecast_gt_npts(series=_arr(), fh=5, params=p, n=100)
        assert pu["kernel_type"] == "uniform"

    def test_exponential_kernel(self):
        p = {"kernel": "exponential"}
        f, fq, pu, err = forecast_gt_npts(series=_arr(), fh=5, params=p, n=100)
        assert pu["kernel_type"] == "exponential"

    def test_climatological_kernel(self):
        p = {"kernel": "climatological"}
        f, fq, pu, err = forecast_gt_npts(series=_arr(), fh=5, params=p, n=100)
        assert pu["kernel_type"] == "uniform"

    def test_context_length(self):
        p = {"context_length": 50}
        f, fq, pu, err = forecast_gt_npts(series=_arr(), fh=5, params=p, n=100)
        assert pu["context_length"] == 50

    def test_no_context_length(self):
        f, fq, pu, err = forecast_gt_npts(series=_arr(), fh=5, params={}, n=100)
        assert pu["context_length"] is None

    def test_seasonal_model(self):
        p = {"use_seasonal_model": False}
        f, fq, pu, err = forecast_gt_npts(series=_arr(), fh=5, params=p, n=100)
        assert pu["use_seasonal_model"] is False


# ===========================================================================
# GluonTSExtraMethod.forecast (lines 567-593)
# ===========================================================================


class TestGluonTSExtraMethodForecast:
    def test_gt_deepar_via_class(self):
        m = GTDeepARMethod()
        res = m.forecast(_series(), horizon=10, seasonality=1, params={})
        assert isinstance(res, ForecastResult)
        assert len(res.forecast) == 10

    def test_gt_sfeedforward_via_class(self):
        m = GTSimpleFeedForwardMethod()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_gt_prophet_via_class(self):
        m = GTProphetMethod()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_gt_tft_via_class(self):
        m = GTTFTMethod()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_gt_wavenet_via_class(self):
        m = GTWaveNetMethod()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_gt_deepnpts_via_class(self):
        m = GTDeepNPTSMethod()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_gt_mqf2_via_class(self):
        m = GTMQF2Method()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_gt_npts_via_class(self):
        m = GTNPTSMethod()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_unsupported_method_name(self):
        class _Bad(GluonTSExtraMethod):
            @property
            def name(self):
                return "unknown_method"

        with pytest.raises(RuntimeError, match="Unsupported"):
            _Bad().forecast(_series(), horizon=5, seasonality=1, params={})

    def test_error_from_impl_raised(self):
        """When impl returns an error string, RuntimeError is raised."""
        orig = _FakeEstimator.train
        _FakeEstimator.train = MagicMock(side_effect=RuntimeError("boom"))
        try:
            m = GTDeepARMethod()
            with pytest.raises(RuntimeError):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _FakeEstimator.train = orig

    def test_none_forecast_raises(self):
        """When impl returns None vals, RuntimeError raised."""
        orig = _FakeEstimator.predict
        _FakeEstimator.predict = lambda self, ds: iter([])
        try:
            m = GTDeepARMethod()
            with pytest.raises(RuntimeError):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _FakeEstimator.predict = orig

    def test_metadata_with_quantiles(self):
        m = GTDeepARMethod()
        res = m.forecast(
            _series(), horizon=10, seasonality=1, params={"quantiles": [0.1, 0.5, 0.9]}
        )
        assert res.metadata is not None
        assert "quantiles" in res.metadata


# ===========================================================================
# Registered class properties
# ===========================================================================


class TestRegisteredClassProperties:
    def test_deepar_name(self):
        assert GTDeepARMethod().name == "gt_deepar"

    def test_sff_name(self):
        assert GTSimpleFeedForwardMethod().name == "gt_sfeedforward"

    def test_prophet_name(self):
        assert GTProphetMethod().name == "gt_prophet"

    def test_tft_name(self):
        assert GTTFTMethod().name == "gt_tft"

    def test_wavenet_name(self):
        assert GTWaveNetMethod().name == "gt_wavenet"

    def test_deepnpts_name(self):
        assert GTDeepNPTSMethod().name == "gt_deepnpts"

    def test_mqf2_name(self):
        assert GTMQF2Method().name == "gt_mqf2"

    def test_npts_name(self):
        assert GTNPTSMethod().name == "gt_npts"

    def test_category(self):
        assert GTDeepARMethod().category == "gluonts_extra"

    def test_required_packages_deepar(self):
        assert "gluonts" in GTDeepARMethod().required_packages
        assert "torch" in GTDeepARMethod().required_packages

    def test_required_packages_prophet(self):
        assert "prophet" in GTProphetMethod().required_packages

    def test_required_packages_mqf2(self):
        assert "cpflows" in GTMQF2Method().required_packages

    def test_required_packages_npts(self):
        pkgs = GTNPTSMethod().required_packages
        assert "gluonts" in pkgs

    def test_supports_features(self):
        feats = GTDeepARMethod().supports_features
        assert feats["price"] is True
        assert feats["return"] is True
        assert feats["volatility"] is False
