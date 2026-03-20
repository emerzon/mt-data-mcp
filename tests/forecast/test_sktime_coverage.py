"""Tests for mtdata.forecast.methods.sktime – coverage for lines 56-122, 129-171, 210-241, 265-270."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub sktime before importing the module under test
# ---------------------------------------------------------------------------


def _make_module(name, attrs=None):
    mod = types.ModuleType(name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    return mod


class _FakeForecaster:
    """Mock sktime forecaster that supports fit/predict/predict_interval."""
# ruff: noqa: E402

    def __init__(self, **kw):
        self._kw = kw
        self._fitted = False

    def fit(self, y, X=None):
        self._fitted = True
        self._y = y
        return self

    def predict(self, fh, X=None):
        return pd.Series(np.ones(len(fh)) * 42.0)

    def predict_interval(self, fh, X=None, coverage=0.9):
        h = len(fh)
        idx = pd.RangeIndex(h)
        cols = pd.MultiIndex.from_tuples(
            [
                ("y", coverage, "lower"),
                ("y", coverage, "upper"),
            ]
        )
        data = np.column_stack([np.ones(h) * 40.0, np.ones(h) * 44.0])
        return pd.DataFrame(data, index=idx, columns=cols)


class _FakeThetaForecaster(_FakeForecaster):
    pass


class _FakeNaiveForecaster(_FakeForecaster):
    pass


class _FakeAutoETS(_FakeForecaster):
    pass


# Build sktime stub modules
_sktime = _make_module("sktime")
_sktime_forecasting = _make_module("sktime.forecasting")
_sktime_forecasting_theta = _make_module(
    "sktime.forecasting.theta", {"ThetaForecaster": _FakeThetaForecaster}
)
_sktime_forecasting_naive = _make_module(
    "sktime.forecasting.naive", {"NaiveForecaster": _FakeNaiveForecaster}
)
_sktime_forecasting_ets = _make_module(
    "sktime.forecasting.ets", {"AutoETS": _FakeAutoETS}
)

_STUBS = {
    "sktime": _sktime,
    "sktime.forecasting": _sktime_forecasting,
    "sktime.forecasting.theta": _sktime_forecasting_theta,
    "sktime.forecasting.naive": _sktime_forecasting_naive,
    "sktime.forecasting.ets": _sktime_forecasting_ets,
}

_originals = {}
for name, mod in _STUBS.items():
    _originals[name] = sys.modules.get(name)
    sys.modules[name] = mod

# Patch _HAS_SKTIME before importing sktime module
import mtdata.forecast.methods.sktime as _sktime_mod

_orig_has_sktime = _sktime_mod._HAS_SKTIME
_sktime_mod._HAS_SKTIME = True

from mtdata.forecast.methods.sktime import (
    GenericSktimeMethod,
    SktThetaMethod,
    SktNaiveMethod,
    SktAutoETSMethod,
)
from mtdata.forecast.interface import ForecastResult


@pytest.fixture(autouse=True, scope="module")
def _restore_sys_modules():
    yield
    for name, orig in _originals.items():
        if orig is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = orig
    _sktime_mod._HAS_SKTIME = _orig_has_sktime


# ===========================================================================
# Helpers
# ===========================================================================


def _series(n=100, freq=None, name="y"):
    if freq:
        idx = pd.date_range("2020-01-01", periods=n, freq=freq)
        return pd.Series(np.random.rand(n) * 100 + 50, index=idx, name=name)
    return pd.Series(np.random.rand(n) * 100 + 50, name=name)


# ===========================================================================
# SktimeMethod base properties
# ===========================================================================


class TestSktimeMethodProperties:
    def test_category(self):
        assert GenericSktimeMethod().category == "sktime"

    def test_required_packages(self):
        assert "sktime" in GenericSktimeMethod().required_packages

    def test_supports_features(self):
        feats = GenericSktimeMethod().supports_features
        assert feats["price"] is True
        assert feats["ci"] is True


# ===========================================================================
# GenericSktimeMethod._get_estimator (lines 196-227)
# ===========================================================================


class TestGenericSktimeGetEstimator:
    def test_default_estimator(self):
        m = GenericSktimeMethod()
        est = m._get_estimator(12, {})
        assert isinstance(est, _FakeThetaForecaster)

    def test_explicit_estimator_path(self):
        m = GenericSktimeMethod()
        est = m._get_estimator(
            12, {"estimator": "sktime.forecasting.naive.NaiveForecaster"}
        )
        assert isinstance(est, _FakeNaiveForecaster)

    def test_sp_injection(self):
        """sp should be injected if estimator accepts it."""
        # Patch signature to accept sp
        _FakeThetaForecaster.__init__ = (
            lambda self, sp=1, **kw: _FakeForecaster.__init__(self, sp=sp, **kw)
        )
        try:
            m = GenericSktimeMethod()
            est = m._get_estimator(12, {})
            assert est._kw.get("sp") == 12
        finally:
            _FakeThetaForecaster.__init__ = _FakeForecaster.__init__

    def test_invalid_estimator_path(self):
        m = GenericSktimeMethod()
        with pytest.raises(ValueError, match="Could not import"):
            m._get_estimator(12, {"estimator": "nonexistent.module.Class"})

    def test_bad_rsplit(self):
        m = GenericSktimeMethod()
        with pytest.raises(ValueError, match="Could not import"):
            m._get_estimator(12, {"estimator": "NoDotsHere"})


# ===========================================================================
# SktimeMethod.forecast (lines 37-181)
# ===========================================================================


class TestSktimeMethodForecast:
    def test_has_sktime_false(self):
        saved = _sktime_mod._HAS_SKTIME
        _sktime_mod._HAS_SKTIME = False
        try:
            m = GenericSktimeMethod()
            with pytest.raises(RuntimeError, match="not installed"):
                m.forecast(_series(), horizon=5, seasonality=12, params={})
        finally:
            _sktime_mod._HAS_SKTIME = saved

    def test_basic_forecast(self):
        m = GenericSktimeMethod()
        res = m.forecast(_series(), horizon=10, seasonality=12, params={})
        assert isinstance(res, ForecastResult)
        assert len(res.forecast) == 10

    def test_forecast_with_datetime_index(self):
        m = GenericSktimeMethod()
        res = m.forecast(_series(freq="h"), horizon=5, seasonality=12, params={})
        assert res.forecast is not None

    def test_forecast_range_index(self):
        s = pd.Series(np.random.rand(50) * 100, index=pd.RangeIndex(50))
        m = GenericSktimeMethod()
        res = m.forecast(s, horizon=5, seasonality=12, params={})
        assert res.forecast is not None

    def test_forecast_no_freq_fallback(self):
        """DatetimeIndex without freq -> infer -> fallback to reset_index."""
        idx = pd.DatetimeIndex(
            ["2020-01-01", "2020-01-03", "2020-01-07", "2020-01-10"] * 10
        )
        s = pd.Series(np.random.rand(40), index=idx)
        m = GenericSktimeMethod()
        res = m.forecast(s, horizon=5, seasonality=1, params={})
        assert res.forecast is not None

    def test_forecast_with_exog(self):
        s = _series()
        X = np.random.rand(100, 2)
        X_fut = np.random.rand(5, 2)
        m = GenericSktimeMethod()
        res = m.forecast(
            s, horizon=5, seasonality=1, params={}, exog_used=X, exog_future=X_fut
        )
        assert res.forecast is not None

    def test_forecast_exog_in_params(self):
        s = _series()
        m = GenericSktimeMethod()
        res = m.forecast(
            s,
            horizon=5,
            seasonality=1,
            params={
                "exog_used": np.random.rand(100, 2),
                "exog_future": np.random.rand(5, 2),
            },
        )
        assert res.forecast is not None

    def test_forecast_dataframe_output(self):
        """When estimator.predict returns DataFrame."""
        orig_predict = _FakeThetaForecaster.predict
        _FakeThetaForecaster.predict = lambda self, fh, X=None: pd.DataFrame(
            {"y": np.ones(len(fh)) * 42.0}
        )
        try:
            m = GenericSktimeMethod()
            res = m.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeThetaForecaster.predict = orig_predict

    def test_forecast_array_output(self):
        """When estimator.predict returns raw array."""
        orig_predict = _FakeThetaForecaster.predict
        _FakeThetaForecaster.predict = lambda self, fh, X=None: np.ones(len(fh)) * 42.0
        try:
            m = GenericSktimeMethod()
            res = m.forecast(_series(), horizon=5, seasonality=1, params={})
            assert res.forecast is not None
        finally:
            _FakeThetaForecaster.predict = orig_predict

    def test_forecast_with_ci(self):
        m = GenericSktimeMethod()
        res = m.forecast(_series(), horizon=5, seasonality=12, params={}, ci_alpha=0.1)
        assert res.ci_values is not None
        lo, hi = res.ci_values
        assert len(lo) == 5
        assert len(hi) == 5

    def test_forecast_ci_in_params(self):
        m = GenericSktimeMethod()
        res = m.forecast(_series(), horizon=5, seasonality=12, params={"ci_alpha": 0.1})
        assert res.ci_values is not None

    def test_forecast_ci_exception_ignored(self):
        """predict_interval exception -> ci_values = None."""
        orig = _FakeThetaForecaster.predict_interval
        _FakeThetaForecaster.predict_interval = MagicMock(
            side_effect=RuntimeError("no CI")
        )
        try:
            m = GenericSktimeMethod()
            res = m.forecast(
                _series(), horizon=5, seasonality=12, params={"ci_alpha": 0.1}
            )
            assert res.ci_values is None
        finally:
            _FakeThetaForecaster.predict_interval = orig

    def test_estimator_exception(self):
        """When estimator.fit raises, RuntimeError propagated."""
        orig = _FakeThetaForecaster.fit
        _FakeThetaForecaster.fit = MagicMock(side_effect=RuntimeError("fit failed"))
        try:
            m = GenericSktimeMethod()
            with pytest.raises(RuntimeError, match="Sktime"):
                m.forecast(_series(), horizon=5, seasonality=1, params={})
        finally:
            _FakeThetaForecaster.fit = orig

    def test_params_used(self):
        m = GenericSktimeMethod()
        res = m.forecast(
            _series(), horizon=5, seasonality=12, params={"some_key": "val"}
        )
        assert res.params_used["seasonality"] == 12
        assert res.params_used["some_key"] == "val"

    def test_exog_future_as_dataframe(self):
        s = _series()
        m = GenericSktimeMethod()
        X_fut = pd.DataFrame(np.random.rand(5, 2), columns=["a", "b"])
        res = m.forecast(s, horizon=5, seasonality=1, params={}, exog_future=X_fut)
        assert res.forecast is not None


# ===========================================================================
# SktThetaMethod (lines 230-241)
# ===========================================================================


class TestSktThetaMethod:
    def test_name(self):
        assert SktThetaMethod().name == "skt_theta"

    def test_get_estimator_default(self):
        m = SktThetaMethod()
        est = m._get_estimator(12, {})
        assert isinstance(est, _FakeThetaForecaster)

    def test_forecast(self):
        m = SktThetaMethod()
        res = m.forecast(_series(), horizon=5, seasonality=12, params={})
        assert res.forecast is not None


# ===========================================================================
# SktNaiveMethod (lines 244-256)
# ===========================================================================


class TestSktNaiveMethod:
    def test_name(self):
        assert SktNaiveMethod().name == "skt_naive"

    def test_get_estimator_default(self):
        m = SktNaiveMethod()
        est = m._get_estimator(12, {})
        assert isinstance(est, _FakeNaiveForecaster)

    def test_default_strategy(self):
        m = SktNaiveMethod()
        est = m._get_estimator(12, {})
        # strategy="last" is default but may not be passed if not in sig
        assert est is not None

    def test_forecast(self):
        m = SktNaiveMethod()
        res = m.forecast(_series(), horizon=5, seasonality=1, params={})
        assert res.forecast is not None


# ===========================================================================
# SktAutoETSMethod (lines 259-270)
# ===========================================================================


class TestSktAutoETSMethod:
    def test_name(self):
        assert SktAutoETSMethod().name == "skt_autoets"

    def test_get_estimator_default(self):
        m = SktAutoETSMethod()
        est = m._get_estimator(12, {})
        assert isinstance(est, _FakeAutoETS)

    def test_forecast(self):
        m = SktAutoETSMethod()
        res = m.forecast(_series(), horizon=5, seasonality=12, params={})
        assert res.forecast is not None
