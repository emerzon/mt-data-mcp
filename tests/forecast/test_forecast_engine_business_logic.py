from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast import forecast_engine as fe
from mtdata.forecast import forecast_preprocessing as fp
from mtdata.forecast.interface import ForecastResult
from mtdata.utils.utils import _format_time_minimal


def _df(n: int = 20) -> pd.DataFrame:
    t0 = 1_700_100_000
    times = np.arange(t0, t0 + n * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 105.0, n, dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.4,
            "close": close,
            "volume": np.linspace(1000.0, 1200.0, n),
        }
    )


def test_normalize_weights_and_lookback_helpers():
    assert fe._normalize_weights(None, 2) is None
    assert fe._normalize_weights([1, 1], 2).tolist() == [0.5, 0.5]
    assert fe._normalize_weights("1,3", 2).tolist() == [0.25, 0.75]
    assert fe._normalize_weights([1], 2) is None
    assert fe._normalize_weights([-1, 0], 2) is None

    assert fe._calculate_lookback_bars("theta", horizon=4, lookback=10, seasonality=24, timeframe="H1") == 12
    assert fe._calculate_lookback_bars("analog", horizon=4, lookback=None, seasonality=24, timeframe="H1") == 5131
    assert fe._calculate_lookback_bars(
        "analog", horizon=4, lookback=None, seasonality=24, timeframe="H1", params={"window_size": 256}
    ) == 5515
    assert fe._calculate_lookback_bars(
        "analog", horizon=4, lookback=10, seasonality=24, timeframe="H1", params={"window_size": 256}
    ) == 5515
    assert fe._calculate_lookback_bars(
        "analog",
        horizon=4,
        lookback=6000,
        seasonality=24,
        timeframe="H1",
        params={"window_size": 64, "search_depth": 500},
    ) == 6002
    assert fe._calculate_lookback_bars("seasonal_naive", horizon=4, lookback=None, seasonality=12, timeframe="H1") == 36
    assert fe._calculate_lookback_bars("fourier_ols", horizon=4, lookback=None, seasonality=24, timeframe="H1") >= 300


def test_preprocessing_helpers_and_output_format():
    df = _df(6)
    base_col = fp._prepare_base_data(df, quantity="return")
    assert base_col == "__log_return"
    assert "__log_return" in df.columns

    base_col = fp._prepare_base_data(df, quantity="volatility")
    assert base_col == "__squared_return"
    assert "__squared_return" in df.columns

    calls = {"parse": 0, "apply": 0}
    df["ema_10"] = np.linspace(1.0, 2.0, len(df))
    out_col = fp._apply_features_and_target_spec(
        df,
        features={"ti": "ema:10"},
        target_spec={"column": "ema_10", "transform": "log"},
        base_col="close",
        parse_ti_specs=lambda spec: calls.__setitem__("parse", calls["parse"] + 1) or [{"s": spec}],
        apply_ta_indicators=lambda data, spec: calls.__setitem__("apply", calls["apply"] + 1) or ["ema_10"],
    )
    assert out_col == "__target_ema_10"
    assert calls["parse"] == 1
    assert calls["apply"] == 1

    forecast_values = np.array([0.01, 0.02, -0.01], dtype=float)
    reconstructed = np.array([101.0, 103.0, 102.0], dtype=float)
    ci = np.array([[0.0, 0.01, -0.02], [0.02, 0.03, 0.00]], dtype=float)
    with patch("mtdata.forecast.forecast_engine._use_client_tz", return_value=False):
        res = fe._format_forecast_output(
            forecast_values=forecast_values,
            last_epoch=float(df["time"].iloc[-1]),
            tf_secs=3600,
            horizon=3,
            base_col="close",
            df=df,
            ci_alpha=0.1,
            ci_values=ci,
            method="naive",
            quantity="return",
            denoise_used=False,
            metadata={"meta": 1},
            digits=5,
            forecast_return_values=forecast_values,
            reconstructed_prices=reconstructed,
        )
    assert res["success"] is True
    assert res["forecast_return"] == [0.01, 0.02, -0.01]
    assert res["forecast_price"] == [101.0, 103.0, 102.0]
    assert res["forecast_time"] == [
        _format_time_minimal(float(df["time"].iloc[-1]) + 3600.0),
        _format_time_minimal(float(df["time"].iloc[-1]) + 7200.0),
        _format_time_minimal(float(df["time"].iloc[-1]) + 10800.0),
    ]
    assert "forecast" not in res
    assert res["ci_alpha"] == 0.1
    assert res["ci_status"] == "available"
    assert res["ci_available"] is True
    assert "ci_requested" not in res
    assert "ci_alpha_requested" not in res
    assert "ci_unavailable" not in res
    assert res["lower_return"] == [0.0, 0.01, -0.02]
    assert res["upper_return"] == [0.02, 0.03, 0.0]
    assert "lower_price" not in res
    assert "upper_price" not in res
    assert res["digits"] == 5
    assert res["meta"] == 1
    assert res["last_price"] == float(df["close"].iloc[-1])
    assert "last_price_close" not in res
    assert res["last_price_source"] == "candle_close"

    with patch("mtdata.forecast.forecast_engine._use_client_tz", return_value=False):
        no_ci = fe._format_forecast_output(
            forecast_values=np.array([101.0, 102.0], dtype=float),
            last_epoch=float(df["time"].iloc[-1]),
            tf_secs=3600,
            horizon=2,
            base_col="close",
            df=df,
            ci_alpha=0.05,
            ci_values=None,
            method="theta",
            quantity="price",
            denoise_used=False,
        )
    assert no_ci["ci_status"] == "unavailable"
    assert no_ci["ci_alpha"] == 0.05
    assert no_ci["ci_available"] is False
    assert "ci_unavailable" not in no_ci
    assert "ci_requested" not in no_ci
    assert "ci_alpha_requested" not in no_ci
    assert "warnings" in no_ci
    assert "Point forecast only" in no_ci["warnings"][0]
    assert "forecast_conformal_intervals" in no_ci["warnings"][0]
    assert no_ci["forecast_price"] == [101.0, 102.0]
    assert "forecast" not in no_ci
    assert "lower_price" not in no_ci
    assert "upper_price" not in no_ci
    assert no_ci["last_price"] == float(df["close"].iloc[-1])
    assert "last_price_close" not in no_ci
    assert no_ci["last_price_source"] == "candle_close"


def test_prepare_ensemble_cv_uses_valid_rows_only(monkeypatch):
    series = pd.Series(np.arange(1.0, 21.0))

    def fake_dispatch_with_error(method_name, train, horizon, seasonality, params):
        if method_name == "broken":
            return None, {"method": "broken", "error": "unsupported", "error_type": "ValueError"}
        return np.array([float(train.iloc[-1] + 1.0)], dtype=float), None

    monkeypatch.setattr(fe, "_ensemble_dispatch_with_error", fake_dispatch_with_error)

    x, y = fe._prepare_ensemble_cv(
        series=series,
        methods=["naive", "broken"],
        horizon=1,
        seasonality=1,
        params_map={},
        cv_points=4,
        min_train=3,
    )
    # Sparse rows: "naive" column is valid, "broken" column is NaN
    assert x.shape[0] == 4
    assert x.shape[1] == 2
    assert np.all(np.isfinite(x[:, 0]))   # naive column fully valid
    assert np.all(np.isnan(x[:, 1]))       # broken column all NaN
    assert y.shape[0] == 4

    x, y = fe._prepare_ensemble_cv(
        series=series,
        methods=["naive", "theta"],
        horizon=1,
        seasonality=1,
        params_map={},
        cv_points=3,
        min_train=3,
    )
    assert x.shape[0] == 3
    assert x.shape[1] == 2
    assert y.shape[0] == 3


def test_prepare_ensemble_cv_uses_all_horizon_steps(monkeypatch):
    series = pd.Series(np.arange(1.0, 11.0))

    def fake_dispatch_with_error(method_name, train, horizon, seasonality, params):
        base = float(train.iloc[-1])
        return np.array([base + 1.0, base + 2.0], dtype=float), None

    monkeypatch.setattr(fe, "_ensemble_dispatch_with_error", fake_dispatch_with_error)

    x, y = fe._prepare_ensemble_cv(
        series=series,
        methods=["naive", "theta"],
        horizon=2,
        seasonality=1,
        params_map={},
        cv_points=2,
        min_train=3,
    )

    assert x.shape == (4, 2)
    assert y.tolist() == [7.0, 8.0, 8.0, 9.0]


def test_prepare_ensemble_cv_records_dispatch_errors_without_function_state(monkeypatch):
    class GoodForecaster:
        def forecast(self, series, horizon, seasonality, params):
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1] + 1.0)], dtype=float),
                params_used={},
                metadata={},
            )

    class BadForecaster:
        def forecast(self, series, horizon, seasonality, params):
            raise RuntimeError("boom")

    class FakeRegistry:
        @staticmethod
        def get(name):
            if name == "bad":
                return BadForecaster()
            return GoodForecaster()

    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)

    failures = []
    x, y = fe._prepare_ensemble_cv(
        series=pd.Series(np.arange(1.0, 10.0)),
        methods=["bad", "naive"],
        horizon=1,
        seasonality=1,
        params_map={},
        cv_points=1,
        min_train=3,
        failure_sink=failures,
    )

    # Sparse: "bad" column is NaN, "naive" column is valid
    assert x.shape[1] == 2
    assert x.shape[0] >= 1
    assert np.all(np.isnan(x[:, 0]))      # bad column all NaN
    assert np.all(np.isfinite(x[:, 1]))    # naive column valid
    assert failures[0]["method"] == "bad"
    assert failures[0]["error"] == "boom"
    assert failures[0]["error_type"] == "RuntimeError"
    assert getattr(fe._ensemble_dispatch_with_error, "_last_error", None) is None


def test_prepare_ensemble_cv_sparse_failure_preserves_valid_methods(monkeypatch):
    """When one method fails intermittently, the CV matrix keeps valid entries
    and marks failures as NaN, instead of discarding the entire anchor."""
    series = pd.Series(np.arange(1.0, 21.0))
    call_count = {"flaky": 0}

    def fake_dispatch_with_error(method_name, train, horizon, seasonality, params):
        if method_name == "flaky":
            call_count["flaky"] += 1
            if call_count["flaky"] % 2 == 0:
                return None, {"error": "intermittent", "error_type": "RuntimeError"}
        return np.array([float(train.iloc[-1] + 1.0)], dtype=float), None

    monkeypatch.setattr(fe, "_ensemble_dispatch_with_error", fake_dispatch_with_error)

    failures: list = []
    x, y = fe._prepare_ensemble_cv(
        series=series,
        methods=["good", "flaky"],
        horizon=1,
        seasonality=1,
        params_map={},
        cv_points=4,
        min_train=3,
        failure_sink=failures,
    )

    assert x.shape[0] == 4
    assert x.shape[1] == 2
    # "good" column always valid
    assert np.all(np.isfinite(x[:, 0]))
    # "flaky" column has a mix of valid and NaN
    n_nan = np.sum(np.isnan(x[:, 1]))
    n_valid = np.sum(np.isfinite(x[:, 1]))
    assert n_nan > 0
    assert n_valid > 0
    assert len(failures) == n_nan


def test_prepare_ensemble_cv_all_methods_fail_returns_empty():
    """When ALL methods fail at every anchor, the matrix is empty."""
    series = pd.Series(np.arange(1.0, 21.0))

    def always_fail(method_name, train, horizon, seasonality, params):
        return None, {"error": "fail", "error_type": "ValueError"}

    from mtdata.forecast.methods.ensemble import _prepare_ensemble_cv_default
    x, y = _prepare_ensemble_cv_default(
        series, ["a", "b"], 1, 1, {}, 4, 3, always_fail,
    )
    assert x.shape == (0, 2)
    assert y.shape == (0,)


def test_forecast_engine_validation_and_top_level_errors(monkeypatch):
    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive", "ensemble"))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))

    assert "Invalid timeframe" in fe.forecast_engine(symbol="EURUSD", timeframe="BAD")["error"]

    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {})
    assert fe.forecast_engine(symbol="EURUSD", timeframe="H1")["error"] == "Unsupported timeframe seconds for H1"
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})

    assert "Invalid method" in fe.forecast_engine(symbol="EURUSD", timeframe="H1", method="theta")["error"]
    assert fe.forecast_engine(symbol="EURUSD", timeframe="H1", method="naive", quantity="volatility")["error"] == "Use forecast_volatility for volatility models"

    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: (_ for _ in ()).throw(ValueError("bad params")))
    out = fe.forecast_engine(symbol="EURUSD", timeframe="H1", method="naive")
    assert out["error"].startswith("Forecast engine failed: bad params")


def test_forecast_engine_prefetched_non_ensemble_success_and_failures(monkeypatch):
    class GoodForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            return ForecastResult(
                forecast=np.array([0.01, 0.02], dtype=float),
                ci_values=(np.array([0.0, 0.0]), np.array([0.1, 0.1])),
                params_used={"used": True},
                metadata={"engine_meta": 1},
            )

    class NoneForecaster:
        def forecast(self, *args, **kwargs):
            return ForecastResult(forecast=None, params_used={}, metadata={})

    class ErrorForecaster:
        def forecast(self, *args, **kwargs):
            raise RuntimeError("method exploded")

    class FakeRegistry:
        current = GoodForecaster

        @staticmethod
        def get(name):
            return FakeRegistry.current()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive", "ensemble"))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: SimpleNamespace(digits=5))
    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not fetch")))

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=2,
        quantity="return",
        params={"alpha": 1},
        prefetched_df=_df(20),
    )
    assert out["success"] is True
    assert out["forecast_return"] == [0.01, 0.02]
    assert len(out["forecast_price"]) == 2
    assert out["digits"] == 5
    assert out["params_used"] == {"used": True}
    assert "diagnostics" in out
    assert out["diagnostics"]["history_bars_used"] == 20
    assert out["diagnostics"]["target_points_used"] >= 1
    assert out["diagnostics"]["target_points_used"] <= 20
    assert out["diagnostics"]["seasonality_used"] == 24
    assert out["diagnostics"]["quantity"] == "return"
    assert out["diagnostics"]["base_col_used"] == "__log_return"
    assert out["diagnostics"]["lookback_bars_requested"] is None
    assert out["diagnostics"]["lookback_bars_fetched"] >= 20

    FakeRegistry.current = NoneForecaster
    out = fe.forecast_engine(symbol="EURUSD", timeframe="H1", method="naive", prefetched_df=_df(20))
    assert out["error"] == "Method 'naive' returned no forecast values"

    FakeRegistry.current = ErrorForecaster
    out = fe.forecast_engine(symbol="EURUSD", timeframe="H1", method="naive", prefetched_df=_df(20))
    assert out["error"].startswith("Forecast method 'naive' failed: method exploded")


def test_run_registered_forecast_method_uses_method_prepare_hook(monkeypatch):
    captured = {}

    class HookedForecaster:
        def prepare_forecast_call(self, params, call_kwargs, context):
            captured["context"] = context
            params = dict(params)
            call_kwargs = dict(call_kwargs)
            params["prepared"] = context.symbol
            call_kwargs["prepared_flag"] = context.base_col
            return params, call_kwargs

        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["params"] = dict(params)
            captured["kwargs"] = dict(kwargs)
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={"prepared": True},
                metadata={"hooked": True},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return HookedForecaster()

    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)

    forecast, ci_values, metadata = fe._run_registered_forecast_method(
        method_l="theta",
        method="theta",
        df=_df(20),
        target_series=pd.Series([1.0, 2.0, 3.0], name="close"),
        horizon=1,
        seasonality=24,
        params={"alpha": 1},
        ci_alpha=0.05,
        as_of="2024-01-01",
        quantity_l="price",
        symbol="EURUSD",
        timeframe="H1",
        base_col="close",
        denoise_spec_used={"method": "ema"},
        X=np.array([[1.0], [2.0], [3.0]], dtype=float),
        future_exog=np.array([[4.0]], dtype=float),
    )

    assert forecast.tolist() == [3.0]
    assert ci_values is None
    assert metadata["hooked"] is True
    assert metadata["params_used"] == {"prepared": True}
    assert captured["params"]["prepared"] == "EURUSD"
    assert captured["kwargs"]["prepared_flag"] == "close"
    assert captured["kwargs"]["exog_used"].shape == (3, 1)
    assert captured["context"].timeframe == "H1"
    assert captured["context"].quantity == "price"
    assert captured["context"].future_exog.shape == (1, 1)


def test_forecast_engine_preserves_prefetched_denoised_base_column(monkeypatch):
    captured = {}

    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["series_name"] = series.name
            captured["last_value"] = float(series.iloc[-1])
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    df = _df(20)
    df["close_dn"] = df["close"] * 10.0

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        prefetched_df=df,
        prefetched_base_col="close_dn",
        prefetched_denoise_spec={"method": "ema"},
    )

    assert out["success"] is True
    assert out["base_col"] == "close_dn"
    assert out["forecast_price"] == [float(df["close_dn"].iloc[-1])]
    assert out["denoise_applied"] is True
    assert captured["series_name"] == "close_dn"
    assert captured["last_value"] == float(df["close_dn"].iloc[-1])


def test_forecast_engine_reconstructs_return_prices_from_prefetched_denoised_base(monkeypatch):
    captured = {}

    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["series_name"] = series.name
            captured["last_value"] = float(series.iloc[-1])
            return ForecastResult(
                forecast=np.array([0.0], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    df = _df(20)
    df["close_dn"] = df["close"] * 10.0

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        quantity="return",
        prefetched_df=df,
        prefetched_base_col="close_dn",
        prefetched_denoise_spec={"method": "ema"},
    )

    assert out["success"] is True
    assert out["base_col"] == "__log_return"
    assert out["forecast_return"] == [0.0]
    assert out["forecast_price"] == [float(df["close_dn"].iloc[-1])]
    assert out["denoise_applied"] is True
    assert captured["series_name"] == "__log_return"


def test_forecast_engine_applies_denoise_to_prefetched_raw_history(monkeypatch):
    captured = {}

    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["series_name"] = series.name
            captured["last_value"] = float(series.iloc[-1])
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    def fake_apply_denoise(df, spec, default_when=None):
        df["close_dn"] = df["close"] * 10.0
        return ["close_dn"]

    monkeypatch.setattr(fe, "_normalize_denoise_spec", lambda spec, default_when=None: {"method": "ema", "columns": ["close"]})
    monkeypatch.setattr(fe, "_apply_denoise", fake_apply_denoise)

    df = _df(20)
    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        denoise={"method": "ema"},
        prefetched_df=df,
    )

    assert out["success"] is True
    assert out["base_col"] == "close_dn"
    assert out["denoise_applied"] is True
    assert captured["series_name"] == "close_dn"
    assert captured["last_value"] == float(df["close"].iloc[-1] * 10.0)


def test_forecast_engine_surfaces_denoise_warnings(monkeypatch):
    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    df = _df(20)
    df.attrs["denoise_warnings"] = [
        "Denoise method 'wavelet' requires PyWavelets, but it is not installed."
    ]

    monkeypatch.setattr(
        fe,
        "_resolve_history_context",
        lambda **kwargs: (df, "close", {"method": "wavelet"}),
    )

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        denoise={"method": "wavelet"},
    )

    assert out["success"] is True
    assert "warnings" in out
    assert any("requires PyWavelets" in warning for warning in out["warnings"])


def test_forecast_engine_builds_exog_and_aligns_for_returns(monkeypatch):
    captured = {}

    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["series_len"] = len(series)
            captured["exog_used"] = kwargs.get("exog_used")
            captured["exog_future"] = exog_future
            return ForecastResult(
                forecast=np.ones(int(horizon), dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("theta",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=5,
        quantity="return",
        params={"alpha": 1},
        features={"include": "open,high low", "future_covariates": "hour,dow,fourier:24,is_weekend"},
        prefetched_df=_df(30),
    )

    assert out["success"] is True
    assert captured["series_len"] == 29
    assert captured["exog_used"].shape == (29, 10)
    assert captured["exog_future"].shape == (5, 10)


def test_forecast_engine_dimred_failure_falls_back_to_raw_features(monkeypatch):
    captured = {}

    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["exog_used"] = kwargs.get("exog_used")
            captured["exog_future"] = exog_future
            return ForecastResult(
                forecast=np.ones(int(horizon), dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("theta",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)
    monkeypatch.setattr(fp, "_create_dimred_reducer", lambda method, params: (_ for _ in ()).throw(RuntimeError("dimred failed")))

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=4,
        features={"include": "open,high"},
        dimred_method="pca",
        prefetched_df=_df(20),
    )

    assert out["success"] is True
    assert captured["exog_used"].shape == (20, 2)
    assert captured["exog_future"].shape == (4, 2)


def test_forecast_engine_forwards_ci_alpha_in_params_and_kwargs(monkeypatch):
    captured = {}

    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["params"] = dict(params)
            captured["kwargs"] = dict(kwargs)
            return ForecastResult(
                forecast=np.array([1.0], dtype=float),
                ci_values=(np.array([0.9], dtype=float), np.array([1.1], dtype=float)),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: SimpleNamespace(digits=5))

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        ci_alpha=0.1,
        prefetched_df=_df(20),
    )

    assert out["success"] is True
    assert captured["params"]["ci_alpha"] == 0.1
    assert captured["kwargs"]["ci_alpha"] == 0.1


def test_forecast_engine_warns_when_ci_requested_but_method_has_no_intervals(monkeypatch):
    class NoCIForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            return ForecastResult(
                forecast=np.array([1.0], dtype=float),
                ci_values=None,
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return NoCIForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        ci_alpha=0.1,
        prefetched_df=_df(20),
    )

    assert out["success"] is True
    assert out["ci_status"] == "unavailable"
    assert out["ci_alpha"] == 0.1
    assert out["ci_available"] is False
    assert "ci_unavailable" not in out
    assert "ci_requested" not in out
    assert "ci_alpha_requested" not in out
    assert "warnings" in out
    assert "Point forecast only" in out["warnings"][0]
    assert "EURUSD" in out["warnings"][0]
    assert "--timeframe H1" in out["warnings"][0]
    assert " SYMBOL " not in out["warnings"][0]
    assert "lower_price" not in out
    assert "upper_price" not in out


def test_forecast_engine_injects_context_for_analog(monkeypatch):
    captured = {}

    class CaptureForecaster(fe._analog_methods.AnalogMethod):
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["params"] = dict(params)
            captured["kwargs"] = dict(kwargs)
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("analog",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="analog",
        horizon=1,
        as_of="2024-01-01",
        params={"window_size": 32},
        prefetched_df=_df(20),
    )

    assert out["success"] is True
    assert captured["params"]["window_size"] == 32
    assert captured["params"]["symbol"] == "EURUSD"
    assert captured["params"]["timeframe"] == "H1"
    assert captured["params"]["as_of"] == "2024-01-01"
    assert captured["params"]["base_col"] == "close"
    assert captured["kwargs"]["as_of"] == "2024-01-01"
    assert isinstance(captured["kwargs"]["history_df"], pd.DataFrame)
    assert captured["kwargs"]["history_base_col"] == "close"
    assert captured["kwargs"]["history_denoise_spec"] is None


def test_forecast_engine_fetches_sufficient_history_for_analog(monkeypatch):
    captured = {}

    class CaptureForecaster(fe._analog_methods.AnalogMethod):
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    def fake_fetch_history(symbol, timeframe, need, as_of):
        captured["fetch"] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "need": int(need),
            "as_of": as_of,
        }
        return _df(int(need))

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("analog",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "_fetch_history", fake_fetch_history)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="analog",
        horizon=12,
        params={"window_size": 64, "search_depth": 5000, "top_k": 20},
        ci_alpha=None,
    )

    assert out["success"] is True
    assert captured["fetch"] == {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "need": 5139,
        "as_of": None,
    }


def test_forecast_engine_injects_denoise_context_for_analog(monkeypatch):
    captured = {}

    class CaptureForecaster(fe._analog_methods.AnalogMethod):
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["params"] = dict(params)
            captured["kwargs"] = dict(kwargs)
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("analog",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    df = _df(40)
    df["close_dn"] = df["close"] * 10.0

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="analog",
        horizon=1,
        prefetched_df=df,
        prefetched_base_col="close_dn",
        prefetched_denoise_spec={"method": "ema", "params": {"span": 5}},
    )

    assert out["success"] is True
    assert captured["params"]["base_col"] == "close_dn"
    assert captured["params"]["denoise"]["method"] == "ema"
    assert captured["params"]["denoise"]["params"] == {"span": 5}
    assert captured["params"]["denoise"]["columns"] == ["close"]
    assert captured["kwargs"]["history_base_col"] == "close_dn"
    assert captured["kwargs"]["history_denoise_spec"]["method"] == "ema"


def test_forecast_engine_analog_rejects_prefetched_history_shorter_than_window(monkeypatch):
    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("analog",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    class FakeRegistry:
        @staticmethod
        def get(name):
            from mtdata.forecast.methods.analog import AnalogMethod

            return AnalogMethod()

    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="analog",
        horizon=1,
        params={"window_size": 32},
        prefetched_df=_df(20),
    )

    assert out["error"].startswith("Forecast method 'analog' failed: Analog method requires at least 32 price points")


def test_forecast_engine_adds_broker_time_check_for_live_data(monkeypatch):
    class FakeRegistry:
        @staticmethod
        def get(name):
            return SimpleNamespace(
                forecast=lambda *args, **kwargs: ForecastResult(
                    forecast=np.array([101.0], dtype=float),
                    params_used={"used": True},
                    metadata={},
                )
            )

    captured = {}

    def fake_inspect(symbol, probe_timeframe, ttl_seconds):
        captured["symbol"] = symbol
        captured["probe_timeframe"] = probe_timeframe
        captured["ttl_seconds"] = ttl_seconds
        return {"status": "ok", "reason": None, "probe_timeframe": probe_timeframe}

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: _df(20))
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)
    monkeypatch.setattr(fe, "get_cached_mt5_time_alignment", fake_inspect)
    monkeypatch.setattr(
        fe,
        "mt5_config",
        SimpleNamespace(broker_time_check_enabled=True, broker_time_check_ttl_seconds=45),
    )

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        ci_alpha=None,
    )

    assert out["success"] is True
    assert captured == {"symbol": "EURUSD", "probe_timeframe": "M1", "ttl_seconds": 45}
    assert out["diagnostics"]["broker_time_check"]["status"] == "ok"


def test_forecast_engine_surfaces_broker_time_misalignment_warning(monkeypatch):
    class FakeRegistry:
        @staticmethod
        def get(name):
            return SimpleNamespace(
                forecast=lambda *args, **kwargs: ForecastResult(
                    forecast=np.array([101.0], dtype=float),
                    params_used={"used": True},
                    metadata={},
                )
            )

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: _df(20))
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)
    monkeypatch.setattr(
        fe,
        "mt5_config",
        SimpleNamespace(broker_time_check_enabled=True, broker_time_check_ttl_seconds=60),
    )
    monkeypatch.setattr(
        fe,
        "get_cached_mt5_time_alignment",
        lambda symbol, probe_timeframe, ttl_seconds: {
            "status": "misaligned",
            "reason": "timezone_mismatch",
            "warning": "MT5 broker-time sanity check failed: inferred broker offset is 10800s but configuration resolves to 7200s",
        },
    )

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        ci_alpha=None,
    )

    assert out["success"] is True
    assert out["diagnostics"]["broker_time_check"]["status"] == "misaligned"
    assert out["warnings"] == [
        "MT5 broker-time sanity check failed: inferred broker offset is 10800s but configuration resolves to 7200s"
    ]


def test_forecast_engine_keeps_stale_broker_time_check_diagnostic_only(monkeypatch):
    class FakeRegistry:
        @staticmethod
        def get(name):
            return SimpleNamespace(
                forecast=lambda *args, **kwargs: ForecastResult(
                    forecast=np.array([101.0], dtype=float),
                    params_used={"used": True},
                    metadata={},
                )
            )

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: _df(20))
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)
    monkeypatch.setattr(
        fe,
        "mt5_config",
        SimpleNamespace(broker_time_check_enabled=True, broker_time_check_ttl_seconds=60),
    )
    monkeypatch.setattr(
        fe,
        "get_cached_mt5_time_alignment",
        lambda symbol, probe_timeframe, ttl_seconds: {
            "status": "stale",
            "reason": "market_data_stale",
            "warning": "MT5 broker-time sanity check could not confirm live alignment: market is closed",
        },
    )

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        ci_alpha=None,
    )

    assert out["success"] is True
    assert out["diagnostics"]["broker_time_check"]["status"] == "stale"
    assert "warnings" not in out


def test_forecast_engine_skips_broker_time_check_for_prefetched_and_asof(monkeypatch):
    class FakeRegistry:
        @staticmethod
        def get(name):
            return SimpleNamespace(
                forecast=lambda *args, **kwargs: ForecastResult(
                    forecast=np.array([101.0], dtype=float),
                    params_used={"used": True},
                    metadata={},
                )
            )

    calls = {"count": 0}

    def fake_inspect(symbol, probe_timeframe, ttl_seconds):
        calls["count"] += 1
        return {"status": "ok"}

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: _df(20))
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)
    monkeypatch.setattr(fe, "get_cached_mt5_time_alignment", fake_inspect)
    monkeypatch.setattr(
        fe,
        "mt5_config",
        SimpleNamespace(broker_time_check_enabled=True, broker_time_check_ttl_seconds=60),
    )

    out_prefetched = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        ci_alpha=None,
        prefetched_df=_df(20),
    )
    out_asof = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        ci_alpha=None,
        as_of="2024-01-01",
    )

    assert out_prefetched["success"] is True
    assert out_asof["success"] is True
    assert calls["count"] == 0
    assert "broker_time_check" not in out_prefetched["diagnostics"]
    assert "broker_time_check" not in out_asof["diagnostics"]


def test_forecast_engine_target_spec_and_data_validity_errors(monkeypatch):
    class FakeRegistry:
        @staticmethod
        def get(name):
            return SimpleNamespace(
                forecast=lambda *args, **kwargs: ForecastResult(
                    forecast=np.array([1.0], dtype=float), params_used={}, metadata={}
                )
            )

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive", "ensemble"))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "_fetch_history", lambda *args, **kwargs: _df(10))

    monkeypatch.setattr(fe, "build_target_series", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad spec")))
    out = fe.forecast_engine(symbol="EURUSD", timeframe="H1", method="naive", target_spec={"x": 1})
    assert out["error"] == "Invalid target_spec: bad spec"

    dshort = _df(5)
    dshort["close"] = np.nan
    out = fe.forecast_engine(symbol="EURUSD", timeframe="H1", method="naive", prefetched_df=dshort)
    assert "Not enough valid data points" in out["error"]


def test_forecast_engine_target_spec_column_alias_is_applied(monkeypatch):
    captured = {}

    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            captured["last_value"] = float(series.iloc[-1])
            return ForecastResult(
                forecast=np.array([float(series.iloc[-1])], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)

    df = _df(20)
    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=1,
        prefetched_df=df,
        target_spec={"column": "open", "transform": "none"},
    )

    assert out["success"] is True
    assert out["base_col"] == "open"
    assert captured["last_value"] == float(df["open"].iloc[-1])


def test_forecast_engine_reconstructs_custom_simple_return_targets(monkeypatch):
    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            return ForecastResult(
                forecast=np.array([0.10, 0.10], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    df = _df(20)
    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=2,
        quantity="return",
        prefetched_df=df,
        target_spec={"base": "close", "transform": "return", "k": 1},
    )

    assert out["success"] is True
    assert out["forecast_return"] == [0.1, 0.1]
    np.testing.assert_allclose(out["forecast_price"], np.array([115.5, 127.05]))


def test_forecast_engine_reconstructs_custom_k_lag_return_targets(monkeypatch):
    class CaptureForecaster:
        def forecast(self, series, horizon, seasonality, params, exog_future=None, **kwargs):
            return ForecastResult(
                forecast=np.array([0.10, 0.10, 0.10], dtype=float),
                params_used={},
                metadata={},
            )

    class FakeRegistry:
        @staticmethod
        def get(name):
            return CaptureForecaster()

    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "ForecastRegistry", FakeRegistry)
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    df = _df(20)
    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=3,
        quantity="return",
        prefetched_df=df,
        target_spec={"base": "close", "transform": "return", "k": 2},
    )

    closes = df["close"].to_numpy(dtype=float)
    expected = np.array(
        [
            closes[-2] * 1.1,
            closes[-1] * 1.1,
            (closes[-2] * 1.1) * 1.1,
        ],
        dtype=float,
    )

    assert out["success"] is True
    assert out["forecast_return"] == [0.1, 0.1, 0.1]
    np.testing.assert_allclose(np.asarray(out["forecast_price"], dtype=float), expected)


def test_forecast_engine_ensemble_paths(monkeypatch):
    monkeypatch.setattr(fe, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(fe, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("ensemble", "naive", "theta"))
    monkeypatch.setattr(fe, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "get_symbol_info_cached", lambda symbol: None)

    def fake_dispatch(name, series, horizon, seasonality, params):
        if name == "naive":
            return np.array([1.0, 2.0], dtype=float)
        if name == "theta":
            return np.array([2.0, 4.0], dtype=float)
        return None

    def fake_dispatch_with_error(name, series, horizon, seasonality, params):
        fc = fake_dispatch(name, series, horizon, seasonality, params)
        if fc is None:
            return None, {"method": name, "error": "unsupported", "error_type": "ValueError"}
        return fc, None

    monkeypatch.setattr(fe, "_ensemble_dispatch_method", fake_dispatch)
    monkeypatch.setattr(fe, "_ensemble_dispatch_with_error", fake_dispatch_with_error)
    monkeypatch.setattr(fe, "_prepare_ensemble_cv", lambda *args, **kwargs: (np.empty((0, 2)), np.empty((0,))))

    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="ensemble",
        horizon=2,
        params={"methods": "naive,theta", "weights": "1,3", "mode": "bma"},
        prefetched_df=_df(20),
    )
    assert out["success"] is True
    assert out["forecast_price"] == [1.75, 3.5]
    assert out["ensemble"]["mode_used"] == "average"
    assert out["ensemble"]["methods"] == ["naive", "theta"]

    monkeypatch.setattr(fe, "_ensemble_dispatch_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(fe, "_ensemble_dispatch_with_error", lambda name, *args, **kwargs: (None, {"method": name, "error": "unsupported", "error_type": "ValueError"}))
    out = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="ensemble",
        horizon=2,
        params={"methods": "naive,theta"},
        prefetched_df=_df(20),
    )
    assert out["error"] == "Ensemble failed: no component forecasts"
