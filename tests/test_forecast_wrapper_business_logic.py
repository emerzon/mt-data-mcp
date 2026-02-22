from __future__ import annotations

import numpy as np
import pandas as pd

import mtdata.forecast.forecast as ff
import mtdata.forecast.forecast_engine as fe
import mtdata.forecast.volatility as fv


def _sample_df(n: int = 40) -> pd.DataFrame:
    t0 = 1_700_000_000
    times = np.arange(t0, t0 + n * 3600, 3600, dtype=float)
    close = np.linspace(100.0, 110.0, n, dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, n),
        }
    )


def test_create_dimred_reducer_selectkbest_and_identity():
    reducer, meta = ff._create_dimred_reducer("selectkbest", {"k": "bad"})
    out = reducer.fit_transform(np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=float))
    assert out.shape == (3, 2)
    assert meta["k"] == 5

    reducer, meta = ff._create_dimred_reducer("unknown", {})
    out = reducer.fit_transform([[1, 2], [3, 4]])
    assert out == [[1, 2], [3, 4]]
    assert meta == {"method": "identity"}


def test_forecast_validates_timeframe_and_seconds(monkeypatch):
    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    assert "Invalid timeframe" in ff.forecast(symbol="EURUSD", timeframe="BAD")["error"]

    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {})
    assert ff.forecast(symbol="EURUSD", timeframe="H1")["error"] == "Unsupported timeframe seconds for H1"


def test_forecast_routes_to_volatility_endpoint(monkeypatch):
    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(fv, "forecast_volatility", lambda **kwargs: {"volatility": True, "method": kwargs["method"]})

    out = ff.forecast(symbol="EURUSD", timeframe="H1", quantity="volatility", method="theta")
    assert out == {"volatility": True, "method": "theta"}

    out = ff.forecast(symbol="EURUSD", timeframe="H1", quantity="price", method="vol_garch")
    assert out == {"volatility": True, "method": "vol_garch"}


def test_forecast_handles_fetch_errors_and_short_history(monkeypatch):
    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(ff, "_parse_kv_or_json", lambda v: dict(v or {}))

    monkeypatch.setattr(ff, "_fetch_history", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fetch failed")))
    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="theta")
    assert out["error"] == "fetch failed"

    monkeypatch.setattr(ff, "_fetch_history", lambda *args, **kwargs: _sample_df(2))
    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="theta")
    assert out["error"] == "Not enough closed bars to compute forecast"


def test_forecast_rejects_seasonal_naive_without_positive_seasonality(monkeypatch):
    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(ff, "_parse_kv_or_json", lambda v: {"seasonality": 0})

    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="seasonal_naive")
    assert "seasonal_naive requires a positive 'seasonality'" in out["error"]


def test_forecast_features_builds_exog_and_aligns_for_returns(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"success": True, "method": kwargs["method"], "forecast_price": [1.0] * int(kwargs["horizon"])}

    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(ff, "_fetch_history", lambda *args, **kwargs: _sample_df(30))
    monkeypatch.setattr(ff, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(fe, "forecast_engine", fake_engine)

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=5,
        quantity="return",
        params={"alpha": 1},
        features={"include": "open,high low", "future_covariates": "hour,dow,fourier:24,is_weekend"},
    )

    assert out["success"] is True
    assert captured["params"]["alpha"] == 1
    assert captured["params"]["symbol"] == "EURUSD"
    assert captured["params"]["timeframe"] == "H1"
    assert captured["exog_used"].shape[0] == 29
    assert captured["exog_used"].shape[1] == 10
    assert captured["exog_future"].shape == (5, 10)


def test_forecast_dimred_failure_falls_back_and_engine_error_passthrough(monkeypatch):
    captured = {}

    def fake_engine(**kwargs):
        captured.update(kwargs)
        return {"error": "engine failure"}

    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(ff, "_fetch_history", lambda *args, **kwargs: _sample_df(20))
    monkeypatch.setattr(ff, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(ff, "_create_dimred_reducer", lambda method, params: (_ for _ in ()).throw(RuntimeError("dimred failed")))
    monkeypatch.setattr(fe, "forecast_engine", fake_engine)

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=4,
        features={"include": "open,high"},
        dimred_method="pca",
    )

    assert out["error"] == "engine failure"
    assert captured["exog_used"] is not None
    assert captured["exog_used"].shape[1] == 2


def test_forecast_target_spec_transform_and_alias_paths(monkeypatch):
    captured = {"ti_parse": 0, "ti_apply": 0}

    def fake_engine(**kwargs):
        return {"success": True, "forecast_price": [1.0] * int(kwargs["horizon"])}

    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(ff, "_parse_kv_or_json", lambda v: dict(v or {}))
    monkeypatch.setattr(ff, "_fetch_history", lambda *args, **kwargs: _sample_df(6))
    monkeypatch.setattr(fe, "forecast_engine", fake_engine)
    monkeypatch.setattr(ff, "_parse_ti_specs_util", lambda s: captured.__setitem__("ti_parse", captured["ti_parse"] + 1) or [{"spec": s}])
    monkeypatch.setattr(ff, "_apply_ta_indicators_util", lambda *args, **kwargs: captured.__setitem__("ti_apply", captured["ti_apply"] + 1))

    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=2,
        target_spec={"base": "close", "transform": "return", "k": 3},
    )
    assert out["error"] == "Not enough data for transformed target"

    monkeypatch.setattr(ff, "_fetch_history", lambda *args, **kwargs: _sample_df(20))
    out = ff.forecast(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        horizon=3,
        target_spec={"base": "typical", "transform": "none", "indicators": "sma:close:5"},
    )
    assert out["success"] is True
    assert captured["ti_parse"] == 1
    assert captured["ti_apply"] == 1


def test_forecast_returns_traceback_on_unexpected_exception(monkeypatch):
    monkeypatch.setattr(ff, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(ff, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(ff, "_parse_kv_or_json", lambda v: (_ for _ in ()).throw(ValueError("parse exploded")))

    out = ff.forecast(symbol="EURUSD", timeframe="H1", method="theta")

    assert out["error"].startswith("Forecast failed: parse exploded")
    assert "traceback" in out
