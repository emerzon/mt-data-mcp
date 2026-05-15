from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mtdata.forecast.requests import ForecastVolatilityEstimateRequest
from mtdata.forecast.use_cases import run_forecast_volatility_estimate
from mtdata.forecast import volatility as vol


def _rates(n: int = 360, start: int = 1_700_000_000, step: int = 3600):
    close = np.linspace(100.0, 120.0, n, dtype=float)
    open_ = close - 0.1
    high = close + 0.3
    low = close - 0.4
    out = []
    for i in range(n):
        out.append(
            {
                "time": float(start + i * step),
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "tick_volume": 100,
                "spread": 1,
                "real_volume": 100,
            }
        )
    return out


def test_volatility_metadata_and_helper_functions(monkeypatch):
    monkeypatch.setattr(vol, "_ARCH_AVAILABLE", False)
    methods = vol.get_volatility_methods_data()["methods"]
    by_name = {m["method"]: m for m in methods}

    assert by_name["ewma"]["available"] is True
    assert by_name["garch"]["available"] is False
    assert "arch" in by_name["garch"]["requires"]
    assert by_name["theta"]["available"] is True

    assert vol._bars_per_year("H1") == 6048.0
    assert math.isnan(vol._bars_per_year("BAD"))

    assert vol._kernel_weight("bartlett", 1, 4) > 0
    assert vol._kernel_weight("parzen", 1, 4) > 0
    assert vol._kernel_weight("tukey_hanning", 1, 4) > 0

    assert math.isnan(vol._realized_kernel_variance(np.array([0.1, 0.2]), bandwidth=None))
    rk = vol._realized_kernel_variance(np.array([0.1, -0.2, 0.05, 0.03, -0.01]), bandwidth=2, kernel="bartlett")
    assert math.isfinite(rk)
    assert rk >= 0.0

    p = vol._parkinson_sigma_sq(np.array([2.0, 3.0]), np.array([1.0, 1.5]))
    gk = vol._garman_klass_sigma_sq(np.array([1.2, 2.0]), np.array([2.0, 3.0]), np.array([1.0, 1.5]), np.array([1.8, 2.8]))
    rs = vol._rogers_satchell_sigma_sq(np.array([1.2, 2.0]), np.array([2.0, 3.0]), np.array([1.0, 1.5]), np.array([1.8, 2.8]))
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(gk))
    assert np.all(np.isfinite(rs))


def test_finalize_volatility_output_compact_omits_explanatory_fields():
    payload = {
        "success": True,
        "sigma_bar_return": 0.01,
        "sigma_annual_return": 0.5,
        "horizon_sigma_return": 0.02,
        "horizon_sigma_annual": 0.8,
        "params_used": {"lookback": 100, "lambda_": 0.94},
        "params_explained": {"lambda_": "decay explanation"},
    }

    compact = vol._finalize_volatility_output(payload, detail="compact")
    full = vol._finalize_volatility_output(payload, detail="full")

    assert compact["volatility_per_bar"] == pytest.approx(0.01)
    assert compact["volatility_horizon"] == pytest.approx(0.02)
    assert compact["volatility_unit"] == "return_fraction"
    assert "sigma_bar_return" not in compact
    assert "horizon_sigma_return" not in compact
    assert "params_used" not in compact
    assert "params_explained" not in compact
    assert "volatility_interpretation" not in compact
    assert full["sigma_bar_return"] == pytest.approx(0.01)
    assert full["params_used"]["lookback"] == 100
    assert set(full["volatility_interpretation"]) == {
        "volatility_per_bar",
        "volatility_annualized",
        "volatility_horizon",
        "volatility_horizon_annualized",
        "volatility_unit",
    }
    assert "sqrt-time scaling" in full["volatility_interpretation"]["volatility_horizon_annualized"]


def test_forecast_volatility_estimate_preserves_impl_payload_without_public_stripper():
    def fake_forecast_volatility(**_kwargs):
        return {
            "success": True,
            "volatility_per_bar": 0.01,
            "volatility_annualized": 0.5,
            "volatility_horizon": 0.02,
            "volatility_horizon_annualized": 0.8,
            "sigma_bar_return": 0.01,
            "sigma_annual_return": 0.5,
            "horizon_sigma_return": 0.02,
            "horizon_sigma_annual": 0.8,
            "volatility_interpretation": {
                "volatility_per_bar": "per bar",
            },
        }

    out = run_forecast_volatility_estimate(
        ForecastVolatilityEstimateRequest(symbol="EURUSD", detail="full"),
        forecast_volatility_impl=fake_forecast_volatility,
    )

    assert out["volatility_per_bar"] == pytest.approx(0.01)
    assert out["volatility_horizon"] == pytest.approx(0.02)
    assert out["sigma_bar_return"] == pytest.approx(0.01)
    assert out["sigma_annual_return"] == pytest.approx(0.5)
    assert out["horizon_sigma_return"] == pytest.approx(0.02)
    assert out["horizon_sigma_annual"] == pytest.approx(0.8)
    assert out["volatility_interpretation"] == {"volatility_per_bar": "per bar"}


def test_forecast_volatility_validations(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="BAD")
    assert "Invalid timeframe" in out["error"]

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="nope")  # type: ignore[arg-type]
    assert out["error"] == "Invalid method: nope"

    monkeypatch.setattr(vol, "_ARCH_AVAILABLE", False)
    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="garch")
    assert "requires 'arch' package" in out["error"]


def test_forecast_volatility_general_theta_and_proxy_errors(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(vol, "_mt5_copy_rates_from", lambda *args, **kwargs: _rates(360))
    monkeypatch.setattr(vol.mt5, "symbol_info", lambda _symbol: SimpleNamespace(visible=False))
    monkeypatch.setattr(vol.mt5, "symbol_info_tick", lambda _symbol: SimpleNamespace(time=1_700_100_000))
    monkeypatch.setattr("mtdata.utils.mt5._mt5_epoch_to_utc", lambda t: float(t))
    monkeypatch.setattr(vol.mt5, "symbol_select", lambda _symbol, _visible: True)
    monkeypatch.setattr(vol.mt5, "last_error", lambda: (0, "ok"))

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="theta", proxy=None)
    assert "require 'proxy'" in out["error"]

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="theta", proxy="bad_proxy")  # type: ignore[arg-type]
    assert "Unsupported proxy" in out["error"]

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=4,
        method="theta",
        proxy="squared_return",
        params={"alpha": 0.3},
    )
    assert out["success"] is True
    assert out["method"] == "theta"
    assert out["proxy"] == "squared_return"
    assert out["horizon_sigma_return"] > 0
    assert out["volatility_per_bar"] == out["sigma_bar_return"]
    assert out["volatility_horizon"] == out["horizon_sigma_return"]
    assert "volatility_horizon" in out["volatility_interpretation"]
    expected_bpy = vol._bars_per_year("H1")
    assert out["sigma_annual_return"] == pytest.approx(out["sigma_bar_return"] * math.sqrt(expected_bpy))
    assert out["horizon_sigma_annual"] == pytest.approx(
        out["horizon_sigma_return"] * math.sqrt(expected_bpy / 4)
    )


def test_forecast_volatility_direct_methods_and_short_data(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(vol.mt5, "symbol_info", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(vol.mt5, "symbol_info_tick", lambda _symbol: SimpleNamespace(time=1_700_100_000))
    monkeypatch.setattr("mtdata.utils.mt5._mt5_epoch_to_utc", lambda t: float(t))
    monkeypatch.setattr(vol.mt5, "last_error", lambda: (0, "ok"))
    monkeypatch.setattr(vol, "_mt5_copy_rates_from", lambda *args, **kwargs: _rates(240))

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=5,
        method="ewma",
        params='{"lookback": 80, "lambda_": 0.9}',
    )
    assert out["success"] is True
    assert out["method"] == "ewma"
    assert out["params_used"]["lookback"] == 80
    assert out["params_used"]["lambda_source"] == "lambda_"
    assert out["params_used"]["decay_factor"] == pytest.approx(0.9)
    assert "params_explained" in out
    assert "lambda_" in out["params_explained"]
    assert "decay_factor" in out["params_explained"]
    expected_bpy = vol._bars_per_year("H1")
    assert out["sigma_annual_return"] == pytest.approx(out["sigma_bar_return"] * math.sqrt(expected_bpy))
    assert out["horizon_sigma_annual"] == pytest.approx(
        out["horizon_sigma_return"] * math.sqrt(expected_bpy / 5)
    )
    assert out["horizon_sigma_annual"] == pytest.approx(out["sigma_annual_return"], rel=1e-6)

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="realized_kernel",
        params={"window": 60, "kernel": "bartlett", "bandwidth": 5},
    )
    assert out["success"] is True
    assert out["params_used"]["kernel"] == "bartlett"
    assert out["volatility_horizon"] == out["volatility_per_bar"]
    assert "horizon=1" in out["volatility_interpretation"]["horizon_note"]

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="parkinson",
        params={"window": 20},
    )
    assert out["success"] is True
    assert out["method"] == "parkinson"

    monkeypatch.setattr(vol, "_mt5_copy_rates_from", lambda *args, **kwargs: _rates(5))
    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="ewma")
    assert "Insufficient returns" in out["error"]


def test_forecast_volatility_yang_zhang_weights_overnight_variance(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(vol.mt5, "symbol_info", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(vol.mt5, "symbol_info_tick", lambda _symbol: SimpleNamespace(time=1_700_100_000))
    monkeypatch.setattr("mtdata.utils.mt5._mt5_epoch_to_utc", lambda t: float(t))
    monkeypatch.setattr(vol.mt5, "last_error", lambda: (0, "ok"))

    rows = [
        (100.0, 110.0),
        (130.0, 140.0),
        (126.0, 128.0),
        (150.0, 151.0),
        (149.0, 170.0),
        (171.0, 172.0),
        (173.0, 174.0),
    ]
    bars = []
    for idx, (open_, close) in enumerate(rows):
        bars.append(
            {
                "time": float(1_700_000_000 + idx * 3600),
                "open": open_,
                "high": max(open_, close),
                "low": min(open_, close),
                "close": close,
                "tick_volume": 100,
                "spread": 1,
                "real_volume": 100,
            }
        )
    monkeypatch.setattr(vol, "_mt5_copy_rates_from", lambda *args, **kwargs: bars)

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="yang_zhang",
        params={"window": 4},
    )

    used_bars = bars[:-1]
    open_ = np.array([bar["open"] for bar in used_bars], dtype=float)
    high = np.array([bar["high"] for bar in used_bars], dtype=float)
    low = np.array([bar["low"] for bar in used_bars], dtype=float)
    close = np.array([bar["close"] for bar in used_bars], dtype=float)
    oc = np.log(np.maximum(open_[1:], 1e-12)) - np.log(np.maximum(close[:-1], 1e-12))
    co = np.log(np.maximum(close[1:], 1e-12)) - np.log(np.maximum(open_[1:], 1e-12))
    rs = (
        (np.log(np.maximum(high[1:], 1e-12)) - np.log(np.maximum(close[1:], 1e-12)))
        * (np.log(np.maximum(high[1:], 1e-12)) - np.log(np.maximum(open_[1:], 1e-12)))
        + (np.log(np.maximum(low[1:], 1e-12)) - np.log(np.maximum(close[1:], 1e-12)))
        * (np.log(np.maximum(low[1:], 1e-12)) - np.log(np.maximum(open_[1:], 1e-12)))
    )
    window = 4
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    oc_var = float(np.var(oc[-window:], ddof=0))
    co_var = float(np.var(co[-window:], ddof=0))
    rs_mean = float(np.mean(rs[-window:]))
    expected_sigma2 = oc_var + k * co_var + (1 - k) * rs_mean
    wrong_sigma2 = co_var + k * oc_var + (1 - k) * rs_mean

    assert out["success"] is True
    assert rs_mean == pytest.approx(0.0)
    assert expected_sigma2 > wrong_sigma2
    assert out["sigma_bar_return"] == pytest.approx(math.sqrt(expected_sigma2))


def test_forecast_volatility_ensemble_aggregates_component_methods(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(vol.mt5, "symbol_info", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(vol.mt5, "symbol_info_tick", lambda _symbol: SimpleNamespace(time=1_700_100_000))
    monkeypatch.setattr("mtdata.utils.mt5._mt5_epoch_to_utc", lambda t: float(t))
    monkeypatch.setattr(vol.mt5, "last_error", lambda: (0, "ok"))
    monkeypatch.setattr(vol, "_mt5_copy_rates_from", lambda *args, **kwargs: _rates(240))

    ewma = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", horizon=5, method="ewma")
    rolling_std = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", horizon=5, method="rolling_std")
    ensemble = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        horizon=5,
        method="ensemble",
        params={
            "methods": ["ewma", "rolling_std"],
            "aggregator": "mean",
            "expose_components": True,
        },
    )

    assert ewma["success"] is True
    assert rolling_std["success"] is True
    assert ensemble["success"] is True
    assert ensemble["method"] == "ensemble"
    assert ensemble["params_used"]["methods"] == ["ewma", "rolling_std"]
    assert len(ensemble["components"]) == 2
    assert ensemble["sigma_bar_return"] == pytest.approx(
        (float(ewma["sigma_bar_return"]) + float(rolling_std["sigma_bar_return"])) / 2.0
    )
    assert ensemble["horizon_sigma_return"] == pytest.approx(
        (float(ewma["horizon_sigma_return"]) + float(rolling_std["horizon_sigma_return"])) / 2.0
    )
    expected_bpy = vol._bars_per_year("H1")
    assert ensemble["sigma_annual_return"] == pytest.approx(
        ensemble["sigma_bar_return"] * math.sqrt(expected_bpy)
    )
    assert ensemble["horizon_sigma_annual"] == pytest.approx(
        ensemble["horizon_sigma_return"] * math.sqrt(expected_bpy / 5)
    )
    assert ensemble["horizon_sigma_annual"] == pytest.approx(ensemble["sigma_annual_return"], rel=1e-6)
