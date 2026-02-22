from __future__ import annotations

from types import SimpleNamespace

import math
import numpy as np

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

    assert vol._bars_per_year("H1") > 8000
    assert vol._bars_per_year("BAD") == 0

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
    monkeypatch.setattr(vol, "_mt5_epoch_to_utc", lambda t: float(t))
    monkeypatch.setattr(vol.mt5, "symbol_select", lambda _symbol, _visible: True)
    monkeypatch.setattr(vol.mt5, "last_error", lambda: (0, "ok"))

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="theta", proxy=None)
    assert "require 'proxy'" in out["error"]

    out = vol.forecast_volatility(symbol="EURUSD", timeframe="H1", method="theta", proxy="bad_proxy")  # type: ignore[arg-type]
    assert "Unsupported proxy" in out["error"]

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="theta",
        proxy="squared_return",
        params={"alpha": 0.3},
    )
    assert out["success"] is True
    assert out["method"] == "theta"
    assert out["proxy"] == "squared_return"
    assert out["horizon_sigma_return"] > 0


def test_forecast_volatility_direct_methods_and_short_data(monkeypatch):
    monkeypatch.setattr(vol, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(vol, "TIMEFRAME_SECONDS", {"H1": 3600})
    monkeypatch.setattr(vol, "_ensure_symbol_ready", lambda _symbol: None)
    monkeypatch.setattr(vol.mt5, "symbol_info", lambda _symbol: SimpleNamespace(visible=True))
    monkeypatch.setattr(vol.mt5, "symbol_info_tick", lambda _symbol: SimpleNamespace(time=1_700_100_000))
    monkeypatch.setattr(vol, "_mt5_epoch_to_utc", lambda t: float(t))
    monkeypatch.setattr(vol.mt5, "last_error", lambda: (0, "ok"))
    monkeypatch.setattr(vol, "_mt5_copy_rates_from", lambda *args, **kwargs: _rates(240))

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="ewma",
        params='{"lookback": 80, "lambda_": 0.9}',
    )
    assert out["success"] is True
    assert out["method"] == "ewma"
    assert out["params_used"]["lookback"] == 80

    out = vol.forecast_volatility(
        symbol="EURUSD",
        timeframe="H1",
        method="realized_kernel",
        params={"window": 60, "kernel": "bartlett", "bandwidth": 5},
    )
    assert out["success"] is True
    assert out["params_used"]["kernel"] == "bartlett"

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
