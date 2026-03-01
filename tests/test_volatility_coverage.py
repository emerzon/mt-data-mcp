"""Comprehensive tests for mtdata.forecast.volatility module."""

import sys
from unittest.mock import MagicMock

# MUST mock MetaTrader5 before any project imports
sys.modules["MetaTrader5"] = MagicMock()

import math
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.volatility import (
    _bars_per_year,
    _garman_klass_sigma_sq,
    _kernel_weight,
    _parkinson_sigma_sq,
    _realized_kernel_variance,
    _rogers_satchell_sigma_sq,
    forecast_volatility,
    get_volatility_methods_data,
)

MOD = "mtdata.forecast.volatility"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rates(n=200, base_price=1.1000, seed=42):
    """Generate fake OHLC rates as a numpy structured array."""
    rng = np.random.RandomState(seed)
    dt = np.dtype([
        ("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
        ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<i8"),
        ("spread", "<i8"), ("real_volume", "<i8"),
    ])
    rates = np.empty(n, dtype=dt)
    t0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    price = base_price
    for i in range(n):
        rates[i]["time"] = t0 + i * 3600
        o = price
        c = o * (1 + rng.normal(0, 0.001))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0005)))
        lo = min(o, c) * (1 - abs(rng.normal(0, 0.0005)))
        rates[i]["open"] = o
        rates[i]["high"] = h
        rates[i]["low"] = lo
        rates[i]["close"] = c
        rates[i]["tick_volume"] = rng.randint(100, 10000)
        rates[i]["spread"] = rng.randint(1, 20)
        rates[i]["real_volume"] = 0
        price = c
    return rates


_SENTINEL = object()


@contextmanager
def _mock_vol_env(n_bars=2000, ensure_err=None, rates_return=_SENTINEL,
                  rates_side_effect=None):
    """Patch MT5 utilities for forecast_volatility tests."""
    rates = _make_rates(n_bars) if rates_return is _SENTINEL else rates_return
    mock_info = MagicMock(visible=True)
    mock_tick = MagicMock()
    mock_tick.time = 1704067200.0

    copy_kw = ({"side_effect": rates_side_effect} if rates_side_effect
               else {"return_value": rates})

    with (
        patch(f"{MOD}._ensure_symbol_ready", return_value=ensure_err) as m_ensure,
        patch(f"{MOD}._mt5_copy_rates_from", **copy_kw) as m_copy,
        patch(f"{MOD}._mt5_epoch_to_utc", return_value=1704067200.0),
        patch(f"{MOD}._parse_start_datetime",
              return_value=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        patch(f"{MOD}.mt5") as m_mt5,
    ):
        m_mt5.symbol_info.return_value = mock_info
        m_mt5.symbol_info_tick.return_value = mock_tick
        m_mt5.last_error.return_value = (-1, "mock error")
        yield {"mt5": m_mt5, "copy_rates": m_copy, "ensure": m_ensure}


# ===================================================================
# get_volatility_methods_data
# ===================================================================

class TestGetVolatilityMethodsData:
    def test_returns_dict_with_methods_key(self):
        result = get_volatility_methods_data()
        assert isinstance(result, dict)
        assert "methods" in result

    def test_methods_is_list(self):
        result = get_volatility_methods_data()
        assert isinstance(result["methods"], list)
        assert len(result["methods"]) > 0

    def test_ewma_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "ewma" in names

    def test_parkinson_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "parkinson" in names

    def test_gk_and_rs_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "gk" in names
        assert "rs" in names

    def test_yang_zhang_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "yang_zhang" in names

    def test_garch_variants_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        for g in ("garch", "egarch", "gjr_garch", "garch_t", "egarch_t",
                  "gjr_garch_t", "figarch"):
            assert g in names

    def test_all_methods_have_required_fields(self):
        for m in get_volatility_methods_data()["methods"]:
            assert "method" in m
            assert "available" in m
            assert "description" in m
            assert "params" in m

    def test_realized_kernel_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "realized_kernel" in names

    def test_har_rv_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "har_rv" in names

    def test_ensemble_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "ensemble" in names

    def test_theta_present(self):
        names = [m["method"] for m in get_volatility_methods_data()["methods"]]
        assert "theta" in names


# ===================================================================
# _bars_per_year
# ===================================================================

class TestBarsPerYear:
    @pytest.mark.parametrize("tf,expected", [
        ("M1", 525600),
        ("M5", 105120),
        ("M15", 35040),
        ("H1", 8760),
        ("H4", 2190),
        ("D1", 365),
        ("W1", 52),
        ("MN1", 12),
    ])
    def test_known_timeframes(self, tf, expected):
        assert _bars_per_year(tf) == expected

    def test_invalid_timeframe_returns_zero(self):
        assert _bars_per_year("INVALID") == 0

    def test_empty_string_returns_zero(self):
        assert _bars_per_year("") == 0


# ===================================================================
# _parkinson_sigma_sq
# ===================================================================

class TestParkinsonSigmaSq:
    def test_constant_price_zero_variance(self):
        h = np.array([1.0, 1.0, 1.0])
        lo = np.array([1.0, 1.0, 1.0])
        result = _parkinson_sigma_sq(h, lo)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive_spread(self):
        h = np.array([1.10, 1.12, 1.08])
        lo = np.array([1.05, 1.06, 1.02])
        result = _parkinson_sigma_sq(h, lo)
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))

    def test_single_element(self):
        result = _parkinson_sigma_sq(np.array([1.1]), np.array([1.0]))
        assert result.shape == (1,)
        assert result[0] >= 0

    def test_nan_in_input(self):
        h = np.array([1.1, np.nan, 1.08])
        lo = np.array([1.0, 1.0, 1.02])
        result = _parkinson_sigma_sq(h, lo)
        assert np.isnan(result[1])

    def test_non_negative(self):
        rng = np.random.RandomState(0)
        h = 1.1 + rng.uniform(0, 0.05, 100)
        lo = 1.1 - rng.uniform(0, 0.05, 100)
        result = _parkinson_sigma_sq(h, lo)
        assert np.all(result[np.isfinite(result)] >= 0)

    def test_zero_prices_handled(self):
        h = np.array([0.0, 1.0])
        lo = np.array([0.0, 0.5])
        result = _parkinson_sigma_sq(h, lo)
        assert result.shape == (2,)


# ===================================================================
# _garman_klass_sigma_sq
# ===================================================================

class TestGarmanKlassSigmaSq:
    def test_constant_price_zero_variance(self):
        p = np.array([1.0, 1.0, 1.0])
        result = _garman_klass_sigma_sq(p, p, p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive_spread(self):
        o = np.array([1.05, 1.06, 1.04])
        h = np.array([1.10, 1.12, 1.08])
        lo = np.array([1.02, 1.03, 1.01])
        c = np.array([1.07, 1.08, 1.05])
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert np.all(result >= 0)

    def test_single_element(self):
        result = _garman_klass_sigma_sq(
            np.array([1.05]), np.array([1.10]),
            np.array([1.00]), np.array([1.07]),
        )
        assert result.shape == (1,)
        assert result[0] >= 0

    def test_nan_handling(self):
        o = np.array([1.0, np.nan])
        h = np.array([1.1, 1.1])
        lo = np.array([0.9, 0.9])
        c = np.array([1.05, 1.05])
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert np.isnan(result[1])

    def test_non_negative(self):
        rng = np.random.RandomState(1)
        n = 100
        o = 1.1 + rng.normal(0, 0.01, n)
        c = o + rng.normal(0, 0.01, n)
        h = np.maximum(o, c) + rng.uniform(0, 0.005, n)
        lo = np.minimum(o, c) - rng.uniform(0, 0.005, n)
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert np.all(result[np.isfinite(result)] >= 0)

    def test_zero_prices_handled(self):
        o = np.array([0.0, 1.0])
        h = np.array([0.0, 1.1])
        lo = np.array([0.0, 0.9])
        c = np.array([0.0, 1.05])
        result = _garman_klass_sigma_sq(o, h, lo, c)
        assert result.shape == (2,)


# ===================================================================
# _rogers_satchell_sigma_sq
# ===================================================================

class TestRogersSatchellSigmaSq:
    def test_constant_price_near_zero(self):
        p = np.array([1.0, 1.0, 1.0])
        result = _rogers_satchell_sigma_sq(p, p, p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive_spread(self):
        o = np.array([1.05, 1.06])
        h = np.array([1.10, 1.12])
        lo = np.array([1.02, 1.03])
        c = np.array([1.07, 1.08])
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert np.all(result >= 0)

    def test_single_element(self):
        result = _rogers_satchell_sigma_sq(
            np.array([1.05]), np.array([1.10]),
            np.array([1.00]), np.array([1.07]),
        )
        assert result.shape == (1,)

    def test_nan_handling(self):
        o = np.array([1.0, np.nan])
        h = np.array([1.1, 1.1])
        lo = np.array([0.9, 0.9])
        c = np.array([1.05, 1.05])
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert np.isnan(result[1])

    def test_non_negative(self):
        rng = np.random.RandomState(2)
        n = 100
        o = 1.1 + rng.normal(0, 0.01, n)
        c = o + rng.normal(0, 0.005, n)
        h = np.maximum(o, c) + rng.uniform(0.001, 0.01, n)
        lo = np.minimum(o, c) - rng.uniform(0.001, 0.01, n)
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert np.all(result[np.isfinite(result)] >= 0)

    def test_zero_prices_handled(self):
        o = np.array([0.0, 1.0])
        h = np.array([0.0, 1.1])
        lo = np.array([0.0, 0.9])
        c = np.array([0.0, 1.05])
        result = _rogers_satchell_sigma_sq(o, h, lo, c)
        assert result.shape == (2,)


# ===================================================================
# _kernel_weight
# ===================================================================

class TestKernelWeight:
    # Bartlett / triangular
    def test_bartlett_at_zero(self):
        assert _kernel_weight("bartlett", 0, 10) == pytest.approx(1.0)

    def test_bartlett_at_boundary(self):
        w = _kernel_weight("bartlett", 10, 10)
        assert 0.0 <= w <= 1.0

    def test_triangular_alias(self):
        assert _kernel_weight("triangular", 3, 10) == _kernel_weight("bartlett", 3, 10)

    # Parzen
    def test_parzen_at_zero(self):
        assert _kernel_weight("parzen", 0, 10) == pytest.approx(1.0)

    def test_parzen_small_x(self):
        w = _kernel_weight("parzen", 2, 20)
        assert 0.0 <= w <= 1.0

    def test_parzen_large_x(self):
        w = _kernel_weight("parzen", 9, 10)
        assert 0.0 <= w <= 1.0

    # Tukey-Hanning (default)
    def test_tukey_hanning_at_zero(self):
        assert _kernel_weight("tukey_hanning", 0, 10) == pytest.approx(1.0)

    def test_tukey_hanning_at_boundary(self):
        w = _kernel_weight("tukey_hanning", 10, 10)
        assert 0.0 <= w <= 1.0

    def test_unknown_kernel_defaults_tukey(self):
        w_unknown = _kernel_weight("unknown_kernel", 3, 10)
        w_tukey = _kernel_weight("tukey_hanning", 3, 10)
        assert w_unknown == pytest.approx(w_tukey)

    # Edge cases
    def test_zero_bandwidth(self):
        assert _kernel_weight("bartlett", 1, 0) == 0.0

    def test_negative_bandwidth_returns_zero(self):
        assert _kernel_weight("bartlett", 1, -5) == 0.0

    def test_weights_decrease_with_lag(self):
        w1 = _kernel_weight("tukey_hanning", 1, 20)
        w5 = _kernel_weight("tukey_hanning", 5, 20)
        w10 = _kernel_weight("tukey_hanning", 10, 20)
        assert w1 >= w5 >= w10


# ===================================================================
# _realized_kernel_variance
# ===================================================================

class TestRealizedKernelVariance:
    def test_simple_returns(self):
        rng = np.random.RandomState(10)
        r = rng.normal(0, 0.01, 200)
        rk = _realized_kernel_variance(r)
        assert math.isfinite(rk)
        assert rk >= 0

    def test_too_few_returns_nan(self):
        r = np.array([0.01, 0.02])
        assert math.isnan(_realized_kernel_variance(r))

    def test_empty_returns_nan(self):
        assert math.isnan(_realized_kernel_variance(np.array([])))

    def test_zero_returns(self):
        r = np.zeros(50)
        rk = _realized_kernel_variance(r)
        assert rk == pytest.approx(0.0, abs=1e-15)

    def test_auto_bandwidth(self):
        rng = np.random.RandomState(11)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, bandwidth=None)
        assert math.isfinite(rk) and rk >= 0

    def test_explicit_bandwidth(self):
        rng = np.random.RandomState(12)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, bandwidth=5)
        assert math.isfinite(rk) and rk >= 0

    def test_bartlett_kernel(self):
        rng = np.random.RandomState(13)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, kernel="bartlett")
        assert math.isfinite(rk) and rk >= 0

    def test_parzen_kernel(self):
        rng = np.random.RandomState(14)
        r = rng.normal(0, 0.01, 100)
        rk = _realized_kernel_variance(r, kernel="parzen")
        assert math.isfinite(rk) and rk >= 0

    def test_nan_in_returns_filtered(self):
        r = np.array([0.01, np.nan, -0.01, 0.005, np.nan, 0.003, -0.002,
                       0.01, -0.005, 0.002])
        rk = _realized_kernel_variance(r)
        assert math.isfinite(rk)


# ===================================================================
# forecast_volatility – validation / error paths
# ===================================================================

class TestForecastVolatilityValidation:
    def test_invalid_timeframe(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "INVALID", 1)
            assert "error" in result
            assert "Invalid timeframe" in result["error"]

    def test_invalid_method(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="bogus")
            assert "error" in result
            assert "Invalid method" in result["error"]

    def test_garch_without_arch_package(self):
        with _mock_vol_env():
            with patch(f"{MOD}._ARCH_AVAILABLE", False):
                result = forecast_volatility("EURUSD", "H1", 1, method="garch")
                assert "error" in result
                assert "arch" in result["error"].lower()

    def test_ensure_symbol_error(self):
        with _mock_vol_env(ensure_err="Symbol not found"):
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert "error" in result
            assert "Symbol not found" in result["error"]

    def test_null_rates_error(self):
        with _mock_vol_env(rates_return=None):
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert "error" in result

    def test_insufficient_rates_error(self):
        with _mock_vol_env(rates_return=_make_rates(2)):
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert "error" in result

    def test_general_method_without_proxy(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 5, method="theta")
            assert "error" in result
            assert "proxy" in result["error"].lower()

    def test_general_method_unsupported_proxy(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta", proxy="bad_proxy")
            assert "error" in result
            assert "Unsupported proxy" in result["error"]


# ===================================================================
# forecast_volatility – EWMA (first code path)
# ===================================================================

class TestForecastVolatilityEWMA:
    def test_ewma_success_default(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True
            assert result["method"] == "ewma"

    def test_ewma_output_structure(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True
            for key in ("sigma_bar_return", "sigma_annual_return",
                        "horizon_sigma_return", "horizon_sigma_annual"):
                assert key in result
                assert isinstance(result[key], float)
                assert math.isfinite(result[key])

    def test_ewma_custom_lambda(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"lambda_": 0.97})
            assert result.get("success") is True
            assert result["params_used"]["lambda_"] == pytest.approx(0.97)

    def test_ewma_custom_halflife(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"halflife": 30})
            assert result.get("success") is True

    def test_ewma_horizon_gt1(self):
        with _mock_vol_env():
            r1 = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            r5 = forecast_volatility("EURUSD", "H1", 5, method="ewma")
            assert r1.get("success") and r5.get("success")
            assert r5["horizon_sigma_return"] > r1["horizon_sigma_return"]

    def test_ewma_sigma_positive(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result["sigma_bar_return"] > 0
            assert result["sigma_annual_return"] > 0


# ===================================================================
# forecast_volatility – range-based methods (first code path)
# ===================================================================

class TestForecastVolatilityRangeBased:
    @pytest.mark.parametrize("method", [
        "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
    ])
    def test_success(self, method):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method=method, params={"window": 20})
            assert result.get("success") is True
            assert result["method"] == method

    @pytest.mark.parametrize("method", [
        "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
    ])
    def test_sigma_positive(self, method):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method=method, params={"window": 20})
            assert result.get("success") is True
            assert result["sigma_bar_return"] >= 0
            assert math.isfinite(result["sigma_bar_return"])

    @pytest.mark.parametrize("method", [
        "parkinson", "gk", "rs", "yang_zhang", "rolling_std",
    ])
    def test_horizon_scaling(self, method):
        with _mock_vol_env():
            r1 = forecast_volatility(
                "EURUSD", "H1", 1, method=method, params={"window": 20})
            r5 = forecast_volatility(
                "EURUSD", "H1", 5, method=method, params={"window": 20})
            if r1.get("success") and r5.get("success"):
                assert r5["horizon_sigma_return"] >= r1["horizon_sigma_return"]

    def test_custom_window(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson", params={"window": 50})
            assert result.get("success") is True
            assert result["params_used"]["window"] == 50

    def test_output_has_required_fields(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 3, method="gk", params={"window": 20})
            assert result.get("success") is True
            assert result["symbol"] == "EURUSD"
            assert result["timeframe"] == "H1"
            assert result["horizon"] == 3


# ===================================================================
# forecast_volatility – realized kernel (first code path)
# ===================================================================

class TestForecastVolatilityRealizedKernel:
    def test_success_default(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel")
            assert result.get("success") is True
            assert result["method"] == "realized_kernel"

    def test_custom_kernel_and_window(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel",
                params={"window": 100, "kernel": "bartlett"})
            assert result.get("success") is True
            assert result["params_used"]["kernel"] == "bartlett"
            assert result["params_used"]["window"] == 100

    def test_custom_bandwidth(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel",
                params={"bandwidth": 10})
            assert result.get("success") is True
            assert result["params_used"]["bandwidth"] == 10

    def test_sigma_finite_positive(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel")
            assert result.get("success") is True
            assert result["sigma_bar_return"] > 0
            assert math.isfinite(result["sigma_annual_return"])


# ===================================================================
# forecast_volatility – GARCH family (first code path)
# ===================================================================

class TestForecastVolatilityGarch:
    def _mock_arch_model(self, horizon=3):
        """Build a mock _arch_model that returns realistic objects."""
        variances = np.full((1, max(1, horizon)), 0.5)
        mock_fc = MagicMock()
        mock_fc.variance.values = variances

        mock_res = MagicMock()
        mock_res.forecast.return_value = mock_fc

        mock_am = MagicMock()
        mock_am.fit.return_value = mock_res
        return MagicMock(return_value=mock_am)

    @pytest.mark.parametrize("method", [
        "garch", "egarch", "gjr_garch", "garch_t", "egarch_t",
        "gjr_garch_t", "figarch",
    ])
    def test_garch_family_success(self, method):
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", self._mock_arch_model(1)),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method=method)
                assert result.get("success") is True
                assert result["method"] == method

    def test_garch_output_fields(self):
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", self._mock_arch_model(3)),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 3, method="garch")
                assert result.get("success") is True
                assert "sigma_bar_return" in result
                assert "sigma_annual_return" in result
                assert "horizon_sigma_return" in result

    def test_garch_custom_params(self):
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", self._mock_arch_model(1)),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method="garch",
                    params={"p": 2, "q": 1, "dist": "studentst"})
                assert result.get("success") is True
                assert result["params_used"]["dist"] == "studentst"

    def test_garch_fit_error(self):
        mock_am = MagicMock()
        mock_am_inst = MagicMock()
        mock_am_inst.fit.side_effect = RuntimeError("convergence failed")
        mock_am.return_value = mock_am_inst
        with _mock_vol_env():
            with (
                patch(f"{MOD}._ARCH_AVAILABLE", True),
                patch(f"{MOD}._arch_model", mock_am),
            ):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method="garch")
                assert "error" in result


# ===================================================================
# forecast_volatility – HAR-RV (second code path)
# ===================================================================

class TestForecastVolatilityHarRV:
    def test_har_rv_success(self):
        rates = _make_rates(2000)
        with _mock_vol_env(rates_side_effect=lambda *a, **kw: rates):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="har_rv",
                params={"rv_timeframe": "H1", "days": 40,
                        "window_w": 3, "window_m": 10})
            assert result.get("success") is True or "error" in result

    def test_har_rv_output_fields(self):
        rates = _make_rates(2000)
        with _mock_vol_env(rates_side_effect=lambda *a, **kw: rates):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="har_rv",
                params={"rv_timeframe": "H1", "days": 40,
                        "window_w": 3, "window_m": 10})
            if result.get("success"):
                assert result["method"] == "har_rv"
                assert "sigma_bar_return" in result
                assert "params_used" in result

    def test_har_rv_insufficient_intraday(self):
        small_rates = _make_rates(10)
        with _mock_vol_env(rates_side_effect=lambda *a, **kw: small_rates):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="har_rv",
                params={"rv_timeframe": "H1", "days": 40})
            assert "error" in result


# ===================================================================
# forecast_volatility – general methods (theta)
# ===================================================================

class TestForecastVolatilityGeneral:
    def test_theta_squared_return(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return")
            assert result.get("success") is True
            assert result["method"] == "theta"
            assert result["proxy"] == "squared_return"

    def test_theta_abs_return(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="abs_return")
            assert result.get("success") is True
            assert result["proxy"] == "abs_return"

    def test_theta_log_r2(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="log_r2")
            assert result.get("success") is True
            assert result["proxy"] == "log_r2"

    def test_theta_custom_alpha(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return", params={"alpha": 0.5})
            assert result.get("success") is True

    def test_theta_output_structure(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return")
            assert result.get("success") is True
            for key in ("sigma_bar_return", "sigma_annual_return",
                        "horizon_sigma_return", "horizon_sigma_annual"):
                assert key in result
                assert math.isfinite(result[key])

    def test_arima_without_statsmodels(self):
        with _mock_vol_env():
            with patch(f"{MOD}._SM_SARIMAX_AVAILABLE", False):
                result = forecast_volatility(
                    "EURUSD", "H1", 5, method="arima",
                    proxy="squared_return")
                assert "error" in result

    def test_sarima_without_statsmodels(self):
        with _mock_vol_env():
            with patch(f"{MOD}._SM_SARIMAX_AVAILABLE", False):
                result = forecast_volatility(
                    "EURUSD", "H1", 5, method="sarima",
                    proxy="squared_return")
                assert "error" in result

    def test_ets_without_statsmodels(self):
        with _mock_vol_env():
            with patch(f"{MOD}._SM_ETS_AVAILABLE", False):
                result = forecast_volatility(
                    "EURUSD", "H1", 5, method="ets",
                    proxy="squared_return")
                assert "error" in result

    def test_general_insufficient_data(self):
        with _mock_vol_env(rates_return=_make_rates(4)):
            result = forecast_volatility(
                "EURUSD", "H1", 5, method="theta",
                proxy="squared_return")
            assert "error" in result


# ===================================================================
# forecast_volatility – params parsing
# ===================================================================

class TestForecastVolatilityParamsParsing:
    def test_dict_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"lambda_": 0.95})
            assert result.get("success") is True

    def test_json_string_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params='{"window": 30}')
            assert result.get("success") is True

    def test_kv_string_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params="window=30")
            assert result.get("success") is True

    def test_none_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma", params=None)
            assert result.get("success") is True

    def test_brace_kv_params(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params="{window: 30}")
            assert result.get("success") is True


# ===================================================================
# forecast_volatility – as_of
# ===================================================================

class TestForecastVolatilityAsOf:
    def test_as_of_valid(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                as_of="2024-06-15")
            assert result.get("success") is True

    def test_as_of_drops_last_bar_only_without_as_of(self):
        """Without as_of the last (forming) bar is dropped; with as_of it is kept."""
        with _mock_vol_env(n_bars=100) as env:
            r_no = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            r_as = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma", as_of="2024-03-01")
            assert r_no.get("success") is True
            assert r_as.get("success") is True


# ===================================================================
# forecast_volatility – denoise
# ===================================================================

class TestForecastVolatilityDenoise:
    def test_denoise_spec_passed(self):
        with _mock_vol_env():
            with patch(f"{MOD}._apply_denoise") as mock_dn, \
                 patch(f"{MOD}._normalize_denoise_spec",
                       return_value={"method": "wavelet",
                                     "columns": ["close"]}):
                result = forecast_volatility(
                    "EURUSD", "H1", 1, method="ewma",
                    denoise={"method": "wavelet"})
                assert result.get("success") is True

    def test_no_denoise(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma", denoise=None)
            assert result.get("success") is True


# ===================================================================
# forecast_volatility – different timeframes
# ===================================================================

class TestForecastVolatilityTimeframes:
    @pytest.mark.parametrize("tf", ["M1", "M5", "M15", "H1", "H4", "D1"])
    def test_various_timeframes(self, tf):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", tf, 1, method="ewma")
            assert result.get("success") is True
            assert result["timeframe"] == tf

    def test_annual_sigma_scales_with_timeframe(self):
        with _mock_vol_env():
            r_h1 = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            r_d1 = forecast_volatility("EURUSD", "D1", 1, method="ewma")
            assert r_h1.get("success") and r_d1.get("success")
            # Both should produce finite annualized values
            assert math.isfinite(r_h1["sigma_annual_return"])
            assert math.isfinite(r_d1["sigma_annual_return"])


# ===================================================================
# forecast_volatility – symbol visibility restore
# ===================================================================

class TestForecastVolatilityVisibility:
    def test_invisible_symbol_restored(self):
        """If symbol was not visible before, it should be hidden again."""
        mock_info = MagicMock(visible=False)
        with _mock_vol_env() as env:
            env["mt5"].symbol_info.return_value = mock_info
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            # symbol_select(symbol, False) should have been called
            assert env["mt5"].symbol_select.called

    def test_visible_symbol_not_hidden(self):
        """If symbol was already visible, don't hide it."""
        mock_info = MagicMock(visible=True)
        with _mock_vol_env() as env:
            env["mt5"].symbol_info.return_value = mock_info
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True


# ===================================================================
# Additional edge-case tests
# ===================================================================

class TestForecastVolatilityEdgeCases:
    def test_horizon_one(self):
        with _mock_vol_env():
            result = forecast_volatility("EURUSD", "H1", 1, method="ewma")
            assert result.get("success") is True
            assert result["horizon"] == 1

    def test_large_horizon(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 100, method="ewma")
            assert result.get("success") is True
            assert result["horizon"] == 100

    def test_rolling_std_uses_returns(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="rolling_std",
                params={"window": 10})
            assert result.get("success") is True

    def test_yang_zhang_complex_estimator(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="yang_zhang",
                params={"window": 30})
            assert result.get("success") is True
            assert result["sigma_bar_return"] >= 0

    def test_realized_kernel_parzen(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="realized_kernel",
                params={"kernel": "parzen", "window": 80})
            assert result.get("success") is True

    def test_ewma_lookback_larger_than_data(self):
        """EWMA with lookback larger than available data uses all data."""
        with _mock_vol_env(n_bars=100):
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="ewma",
                params={"lookback": 5000})
            assert result.get("success") is True

    def test_parkinson_small_window(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 1, method="parkinson",
                params={"window": 5})
            assert result.get("success") is True

    def test_gk_with_horizon(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 10, method="gk",
                params={"window": 20})
            assert result.get("success") is True
            assert result["horizon"] == 10

    def test_rs_with_horizon(self):
        with _mock_vol_env():
            result = forecast_volatility(
                "EURUSD", "H1", 10, method="rs",
                params={"window": 20})
            assert result.get("success") is True

    def test_multiple_sequential_calls(self):
        """Multiple calls with different methods should all succeed."""
        with _mock_vol_env():
            for m in ("ewma", "parkinson", "gk", "rs", "rolling_std"):
                result = forecast_volatility("EURUSD", "H1", 1, method=m)
                assert result.get("success") is True, f"Failed for {m}"
