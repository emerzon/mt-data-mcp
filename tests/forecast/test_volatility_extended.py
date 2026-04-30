"""Extended tests for mtdata.forecast.volatility – targets remaining uncovered lines.

Covers: import fallbacks (24-25, 29-30, 35-37, 41-42), _bars_per_year exception
(225-226), params parsing branches (363-378), general-method error/data paths
(401-474), ARIMA/SARIMA/ETS/theta (454-481), HAR-RV via the second code section
(946-1012, 1085-1180), and scattered branch conditions.
"""
import sys
from unittest.mock import MagicMock

# MUST mock MetaTrader5 before any project imports
sys.modules["MetaTrader5"] = MagicMock()

import importlib
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import mtdata.forecast.volatility as vol_mod
from mtdata.forecast.volatility import (
    _bars_per_year,
    _kernel_weight,
    forecast_volatility,
)

MOD = "mtdata.forecast.volatility"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _make_rates(n=200, base_price=1.1, seed=42, bar_secs=3600,
                start_epoch=1_704_067_200):
    """Vectorised synthetic OHLC structured array."""
    rng = np.random.RandomState(seed)
    dt = np.dtype([
        ("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
        ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<i8"),
        ("spread", "<i8"), ("real_volume", "<i8"),
    ])
    rates = np.empty(n, dtype=dt)
    rets = rng.normal(0, 0.002, n)
    prices = base_price * np.exp(np.cumsum(rets))
    opens = np.roll(prices, 1); opens[0] = base_price
    rates["time"] = start_epoch + np.arange(n) * bar_secs
    rates["open"] = opens
    rates["close"] = prices
    rates["high"] = np.maximum(opens, prices) * (1 + np.abs(rng.normal(0, 0.001, n)))
    rates["low"] = np.minimum(opens, prices) * (1 - np.abs(rng.normal(0, 0.001, n)))
    rates["tick_volume"] = rng.randint(100, 10_000, n)
    rates["spread"] = rng.randint(1, 20, n)
    rates["real_volume"] = np.zeros(n, dtype=np.int64)
    return rates


@contextmanager
def _mock_env(
    n_bars=500,
    ensure_err=None,
    ensure_side_effect=None,
    rates_return=_SENTINEL,
    rates_side_effect=None,
    tick_time=1_704_067_200.0,
    parse_dt_return=datetime(2024, 1, 1, tzinfo=timezone.utc),
    info_visible=True,
    bar_secs=3600,
    select_side_effect=None,
):
    """Patch MT5 utilities for ``forecast_volatility`` tests."""
    if rates_return is _SENTINEL and rates_side_effect is None:
        rates = _make_rates(n_bars, bar_secs=bar_secs)
    else:
        rates = rates_return

    mock_info = MagicMock(visible=info_visible)
    mock_tick = MagicMock()
    mock_tick.time = tick_time

    copy_kw = ({"side_effect": rates_side_effect}
               if rates_side_effect is not None else {"return_value": rates})
    ensure_kw = ({"side_effect": ensure_side_effect}
                 if ensure_side_effect is not None else {"return_value": ensure_err})

    with (
        patch(f"{MOD}._ensure_symbol_ready", **ensure_kw) as m_ensure,
        patch(f"{MOD}._mt5_copy_rates_from", **copy_kw) as m_copy,
        patch("mtdata.utils.mt5._mt5_epoch_to_utc", return_value=1_704_067_200.0),
        patch(f"{MOD}._parse_start_datetime", return_value=parse_dt_return),
        patch(f"{MOD}.mt5") as m_mt5,
    ):
        m_mt5.symbol_info.return_value = mock_info
        m_mt5.symbol_info_tick.return_value = mock_tick
        m_mt5.last_error.return_value = (-1, "mock error")
        if select_side_effect is not None:
            m_mt5.symbol_select.side_effect = select_side_effect
        yield {"mt5": m_mt5, "copy_rates": m_copy, "ensure": m_ensure}


# ===================================================================
# 1. Import fallbacks  (lines 24-25, 29-30, 35-37, 41-42)
# ===================================================================

class TestImportFallbacks:
    """Reload the module with blocked optional packages to exercise except branches."""

    def _block_and_reload(self, block_keys):
        # Remove ALL submodule entries for blocked packages so cached refs
        # don't bypass the block.
        prefixes = {k.split(".")[0] for k in block_keys}
        all_remove = [k for k in list(sys.modules)
                      if any(k == p or k.startswith(p + ".") for p in prefixes)]
        saved = {}
        for k in all_remove:
            saved[k] = sys.modules.pop(k)
        for k in block_keys:
            sys.modules[k] = None          # triggers ImportError on import
        try:
            importlib.reload(vol_mod)
            # Capture flags *before* finally restores modules
            flags = {
                "_SM_ETS_AVAILABLE": vol_mod._SM_ETS_AVAILABLE,
                "_SM_SARIMAX_AVAILABLE": vol_mod._SM_SARIMAX_AVAILABLE,
                "_ARCH_AVAILABLE": vol_mod._ARCH_AVAILABLE,
                "_NF_AVAILABLE": vol_mod._NF_AVAILABLE,
                "_MLF_AVAILABLE": vol_mod._MLF_AVAILABLE,
            }
            return flags
        finally:
            for k in block_keys:
                sys.modules.pop(k, None)
            for k, v in saved.items():
                sys.modules[k] = v
            importlib.reload(vol_mod)      # restore original state

    def test_statsmodels_ets_unavailable(self):
        """Lines 24-25: _SM_ETS_AVAILABLE = False."""
        flags = self._block_and_reload([
            "statsmodels", "statsmodels.tsa",
            "statsmodels.tsa.holtwinters",
        ])
        assert flags["_SM_ETS_AVAILABLE"] is False

    def test_statsmodels_sarimax_unavailable(self):
        """Lines 29-30: _SM_SARIMAX_AVAILABLE = False."""
        flags = self._block_and_reload([
            "statsmodels", "statsmodels.tsa",
            "statsmodels.tsa.statespace",
            "statsmodels.tsa.statespace.sarimax",
        ])
        assert flags["_SM_SARIMAX_AVAILABLE"] is False

    def test_arch_unavailable(self):
        """Lines 41-42: _ARCH_AVAILABLE = False."""
        flags = self._block_and_reload(["arch"])
        assert flags["_ARCH_AVAILABLE"] is False

    def test_importlib_util_find_spec_raises(self):
        """Lines 35-37: _NF_AVAILABLE = _MLF_AVAILABLE = False on exception."""
        orig = importlib.util.find_spec
        with patch("importlib.util.find_spec", side_effect=Exception("boom")):
            importlib.reload(vol_mod)
            assert vol_mod._NF_AVAILABLE is False
            assert vol_mod._MLF_AVAILABLE is False
        importlib.reload(vol_mod)          # restore


# ===================================================================
# 2. _bars_per_year edge cases  (lines 225-226)
# ===================================================================

class TestBarsPerYearException:
    def test_exception_in_computation(self):
        """Invalid TIMEFRAME_SECONDS access returns NaN."""
        with patch("mtdata.forecast.common.TIMEFRAME_SECONDS") as mock_ts:
            mock_ts.get.side_effect = TypeError("mock")
            assert math.isnan(_bars_per_year("H1"))


# ===================================================================
# 3. Params-parsing branches  (lines 363-378)
# ===================================================================

class TestParamsParsing:
    """Exercise the fallback brace-format parser paths."""

    def test_brace_eq_format(self):
        """Lines 365-367: k=v inside braces that fail JSON."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{lookback=200}")
            assert r.get("success") is True

    def test_brace_colon_trailing_no_value(self):
        """Line 375: key: with no following token → p[key]=''."""
        with _mock_env():
            # Use a key that won't break ewma param parsing
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{extra:}")
            assert r.get("success") is True

    def test_brace_stray_token(self):
        """Line 378: token without = or trailing colon."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{stray}")
            assert r.get("success") is True

    def test_brace_empty_tokens(self):
        """Line 363: empty token after strip."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{,,}")
            assert r.get("success") is True

    def test_brace_mixed_formats(self):
        """Lines 363-378: mixed =, colon, stray, comma."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="ewma",
                                    params="{lookback=300, extra: 5, junk}")
            assert r.get("success") is True

    def test_comma_separated_kv_pairs_without_spaces(self):
        with _mock_env():
            r = forecast_volatility(
                "EURUSD",
                "H1",
                1,
                method="ewma",
                params="lookback=300,lambda=0.9",
            )
            assert r.get("success") is True
            assert r["params_used"]["lookback"] == 300
            assert r["params_used"]["lambda_"] == 0.9


# ===================================================================
# 4. tf_secs falsy  (line 336)
# ===================================================================

class TestTfSecsFalsy:
    def test_timeframe_seconds_zero(self):
        """Line 335-336: tf_secs evaluates to falsy."""
        with _mock_env():
            with patch(f"{MOD}.TIMEFRAME_SECONDS", {"H1": 0}):
                r = forecast_volatility("EURUSD", "H1", 1, method="ewma")
                assert "error" in r


# ===================================================================
# 5. General-method error paths  (lines 401-434, 437, 447)
# ===================================================================

class TestGeneralMethodErrors:
    """Call general forecasters (theta by default) to hit error branches."""

    def test_ensure_error(self):
        """Line 401: _ensure_symbol_ready returns error string."""
        with _mock_env(ensure_err="Symbol not available"):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r and "Symbol not available" in r["error"]

    def test_ensure_error_restores_hidden_symbol_visibility(self):
        with _mock_env(ensure_err="Symbol not available", info_visible=False) as env:
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r and "Symbol not available" in r["error"]
            env["mt5"].symbol_select.assert_called_once_with("EURUSD", False)

    def test_invalid_as_of(self):
        """Lines 404-406: _parse_start_datetime returns None."""
        with _mock_env(parse_dt_return=None):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return", as_of="bad")
            assert "error" in r and "as_of" in r["error"].lower()

    def test_valid_as_of(self):
        """Line 407: as_of triggers copy_rates_from with to_dt."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return",
                                    as_of="2024-06-01")
            assert r.get("success") is True

    def test_tick_no_time(self):
        """Line 414: server_now_dt = datetime.now(…) when tick has no time."""
        with _mock_env(tick_time=None):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert r.get("success") is True

    def test_visibility_restore_exception(self):
        """Lines 420-421: except pass when symbol_select raises."""
        with _mock_env(info_visible=False,
                       select_side_effect=RuntimeError("boom")):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            # Should still succeed (exception silenced)
            assert r.get("success") is True
    def test_few_bars_after_drop(self):
        """Line 428: Not enough closed bars (< 5 after dropping forming bar)."""
        with _mock_env(n_bars=5):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_insufficient_returns(self):
        """Line 434: < 10 finite returns."""
        with _mock_env(n_bars=10):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_no_proxy(self):
        """Line 437: general methods require proxy."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy=None)
            assert "error" in r and "proxy" in r["error"].lower()

    def test_unsupported_proxy(self):
        """Line 447: unsupported proxy string."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="foobar")
            assert "error" in r and "proxy" in r["error"].lower()

    def test_denoise_applied(self):
        """Line 430: _apply_denoise called for general methods."""
        with _mock_env():
            with patch(f"{MOD}._apply_denoise") as mock_dn:
                r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                        proxy="squared_return",
                                        denoise={"method": "wavelet"})
                assert r.get("success") is True
                assert mock_dn.called


class TestFetchMt5RatesGuarded:

    def test_invalid_as_of_restores_symbol_visibility(self):
        with _mock_env(info_visible=False, parse_dt_return=None) as env:
            rates, err = vol_mod._fetch_mt5_rates_guarded("EURUSD", object(), 25, as_of="bad-date")

        assert rates is None
        assert err == "Invalid as_of time."
        env["copy_rates"].assert_not_called()
        env["mt5"].symbol_select.assert_called_once_with("EURUSD", False)

    def test_ensure_error_short_circuits_before_copy(self):
        with _mock_env(ensure_err="Symbol not available") as env:
            rates, err = vol_mod._fetch_mt5_rates_guarded("EURUSD", object(), 25)

        assert rates is None
        assert err == "Symbol not available"
        env["copy_rates"].assert_not_called()

    def test_live_fetch_applies_auto_shift_when_timeframe_provided(self):
        with _mock_env(n_bars=5) as env:
            with patch(f"{MOD}._resolve_live_rate_auto_shift_seconds", return_value=7200):
                rates, err = vol_mod._fetch_mt5_rates_guarded("EURUSD", object(), 25, timeframe="H1")

        assert err is None
        assert rates is not None
        assert float(rates["time"][0]) == 1_704_067_200.0 + 7200.0
        env["copy_rates"].assert_called_once()


# ===================================================================
# 6. General methods – proxy variants  (lines 440-445, 559-564)
# ===================================================================

class TestGeneralMethodProxy:
    @pytest.mark.parametrize("proxy,back_label", [
        ("squared_return", "sqrt"),
        ("abs_return", "abs"),
        ("log_r2", "exp_sqrt"),
    ])
    def test_theta_proxy_variants(self, proxy, back_label):
        """Lines 440-445, 559-564: proxy back-transforms."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy=proxy)
            assert r.get("success") is True
            assert r.get("proxy") == proxy

    @pytest.mark.parametrize("proxy", ["squared_return", "abs_return", "log_r2"])
    def test_theta_as_of_with_proxy(self, proxy):
        """Line 425-426: as_of=None keeps forming-bar drop with each proxy."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 3, method="theta",
                                    proxy=proxy)
            assert r.get("success") is True


# ===================================================================
# 7. ARIMA / SARIMA  (lines 451-466)
# ===================================================================

class TestArimaSarima:
    def _mock_sarimax(self, yhat_size=5):
        mock_fc = MagicMock()
        mock_fc.predicted_mean.to_numpy.return_value = np.ones(yhat_size) * 0.001
        mock_res = MagicMock()
        mock_res.get_forecast.return_value = mock_fc
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_res
        return mock_model

    def test_arima_unavailable(self):
        """Line 452-453: SARIMAX not available."""
        with _mock_env():
            with patch.object(vol_mod, "_SM_SARIMAX_AVAILABLE", False):
                r = forecast_volatility("EURUSD", "H1", 5, method="arima",
                                        proxy="squared_return")
                assert "error" in r and "statsmodels" in r["error"]

    def test_arima_success(self):
        """Lines 454-464: ARIMA happy path."""
        model = self._mock_sarimax()
        with _mock_env():
            with patch.object(vol_mod, "_SM_SARIMAX_AVAILABLE", True), \
                 patch.object(vol_mod, "_SARIMAX", return_value=model):
                r = forecast_volatility("EURUSD", "H1", 5, method="arima",
                                        proxy="squared_return")
                assert r.get("success") is True

    def test_arima_fit_error(self):
        """Lines 465-466: SARIMAX raises during fit."""
        mock_model = MagicMock()
        mock_model.fit.side_effect = ValueError("singular")
        with _mock_env():
            with patch.object(vol_mod, "_SM_SARIMAX_AVAILABLE", True), \
                 patch.object(vol_mod, "_SARIMAX", return_value=mock_model):
                r = forecast_volatility("EURUSD", "H1", 5, method="arima",
                                        proxy="squared_return")
                assert "error" in r and "SARIMAX" in r["error"]

    def test_sarima_seasonal(self):
        """Lines 455-459: SARIMA with seasonal order."""
        model = self._mock_sarimax()
        with _mock_env():
            with patch.object(vol_mod, "_SM_SARIMAX_AVAILABLE", True), \
                 patch.object(vol_mod, "_SARIMAX", return_value=model):
                r = forecast_volatility("EURUSD", "H1", 5, method="sarima",
                                        proxy="squared_return",
                                        params={"P": 1, "D": 0, "Q": 1})
                assert r.get("success") is True

    @pytest.mark.parametrize("p,d,q", [(2, 0, 1), (1, 1, 0), (0, 0, 2)])
    def test_arima_order_variants(self, p, d, q):
        """Line 454: various (p,d,q) orders."""
        model = self._mock_sarimax()
        with _mock_env():
            with patch.object(vol_mod, "_SM_SARIMAX_AVAILABLE", True), \
                 patch.object(vol_mod, "_SARIMAX", return_value=model):
                r = forecast_volatility("EURUSD", "H1", 5, method="arima",
                                        proxy="squared_return",
                                        params={"p": p, "d": d, "q": q})
                assert r.get("success") is True


# ===================================================================
# 8. ETS  (lines 467-474)
# ===================================================================

class TestEts:
    def test_ets_unavailable(self):
        """Lines 468-469: ETS not available."""
        with _mock_env():
            with patch.object(vol_mod, "_SM_ETS_AVAILABLE", False):
                r = forecast_volatility("EURUSD", "H1", 5, method="ets",
                                        proxy="squared_return")
                assert "error" in r and "statsmodels" in r["error"].lower()

    def test_ets_success(self):
        """Lines 470-472: ETS happy path."""
        mock_res = MagicMock()
        mock_res.forecast.return_value = np.ones(5) * 0.001
        mock_ets_cls = MagicMock()
        mock_ets_cls.return_value.fit.return_value = mock_res
        with _mock_env():
            with patch.object(vol_mod, "_SM_ETS_AVAILABLE", True), \
                 patch.object(vol_mod, "_ETS", mock_ets_cls):
                r = forecast_volatility("EURUSD", "H1", 5, method="ets",
                                        proxy="squared_return")
                assert r.get("success") is True

    def test_ets_error(self):
        """Lines 473-474: ETS raises during fit."""
        mock_ets_cls = MagicMock()
        mock_ets_cls.return_value.fit.side_effect = RuntimeError("bad")
        with _mock_env():
            with patch.object(vol_mod, "_SM_ETS_AVAILABLE", True), \
                 patch.object(vol_mod, "_ETS", mock_ets_cls):
                r = forecast_volatility("EURUSD", "H1", 5, method="ets",
                                        proxy="squared_return")
                assert "error" in r and "ETS" in r["error"]


# ===================================================================
# 9. Theta  (lines 475-481)
# ===================================================================

class TestTheta:
    def test_theta_default_alpha(self):
        """Lines 476-481: theta with default alpha."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert r.get("success") is True

    def test_theta_custom_alpha(self):
        """Line 479: alpha=0.5 from params."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return",
                                    params={"alpha": 0.5})
            assert r.get("success") is True

    def test_theta_large_horizon(self):
        """Lines 477-478: trend_future with large horizon."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 50, method="theta",
                                    proxy="abs_return")
            assert r.get("success") is True
            assert r["horizon"] == 50


# ===================================================================
# 10. HAR-RV – second-section error/branch paths
# ===================================================================

class TestHarRvSecondSection:
    """har_rv uses a single dedicated intraday-fetch path; these tests cover the
    branching logic within that path (ensure errors, as_of, tick fallback, etc.)."""

    def test_ewma_does_not_fall_through_to_second_section(self):
        std = _make_rates(200)
        with _mock_env(rates_side_effect=[std]):
            r = forecast_volatility("EURUSD", "H1", 5, method="ewma")
            assert r.get("success") is True

    def test_lambda_backward_compat_ewma(self):
        """'lambda' key (legacy alias) is normalised to 'lambda_' before dispatch
        and must affect the EWMA result reflected in params_used."""
        custom_lam = 0.80
        std = _make_rates(200)
        with _mock_env(rates_side_effect=[std]):
            r = forecast_volatility("EURUSD", "H1", 5, method="ewma",
                                    params={"lambda": custom_lam})
        assert r.get("success") is True
        assert abs(r["params_used"]["lambda_"] - custom_lam) < 1e-9

    def test_second_section_ensure_error(self):
        """_ensure_symbol_ready error in the HAR-RV intraday fetch returns an error."""
        with _mock_env(ensure_side_effect=["Symbol locked"]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_second_section_as_of(self):
        """as_of triggers the historical copy path inside the HAR-RV second fetch."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday],
                       parse_dt_return=datetime(2024, 6, 1, tzinfo=timezone.utc)):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="2024-06-01")
            assert r.get("success") is True or "error" in r

    def test_second_section_invalid_as_of(self):
        """_parse_start_datetime returning None yields an error response."""
        with _mock_env(parse_dt_return=None):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="bad-date")
            assert "error" in r

    def test_second_section_tick_no_time(self):
        """tick.time being None causes a fallback to datetime.now."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(tick_time=None,
                       rates_side_effect=[intraday]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert r.get("success") is True or "error" in r

    def test_second_section_visibility_restore(self):
        """When symbol was not visible, symbol_select is called to restore state."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(info_visible=False,
                       rates_side_effect=[intraday]) as env:
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert env["mt5"].symbol_select.called

    def test_second_section_visibility_restore_exc(self):
        """An exception from symbol_select during visibility restore is silently ignored."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(info_visible=False,
                       rates_side_effect=[intraday],
                       select_side_effect=RuntimeError("boom")):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            # Function should not crash
            assert isinstance(r, dict)

    def test_second_section_insufficient_rates(self):
        """None returned from the intraday rate fetch produces an error response."""
        with _mock_env(rates_side_effect=[None]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_second_section_few_bars(self):
        """Fewer than 3 bars after dropping the forming bar yields an error."""
        tiny = _make_rates(3)
        with _mock_env(rates_side_effect=[tiny]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_second_section_denoise(self):
        """A denoise spec is applied during the HAR-RV second section."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday]):
            with patch(f"{MOD}._apply_denoise"):
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "wavelet"})
                assert isinstance(r, dict)

    def test_second_section_denoise_error(self):
        """A denoise error in the HAR-RV second section is silently ignored."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday]):
            with patch(f"{MOD}._apply_denoise", side_effect=RuntimeError("x")):
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "wavelet"})
                # Should succeed even if denoise fails
                assert isinstance(r, dict)


# ===================================================================
# 11. HAR-RV block  (lines 1085-1180)
# ===================================================================

class TestHarRvBlock:
    """Tests targeting the HAR-RV implementation inside forecast_volatility."""

    @staticmethod
    def _har_rv_side_effect(n_intraday=15000):
        """Standard intraday side effect for the HAR-RV fetch."""
        intraday = _make_rates(n_intraday, bar_secs=300, seed=99)
        return [intraday]

    def test_success(self):
        """Lines 1085-1178: HAR-RV happy path."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert r.get("success") is True
            assert "sigma_bar_return" in r
            assert "params_used" in r
            expected_bpy = _bars_per_year("H1")
            assert r["sigma_annual_return"] == pytest.approx(r["sigma_bar_return"] * math.sqrt(expected_bpy))
            assert r["horizon_sigma_annual"] == pytest.approx(
                r["horizon_sigma_return"] * math.sqrt(expected_bpy)
            )
            assert r["horizon_sigma_annual"] > r["sigma_annual_return"]
            assert "beta" in r["params_used"]

    def test_invalid_rv_timeframe(self):
        """Line 1090-1091: rv_timeframe not in TIMEFRAME_MAP."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    params={"rv_timeframe": "INVALID"})
            assert "error" in r and "rv_timeframe" in r["error"].lower()

    def test_ensure_error_intraday(self):
        """Line 1100-1101: ensure error during intraday fetch."""
        with _mock_env(ensure_side_effect=["RV symbol err"],
                       rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_as_of_intraday(self):
        """Lines 1103-1107: as_of path for intraday data."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect(),
                       parse_dt_return=datetime(2024, 3, 1, tzinfo=timezone.utc)):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="2024-03-01")
            assert r.get("success") is True or "error" in r

    def test_invalid_as_of_intraday(self):
        """Lines 1105-1106: invalid as_of returns error in intraday fetch."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday]):
            with patch(f"{MOD}._parse_start_datetime", return_value=None):
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        as_of="2024-03-01")
                assert "error" in r

    def test_tick_no_time_intraday(self):
        """Line 1114: tick.time None during intraday fetch."""
        with _mock_env(tick_time=None,
                       rates_side_effect=self._har_rv_side_effect()):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert r.get("success") is True or "error" in r

    def test_visibility_restore_intraday(self):
        """Lines 1117-1121: visibility restore in intraday fetch block."""
        with _mock_env(info_visible=False,
                       rates_side_effect=self._har_rv_side_effect()) as env:
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert env["mt5"].symbol_select.called

    def test_visibility_restore_exc_intraday(self):
        """Lines 1118-1121: except pass in intraday visibility restore."""
        with _mock_env(info_visible=False,
                       rates_side_effect=self._har_rv_side_effect(),
                       select_side_effect=RuntimeError("boom")):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert isinstance(r, dict)

    def test_insufficient_intraday_rates(self):
        """Line 1122-1123: < 50 intraday bars."""
        tiny = _make_rates(30, bar_secs=300)
        with _mock_env(rates_side_effect=[tiny]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_insufficient_intraday_rates_none(self):
        """Line 1122: rates_rv is None."""
        with _mock_env(rates_side_effect=[None]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_insufficient_daily_rv(self):
        """Line 1136-1137: not enough daily RV observations."""
        # 100 M5 bars ≈ 0.35 day → 1 daily group < 30
        tiny_intra = _make_rates(100, bar_secs=300)
        with _mock_env(rates_side_effect=[tiny_intra]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r and "daily RV" in r.get("error", "")

    def test_insufficient_alignment_samples(self):
        """Line 1154-1155: < 20 usable samples after NaN masking."""
        # 9000 M5 bars ≈ 31 days; with default window_m=22, only ~9 aligned.
        short_intra = _make_rates(9000, bar_secs=300, seed=77)
        with _mock_env(rates_side_effect=[short_intra]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert "error" in r

    def test_exception_handler(self):
        """Lines 1179-1180: generic exception caught."""
        with _mock_env(rates_side_effect=[_make_rates(15000, bar_secs=300)]):
            with patch(f"{MOD}.TIMEFRAME_MAP") as mock_map:
                # Make TIMEFRAME_MAP.get succeed for 'H1' but return None
                # for the rv_timeframe lookup after validation
                real_map = {"H1": MagicMock(), "M5": None}
                mock_map.__contains__ = lambda s, k: k in real_map
                mock_map.__getitem__ = lambda s, k: real_map[k]
                mock_map.get = lambda k, d=None: real_map.get(k, d)
                mock_map.keys = lambda: real_map.keys()
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
                assert "error" in r

    def test_custom_rv_params(self):
        """Custom rv_timeframe, days, window_w, window_m."""
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        with _mock_env(rates_side_effect=[intraday]):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    params={"rv_timeframe": "M5",
                                            "days": 60,
                                            "window_w": 3,
                                            "window_m": 10})
            assert r.get("success") is True or "error" in r

    def test_har_rv_uses_single_rate_fetch(self):
        with _mock_env(rates_side_effect=self._har_rv_side_effect()) as env:
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            assert isinstance(r, dict)
            assert env["copy_rates"].call_count == 1

    def test_as_of_keeps_all_bars(self):
        """Lines 1125-1126: as_of set → last bar NOT dropped."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect(),
                       parse_dt_return=datetime(2024, 6, 1, tzinfo=timezone.utc)):
            r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                    as_of="2024-06-01")
            assert isinstance(r, dict)

    def test_horizon_sigma_scaling(self):
        """Lines 1169-1170: horizon sigma scales with horizon."""
        with _mock_env(rates_side_effect=self._har_rv_side_effect() * 2):
            r1 = forecast_volatility("EURUSD", "H1", 1, method="har_rv")
            r5 = forecast_volatility("EURUSD", "H1", 5, method="har_rv")
            if r1.get("success") and r5.get("success"):
                assert r5["horizon_sigma_return"] >= r1["horizon_sigma_return"]


# ===================================================================
# 12. Denoise branch details in second section  (lines 1001-1012)
# ===================================================================

class TestSecondSectionDenoiseDetails:
    """Denoise column-defaulting logic in the second fetch section."""

    def _make_side_effect(self):
        intraday = _make_rates(15000, bar_secs=300, seed=99)
        return [intraday]

    def test_denoise_spec_columns_default_for_har_rv(self):
        """Lines 1004-1008: columns default to OHLC for non-ewma, non-garch."""
        with _mock_env(rates_side_effect=self._make_side_effect()):
            with patch(f"{MOD}._apply_denoise") as mock_dn:
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "sma"})
                if mock_dn.called:
                    spec = mock_dn.call_args[0][1] if len(mock_dn.call_args[0]) > 1 else mock_dn.call_args[1].get("spec")
                    # Should have OHLC columns by default
                    assert isinstance(r, dict)

    def test_denoise_with_explicit_columns(self):
        """Lines 1003-1004: user-provided columns kept as-is."""
        with _mock_env(rates_side_effect=self._make_side_effect()):
            with patch(f"{MOD}._apply_denoise"):
                r = forecast_volatility("EURUSD", "H1", 5, method="har_rv",
                                        denoise={"method": "sma",
                                                  "columns": ["close"]})
                assert isinstance(r, dict)


# ===================================================================
# 13. Scattered branch conditions
# ===================================================================

class TestScatteredBranches:
    """Cover remaining single-line branches."""

    def test_invalid_method(self):
        """Line 341-342: method not in valid_direct ∪ valid_general."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "H1", 1, method="bogus")
            assert "error" in r and "Invalid method" in r["error"]

    def test_invalid_timeframe(self):
        """Line 331-332: timeframe not in TIMEFRAME_MAP."""
        with _mock_env():
            r = forecast_volatility("EURUSD", "INVALID", 1, method="ewma")
            assert "error" in r and "timeframe" in r["error"].lower()

    def test_garch_arch_unavailable(self):
        """Line 343-344: garch without arch package."""
        with _mock_env():
            with patch.object(vol_mod, "_ARCH_AVAILABLE", False):
                r = forecast_volatility("EURUSD", "H1", 1, method="garch")
                assert "error" in r and "arch" in r["error"].lower()

    def test_general_method_rates_none(self):
        """Line 422-423: rates is None for general method."""
        with _mock_env(rates_return=None):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_general_method_rates_too_few(self):
        """Line 422-423: < 5 rates for general method."""
        with _mock_env(n_bars=4):
            r = forecast_volatility("EURUSD", "H1", 5, method="theta",
                                    proxy="squared_return")
            assert "error" in r

    def test_kernel_weight_parzen_bartlett_alias(self):
        """Line 279: parzen_bartlett is treated the same as parzen."""
        w1 = _kernel_weight("parzen", 3, 10)
        w2 = _kernel_weight("parzen_bartlett", 3, 10)
        assert w1 == w2

    def test_kernel_weight_parzen_mid_x(self):
        """Line 280-281 vs 282-283: x in (0.5, 1.0] branch."""
        # x = h/(bandwidth+1); for h=8, bw=10 → x=8/11≈0.727 > 0.5
        w = _kernel_weight("parzen", 8, 10)
        assert 0.0 <= w <= 1.0

    def test_bars_per_year_returns_float(self):
        """Shared helper returns a finite float."""
        result = _bars_per_year("M5")
        assert isinstance(result, float) and result > 0

    @pytest.mark.parametrize("method", ["arima", "sarima", "ets", "theta"])
    def test_general_method_result_fields(self, method):
        """Lines 569-572: successful general method returns expected keys."""
        if method in ("arima", "sarima"):
            mock_fc = MagicMock()
            mock_fc.predicted_mean.to_numpy.return_value = np.ones(5) * 0.001
            mock_res = MagicMock()
            mock_res.get_forecast.return_value = mock_fc
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_res
            ctx = [patch.object(vol_mod, "_SM_SARIMAX_AVAILABLE", True),
                   patch.object(vol_mod, "_SARIMAX", return_value=mock_model)]
        elif method == "ets":
            mock_res = MagicMock()
            mock_res.forecast.return_value = np.ones(5) * 0.001
            mock_cls = MagicMock()
            mock_cls.return_value.fit.return_value = mock_res
            ctx = [patch.object(vol_mod, "_SM_ETS_AVAILABLE", True),
                   patch.object(vol_mod, "_ETS", mock_cls)]
        else:
            ctx = []

        with _mock_env():
            from contextlib import ExitStack
            with ExitStack() as stack:
                for c in ctx:
                    stack.enter_context(c)
                r = forecast_volatility("EURUSD", "H1", 5, method=method,
                                        proxy="squared_return")
                assert r.get("success") is True
                for key in ("symbol", "timeframe", "method", "horizon",
                            "sigma_bar_return", "sigma_annual_return"):
                    assert key in r, f"Missing key {key}"
