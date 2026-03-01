"""Tests for core/pivot.py — pivot_compute_points tool.

Covers lines 25-273 by mocking MT5 calls and _symbol_ready_guard.
"""
import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from contextlib import contextmanager
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rate(open_=1.1000, high=1.1050, low=1.0950, close=1.1020, time_=1_700_000_000.0):
    """Return a dict mimicking a structured MT5 rate row."""
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "time": time_, "tick_volume": 100, "spread": 10, "real_volume": 0,
    }


def _make_symbol_info(digits=5, visible=True):
    info = MagicMock()
    info.digits = digits
    info.visible = visible
    return info


def _make_tick(time_val=1_700_000_000.0):
    t = MagicMock()
    t.time = time_val
    return t


@contextmanager
def _mock_symbol_guard(error=None, info=None):
    """Context manager that replaces _symbol_ready_guard."""
    @contextmanager
    def _guard(symbol):
        yield error, info
    with patch("mtdata.core.pivot._symbol_ready_guard", _guard):
        yield


# Unwrap the decorated function
def _get_pivot_fn():
    from mtdata.core.pivot import pivot_compute_points
    fn = pivot_compute_points
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


_TF_MAP_PATCH = "mtdata.core.pivot.TIMEFRAME_MAP"
_TF_SECS_PATCH = "mtdata.core.pivot.TIMEFRAME_SECONDS"
_MT5 = "mtdata.core.pivot.mt5"
_GUARD = "mtdata.core.pivot._symbol_ready_guard"
_FMT = "mtdata.core.pivot._format_time_minimal"
_FMT_LOCAL = "mtdata.core.pivot._format_time_minimal_local"
_USE_CTZ = "mtdata.core.pivot._use_client_tz"
_EPOCH = "mtdata.core.pivot._mt5_epoch_to_utc"
_COPY_RATES = "mtdata.core.pivot._mt5_copy_rates_from"


class TestPivotInvalidInputs:

    def test_invalid_timeframe(self):
        fn = _get_pivot_fn()
        with patch(_TF_MAP_PATCH, {"D1": 1}):
            res = fn("EURUSD", timeframe="INVALID")
        assert "error" in res
        assert "Invalid timeframe" in res["error"]

    def test_unsupported_timeframe_seconds(self):
        fn = _get_pivot_fn()
        with patch(_TF_MAP_PATCH, {"D1": 1, "X": 99}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}):
            res = fn("EURUSD", timeframe="X")
        assert "error" in res
        assert "Unsupported timeframe" in res["error"]


class TestPivotSymbolGuardError:

    def test_symbol_error(self):
        fn = _get_pivot_fn()
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             _mock_symbol_guard(error="Symbol not found"):
            res = fn("BADSY", timeframe="D1")
        assert res["error"] == "Symbol not found"


class TestPivotNoRates:

    def test_rates_none(self):
        fn = _get_pivot_fn()
        info = _make_symbol_info()

        @contextmanager
        def _guard(symbol):
            yield None, info

        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: x), \
             patch(_COPY_RATES, return_value=None), \
             patch(f"{_MT5}.last_error", return_value=(0, "no data")):
            res = fn("EURUSD", timeframe="D1")
        assert "error" in res

    def test_rates_empty(self):
        fn = _get_pivot_fn()
        info = _make_symbol_info()

        @contextmanager
        def _guard(symbol):
            yield None, info

        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: x), \
             patch(_COPY_RATES, return_value=np.array([])), \
             patch(f"{_MT5}.last_error", return_value=(0, "")):
            res = fn("EURUSD", timeframe="D1")
        assert "error" in res


class TestPivotSingleBar:

    def _run(self, rate, now_ts, tf_secs=86400):
        fn = _get_pivot_fn()
        info = _make_symbol_info()

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = np.array([rate])
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": tf_secs}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick(now_ts)), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=rates), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"):
            return fn("EURUSD", timeframe="D1")

    def test_completed_single_bar(self):
        rate = _make_rate(time_=1_700_000_000.0)
        # now_ts > bar_time + tf_secs → bar is completed
        res = self._run(rate, now_ts=1_700_100_000.0)
        assert res.get("success") is True

    def test_uncompleted_single_bar(self):
        rate = _make_rate(time_=1_700_000_000.0)
        # now_ts < bar_time + tf_secs → bar not yet completed
        res = self._run(rate, now_ts=1_700_000_001.0)
        assert "error" in res
        assert "No completed bars" in res["error"]


class TestPivotHappyPath:
    """Full happy-path test with two bars (rates[-2] used as source)."""

    def _run(self, rates_list, digits=5, use_ctz=False):
        fn = _get_pivot_fn()
        info = _make_symbol_info(digits=digits)

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = np.array(rates_list)
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=rates), \
             patch(_USE_CTZ, return_value=use_ctz), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
             patch(_FMT_LOCAL, side_effect=lambda x: f"L{int(x)}"):
            return fn("EURUSD", timeframe="D1")

    def test_classic_levels(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r)
        assert res["success"] is True
        assert isinstance(res["levels"], list)
        level_names = [lv["level"] for lv in res["levels"]]
        assert "PP" in level_names
        assert "R1" in level_names
        assert "S1" in level_names

    def test_all_methods_present(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r)
        # Should have columns for classic, fibonacci, camarilla, woodie, demark
        for lv in res["levels"]:
            if lv["level"] == "PP":
                assert "classic" in lv
                assert "fibonacci" in lv

    def test_digits_rounding(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r, digits=2)
        for lv in res["levels"]:
            for method in ("classic", "fibonacci"):
                val = lv.get(method)
                if val is not None:
                    # Should be rounded to 2 decimals
                    assert round(val, 2) == val

    def test_utc_timezone_label(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r, use_ctz=False)
        assert res.get("timezone") == "UTC"

    def test_client_tz_no_utc_label(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r, use_ctz=True)
        assert "timezone" not in res

    def test_period_field(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r)
        assert "period" in res
        assert "start" in res["period"]
        assert "end" in res["period"]

    def test_calculation_basis_context(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r, use_ctz=False)
        assert res["calculation_basis"]["source_bar"] == "last completed D1 bar"
        assert res["calculation_basis"]["session_boundary"] == "MT5 broker/session calendar"
        assert res["calculation_basis"]["display_timezone"] == "UTC"

    def test_symbol_timeframe_in_response(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r)
        assert res["symbol"] == "EURUSD"
        assert res["timeframe"] == "D1"


class TestPivotMethods:
    """Test individual pivot method computations."""

    def _levels(self, H, L, C, O):
        """Run pivot and return levels_by_method dict."""
        r = [_make_rate(open_=O, high=H, low=L, close=C, time_=100.0),
             _make_rate(time_=200.0)]
        fn = _get_pivot_fn()
        info = _make_symbol_info(digits=10)

        @contextmanager
        def _guard(symbol):
            yield None, info

        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(r)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"):
            res = fn("EURUSD", timeframe="D1")
        # Convert levels list into a dict: method -> {level -> value}
        by_method: Dict[str, Dict[str, Any]] = {}
        for lv in res.get("levels", []):
            lvl = lv["level"]
            for k, v in lv.items():
                if k == "level":
                    continue
                by_method.setdefault(k, {})[lvl] = v
        return by_method

    def test_classic_pp(self):
        H, L, C, O = 1.1050, 1.0950, 1.1020, 1.1000
        m = self._levels(H, L, C, O)
        expected_pp = (H + L + C) / 3.0
        assert abs(m["classic"]["PP"] - expected_pp) < 1e-6

    def test_classic_r1_s1(self):
        H, L, C, O = 1.1050, 1.0950, 1.1020, 1.1000
        m = self._levels(H, L, C, O)
        pp = (H + L + C) / 3.0
        assert abs(m["classic"]["R1"] - (2 * pp - L)) < 1e-6
        assert abs(m["classic"]["S1"] - (2 * pp - H)) < 1e-6

    def test_fibonacci_levels(self):
        H, L, C, O = 1.1050, 1.0950, 1.1020, 1.1000
        m = self._levels(H, L, C, O)
        pp = (H + L + C) / 3.0
        rng = H - L
        assert abs(m["fibonacci"]["R1"] - (pp + 0.382 * rng)) < 1e-6
        assert abs(m["fibonacci"]["S1"] - (pp - 0.382 * rng)) < 1e-6

    def test_camarilla_levels(self):
        H, L, C, O = 1.1050, 1.0950, 1.1020, 1.1000
        m = self._levels(H, L, C, O)
        rng = H - L
        k = 1.1
        assert abs(m["camarilla"]["R1"] - (C + k * rng / 12.0)) < 1e-6

    def test_woodie_pp(self):
        H, L, C, O = 1.1050, 1.0950, 1.1020, 1.1000
        m = self._levels(H, L, C, O)
        expected = (H + L + 2 * C) / 4.0
        assert abs(m["woodie"]["PP"] - expected) < 1e-6

    def test_demark_c_lt_o(self):
        H, L, C, O = 1.1050, 1.0950, 1.0900, 1.1000  # C < O
        m = self._levels(H, L, C, O)
        X = H + 2 * L + C
        assert abs(m["demark"]["PP"] - X / 4.0) < 1e-6

    def test_demark_c_gt_o(self):
        H, L, C, O = 1.1050, 1.0950, 1.1040, 1.1000  # C > O
        m = self._levels(H, L, C, O)
        X = 2 * H + L + C
        assert abs(m["demark"]["PP"] - X / 4.0) < 1e-6

    def test_demark_c_eq_o(self):
        H, L, C, O = 1.1050, 1.0950, 1.1000, 1.1000  # C == O
        m = self._levels(H, L, C, O)
        X = H + L + 2 * C
        assert abs(m["demark"]["PP"] - X / 4.0) < 1e-6


class TestPivotNanPrices:

    def test_nan_high(self):
        fn = _get_pivot_fn()
        info = _make_symbol_info()
        rate = {"low": 1.09, "close": 1.10, "open": 1.10, "time": 100.0}
        # Missing 'high' → NaN

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = [rate, _make_rate(time_=200.0)]
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(rates)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: str(x)):
            res = fn("EURUSD", timeframe="D1")
        # Should error because H is NaN
        assert "error" in res


class TestPivotTickFallback:

    def test_tick_none(self):
        """When symbol_info_tick returns None, fallback to utcnow."""
        fn = _get_pivot_fn()
        info = _make_symbol_info()

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=None), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(rates)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"):
            res = fn("EURUSD", timeframe="D1")
        assert res.get("success") is True


class TestPivotException:

    def test_general_exception(self):
        fn = _get_pivot_fn()
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, side_effect=RuntimeError("boom")):
            res = fn("EURUSD", timeframe="D1")
        assert "error" in res
        assert "Error computing pivot" in res["error"]


class TestPivotInfoNone:

    def test_info_none_digits_default(self):
        """When _info_before is None, digits should default to 0."""
        fn = _get_pivot_fn()

        @contextmanager
        def _guard(symbol):
            yield None, None  # info is None

        rates = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(rates)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"):
            res = fn("EURUSD", timeframe="D1")
        assert res.get("success") is True


class TestPivotLevelOrdering:

    def test_resistances_before_supports(self):
        """R levels should appear before PP, which appears before S levels."""
        fn = _get_pivot_fn()
        info = _make_symbol_info(digits=5)

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(rates)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"):
            res = fn("EURUSD", timeframe="D1")

        levels = [lv["level"] for lv in res["levels"]]
        # Find PP position
        pp_idx = levels.index("PP")
        r_indices = [i for i, l in enumerate(levels) if l.startswith("R")]
        s_indices = [i for i, l in enumerate(levels) if l.startswith("S")]
        # All R before PP, all S after PP
        assert all(i < pp_idx for i in r_indices)
        assert all(i > pp_idx for i in s_indices)
