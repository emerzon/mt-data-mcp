"""Tests for core/pivot.py — pivot_compute_points tool.

Covers lines 25-273 by mocking MT5 calls and _symbol_ready_guard.
"""
import math
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

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
    raw = pivot_compute_points
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__

    def _call(*args, **kwargs):
        with patch("mtdata.core.pivot.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


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


def test_pivot_compute_points_logs_finish_event(caplog):
    fn = _get_pivot_fn()
    info = _make_symbol_info()

    @contextmanager
    def _guard(symbol):
        yield None, info

    rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])
    with patch(_TF_MAP_PATCH, {"D1": 1}), \
         patch(_TF_SECS_PATCH, {"D1": 86400}), \
         patch(_GUARD, _guard), \
         patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
         patch(_EPOCH, side_effect=lambda x: float(x)), \
         patch(_COPY_RATES, return_value=rates), \
         patch(_USE_CTZ, return_value=False), \
         patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
         caplog.at_level("DEBUG", logger="mtdata.core.pivot"):
        res = fn("EURUSD", timeframe="D1")

    assert res["success"] is True
    assert any(
        "event=finish operation=pivot_compute_points success=True" in record.message
        for record in caplog.records
    )


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
            res = fn("EURUSD", timeframe="D1", detail="standard")
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
            res = fn("EURUSD", timeframe="D1", detail="standard")
        assert "error" in res


class TestPivotSingleBar:

    def _run(self, rate, now_ts, tf_secs=86400):
        fn = _get_pivot_fn()
        info = _make_symbol_info()
        current_dt = datetime.fromtimestamp(now_ts, tz=timezone.utc)

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
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_dt
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
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
    """Full happy-path tests around completed-bar source selection."""

    def _run(self, rates_list, digits=5, use_ctz=False, detail="compact"):
        fn = _get_pivot_fn()
        info = _make_symbol_info(digits=digits)
        current_dt = datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)

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
             patch(_FMT_LOCAL, side_effect=lambda x: f"L{int(x)}"), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_dt
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
            return fn("EURUSD", timeframe="D1", detail=detail)

    def test_classic_levels(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r)
        assert res["success"] is True
        assert res["detail"] == "compact"
        assert res["method"] == "classic"
        assert "PP" in res["levels"]
        assert "R1" in res["levels"]
        assert "S1" in res["levels"]

    def test_compact_warns_on_degenerate_rounded_levels(self):
        r = [
            _make_rate(time_=100.0),
            _make_rate(
                time_=200.0,
                open_=1.1720,
                high=1.17227,
                low=1.17171,
                close=1.17225,
            ),
        ]

        res = self._run(r, digits=3, detail="compact")

        assert res["success"] is True
        assert res["levels_degenerate"] is True
        assert res["digits"] == 3
        assert res["source_range"] == 0.00056
        assert res["price_increment"] == 0.001
        assert res["unique_level_count"] <= 3
        assert "Source bar range" in res["reason"]
        assert "Pivot levels may appear identical after rounding." in res["reason"]

    def test_all_methods_present(self):
        fn = _get_pivot_fn()
        info = _make_symbol_info(digits=5)
        current_dt = datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=rates), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
             patch(_FMT_LOCAL, side_effect=lambda x: f"L{int(x)}"), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_dt
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
            res = fn("EURUSD", timeframe="D1", detail="standard")

        assert res["detail"] == "standard"
        # Should have columns for classic, fibonacci, camarilla, woodie, demark
        for lv in res["levels"]:
            if lv["level"] == "PP":
                assert "classic" in lv
                assert "fibonacci" in lv

    def test_full_detail_includes_methods(self):
        fn = _get_pivot_fn()
        info = _make_symbol_info(digits=5)
        current_dt = datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = np.array([_make_rate(time_=100.0), _make_rate(time_=200.0)])
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick()), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=rates), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
             patch(_FMT_LOCAL, side_effect=lambda x: f"L{int(x)}"), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_dt
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
            res = fn("EURUSD", timeframe="D1", detail="full")

        assert res["detail"] == "full"
        assert isinstance(res["methods"], list)

    def test_digits_rounding(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r, digits=2, detail="standard")
        for lv in res["levels"]:
            for method in ("classic", "fibonacci"):
                val = lv.get(method)
                if val is not None:
                    # Should be rounded to 2 decimals
                    assert round(val, 2) == val

    def test_level_rows_omit_null_method_cells(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r, detail="standard")
        for lv in res["levels"]:
            for key, val in lv.items():
                if key == "level":
                    continue
                assert val is not None

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

    def test_uses_latest_closed_bar_when_fetch_has_no_live_tail(self):
        earlier = _make_rate(open_=1.1000, high=1.1050, low=1.0950, close=1.1020, time_=100.0)
        latest = _make_rate(open_=1.2000, high=1.2500, low=1.1500, close=1.2200, time_=200.0)

        res = self._run([earlier, latest], use_ctz=False, detail="standard")

        assert res["success"] is True
        assert res["period"]["start"] == "T200"
        pp_row = next(row for row in res["levels"] if row["level"] == "PP")
        assert pp_row["classic"] == pytest.approx(round((1.2500 + 1.1500 + 1.2200) / 3.0, 5))

    def test_period_field_uses_already_normalized_bar_time(self):
        src_time = 1_704_067_200.0
        r = [
            _make_rate(time_=src_time),
            _make_rate(time_=src_time + 86_400.0),
            _make_rate(time_=src_time + 172_800.0),
        ]
        fn = _get_pivot_fn()
        info = _make_symbol_info()

        @contextmanager
        def _guard(symbol):
            yield None, info

        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick(src_time + 180_001.0)), \
             patch(_EPOCH, side_effect=lambda x: float(x) - 7200.0), \
             patch(_COPY_RATES, return_value=np.array(r)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(src_time + 172_801.0, tz=timezone.utc)
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
            res = fn("EURUSD", timeframe="D1", detail="standard")

        assert res["success"] is True
        assert res["period"]["start"] == f"T{int(src_time + 86400.0)}"
        assert res["period"]["end"] == f"T{int(src_time + 172800.0)}"

    def test_stale_tick_uses_current_time_for_completion_check(self):
        src_time = 1_704_067_200.0
        r = [_make_rate(time_=src_time), _make_rate(time_=src_time + 86_400.0)]
        fn = _get_pivot_fn()
        info = _make_symbol_info()
        current_dt = datetime.fromtimestamp(src_time + (7 * 86_400.0), tz=timezone.utc)

        @contextmanager
        def _guard(symbol):
            yield None, info

        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick(src_time + 60.0)), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(r)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_dt
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
            res = fn("EURUSD", timeframe="D1")

        assert res["success"] is True
        assert res["period"]["start"] == f"T{int(src_time + 86400.0)}"

    def test_calculation_basis_context(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r, use_ctz=False, detail="standard")
        assert res["calculation_basis"]["source_bar"] == "last completed D1 bar"
        assert res["calculation_basis"]["session_boundary"] == "MT5 broker/session calendar"
        assert res["calculation_basis"]["display_timezone"] == "UTC"
        assert "null cells mean that pivot method does not define that level." in res["levels_note"]
        assert "Camarilla levels are centered on the close price" in res["levels_note"]

    def test_symbol_timeframe_in_response(self):
        r = [_make_rate(time_=100.0), _make_rate(time_=200.0)]
        res = self._run(r)
        assert res["symbol"] == "EURUSD"
        assert res["timeframe"] == "D1"


class TestPivotMethods:
    """Test individual pivot method computations."""

    def _levels(self, H, L, C, O):
        """Run pivot and return levels_by_method dict."""
        first_time = 1_700_000_000.0
        second_time = first_time + 86_400.0
        r = [
            _make_rate(open_=O, high=H, low=L, close=C, time_=first_time),
            _make_rate(time_=second_time),
        ]
        fn = _get_pivot_fn()
        info = _make_symbol_info(digits=10)
        current_dt = datetime.fromtimestamp(second_time + 1.0, tz=timezone.utc)

        @contextmanager
        def _guard(symbol):
            yield None, info

        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick(second_time + 1.0)), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(r)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: f"T{int(x)}"), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_dt
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
            res = fn("EURUSD", timeframe="D1", detail="standard")
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
        first_time = 1_700_000_000.0
        second_time = first_time + 86_400.0
        rate = {"low": 1.09, "close": 1.10, "open": 1.10, "time": first_time}
        # Missing 'high' → NaN
        current_dt = datetime.fromtimestamp(second_time + 1.0, tz=timezone.utc)

        @contextmanager
        def _guard(symbol):
            yield None, info

        rates = [rate, _make_rate(time_=second_time)]
        with patch(_TF_MAP_PATCH, {"D1": 1}), \
             patch(_TF_SECS_PATCH, {"D1": 86400}), \
             patch(_GUARD, _guard), \
             patch(f"{_MT5}.symbol_info_tick", return_value=_make_tick(second_time + 1.0)), \
             patch(_EPOCH, side_effect=lambda x: float(x)), \
             patch(_COPY_RATES, return_value=np.array(rates)), \
             patch(_USE_CTZ, return_value=False), \
             patch(_FMT, side_effect=lambda x: str(x)), \
             patch("mtdata.core.pivot.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_dt
            mock_datetime.fromtimestamp.side_effect = lambda *args, **kwargs: datetime.fromtimestamp(*args, **kwargs)
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
            res = fn("EURUSD", timeframe="D1", detail="standard")
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
            res = fn("EURUSD", timeframe="D1", detail="standard")

        levels = [lv["level"] for lv in res["levels"]]
        # Find PP position
        pp_idx = levels.index("PP")
        r_indices = [i for i, l in enumerate(levels) if l.startswith("R")]
        s_indices = [i for i, l in enumerate(levels) if l.startswith("S")]
        # All R before PP, all S after PP
        assert all(i < pp_idx for i in r_indices)
        assert all(i > pp_idx for i in s_indices)
