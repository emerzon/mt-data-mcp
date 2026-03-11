"""Tests for core/causal.py — extended coverage for MT5-dependent functions (mocked)."""

import math
import time
import warnings
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest

from mtdata.core import causal as causal_mod
from mtdata.core.causal import (
    _parse_symbols,
    _expand_symbols_for_group,
    _fetch_series,
    _transform_frame,
    _standardize_frame,
    _format_summary,
    causal_discover_signals,
)
from mtdata.utils.mt5 import MT5ConnectionError


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(causal_mod, "ensure_mt5_connection_or_raise", lambda: None)


# ---------------------------------------------------------------------------
# _expand_symbols_for_group (lines 30-50)
# ---------------------------------------------------------------------------


class TestExpandSymbolsForGroup:
    @patch("mtdata.core.causal._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.mt5")
    def test_symbol_not_found(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = None
        syms, err, gp = _expand_symbols_for_group("BADPAIR")
        assert syms == []
        assert "not found" in err

    @patch("mtdata.core.causal._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.mt5")
    def test_symbols_get_none(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        mock_mt5.symbols_get.return_value = None
        mock_mt5.last_error.return_value = (0, "err")
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert syms == []
        assert "Failed to load" in err

    @patch("mtdata.core.causal._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.mt5")
    def test_single_member_returns_warning(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        sym_obj = MagicMock()
        sym_obj.name = "EURUSD"
        sym_obj.visible = True
        mock_mt5.symbols_get.return_value = [sym_obj]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert len(syms) == 1
        assert "fewer than two" in err

    @patch("mtdata.core.causal._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.mt5")
    def test_multiple_members(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        s1 = MagicMock(); s1.name = "EURUSD"; s1.visible = True
        s2 = MagicMock(); s2.name = "GBPUSD"; s2.visible = True
        mock_mt5.symbols_get.return_value = [s1, s2]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert err is None
        assert "EURUSD" in syms and "GBPUSD" in syms

    @patch("mtdata.core.causal._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.mt5")
    def test_invisible_non_anchor_skipped(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        s1 = MagicMock(); s1.name = "EURUSD"; s1.visible = True
        s2 = MagicMock(); s2.name = "GBPUSD"; s2.visible = False
        s3 = MagicMock(); s3.name = "USDJPY"; s3.visible = True
        mock_mt5.symbols_get.return_value = [s1, s2, s3]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert "GBPUSD" not in syms
        assert "USDJPY" in syms

    @patch("mtdata.core.causal._extract_group_path_util", return_value="Forex\\Majors")
    @patch("mtdata.core.causal.mt5")
    def test_anchor_not_in_list_gets_inserted(self, mock_mt5, mock_gp):
        mock_mt5.symbol_info.return_value = MagicMock()
        s1 = MagicMock(); s1.name = "GBPUSD"; s1.visible = True
        s2 = MagicMock(); s2.name = "USDJPY"; s2.visible = True
        mock_mt5.symbols_get.return_value = [s1, s2]
        syms, err, gp = _expand_symbols_for_group("EURUSD")
        assert syms[0] == "EURUSD"


# ---------------------------------------------------------------------------
# _fetch_series (lines 54-78)
# ---------------------------------------------------------------------------


class TestFetchSeries:
    @patch("mtdata.core.causal._mt5_copy_rates_from")
    @patch("mtdata.core.causal._ensure_symbol_ready", return_value="symbol not ready")
    def test_symbol_not_ready(self, mock_ensure, mock_copy):
        series, err = _fetch_series("BAD", None, 100)
        assert err == "symbol not ready"
        assert series.empty

    @patch("mtdata.core.causal.time.sleep")
    @patch("mtdata.core.causal._mt5_copy_rates_from", return_value=None)
    @patch("mtdata.core.causal._ensure_symbol_ready", return_value=None)
    def test_all_retries_fail(self, mock_ensure, mock_copy, mock_sleep):
        series, err = _fetch_series("EURUSD", None, 100, retries=2, pause=0.0)
        assert "Failed to fetch data" in err
        assert "after 2 retries" in err

    @patch("mtdata.core.causal.time.sleep")
    @patch("mtdata.core.causal._mt5_copy_rates_from", return_value=None)
    @patch("mtdata.core.causal._ensure_symbol_ready", return_value=None)
    def test_single_retry_message(self, mock_ensure, mock_copy, mock_sleep):
        series, err = _fetch_series("X", None, 100, retries=1, pause=0.0)
        assert "after" not in err

    @patch("mtdata.core.causal._mt5_copy_rates_from")
    @patch("mtdata.core.causal._ensure_symbol_ready", return_value=None)
    def test_success(self, mock_ensure, mock_copy):
        data = np.array([(1000, 1.1, 1.2, 1.0, 1.15, 100, 10, 0),
                         (2000, 1.15, 1.25, 1.05, 1.20, 200, 20, 0)],
                        dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                               ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])
        mock_copy.return_value = data
        series, err = _fetch_series("EURUSD", None, 100)
        assert err is None
        assert len(series) == 2
        assert series.iloc[0] == 1.15

    @patch("mtdata.core.causal.time.sleep")
    @patch("mtdata.core.causal._mt5_copy_rates_from")
    @patch("mtdata.core.causal._ensure_symbol_ready", return_value=None)
    def test_empty_df_retries(self, mock_ensure, mock_copy, mock_sleep):
        mock_copy.return_value = np.array([])
        series, err = _fetch_series("X", None, 50, retries=2, pause=0.0)
        assert "Failed" in err

    @patch("mtdata.core.causal._mt5_copy_rates_from")
    @patch("mtdata.core.causal._ensure_symbol_ready", return_value=None)
    def test_truncates_excess_data(self, mock_ensure, mock_copy):
        times = list(range(1000, 1000 + 200))
        closes = [1.1 + i * 0.001 for i in range(200)]
        data = np.array(list(zip(times, closes, closes, closes, closes,
                                 [100]*200, [1]*200, [0]*200)),
                        dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
                               ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")])
        mock_copy.return_value = data
        series, err = _fetch_series("EURUSD", None, 50)
        assert err is None
        assert len(series) == 50


# ---------------------------------------------------------------------------
# _format_summary (lines 117-139)
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_empty_rows(self):
        assert "No valid pairings" in _format_summary([], ["A", "B"], "log_return", 0.05)

    def test_causal_link(self):
        rows = [{"effect": "B", "cause": "A", "lag": 2, "p_value": 0.01, "samples": 100}]
        text = _format_summary(rows, ["A", "B"], "log_return", 0.05)
        assert "causal" in text
        assert "B <- A" in text

    def test_no_link(self):
        rows = [{"effect": "B", "cause": "A", "lag": 1, "p_value": 0.99, "samples": 50}]
        text = _format_summary(rows, ["A", "B"], "log_return", 0.05)
        assert "no-link" in text

    def test_group_hint(self):
        rows = [{"effect": "B", "cause": "A", "lag": 1, "p_value": 0.02, "samples": 80}]
        text = _format_summary(rows, ["A", "B"], "log_return", 0.05, group_hint="Forex\\Majors")
        assert "Forex\\Majors" in text

    def test_sorting(self):
        rows = [
            {"effect": "B", "cause": "A", "lag": 1, "p_value": 0.50, "samples": 80},
            {"effect": "A", "cause": "B", "lag": 2, "p_value": 0.01, "samples": 80},
        ]
        text = _format_summary(rows, ["A", "B"], "pct", 0.05)
        lines = text.split("\n")
        # skip header lines (contain "Effect <- Cause"); find data lines with " | "
        data_lines = [l for l in lines if "<-" in l and "|" in l and "Effect" not in l]
        assert "A <- B" in data_lines[0]


# ---------------------------------------------------------------------------
# causal_discover_signals (lines 164-264, the main tool)
# ---------------------------------------------------------------------------


class TestCausalDiscoverSignals:
    def _unwrapped(self):
        fn = causal_discover_signals
        while hasattr(fn, '__wrapped__'):
            fn = fn.__wrapped__
        return fn

    def test_connection_error_payload(self, monkeypatch):
        def fail_connection():
            raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

        monkeypatch.setattr(causal_mod, "ensure_mt5_connection_or_raise", fail_connection)

        result = self._unwrapped()("EURUSD,GBPUSD")

        assert result == {"error": "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."}

    def test_empty_symbols(self):
        result = self._unwrapped()("")
        assert result["success"] is False
        assert "Provide at least one symbol" in result["error"]
        assert result["error_code"] == "invalid_input"

    @patch("mtdata.core.causal._expand_symbols_for_group", return_value=([], "Symbol X not found", None))
    def test_single_symbol_expand_error(self, mock_expand):
        result = self._unwrapped()("X")
        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["error_code"] == "symbol_group_error"

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_insufficient_data(self, mock_fetch):
        mock_fetch.return_value = (pd.Series(dtype=float), "No data")
        result = self._unwrapped()("A,B")
        assert result["success"] is False
        assert ("No data" in result["error"]) or ("Not enough" in result["error"])
        assert result["error_code"] in {"data_fetch_failed", "insufficient_symbols"}

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_insufficient_overlap_includes_per_symbol_diagnostics(self, mock_fetch):
        idx_a = pd.date_range("2024-01-01", periods=50, freq="h")
        idx_b = pd.date_range("2024-02-01", periods=50, freq="h")
        series_map = {
            "BTCUSD": pd.Series(np.linspace(1.0, 2.0, 50), index=idx_a),
            "ETHUSD": pd.Series(np.linspace(2.0, 3.0, 50), index=idx_b),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        result = self._unwrapped()("BTCUSD,ETHUSD", max_lag=5, transform="diff", normalize=False)

        assert result["success"] is False
        assert result["error_code"] == "insufficient_overlap"
        details_text = " ".join(str(x) for x in result.get("details", []))
        assert "BTCUSD: 50 rows" in details_text
        assert "ETHUSD: 50 rows" in details_text
        assert "aligned: 0" in details_text
        assert "minimum 11 required" in details_text

        meta = result.get("meta", {})
        assert meta.get("symbol_rows", {}).get("BTCUSD") == 50
        assert meta.get("symbol_rows", {}).get("ETHUSD") == 50
        assert meta.get("samples_aligned_raw") == 0
        assert meta.get("minimum_samples_required") == 11
        assert meta.get("pair_overlaps", {}).get("BTCUSD-ETHUSD") == 0

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_alignment_detail_includes_pair_bottleneck_when_samples_shrink(self, mock_fetch, mock_granger):
        idx_a = pd.date_range("2024-01-01", periods=100, freq="h")
        idx_b = pd.date_range("2024-01-01", periods=100, freq="h")
        idx_c = pd.date_range("2024-01-02", periods=100, freq="h")
        series_map = {
            "EURUSD": pd.Series(np.linspace(1.0, 2.0, 100), index=idx_a),
            "GBPUSD": pd.Series(np.linspace(2.0, 3.0, 100), index=idx_b),
            "USDJPY": pd.Series(np.linspace(3.0, 4.0, 100), index=idx_c),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.3, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.4, 10, 1)}, None),
        }

        result = self._unwrapped()("EURUSD,GBPUSD,USDJPY", max_lag=2, transform="diff", normalize=False)
        assert result["success"] is True
        meta = result.get("meta", {})
        detail = meta.get("alignment_detail", {})
        assert isinstance(detail, dict)
        pair_overlaps = detail.get("pair_overlaps", {})
        assert pair_overlaps.get("EURUSD-GBPUSD") == 100
        assert pair_overlaps.get("EURUSD-USDJPY") == 76
        assert pair_overlaps.get("GBPUSD-USDJPY") == 76
        assert detail.get("bottleneck_pair") in {"EURUSD-USDJPY", "GBPUSD-USDJPY"}
        assert int(detail.get("aligned_rows", 0)) == 76

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_auto_prunes_symbol_with_no_overlap(self, mock_fetch, mock_granger):
        idx_ab = pd.date_range("2024-01-01", periods=80, freq="h")
        idx_c = pd.date_range("2024-03-01", periods=80, freq="h")
        series_map = {
            "EURUSD": pd.Series(np.linspace(1.0, 2.0, 80), index=idx_ab),
            "GBPUSD": pd.Series(np.linspace(2.0, 3.0, 80), index=idx_ab),
            "USDJPY": pd.Series(np.linspace(3.0, 4.0, 80), index=idx_c),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.02, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.03, 10, 1)}, None),
        }

        result = self._unwrapped()("EURUSD,GBPUSD,USDJPY", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is True
        assert result["meta"]["symbols_used"] == ["EURUSD", "GBPUSD"]
        assert result["meta"]["pruned_symbols"] == ["USDJPY"]
        assert result["meta"]["samples_aligned_raw_after_pruning"] == 80
        assert "pair_overlaps_after_pruning" in result["meta"]
        warnings_out = result.get("warnings", [])
        assert any("Dropped USDJPY due to insufficient overlap" in warning for warning in warnings_out)

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {})
    def test_invalid_timeframe(self):
        result = self._unwrapped()("A,B", timeframe="BAD")
        assert result["success"] is False
        assert "Invalid timeframe" in result["error"]
        assert result["error_code"] == "invalid_timeframe"

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    def test_max_lag_zero(self):
        result = self._unwrapped()("A,B", max_lag=0)
        assert result["success"] is False
        assert "max_lag must be at least 1" in result["error"]
        assert result["error_code"] == "invalid_input"

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_success_returns_structured_payload(self, mock_fetch, mock_granger, caplog):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        def _granger_side_effect(*args, **kwargs):
            warnings.warn("'verbose' is deprecated", FutureWarning)
            return {
                1: ({"ssr_ftest": (1.0, 0.02, 10, 1)}, None),
                2: ({"ssr_ftest": (1.0, 0.03, 10, 1)}, None),
            }

        mock_granger.side_effect = _granger_side_effect

        with warnings.catch_warnings(record=True) as records, caplog.at_level(
            "INFO",
            logger="mtdata.core.causal",
        ):
            warnings.simplefilter("always")
            result = self._unwrapped()("A,B", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is True
        assert "data" in result
        assert "meta" in result
        assert result["data"]["count_links"] >= 1
        assert isinstance(result["data"]["links"], list)
        assert "summary_text" in result["data"]
        assert result["meta"]["pairs_tested"] >= 1
        assert not any("verbose" in str(w.message).lower() for w in records)
        assert any(
            "event=finish operation=causal_discover_signals success=True" in record.message
            for record in caplog.records
        )
