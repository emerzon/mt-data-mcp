"""Tests for core/causal.py — extended coverage for MT5-dependent functions (mocked)."""

import math
import time
import warnings
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core import causal as causal_mod
from mtdata.core.causal import (
    _expand_symbols_for_group,
    _expand_symbols_for_group_path,
    _fetch_series,
    _format_summary,
    _pair_overlap_symbols,
    _parse_symbols,
    _standardize_frame,
    _transform_frame,
    causal_discover_signals,
    cointegration_test,
    correlation_matrix,
)
from mtdata.utils.mt5 import MT5ConnectionError


@pytest.fixture(autouse=True)
def _skip_mt5_connection(monkeypatch):
    monkeypatch.setattr(causal_mod, "ensure_mt5_connection_or_raise", lambda: None)


def test_pair_overlap_symbols_handles_hyphenated_symbols():
    assert _pair_overlap_symbols("BTC-USD-ETH-USD", ["BTC-USD", "ETH-USD"]) == ("BTC-USD", "ETH-USD")


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


class TestExpandSymbolsForGroupPath:
    def _symbol(self, name: str, group_path: str, *, visible: bool = True):
        sym = MagicMock()
        sym.name = name
        sym.visible = visible
        sym.group_path = group_path
        return sym

    @patch("mtdata.core.causal._extract_group_path_util", side_effect=lambda symbol: getattr(symbol, "group_path", None))
    def test_exact_match_returns_visible_members(self, _mock_group):
        gateway = MagicMock()
        gateway.symbols_get.return_value = [
            self._symbol("EURUSD", "Forex\\Majors"),
            self._symbol("GBPUSD", "Forex\\Majors"),
            self._symbol("USDJPY", "Forex\\Majors", visible=False),
            self._symbol("BTCUSD", "Crypto\\Majors"),
        ]

        syms, err, gp = _expand_symbols_for_group_path("Forex\\Majors", gateway=gateway)

        assert err is None
        assert gp == "Forex\\Majors"
        assert syms == ["EURUSD", "GBPUSD"]

    @patch("mtdata.core.causal._extract_group_path_util", side_effect=lambda symbol: getattr(symbol, "group_path", None))
    def test_ambiguous_partial_match_returns_error(self, _mock_group):
        gateway = MagicMock()
        gateway.symbols_get.return_value = [
            self._symbol("EURUSD", "Forex\\Majors"),
            self._symbol("BTCUSD", "Crypto\\Majors"),
        ]

        syms, err, gp = _expand_symbols_for_group_path("Majors", gateway=gateway)

        assert syms == []
        assert gp is None
        assert "matched multiple visible MT5 symbol groups" in err


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

    @patch("mtdata.core.causal._mt5_copy_rates_from")
    @patch("mtdata.core.causal._ensure_symbol_ready", return_value=None)
    def test_deduplicates_duplicate_timestamps(self, mock_ensure, mock_copy):
        data = np.array(
            [
                (1000, 1.1, 1.2, 1.0, 1.15, 100, 10, 0),
                (1000, 1.15, 1.25, 1.05, 1.20, 200, 20, 0),
                (2000, 1.2, 1.3, 1.1, 1.25, 300, 30, 0),
            ],
            dtype=[
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "i8"),
                ("spread", "i4"),
                ("real_volume", "i8"),
            ],
        )
        mock_copy.return_value = data

        series, err = _fetch_series("EURUSD", None, 100)

        assert err is None
        assert len(series) == 2
        assert not series.index.has_duplicates
        assert series.iloc[0] == 1.20


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

    @patch("mtdata.core.causal._expand_symbols_for_group", return_value=(["BTCUSD", "ETHUSD", "LTCUSD"], None, "Crypto"))
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_single_symbol_auto_expand_does_not_succeed_without_anchor(self, mock_fetch, _mock_expand):
        idx_anchor = pd.date_range("2024-01-01", periods=80, freq="h")
        idx_peers = pd.date_range("2024-03-01", periods=80, freq="h")
        series_map = {
            "BTCUSD": pd.Series(np.linspace(1.0, 2.0, 80), index=idx_anchor),
            "ETHUSD": pd.Series(np.linspace(2.0, 3.0, 80), index=idx_peers),
            "LTCUSD": pd.Series(np.linspace(3.0, 4.0, 80), index=idx_peers),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("BTCUSD", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is False
        assert result["error_code"] == "insufficient_overlap"
        assert result["meta"]["symbols_input"] == ["BTCUSD"]
        assert result["meta"]["symbols_expanded"] == ["BTCUSD", "ETHUSD", "LTCUSD"]

    @patch("mtdata.core.causal._expand_symbols_for_group", return_value=(["BTCUSD", "ETHUSD"], None, "Crypto"))
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_single_symbol_auto_expand_fails_when_anchor_fetch_is_missing(self, mock_fetch, _mock_expand):
        idx_peer = pd.date_range("2024-01-01", periods=80, freq="h")

        def _fetch_side_effect(symbol, timeframe, count):
            if symbol == "BTCUSD":
                return pd.Series(dtype=float), "Failed to fetch data for BTCUSD"
            return pd.Series(np.linspace(2.0, 3.0, 80), index=idx_peer), None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("BTCUSD", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is False
        assert result["error_code"] == "anchor_symbol_missing"
        assert "BTCUSD" in result["error"]
        assert "Failed to fetch data for BTCUSD" in " ".join(result.get("warnings", []))

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
        assert "summary_text" not in result["data"]
        assert result["meta"]["pairs_tested"] >= 1
        assert result["meta"]["p_value_correction"] == "bonferroni_across_lags"
        assert not any("verbose" in str(w.message).lower() for w in records)
        assert mock_granger.call_args.kwargs.get("verbose") is False
        assert any(
            "event=finish operation=causal_discover_signals success=True" in record.message
            for record in caplog.records
        )

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_granger_stdout_is_suppressed(self, mock_fetch, mock_granger, capsys):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        def _granger_side_effect(*args, **kwargs):
            print("Granger Causality")
            return {
                1: ({"ssr_ftest": (1.0, 0.02, 10, 1)}, None),
            }

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.side_effect = _granger_side_effect

        result = self._unwrapped()("A,B", max_lag=1, transform="diff", normalize=False)

        assert result["success"] is True
        assert "Granger Causality" not in capsys.readouterr().out

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_best_lag_p_value_is_bonferroni_adjusted(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.return_value = {
            1: ({"ssr_ftest": (1.0, 0.02, 10, 1)}, None),
            2: ({"ssr_ftest": (1.0, 0.03, 10, 1)}, None),
        }

        result = self._unwrapped()("A,B", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is True
        link = result["data"]["links"][0]
        assert link["lag"] == 1
        assert link["p_value_raw"] == pytest.approx(0.02)
        assert link["lag_tests_run"] == 2
        assert link["p_value"] == pytest.approx(0.04)
        assert link["significant"] is True

    @patch("statsmodels.tsa.stattools.grangercausalitytests")
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_granger_failures_are_surfaced_in_metadata(self, mock_fetch, mock_granger):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        base = np.linspace(1.0, 2.0, 80)
        series_map = {
            "A": pd.Series(base, index=idx),
            "B": pd.Series(base * 1.01 + 0.001, index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect
        mock_granger.side_effect = RuntimeError("singular matrix")

        result = self._unwrapped()("A,B", max_lag=2, transform="diff", normalize=False)

        assert result["success"] is True
        assert result["meta"]["pairs_failed"] >= 1
        assert result["meta"]["pair_failures"][0]["error_type"] == "RuntimeError"
        assert "warnings" in result


class TestCorrelationMatrix:
    def _unwrapped(self):
        fn = correlation_matrix
        while hasattr(fn, '__wrapped__'):
            fn = fn.__wrapped__
        return fn

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_method(self):
        result = self._unwrapped()("A,B", method="kendall")
        assert result["success"] is False
        assert result["error_code"] == "invalid_method"

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_transform(self):
        result = self._unwrapped()("A,B", transform="mystery")
        assert result["success"] is False
        assert result["error_code"] == "invalid_transform"

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    def test_min_overlap_too_small(self):
        result = self._unwrapped()("A,B", min_overlap=1)
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "min_overlap" in result["error"]

    def test_symbols_and_group_are_mutually_exclusive(self):
        result = self._unwrapped()("A,B", group="Forex\\Majors")
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "either symbols or group" in result["error"]

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_success_returns_matrix_and_ranked_pairs(self, mock_fetch, caplog):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.015, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "B": pd.Series(80.0 * np.exp(np.cumsum((rets * 0.95) + 0.0005)), index=idx),
            "C": pd.Series(120.0 * np.exp(np.cumsum(-rets)), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        with caplog.at_level("INFO", logger="mtdata.core.causal"):
            result = self._unwrapped()(
                "A,B,C",
                method="pearson",
                transform="log_return",
                min_overlap=30,
            )

        assert result["success"] is True
        data = result["data"]
        assert data["count_pairs"] == 3
        assert data["matrix"]["A"]["A"] == pytest.approx(1.0)
        assert data["matrix"]["A"]["B"] > 0.95
        assert data["matrix"]["A"]["C"] < -0.95
        assert data["pairs"][0]["abs_correlation"] >= data["pairs"][1]["abs_correlation"]
        assert data["strongest_positive"]
        assert data["strongest_negative"]
        assert result["meta"]["pairs_computed"] == 3
        assert any(
            "event=finish operation=correlation_matrix success=True" in record.message
            for record in caplog.records
        )

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_pairwise_overlap_allows_partial_success(self, mock_fetch):
        idx_ab = pd.date_range("2024-01-01", periods=80, freq="h")
        idx_c = pd.date_range("2024-03-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.01, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx_ab),
            "B": pd.Series(90.0 * np.exp(np.cumsum((rets * 1.02) + 0.0002)), index=idx_ab),
            "C": pd.Series(110.0 * np.exp(np.cumsum(rets)), index=idx_c),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B,C", min_overlap=20)

        assert result["success"] is True
        assert result["data"]["count_pairs"] == 1
        assert result["data"]["matrix"]["A"]["B"] is not None
        assert result["data"]["matrix"]["A"]["C"] is None
        assert result["data"]["matrix"]["B"]["C"] is None
        assert result["meta"]["pairs_computed"] == 1
        assert result["meta"]["pair_overlaps"]["A-C"] == 0
        assert result["meta"]["pair_overlaps"]["B-C"] == 0

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_partial_fetch_failures_are_preserved_as_warnings(self, mock_fetch):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.01, 80)
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "B": pd.Series(95.0 * np.exp(np.cumsum((rets * 0.9) + 0.0003)), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            if symbol == "C":
                return pd.Series(dtype=float), "Failed to fetch data for C"
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B,C")

        assert result["success"] is True
        assert result["data"]["count_pairs"] == 1
        assert result["meta"]["symbols_used"] == ["A", "B"]
        assert "warnings" in result
        assert any("Failed to fetch data for C" in warning for warning in result["warnings"])

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._expand_symbols_for_group_path", return_value=(["EURUSD", "GBPUSD"], None, "Forex\\Majors"))
    @patch("mtdata.core.causal._fetch_series")
    def test_group_argument_expands_symbols(self, mock_fetch, _mock_expand):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        rets = np.linspace(-0.01, 0.01, 80)
        series_map = {
            "EURUSD": pd.Series(100.0 * np.exp(np.cumsum(rets)), index=idx),
            "GBPUSD": pd.Series(90.0 * np.exp(np.cumsum((rets * 0.98) + 0.0001)), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()(group="Forex\\Majors", min_overlap=20)

        assert result["success"] is True
        assert result["meta"]["group_input"] == "Forex\\Majors"
        assert result["meta"]["group_resolved"] == "Forex\\Majors"
        assert result["meta"]["symbols_expanded"] == ["EURUSD", "GBPUSD"]
        assert result["data"]["count_pairs"] == 1

    @patch("mtdata.core.causal._expand_symbols_for_group_path", return_value=([], "Group 'Forex' matched multiple visible MT5 symbol groups: Forex\\Majors, Forex\\Minors", None))
    def test_group_argument_surfaces_resolution_error(self, _mock_expand):
        result = self._unwrapped()(group="Forex")

        assert result["success"] is False
        assert result["error_code"] == "symbol_group_error"
        assert "matched multiple visible MT5 symbol groups" in result["error"]

    @patch("mtdata.core.causal._expand_symbols_for_group", return_value=(["BTCUSD", "ETHUSD"], None, "Crypto"))
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_single_symbol_auto_expand_fails_when_anchor_missing(self, mock_fetch, _mock_expand):
        idx = pd.date_range("2024-01-01", periods=80, freq="h")
        series_eth = pd.Series(100.0 * np.exp(np.cumsum(np.linspace(-0.01, 0.01, 80))), index=idx)

        def _fetch_side_effect(symbol, timeframe, count):
            if symbol == "BTCUSD":
                return pd.Series(dtype=float), "Failed to fetch data for BTCUSD"
            return series_eth, None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("BTCUSD")

        assert result["success"] is False
        assert result["error_code"] == "anchor_symbol_missing"
        assert "BTCUSD" in result["error"]
        assert any("Failed to fetch data for BTCUSD" in warning for warning in result["warnings"])

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_insufficient_overlap_includes_pair_details(self, mock_fetch):
        idx_a = pd.date_range("2024-01-01", periods=50, freq="h")
        idx_b = pd.date_range("2024-02-01", periods=50, freq="h")
        series_map = {
            "A": pd.Series(100.0 * np.exp(np.cumsum(np.linspace(-0.01, 0.01, 50))), index=idx_a),
            "B": pd.Series(80.0 * np.exp(np.cumsum(np.linspace(-0.01, 0.01, 50))), index=idx_b),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B", min_overlap=30)

        assert result["success"] is False
        assert result["error_code"] == "insufficient_overlap"
        assert "A-B: 0 rows (minimum 30 required)" in " ".join(result.get("details", []))
        assert result["meta"]["pair_overlaps"]["A-B"] == 0


class TestCointegrationTest:
    def _unwrapped(self):
        fn = cointegration_test
        while hasattr(fn, '__wrapped__'):
            fn = fn.__wrapped__
        return fn

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_transform(self):
        result = self._unwrapped()("A,B", transform="returns")
        assert result["success"] is False
        assert result["error_code"] == "invalid_transform"

    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    def test_invalid_trend(self):
        result = self._unwrapped()("A,B", trend="bad")
        assert result["success"] is False
        assert result["error_code"] == "invalid_trend"

    def test_symbols_and_group_are_mutually_exclusive(self):
        result = self._unwrapped()("A,B", group="Forex\\Majors")
        assert result["success"] is False
        assert result["error_code"] == "invalid_input"
        assert "either symbols or group" in result["error"]

    @patch("statsmodels.tsa.stattools.coint", return_value=(-4.5, 0.01, [-3.9, -3.3, -3.0]))
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._expand_symbols_for_group_path", return_value=(["A", "B"], None, "Forex\\Majors"))
    @patch("mtdata.core.causal._fetch_series")
    def test_group_argument_returns_cointegrated_pair(self, mock_fetch, _mock_expand, _mock_coint):
        idx = pd.date_range("2024-01-01", periods=120, freq="h")
        base = np.cumsum(np.linspace(-0.01, 0.01, 120))
        series_map = {
            "A": pd.Series(100.0 * np.exp(base), index=idx),
            "B": pd.Series(50.0 * np.exp(base * 0.98), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()(group="Forex\\Majors", min_overlap=40)

        assert result["success"] is True
        assert result["meta"]["group_resolved"] == "Forex\\Majors"
        assert result["data"]["count_pairs"] == 1
        assert result["data"]["count_cointegrated"] == 1
        pair = result["data"]["pairs"][0]
        assert pair["cointegrated"] is True
        assert pair["p_value"] == pytest.approx(0.01)
        assert pair["critical_values"]["5%"] == pytest.approx(-3.3)
        assert pair["hedge_ratio"] is not None

    @patch("statsmodels.tsa.stattools.coint", side_effect=RuntimeError("singular matrix"))
    @patch("mtdata.core.causal.TIMEFRAME_MAP", {"H1": 1})
    @patch("mtdata.core.causal._fetch_series")
    def test_failures_surface_test_failed_error(self, mock_fetch, _mock_coint):
        idx = pd.date_range("2024-01-01", periods=120, freq="h")
        base = np.cumsum(np.linspace(-0.01, 0.01, 120))
        series_map = {
            "A": pd.Series(100.0 * np.exp(base), index=idx),
            "B": pd.Series(50.0 * np.exp(base * 0.98), index=idx),
        }

        def _fetch_side_effect(symbol, timeframe, count):
            return series_map[symbol], None

        mock_fetch.side_effect = _fetch_side_effect

        result = self._unwrapped()("A,B", min_overlap=40)

        assert result["success"] is False
        assert result["error_code"] == "test_failed"
        assert "Cointegration tests failed" in result["error"]
        assert result["meta"]["pairs_failed"] >= 1
        assert "warnings" in result
