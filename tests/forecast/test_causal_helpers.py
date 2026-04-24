"""Tests for core/causal.py — pure helper functions (no MT5)."""
import numpy as np
import pandas as pd
import pytest

from mtdata.core.causal import (
    _TRANSFORM_LEGEND,
    _build_correlation_matrix,
    _build_correlation_summary,
    _format_summary,
    _normalize_cointegration_transform,
    _normalize_cointegration_trend,
    _normalize_correlation_method,
    _normalize_transform_name,
    _parse_symbols,
    _standardize_frame,
    _transform_cointegration_frame,
    _transform_frame,
)
from mtdata.shared.parameter_contracts import normalize_symbol_selector_aliases


class TestParseSymbols:
    def test_comma_separated(self):
        assert _parse_symbols("EURUSD, GBPUSD, USDJPY") == ["EURUSD", "GBPUSD", "USDJPY"]

    def test_semicolon_separated(self):
        assert _parse_symbols("EURUSD;GBPUSD") == ["EURUSD", "GBPUSD"]

    def test_deduplication(self):
        assert _parse_symbols("EURUSD, GBPUSD, EURUSD") == ["EURUSD", "GBPUSD"]

    def test_empty_string(self):
        assert _parse_symbols("") == []

    def test_whitespace(self):
        assert _parse_symbols("  EURUSD  ,  GBPUSD  ") == ["EURUSD", "GBPUSD"]

    def test_mixed_delimiters(self):
        assert _parse_symbols("A;B,C;D") == ["A", "B", "C", "D"]


class TestNormalizeSymbolSelectorAliases:
    def test_symbol_alias_populates_plural_contract(self):
        tokens, meta, error = normalize_symbol_selector_aliases(
            symbol="EURUSD",
            symbols=None,
            parse_selector=_parse_symbols,
        )

        assert error is None
        assert tokens == ["EURUSD"]
        assert meta["symbol_input"] == ["EURUSD"]
        assert meta["symbols_input"] == ["EURUSD"]

    def test_conflicting_symbol_aliases_are_rejected(self):
        tokens, meta, error = normalize_symbol_selector_aliases(
            symbol="EURUSD",
            symbols="GBPUSD",
            parse_selector=_parse_symbols,
        )

        assert tokens == []
        assert "Provide either symbol or symbols" in str(error)
        assert meta["symbol_input"] == ["EURUSD"]
        assert meta["symbols_input"] == ["GBPUSD"]


class TestTransformFrame:
    def _df(self):
        return pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1], "B": [50.0, 55.0, 60.5, 66.55]})

    def test_log_return(self):
        result = _transform_frame(self._df(), "log_return")
        assert len(result) == 3  # one row dropped from diff
        assert all(np.isfinite(result["A"]))

    def test_logret_alias(self):
        r1 = _transform_frame(self._df(), "logret")
        r2 = _transform_frame(self._df(), "log_return")
        pd.testing.assert_frame_equal(r1, r2)

    def test_pct_change(self):
        result = _transform_frame(self._df(), "pct")
        assert len(result) == 3

    def test_diff(self):
        result = _transform_frame(self._df(), "diff")
        assert len(result) == 3

    def test_no_transform(self):
        df = self._df()
        result = _transform_frame(df, "none")
        pd.testing.assert_frame_equal(result, df)

    def test_with_zeros(self):
        df = pd.DataFrame({"A": [0.0, 1.0, 2.0]})
        result = _transform_frame(df, "log_return")
        # Zero prices → NaN in log → dropped
        assert len(result) <= 2


class TestStandardizeFrame:
    def test_basic(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, 40.0]})
        result = _standardize_frame(df)
        # Each column should be roughly zero-mean unit-variance
        assert abs(result["A"].mean()) < 1e-10
        assert abs(result["B"].mean()) < 1e-10

    def test_empty_frame(self):
        df = pd.DataFrame()
        result = _standardize_frame(df)
        assert result.empty

    def test_constant_column_preserved(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [5.0, 5.0, 5.0]})
        result = _standardize_frame(df)
        # Constant column should be preserved as-is
        pd.testing.assert_series_equal(result["B"], df["B"], check_names=True)


class TestCorrelationHelpers:
    def test_normalize_correlation_method_aliases(self):
        assert _normalize_correlation_method("pearson") == "pearson"
        assert _normalize_correlation_method("Linear") == "pearson"
        assert _normalize_correlation_method("rank") == "spearman"
        assert _normalize_correlation_method("kendall") is None

    def test_normalize_transform_aliases(self):
        assert _normalize_transform_name("logret") == "log_return"
        assert _normalize_transform_name("pct_change") == "pct"
        assert _normalize_transform_name("raw") == "level"
        assert _normalize_transform_name("mystery") is None

    def test_normalize_cointegration_aliases(self):
        assert _normalize_cointegration_transform("log") == "log_level"
        assert _normalize_cointegration_transform("raw") == "level"
        assert _normalize_cointegration_transform("mystery") is None
        assert _normalize_cointegration_trend("constant") == "c"
        assert _normalize_cointegration_trend("none") == "n"
        assert _normalize_cointegration_trend("bad") is None

    def test_pct_transform_legend_matches_pct_change_scale(self):
        assert _TRANSFORM_LEGEND["pct"]["formula"] == "(close_t - close_t-1) / close_t-1"
        assert "1% gain" in _TRANSFORM_LEGEND["pct"]["use_case"]

    def test_transform_cointegration_frame_supports_log_levels(self):
        df = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [50.0, 0.0, 52.0]})
        result = _transform_cointegration_frame(df, "log_level")
        assert np.isfinite(result["A"]).all()
        assert np.isnan(result["B"]).sum() >= 1

    def test_build_correlation_matrix_is_symmetric(self):
        matrix = _build_correlation_matrix(
            ["EURUSD", "GBPUSD", "USDJPY"],
            [
                {"left": "EURUSD", "right": "GBPUSD", "correlation": 0.81},
                {"left": "EURUSD", "right": "USDJPY", "correlation": -0.42},
            ],
        )

        assert matrix["EURUSD"]["EURUSD"] == pytest.approx(1.0)
        assert matrix["EURUSD"]["GBPUSD"] == pytest.approx(0.81)
        assert matrix["GBPUSD"]["EURUSD"] == pytest.approx(0.81)
        assert matrix["USDJPY"]["EURUSD"] == pytest.approx(-0.42)
        assert matrix["GBPUSD"]["USDJPY"] is None

    def test_build_correlation_summary_splits_positive_and_negative(self):
        rows = [
            {"left": "A", "right": "B", "correlation": 0.91, "samples": 100},
            {"left": "A", "right": "C", "correlation": -0.87, "samples": 95},
            {"left": "B", "right": "C", "correlation": 0.50, "samples": 90},
        ]

        summary = _build_correlation_summary(rows, top_n=1)

        assert summary["strongest_absolute"][0]["left"] == "A"
        assert summary["strongest_absolute"][0]["right"] == "B"
        assert summary["strongest_positive"][0]["correlation"] == pytest.approx(0.91)
        assert summary["strongest_negative"][0]["correlation"] == pytest.approx(-0.87)


class TestFormatSummary:
    def test_empty_rows(self):
        result = _format_summary([], ["EURUSD"], "log_return", 0.05)
        assert "No valid pairings" in result

    def test_with_rows(self):
        rows = [
            {"effect": "EURUSD", "cause": "GBPUSD", "lag": 3, "p_value": 0.01, "samples": 100},
            {"effect": "EURUSD", "cause": "USDJPY", "lag": 2, "p_value": 0.20, "samples": 100},
        ]
        result = _format_summary(rows, ["EURUSD", "GBPUSD", "USDJPY"], "log_return", 0.05)
        assert "causal" in result
        assert "no-link" in result
        assert "EURUSD <- GBPUSD" in result

    def test_group_hint(self):
        rows = [{"effect": "A", "cause": "B", "lag": 1, "p_value": 0.01, "samples": 50}]
        result = _format_summary(rows, ["A", "B"], "diff", 0.05, group_hint="Forex\\Major")
        assert "Forex\\Major" in result

    def test_sorted_by_pvalue(self):
        rows = [
            {"effect": "A", "cause": "B", "lag": 1, "p_value": 0.50, "samples": 50},
            {"effect": "C", "cause": "D", "lag": 2, "p_value": 0.01, "samples": 50},
        ]
        result = _format_summary(rows, ["A", "B", "C", "D"], "pct", 0.05)
        lines = result.strip().split("\n")
        data_lines = [l for l in lines if "<-" in l and "Effect" not in l and "|" in l]
        # First data line should be the one with lower p-value
        assert "C <- D" in data_lines[0]
