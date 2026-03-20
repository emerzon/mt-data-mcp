"""Tests for patterns/candlestick.py — pure helper functions (no MT5)."""

import numpy as np
import pandas as pd
import pytest

from mtdata.patterns.candlestick import (
    _normalize_candlestick_name,
    _parse_min_strength,
    _candlestick_strength_score,
    _is_candlestick_allowed,
    _extract_candlestick_rows,
    _discover_candlestick_pattern_methods,
    _candlestick_span_bars,
)


class TestNormalizeCandlestickName:
    def test_strip_prefix(self):
        assert _normalize_candlestick_name("cdl_doji") == "doji"

    def test_case_insensitive(self):
        assert _normalize_candlestick_name("CDL_Hammer") == "hammer"

    def test_no_prefix(self):
        assert _normalize_candlestick_name("engulfing") == "engulfing"

    def test_remove_underscores(self):
        assert _normalize_candlestick_name("cdl_morning_star") == "morningstar"

    def test_remove_spaces(self):
        assert _normalize_candlestick_name("morning star") == "morningstar"

    def test_empty_string(self):
        assert _normalize_candlestick_name("") == ""


class TestParseMinStrength:
    def test_valid(self):
        assert _parse_min_strength(0.5) == 0.5

    def test_zero(self):
        assert _parse_min_strength(0.0) == 0.0

    def test_one(self):
        assert _parse_min_strength(1.0) == 1.0

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            _parse_min_strength(1.5)

    def test_negative(self):
        with pytest.raises(ValueError):
            _parse_min_strength(-0.1)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            _parse_min_strength("not_a_number")


class TestCandlestickStrengthScore:
    def test_robust_multibar_pattern_scores_higher_than_deprioritized_single_bar(self):
        engulfing = _candlestick_strength_score(
            "cdl_engulfing",
            100.0,
            robust_set={"engulfing"},
            deprioritize={"doji"},
        )
        doji = _candlestick_strength_score(
            "cdl_doji",
            100.0,
            robust_set={"engulfing"},
            deprioritize={"doji"},
        )

        assert engulfing == pytest.approx(0.95)
        assert doji == pytest.approx(0.55)
        assert engulfing > doji

    def test_larger_raw_signal_receives_bonus(self):
        base = _candlestick_strength_score(
            "cdl_alpha",
            100.0,
            robust_set=set(),
            deprioritize=set(),
        )
        boosted = _candlestick_strength_score(
            "cdl_alpha",
            200.0,
            robust_set=set(),
            deprioritize=set(),
        )

        assert base == pytest.approx(0.75)
        assert boosted == pytest.approx(0.95)


class TestIsCandlestickAllowed:
    def test_no_filters(self):
        assert _is_candlestick_allowed(
            "doji", robust_only=False, robust_set=set(), whitelist_set=None
        )

    def test_whitelist_pass(self):
        assert _is_candlestick_allowed(
            "doji", robust_only=False, robust_set=set(), whitelist_set={"doji"}
        )

    def test_whitelist_fail(self):
        assert not _is_candlestick_allowed(
            "hammer", robust_only=False, robust_set=set(), whitelist_set={"doji"}
        )

    def test_robust_pass(self):
        assert _is_candlestick_allowed(
            "doji", robust_only=True, robust_set={"doji"}, whitelist_set=None
        )

    def test_robust_fail(self):
        assert not _is_candlestick_allowed(
            "hammer", robust_only=True, robust_set={"doji"}, whitelist_set=None
        )


class TestExtractCandlestickRows:
    def _make_data(self):
        """Create synthetic DataFrames mimicking pandas_ta pattern columns."""
        n = 10
        df_tail = pd.DataFrame(
            {
                "time": [f"2024-01-{i + 1:02d}" for i in range(n)],
                "close": np.linspace(100, 110, n),
            }
        )
        # Pattern columns with signal values (100 = bullish, -100 = bearish, 0 = no signal)
        temp_tail = pd.DataFrame(
            {
                "cdl_doji": [0, 0, 100, 0, 0, -100, 0, 0, 0, 0],
                "cdl_hammer": [0, 0, 0, 0, 100, 0, 0, 0, 0, 0],
            },
            dtype=float,
        )
        return df_tail, temp_tail

    def test_basic_extraction(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji", "cdl_hammer"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize=set(),
        )
        assert len(rows) > 0
        assert any("Bullish" in str(r[1]) for r in rows)

    def test_empty_pattern_cols(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            [],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize=set(),
        )
        assert rows == []

    def test_high_threshold(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji", "cdl_hammer"],
            threshold=2.0,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize=set(),
        )
        assert rows == []

    def test_min_gap(self):
        df_tail, temp_tail = self._make_data()
        rows_no_gap = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji", "cdl_hammer"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize=set(),
        )
        rows_gap = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji", "cdl_hammer"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=100,
            top_k=5,
            deprioritize=set(),
        )
        assert len(rows_gap) <= len(rows_no_gap)

    def test_bearish_detection(self):
        df_tail, temp_tail = self._make_data()
        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize=set(),
        )
        bearish = [r for r in rows if "Bearish" in str(r[1])]
        assert len(bearish) > 0

    def test_include_metrics_adds_span_context(self):
        df_tail = pd.DataFrame(
            {
                "time": [f"2024-01-{i + 1:02d}" for i in range(5)],
                "close": np.linspace(100, 104, 5),
            }
        )
        temp_tail = pd.DataFrame({"cdl_morning_star": [0, 0, 0, 100, 0]}, dtype=float)

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_morning_star"],
            threshold=0.5,
            robust_only=False,
            robust_set=set(),
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize=set(),
            include_metrics=True,
        )

        assert rows[0][4] == 100
        assert rows[0][6] == "2024-01-02"
        assert rows[0][7] == "2024-01-04"
        assert rows[0][8] == 3

    def test_threshold_uses_semantic_strength_not_raw_signal_only(self):
        df_tail = pd.DataFrame({"time": ["T0", "T1"]})
        temp_tail = pd.DataFrame(
            {
                "cdl_doji": [0.0, 100.0],
                "cdl_engulfing": [0.0, 100.0],
            }
        )

        rows = _extract_candlestick_rows(
            df_tail,
            temp_tail,
            ["cdl_doji", "cdl_engulfing"],
            threshold=0.90,
            robust_only=False,
            robust_set={"engulfing"},
            whitelist_set=None,
            min_gap=0,
            top_k=5,
            deprioritize={"doji"},
        )

        assert rows == [["T1", "Bullish ENGULFING"]]


class TestCandlestickSpanBars:
    def test_defaults_to_single_bar(self):
        assert _candlestick_span_bars("cdl_doji") == 1

    def test_known_multi_bar_pattern(self):
        assert _candlestick_span_bars("cdl_morning_star") == 3


class TestDiscoverCandlestickPatternMethods:
    def test_with_mock_accessor(self):
        class MockTa:
            def cdl_doji(self):
                pass

            def cdl_hammer(self):
                pass

            def sma(self):
                pass  # not a candlestick method

            _private = None

        result = _discover_candlestick_pattern_methods(MockTa())
        assert "cdl_doji" in result
        assert "cdl_hammer" in result
        assert "sma" not in result

    def test_empty_accessor(self):
        class Empty:
            pass

        result = _discover_candlestick_pattern_methods(Empty())
        assert result == ()
