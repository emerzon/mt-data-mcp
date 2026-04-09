"""Tests for common forecast helper utilities."""
from mtdata.forecast.common import (
    default_seasonality as default_seasonality_period,
)
from mtdata.forecast.common import (
    next_times_from_last,
    pd_freq_from_timeframe,
)


class TestDefaultSeasonalityPeriod:
    def test_h1(self):
        assert default_seasonality_period("H1") == 24

    def test_m5(self):
        assert default_seasonality_period("M5") == 288  # 86400/300

    def test_d1(self):
        assert default_seasonality_period("D1") == 5

    def test_unknown(self):
        assert default_seasonality_period("NOPE") == 0


class TestNextTimesFromLast:
    def test_basic(self):
        result = next_times_from_last(1000.0, 60, 3)
        assert result == [1060.0, 1120.0, 1180.0]

    def test_zero_horizon(self):
        assert next_times_from_last(0.0, 60, 0) == []


class TestPdFreqFromTimeframe:
    def test_h4(self):
        assert pd_freq_from_timeframe("H4") == "4h"

    def test_m30(self):
        assert pd_freq_from_timeframe("M30") == "30min"

    def test_fallback(self):
        assert pd_freq_from_timeframe("XX") == "D"
