"""Tests for CandlestickRuntime container."""

from mtdata.patterns.candlestick import CandlestickRuntime


class TestCandlestickRuntime:
    """Runtime init and thread-safety."""

    def test_initial_state(self):
        rt = CandlestickRuntime()
        assert rt.ta is None
        assert rt.mt5 is None
        assert rt.TIMEFRAME_MAP is None
        assert rt._mt5_copy_rates_from is None
        assert rt._rates_to_df is None
        assert rt._symbol_ready_guard is None
        assert rt.ready is False

    def test_manual_assignment(self):
        rt = CandlestickRuntime()
        rt.ta = "fake_ta"
        rt.mt5 = "fake_mt5"
        assert rt.ta == "fake_ta"
        assert rt.mt5 == "fake_mt5"

    def test_ensure_loaded_idempotent(self):
        """After loading, repeated calls don't re-import."""
        rt = CandlestickRuntime()
        # Pre-populate to avoid real imports
        rt.ta = "ta"
        rt.mt5 = "mt5"
        rt.TIMEFRAME_MAP = {"H1": 1}
        rt._mt5_copy_rates_from = lambda: None
        rt._rates_to_df = lambda: None
        rt._symbol_ready_guard = lambda: None
        rt._loaded = True
        # Second call does nothing
        rt.ensure_loaded()
        assert rt.ta == "ta"  # unchanged
