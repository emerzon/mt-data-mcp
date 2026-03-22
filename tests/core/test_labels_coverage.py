"""Tests for core/labels.py — triple barrier labeling (mocked MT5)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper to build a mock OHLC DataFrame
# ---------------------------------------------------------------------------


def _make_df(n: int = 50, base: float = 1.1000, step: float = 0.0005):
    """Create a deterministic OHLC DataFrame with strictly increasing time."""
    times = np.arange(0, n * 3600, 3600, dtype=float)
    closes = np.array([base + i * step for i in range(n)])
    highs = closes + 0.0010
    lows = closes - 0.0010
    return pd.DataFrame({
        "time": times,
        "open": closes - 0.0002,
        "high": highs,
        "low": lows,
        "close": closes,
    })


def _make_flat_df(n: int = 50, price: float = 1.1000):
    """Flat price series — no barrier will be hit."""
    times = np.arange(0, n * 3600, 3600, dtype=float)
    return pd.DataFrame({
        "time": times,
        "open": np.full(n, price),
        "high": np.full(n, price + 0.00001),
        "low": np.full(n, price - 0.00001),
        "close": np.full(n, price),
    })


def _make_down_df(n: int = 50, base: float = 1.2000, step: float = 0.0015):
    """Down-trending OHLC DataFrame for short-side barrier tests."""
    return _make_df(n=n, base=base, step=-abs(step))


_LABELS_MOD = "mtdata.core.labels"


def _get_raw_fn():
    """Get the unwrapped labels_triple_barrier function (bypasses mcp.tool)."""
    from mtdata.core.labels import labels_triple_barrier
    raw = labels_triple_barrier.__wrapped__

    def _call(*args, **kwargs):
        with patch("mtdata.core.labels.ensure_mt5_connection_or_raise", return_value=None):
            return raw(*args, **kwargs)

    return _call


class TestLabelsTripleBarrier:
    """Tests targeting lines 43-146 of labels.py."""

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_basic_pct_barriers(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12)
        assert result["success"] is True
        assert len(result["labels"]) > 0

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_abs_barriers(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60, base=100.0, step=0.5)
        result = _get_raw_fn()("SPX", tp_abs=110.0, sl_abs=90.0, horizon=10)
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_pip_barriers(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pips=50, sl_pips=50, horizon=12)
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_insufficient_history(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(5)
        result = _get_raw_fn()("X", tp_pct=1.0, sl_pct=1.0, horizon=12)
        assert "error" in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_no_barriers_gives_error(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", horizon=12)
        assert "error" in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_non_finite_barrier_gives_error(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_abs=float("nan"), sl_abs=1.0, horizon=12)
        assert "error" in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_label_on_close(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12, label_on="close")
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_label_on_high_low(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12, label_on="high_low")
        assert result["success"] is True

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_summary(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, output="summary")
        assert result["success"] is True
        assert "summary" in result
        assert "counts" in result["summary"]

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_compact(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, output="compact", lookback=10)
        assert result["success"] is True
        assert "summary" in result
        assert len(result["labels"]) <= 10

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_full(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, output="full")
        assert result["success"] is True
        assert "entries" in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_summary_only_flag(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, output="compact", summary_only=True)
        assert result["success"] is True
        assert "summary" in result
        assert "entries" not in result
        assert "labels" not in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_output_summary_only_alias(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, output="summary_only")
        assert result["success"] is True
        assert "summary" in result
        assert "entries" not in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_flat_price_neutral_labels(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_flat_df(60)
        result = _get_raw_fn()("FLAT", tp_pct=5.0, sl_pct=5.0, horizon=5)
        assert result["success"] is True
        assert all(l == 0 for l in result["labels"])

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_minimum_history_keeps_last_valid_entry_bar(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_flat_df(5)
        result = _get_raw_fn()("EURUSD", tp_pct=5.0, sl_pct=5.0, horizon=3)

        assert result["success"] is True
        assert len(result["labels"]) == 2
        assert len(result["entries"]) == 2

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history", side_effect=Exception("MT5 down"))
    def test_exception_returns_error(self, mock_hist, mock_den, mock_pip):
        result = _get_raw_fn()("EURUSD", tp_pct=1.0, sl_pct=1.0, horizon=5)
        assert "error" in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_params_used_in_output(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=1.0, sl_pct=1.0, horizon=5)
        assert "params_used" not in result

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_holding_bars_within_horizon(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=10)
        assert result["success"] is True
        for h in result["holding_bars"]:
            assert 1 <= h <= 10

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_summary_median_holding(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_df(60)
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=5, output="summary")
        assert "median_holding_bars" in result["summary"]

    @patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
    @patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
    @patch(f"{_LABELS_MOD}._fetch_history")
    def test_short_direction_labels_falling_prices_as_take_profit(self, mock_hist, mock_den, mock_pip):
        mock_hist.return_value = _make_down_df(60)
        result = _get_raw_fn()(
            "EURUSD",
            tp_pct=0.2,
            sl_pct=0.2,
            horizon=3,
            direction="short",
            label_on="close",
        )
        assert result["success"] is True
        assert result["direction"] == "short"
        assert result["labels"][0] == 1


@patch(f"{_LABELS_MOD}._get_pip_size", return_value=0.0001)
@patch(f"{_LABELS_MOD}._resolve_denoise_base_col", return_value="close")
@patch(f"{_LABELS_MOD}._fetch_history")
def test_labels_triple_barrier_logs_finish_event(mock_hist, mock_den, mock_pip, caplog):
    mock_hist.return_value = _make_df(60)
    with caplog.at_level("INFO", logger="mtdata.core.labels"):
        result = _get_raw_fn()("EURUSD", tp_pct=0.5, sl_pct=0.5, horizon=12)

    assert result["success"] is True
    assert any(
        "event=finish operation=labels_triple_barrier success=True" in record.message
        for record in caplog.records
    )
