"""Test that strategy_backtest full detail mode includes analytical enrichment."""
from __future__ import annotations

import pandas as pd
import pytest

from mtdata.forecast import backtest as forecast_backtest


def _history_from_closes(closes: list[float]) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        open_price = closes[index - 1] if index > 0 else close
        rows.append(
            {
                "time": 1700000000.0 + (index * 3600.0),
                "open": float(open_price),
                "high": float(max(open_price, close)),
                "low": float(min(open_price, close)),
                "close": float(close),
            }
        )
    return pd.DataFrame(rows)


def test_strategy_backtest_full_detail_includes_equity_curve(monkeypatch):
    """Full detail mode should include equity curve data."""
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
    )

    assert out["success"] is True
    assert "equity_curve" in out, "Full detail mode should include equity_curve"
    assert isinstance(out["equity_curve"], list)
    assert len(out["equity_curve"]) > 0
    for point in out["equity_curve"]:
        assert "time" in point
        assert "equity" in point
        assert isinstance(point["equity"], float)


def test_strategy_backtest_full_detail_includes_trade_distribution(monkeypatch):
    """Full detail mode should include trade distribution statistics."""
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
    )

    assert out["success"] is True
    assert "trade_distribution" in out, "Full detail mode should include trade_distribution"
    dist = out["trade_distribution"]
    
    # At least one category should be present (winning or losing)
    assert "winning" in dist or "losing" in dist or "breakeven" in dist
    
    # Check winning trade stats
    if "winning" in dist:
        assert "count" in dist["winning"]
        assert "avg_return" in dist["winning"]
        assert "max" in dist["winning"]
        assert "min" in dist["winning"]


def test_strategy_backtest_full_detail_includes_monthly_breakdown(monkeypatch):
    """Full detail mode should include monthly performance breakdown."""
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="full",
    )

    assert out["success"] is True
    assert "monthly_breakdown" in out, "Full detail mode should include monthly_breakdown"
    
    if out["monthly_breakdown"]:  # Only check if there is data
        for month_data in out["monthly_breakdown"]:
            assert "month" in month_data
            assert "return" in month_data
            assert "trades" in month_data


def test_strategy_backtest_compact_excludes_analytical_detail(monkeypatch):
    """Compact detail mode should NOT include analytical enrichment."""
    monkeypatch.setattr(
        forecast_backtest,
        "_fetch_history",
        lambda symbol, timeframe, need, as_of=None: _history_from_closes(
            [1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ),
    )

    out = forecast_backtest.strategy_backtest(
        symbol="EURUSD",
        timeframe="H1",
        strategy="sma_cross",
        lookback=8,
        fast_period=2,
        slow_period=3,
        detail="compact",
    )

    assert out["success"] is True
    assert "trades" not in out, "Compact mode should not include trades"
    assert "equity_curve" not in out, "Compact mode should not include equity_curve"
    assert "trade_distribution" not in out, "Compact mode should not include trade_distribution"
    assert "monthly_breakdown" not in out, "Compact mode should not include monthly_breakdown"
