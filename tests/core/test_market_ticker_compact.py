from mtdata.core.market_depth import _compact_market_ticker_payload


def test_compact_ticker_keeps_absolute_spread_for_non_forex_quotes() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "XAUUSD",
            "point": 0.01,
            "bid": 4061.0,
            "ask": 4061.15,
            "spread": 0.15,
            "spread_points": 15.0,
            "spread_pct": 0.003694,
            "contract_size": 100.0,
            "lot_definition": "1 broker lot equals contract_size contract units.",
            "pricing_basis": "per_1_lot_estimate",
            "units": {"spread": "price", "lot": "broker_lot"},
        }
    )

    assert result["spread"] == 0.15
    assert result["spread_points"] == 15.0
    assert result["spread_pct"] == 0.003694
    assert result["point"] == 0.01
    assert "spread_pips" not in result
    assert "contract_size" not in result
    assert "lot_definition" not in result
    assert "pricing_basis" not in result
    assert "units" not in result


def test_compact_ticker_preserves_delayed_freshness_label() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "EURUSD",
            "freshness": "delayed, tick 1m 3s ago",
            "freshness_state": "delayed",
            "data_age_seconds": 63.0,
            "usable_for_live_trading": False,
        }
    )

    assert result["freshness"] == "delayed, tick 1m 3s ago"
    assert result["freshness_state"] == "delayed"


def test_compact_ticker_preserves_future_timestamp_cause() -> None:
    result = _compact_market_ticker_payload(
        {
            "success": True,
            "symbol": "GBPSGD",
            "freshness": "stale, tick 0s ago",
            "freshness_state": "stale",
            "freshness_reason": "future_timestamp",
            "data_age_seconds": 0.0,
            "usable_for_live_trading": False,
            "timestamp_in_future": True,
            "timestamp_skew_seconds": 6.913,
            "timestamp_warning": "Correct MT5 clock alignment before trading.",
            "warning": "Correct MT5 clock alignment before trading.",
        }
    )

    assert result["freshness_reason"] == "future_timestamp"
    assert result["timestamp_in_future"] is True
    assert result["timestamp_skew_seconds"] == 6.913
    assert result["timestamp_warning"] == "Correct MT5 clock alignment before trading."
    assert "warning" not in result
