"""Test for volume validation issue #36."""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from mtdata.core.trading.requests import TradePlaceRequest
from mtdata.core.trading.use_cases import run_trade_place
from mtdata.core.trading import validation, time
from unittest.mock import MagicMock


def make_mocks():
    """Create mock functions for trade_place."""
    return {
        "normalize_order_type_input": validation._normalize_order_type_input,
        "normalize_pending_expiration": time._normalize_pending_expiration,
        "prevalidate_trade_place_market_input": validation._prevalidate_trade_place_market_input,
        "place_market_order": MagicMock(),
        "place_pending_order": MagicMock(),
        "close_positions": MagicMock(),
        "safe_int_ticket": validation._safe_int_ticket,
    }


def test_trade_place_rejects_zero_volume_in_dry_run():
    """Zero volume should be rejected even in dry run mode."""
    req = TradePlaceRequest(
        symbol='EURUSD',
        volume=0,
        order_type='BUY',
        dry_run=True,
        require_sl_tp=False
    )
    result = run_trade_place(req, **make_mocks())
    
    assert "error" in result
    assert "volume must be positive" in result.get("error", "").lower()
    assert result.get("volume_received") == 0.0


def test_trade_place_rejects_negative_volume_in_dry_run():
    """Negative volume should be rejected even in dry run mode."""
    req = TradePlaceRequest(
        symbol='EURUSD',
        volume=-0.01,
        order_type='BUY',
        dry_run=True,
        require_sl_tp=False
    )
    result = run_trade_place(req, **make_mocks())
    
    assert "error" in result
    assert "volume must be positive" in result.get("error", "").lower()
    assert result.get("volume_received") == -0.01


def test_trade_place_rejects_zero_volume_in_normal_mode():
    """Zero volume should be rejected in normal mode."""
    req = TradePlaceRequest(
        symbol='EURUSD',
        volume=0,
        order_type='BUY',
        dry_run=False,
        require_sl_tp=False
    )
    result = run_trade_place(req, **make_mocks())
    
    assert "error" in result
    assert "volume must be positive" in result.get("error", "").lower()


def test_trade_place_rejects_negative_volume_in_normal_mode():
    """Negative volume should be rejected in normal mode."""
    req = TradePlaceRequest(
        symbol='EURUSD',
        volume=-0.05,
        order_type='SELL',
        dry_run=False,
        require_sl_tp=False
    )
    result = run_trade_place(req, **make_mocks())
    
    assert "error" in result
    assert "volume must be positive" in result.get("error", "").lower()


def test_trade_place_rejects_non_finite_volume():
    """Non-finite volumes (NaN, Inf) should be rejected."""
    import math
    
    req_nan = TradePlaceRequest(
        symbol='EURUSD',
        volume=float('nan'),
        order_type='BUY',
        dry_run=True
    )
    result_nan = run_trade_place(req_nan, **make_mocks())
    assert "error" in result_nan
    assert "finite" in result_nan.get("error", "").lower()
    
    req_inf = TradePlaceRequest(
        symbol='EURUSD',
        volume=float('inf'),
        order_type='BUY',
        dry_run=True
    )
    result_inf = run_trade_place(req_inf, **make_mocks())
    assert "error" in result_inf
    assert "finite" in result_inf.get("error", "").lower()


def test_trade_place_accepts_positive_volume_in_dry_run():
    """Positive volumes should be accepted in dry run."""
    req = TradePlaceRequest(
        symbol='EURUSD',
        volume=0.01,
        order_type='BUY',
        dry_run=True,
        require_sl_tp=False
    )
    result = run_trade_place(req, **make_mocks())
    
    assert result.get("success") is True
    assert result.get("dry_run") is True
    assert result.get("volume") == 0.01


if __name__ == "__main__":
    test_trade_place_rejects_zero_volume_in_dry_run()
    print("✓ Zero volume rejected in dry_run")
    
    test_trade_place_rejects_negative_volume_in_dry_run()
    print("✓ Negative volume rejected in dry_run")
    
    test_trade_place_rejects_zero_volume_in_normal_mode()
    print("✓ Zero volume rejected in normal mode")
    
    test_trade_place_rejects_negative_volume_in_normal_mode()
    print("✓ Negative volume rejected in normal mode")
    
    test_trade_place_rejects_non_finite_volume()
    print("✓ Non-finite volume rejected")
    
    test_trade_place_accepts_positive_volume_in_dry_run()
    print("✓ Positive volume accepted in dry_run")
    
    print("\nAll tests passed!")
