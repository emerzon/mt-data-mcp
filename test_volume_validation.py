#!/usr/bin/env python
"""Quick test for volume validation issue."""
import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mtdata.core.trading.requests import TradePlaceRequest
from mtdata.core.trading.use_cases import run_trade_place
from mtdata.core.trading import validation, time, execution
from unittest.mock import MagicMock

# Mock functions
def mock_normalize_order_type_input(order_type):
    return validation._normalize_order_type_input(order_type)

def mock_normalize_pending_expiration(exp):
    return time._normalize_pending_expiration(exp)

def mock_prevalidate(symbol, volume):
    return validation._prevalidate_trade_place_market_input(symbol, volume)

mock_place_market = MagicMock()
mock_place_pending = MagicMock()
mock_close = MagicMock()
mock_safe_int = validation._safe_int_ticket

# Test 1: Zero volume in dry run with require_sl_tp=False
print("Test 1: Zero volume, dry_run=True, require_sl_tp=False")
req1 = TradePlaceRequest(
    symbol='EURUSD', 
    volume=0, 
    order_type='BUY', 
    dry_run=True, 
    require_sl_tp=False
)
result1 = run_trade_place(
    req1,
    normalize_order_type_input=mock_normalize_order_type_input,
    normalize_pending_expiration=mock_normalize_pending_expiration,
    prevalidate_trade_place_market_input=mock_prevalidate,
    place_market_order=mock_place_market,
    place_pending_order=mock_place_pending,
    close_positions=mock_close,
    safe_int_ticket=mock_safe_int,
)
print(json.dumps(result1, indent=2, default=str))
print()

# Test 2: Negative volume in dry run with require_sl_tp=False
print("Test 2: Negative volume, dry_run=True, require_sl_tp=False")
req2 = TradePlaceRequest(
    symbol='EURUSD', 
    volume=-0.01, 
    order_type='BUY', 
    dry_run=True, 
    require_sl_tp=False
)
result2 = run_trade_place(
    req2,
    normalize_order_type_input=mock_normalize_order_type_input,
    normalize_pending_expiration=mock_normalize_pending_expiration,
    prevalidate_trade_place_market_input=mock_prevalidate,
    place_market_order=mock_place_market,
    place_pending_order=mock_place_pending,
    close_positions=mock_close,
    safe_int_ticket=mock_safe_int,
)
print(json.dumps(result2, indent=2, default=str))
