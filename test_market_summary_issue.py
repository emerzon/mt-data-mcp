#!/usr/bin/env python
"""Test to reproduce the market_status summary issue."""

from datetime import datetime, timezone, timedelta
from src.mtdata.core.market_status import market_status

# Just call the actual tool
result = market_status(detail='compact')

# It should be a dict with success key if not wrapped
if isinstance(result, dict) and 'body' in result:
    result = result['body']

print("Result type:", type(result))
print("Keys:", list(result.keys()) if isinstance(result, dict) else "Not a dict")

if isinstance(result, dict):
    print("\nSummary:", result.get('summary'))
    print("Markets open:", result.get('markets_open'))
    print("Markets closed:", result.get('markets_closed'))
    print("Total markets:", len(result.get('markets', [])))
    
    # Count by status
    markets = result.get('markets', [])
    statuses = {}
    for m in markets:
        status = m.get('status', 'unknown')
        statuses[status] = statuses.get(status, 0) + 1
    
    print("\nStatus breakdown:")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")
    
    # Show the market list
    print("\nMarkets:")
    for m in markets:
        print(f"  {m['symbol']}: {m['status']}")
