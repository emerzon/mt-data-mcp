# Common MCP Tool Errors

This document explains common errors you might encounter when using the MetaTrader MCP server and how to fix them.

## Validation Errors

### "Field required" Error

**Error Message:**
```
1 validation error for InputSchema
symbol
  Field required [type=missing, input_value={}, input_type=dict]
```

**What it means:** You're trying to call a tool without providing a required parameter.

**Solution:** Check which parameters are required for the tool you're calling. Most tools require at least a `symbol` parameter.

**Example:**
- ❌ **Wrong:** `patterns_detect()`
- ✅ **Correct:** `patterns_detect(symbol="EURUSD")`

### Common Required Parameters

- **symbol**: Required for most data, pattern, and trading tools (e.g., "EURUSD", "BTCUSD")
- **timeframe**: Usually optional with a default (e.g., "H1", "M15", "D1")
- **id/ticket**: Required when modifying or closing specific positions/orders

## Examples by Tool Category

### Data Tools
```python
# Fetch candles - symbol is required
data_fetch_candles(symbol="EURUSD", timeframe="H1", limit=100)

# Fetch ticks
data_fetch_ticks(symbol="EURUSD", limit=100)
```

### Pattern Detection
```python
# Candlestick patterns - symbol is required
patterns_detect(symbol="EURUSD", mode="candlestick")

# Classic chart patterns
patterns_detect(symbol="EURUSD", mode="classic", limit=500)
```

### Trading Tools
```python
# Get account info  - no parameters needed
trading_account_info()

# Get positions - symbol is optional
trading_positions_get()  # All positions
trading_positions_get(symbol="EURUSD")  # Specific symbol

# Close position - ticket is required
trading_positions_close(ticket=123456)

# Risk analysis - all parameters optional
trading_risk_analyze()  # Analyze current portfolio
trading_risk_analyze(symbol="EURUSD", desired_risk_pct=2.0, proposed_entry=1.1000, proposed_sl=1.0950)
```

### Forecast Tools
```python
# Barrier probability - symbol is required
forecast_barrier_prob(
    symbol="EURUSD",
    method="mc",  # or "closed_form"
    direction="long",
    tp_abs=1.1100,
    sl_abs=1.0950
)
```

## Tips

1. **Always provide the symbol**: Most tools need to know which instrument you're working with.
2. **Check the tool signature**: Use your MCP client's tool discovery to see required vs optional parameters.
3. **Use defaults**: Many parameters have sensible defaults (e.g., timeframe="H1").
4. **Start simple**: Start with just the required parameters, then add optional ones as needed.
