# Troubleshooting

Common issues when running mtdata and how to resolve them.

**Related:**
- [SETUP.md](SETUP.md) — Installation and configuration
- [CLI.md](CLI.md) — Command usage

---

## Connection Issues

### "Could not connect to MT5" or Empty Data

**Checklist:**
1. MetaTrader 5 terminal is installed and running
2. Terminal is logged in to a broker account
3. Symbol is visible in Market Watch

**Test connection:**
```bash
python cli.py symbols_list --limit 10
```

If this works but candles fail:
- Check symbol spelling (case-sensitive for some brokers)
- Ensure symbol is added to Market Watch in MT5

### Terminal Keeps Disconnecting

**Possible causes:**
- MT5 requires periodic reconnection
- Internet stability issues

**Solution:** The library auto-reconnects, but long-running scripts may need error handling:
```python
try:
    result = tool_function(...)
except Exception as e:
    # Retry logic here
```

---

## Parameter Errors

### "validation error ... Field required"

**Example:**
```
1 validation error for InputSchema
symbol
  Field required [type=missing, input_value={}, input_type=dict]
```

**Cause:** A required parameter is missing.

**Solution:** Check command help and provide required arguments:
```bash
python cli.py forecast_generate --help
python cli.py forecast_generate EURUSD --horizon 12  # symbol is positional
```

### "Invalid choice" for Timeframe/Method/Mode

**Cause:** Invalid value for a constrained parameter.

**Solution:** Use `--help` to see allowed values:
```bash
python cli.py forecast_volatility_estimate --help
python cli.py patterns_detect --help
```

**Common timeframes:** `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`, `MN1`

### "Unknown parameter" in --params

**Example:**
```bash
--params "invalid_param=5"
```

**Solution:** Check method documentation:
```bash
python cli.py forecast_list_methods --format json
python cli.py indicators_describe rsi --format json
```

---

## Model/Library Issues

### "Method X not available"

**Cause:** Required optional dependency not installed.

**Solution:** Check availability:
```bash
python cli.py forecast_list_methods --format json
```

Look for `available: false` and the `requires` field. Install missing packages:
```bash
pip install chronos-forecasting torch  # For Chronos
pip install statsforecast              # For StatsForecast models
pip install arch                       # For GARCH
```

### "Import error" or "Module not found"

**Solution:** Install the package or use a different method:
```bash
pip install <missing_package>
```

Or check if a similar method is available without extra dependencies:
```bash
python cli.py forecast_list_library_models native  # Native methods have minimal deps
```

---

## Output Issues

### Output is Hard to Parse

**Solution:** Use JSON format:
```bash
python cli.py symbols_describe EURUSD --format json
python cli.py forecast_generate EURUSD --horizon 12 --format json
```

Pipe to `jq` for processing:
```bash
python cli.py forecast_generate EURUSD --format json | jq '.forecast'
```

### Output is Too Verbose

**Solution:** Omit `--verbose` flag (default is compact output).

### Missing Columns in Output

**Cause:** Indicator calculation failed or data insufficient.

**Solution:** Check that:
1. Enough bars are fetched for the indicator period
2. Indicator name is spelled correctly

```bash
# Check indicator syntax
python cli.py indicators_describe rsi

# Fetch enough bars (at least period + some buffer)
python cli.py data_fetch_candles EURUSD --limit 200 --indicators "rsi(14)"
```

---

## Data Issues

### No Data Returned

**Possible causes:**
- Symbol not in Market Watch
- Time range has no data (weekend, holiday)
- Broker doesn't provide history for this symbol

**Solution:**
```bash
# Check if symbol exists
python cli.py symbols_list --limit 100

# Try without date filters
python cli.py data_fetch_candles EURUSD --limit 100
```

### Timestamps Look Wrong

**Cause:** Server timezone offset not configured.

**Solution:** Set timezone in `.env`:
```ini
MT5_TIME_OFFSET_MINUTES=120  # If server is UTC+2
```

Or use server timezone name:
```ini
MT5_SERVER_TZ=Europe/Athens
```

To estimate an offset quickly (run during active market hours so ticks are current):
```bash
python scripts/detect_mt5_time_offset.py --symbol EURUSD
```

### Volume is Always Zero

**Cause:** Forex spot typically has indicative volume (tick count, not real volume).

**Note:** This is expected for most forex pairs. Use volume-based indicators cautiously.

---

## Performance Issues

### Command is Slow

**Possible causes:**
- Large `--lookback` or `--limit`
- Complex model (Chronos, GARCH)
- First run of pre-trained model (downloading weights)

**Solutions:**
1. Reduce data size: `--limit 500` instead of `--limit 5000`
2. Use faster methods: `theta` instead of `chronos2`
3. Reduce simulation count: `--params "n_sims=1000"` instead of 5000

### Memory Error

**Cause:** Large data or too many simulations.

**Solution:**
- Reduce `--limit`
- Reduce `n_sims` for barrier analysis
- Use streaming instead of batch when possible

---

## Getting Help

### Search Commands by Topic
```bash
python cli.py --help forecast
python cli.py --help barrier
python cli.py --help indicators
```

### Get Command-Specific Help
```bash
python cli.py forecast_generate --help
python cli.py regime_detect --help
```

### Enable Debug Mode
```bash
MTDATA_CLI_DEBUG=1 python cli.py forecast_generate EURUSD --horizon 12
```

---

## Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| MT5 not connecting | Restart MT5 terminal, ensure login |
| Missing parameter | Check `--help` for required arguments |
| Invalid timeframe | Use: M1, M5, M15, M30, H1, H4, D1, W1, MN1 |
| Method not available | Check `forecast_list_methods` and install deps |
| Output hard to read | Add `--format json` |
| Wrong timestamps | Set `MT5_TIME_OFFSET_MINUTES` in `.env` |
| Command slow | Reduce `--limit`, use faster method |

---

## See Also

- [SETUP.md](SETUP.md) — Installation guide
- [CLI.md](CLI.md) — Command usage
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
