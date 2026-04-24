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
mtdata-cli symbols_list --limit 10
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
mtdata-cli forecast_generate --help
mtdata-cli forecast_generate EURUSD --horizon 12  # symbol is positional
```

### "Invalid choice" for Timeframe/Method/Mode

**Cause:** Invalid value for a constrained parameter.

**Solution:** Use `--help` to see allowed values:
```bash
mtdata-cli forecast_volatility_estimate --help
mtdata-cli patterns_detect --help
```

**Common timeframes:** `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`, `MN1`

### "Unknown parameter" in --params

**Example:**
```bash
--params "invalid_param=5"
```

**Solution:** Check method documentation:
```bash
mtdata-cli forecast_list_methods --json
mtdata-cli indicators_describe rsi --json
```

---

## Model/Library Issues

### "Method X not available"

**Cause:** Required optional dependency not installed.

**Solution:** Check availability:
```bash
mtdata-cli forecast_list_methods --json
```

Look for `available: false` and the `requires` field. Install missing packages:

If you're installing mtdata extras, run the extra commands below from the repository root.

```bash
pip install chronos-forecasting torch  # For Chronos
pip install statsforecast              # For StatsForecast models
pip install arch                       # For GARCH
pip install statsmodels                # For ARIMA/ETS + causal_discover_signals
pip install umap-learn                 # For UMAP dimred (Web UI / analysis)
pip install gluonts[torch]             # For Lag-Llama (pretrained)
pip install QuantLib                   # For barrier option pricing & Heston calibration
pip install optuna                     # For Bayesian hyperparameter tuning
pip install neuralforecast torch       # For NHiTS, TFT, PatchTST, NBEATSx
pip install -e .[forecast-timesfm]     # From the repo root; installs the TimesFM Git-backed extra
# Lag-Llama may require a separate Python env due to upstream pins (see `requirements.txt`).
```

### "Import error" or "Module not found"

**Solution:** Install the package or use a different method:
```bash
pip install <missing_package>
```

Or check if a similar method is available without extra dependencies:
```bash
mtdata-cli forecast_list_library_models native  # Native methods have minimal deps
```

---

## Output Issues

### Output is Hard to Parse

**Solution:** Use JSON format:
```bash
mtdata-cli symbols_describe EURUSD --json
mtdata-cli forecast_generate EURUSD --horizon 12 --json
```

Pipe to `jq` for processing:
```bash
mtdata-cli forecast_generate EURUSD --json | jq '.forecast'
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
mtdata-cli indicators_describe rsi

# Fetch enough bars (at least period + some buffer)
mtdata-cli data_fetch_candles EURUSD --limit 200 --indicators "rsi(14)"
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
mtdata-cli symbols_list --limit 100

# Try without date filters
mtdata-cli data_fetch_candles EURUSD --limit 100
```

### Timestamps Look Wrong

**Cause:** Server timezone offset not configured.

**Solution:** Set one timezone method in `.env`. Prefer an IANA timezone name because it handles DST:
```ini
MT5_SERVER_TZ=Europe/Athens
```

If you only know a fixed broker offset, use minutes from UTC instead:
```ini
MT5_TIME_OFFSET_MINUTES=120  # If server is UTC+2
```

Avoid setting both unless you intentionally want a non-zero `MT5_TIME_OFFSET_MINUTES` value to override `MT5_SERVER_TZ`.

To estimate an offset quickly (run during active market hours so ticks are current):
```bash
python scripts/detect_mt5_time_offset.py --symbol EURUSD
```

Set `MT5_SERVER_TZ` or `MT5_TIME_OFFSET_MINUTES` explicitly before starting `mtdata-webapi`. The Web API no longer auto-detects broker offset or mutates environment variables at startup.

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
mtdata-cli --help forecast
mtdata-cli --help barrier
mtdata-cli --help indicators
```

### Get Command-Specific Help
```bash
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help
```

### Enable Debug Mode
PowerShell:
```powershell
$env:MTDATA_CLI_DEBUG = "1"
mtdata-cli forecast_generate EURUSD --horizon 12
$env:MTDATA_CLI_DEBUG = $null
```

Bash:
```bash
MTDATA_CLI_DEBUG=1 mtdata-cli forecast_generate EURUSD --horizon 12
```

---

## Optional Dependency Issues

### Git-backed Extra Fails to Install

**Symptom:** `pip install -e .[forecast-timesfm]`, `pip install -e .[patterns-ext]`, or `pip install -e .[news-ycnbc]` fails during clone/build on Windows.

**Solution:**
1. Install Visual Studio Build Tools 2022 with the **Desktop development with C++** workload.
2. Make sure Git is installed and available on `PATH`.
3. Install the stable base stack first:
   ```bash
   pip install -r requirements.txt
   ```
4. From the repository root, retry only the extra you need:
   ```bash
   pip install -e .[forecast-timesfm]
   pip install -e .[patterns-ext]
   pip install -e .[news-ycnbc]
   ```

If a Git-backed extra still fails, leave it out and use the rest of mtdata without that integration. For pretrained forecasts, `chronos2` and `chronos_bolt` remain available from the stable install path.

### Optional Native Accelerator Source Build Fails

**Symptom:** `pip install -r requirements-optional-src.txt` fails while building `hnswlib` or `tsdownsample` on Python 3.14.

**What this path is for:**
- `hnswlib`: optional accelerator for `search_engine=hnsw`
- `tsdownsample`: optional accelerator for faster LTTB simplification

**Requirements by package:**
1. `hnswlib`
   - Visual Studio Build Tools 2022 with the **Desktop development with C++** workload
2. `tsdownsample`
   - Visual Studio Build Tools 2022 with the **Desktop development with C++** workload
   - Rust toolchain installed via `rustup`

**Recommended recovery steps:**
```bash
# Install only the package you need first
pip install hnswlib==0.8.0
pip install tsdownsample==0.1.4.1
```

If `tsdownsample` fails and you do not want to install Rust, leave it out. mtdata already falls back to the built-in Python simplification path.

If `hnswlib` fails, leave it out and use `search_engine=ckdtree` instead.

### QuantLib Import Error

**Symptom:** `ModuleNotFoundError: No module named 'QuantLib'` when using `forecast_quantlib_*` commands.

**Solution:**
```bash
pip install QuantLib
```
On Windows, a pre-built wheel is available. On Linux, you may need build tools (`cmake`, `swig`).

### Finviz Errors

**Symptom:** `finviz_*` commands return empty data or connection errors.

**Possible causes:**
- Network/firewall blocking finviz.com
- Rate limiting (finviz throttles rapid requests)

**Note:** `finvizfinance` is a core dependency — it is installed automatically with `pip install -e .`. If you see an import error, your install may be incomplete; reinstall with `pip install -e .` or `pip install -r requirements.txt`. Finviz data is US equities only and delayed 15–20 minutes.

### Optuna Not Available

**Symptom:** `forecast_tune_optuna` fails with import error.

**Solution:**
```bash
pip install optuna
```

### Neural Forecast Models Unavailable

**Symptom:** NHiTS, TFT, PatchTST, or NBEATSx show `available: false` in `forecast_list_methods`.

**Solution:**
```bash
pip install neuralforecast torch
```
These models require PyTorch. GPU is recommended for training speed but not required.

---

## Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| MT5 not connecting | Restart MT5 terminal, ensure login |
| Missing parameter | Check `--help` for required arguments |
| Invalid timeframe | Use: M1, M5, M15, M30, H1, H4, D1, W1, MN1 |
| Method not available | Check `forecast_list_methods` and install deps |
| Output hard to read | Add `--json` |
| Wrong timestamps | Set `MT5_SERVER_TZ` in `.env`, or `MT5_TIME_OFFSET_MINUTES` if you only know a fixed offset |
| Command slow | Reduce `--limit`, use faster method |
| QuantLib import error | `pip install QuantLib` |
| Finviz empty data | Check network / firewall (finvizfinance is pre-installed) |
| Optuna not found | `pip install optuna` |
| Neural models unavailable | `pip install neuralforecast torch` |

---

## See Also

- [SETUP.md](SETUP.md) — Installation guide
- [CLI.md](CLI.md) — Command usage
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [FINVIZ.md](FINVIZ.md) — Finviz fundamental data reference
- [OPTIONS_QUANTLIB.md](OPTIONS_QUANTLIB.md) — Options & QuantLib tools

