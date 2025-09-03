# MetaTrader5 MCP Server

A Model Context Protocol (MCP) server that provides access to MetaTrader5 market data and trading functionality.

## Features

- Connect to MetaTrader5 terminal
- Retrieve symbol information and market data
- Get historical rates (OHLCV) for any timeframe
- Access tick data and market depth (DOM)
- Real-time market data streaming
- Support for all major trading instruments

## Installation

1. Install MetaTrader5 terminal
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and configure your MetaTrader5 credentials:

```bash
cp .env.example .env
```

Edit the `.env` file with your broker credentials (optional - you can also connect without pre-configured credentials).

Timezone configuration:
- If your MT5 server time is not UTC, set `MT5_TIME_OFFSET_MINUTES` to the offset in minutes (can be negative).
- Example: UTC+2 → `MT5_TIME_OFFSET_MINUTES=120`; UTC-4 → `-240`.
- All MT5 timestamps are normalized to UTC using this offset.

## Usage

### Starting the MCP Server

```bash
python server.py
```

### Available Tools

#### Market Data
- `get_symbols(search_term, limit)` - Smart search for trading symbols (CSV output)
- `get_symbol_groups(search_term, limit)` - Get available symbol groups (CSV output)
- `get_symbol_info(symbol)` - Get detailed symbol information
- `get_rates(symbol, timeframe, candles, start_datetime, end_datetime, ohlcv, ti, denoise, simplify)` - Get historical OHLCV data in CSV format
- `get_ticks(symbol, count, start_datetime, simplify)` - Get tick data in CSV format
- `get_market_depth(symbol)` - Get market depth (DOM)
- `get_indicators()` - List indicators (CSV: name,category)
-  - Indicator params + full help (JSON)
- `get_denoise_methods()` - JSON with denoise methods, availability, and parameter docs
- `get_candlestick_patterns(symbol, timeframe, candles)` - Detect candlestick patterns (CSV: time,pattern)
- `get_forecast(symbol, timeframe, method, horizon, lookback?, as_of?, params?, ci_alpha?, target?, denoise?)` - Fast forecasts and optional intervals
- `get_forecast_methods()` - Enumerate forecast methods, availability, and parameter docs

### Example Usage

The server automatically connects to MetaTrader5 when any tool is called. Connection credentials can be configured via environment variables or the server will use the terminal's current login.

```python
# Get all available symbol groups from MT5
groups = get_symbol_groups()

# Get all visible symbols (no search term)
all_symbols = get_symbols()

# Smart search: Find symbols containing "EUR" (EURUSD, JPYEUR, etc.)
eur_symbols = get_symbols(search_term="EUR", limit=20)

# Search by group: Find all major forex pairs
majors = get_symbols(search_term="Majors", limit=10)

# Limit number of results
eur_top5 = get_symbols(search_term="EUR", limit=5)

# Get symbol information
info = get_symbol_info("GBPUSD")

# Get EUR/USD hourly data
rates = get_rates("EURUSD", "H1", 100)

# Get tick data
ticks = get_ticks("EURUSD", 50)

# Downsample output for readability (LTTB):
# Simplification represents all numeric columns (e.g., OHLC/volumes or tick fields) and
# selects a common set of rows that preserves overall shape across them.
# Simplify (default method: LTTB)
# Modes:
# - select (default): keep representative rows
# - approximate: aggregate numeric columns between selected breakpoints (means)
# - resample: time-based bucketing by `bucket_seconds` (means)

# Select (default)
rates_small = get_rates("EURUSD", "H1", 2000, simplify={"method": "lttb", "points": 200})
ticks_small = get_ticks("EURUSD", 2000, simplify={"ratio": 0.1})

# Approximate (segment aggregation)
rates_approx = get_rates("EURUSD", "H1", 2000, simplify={"method": "lttb", "mode": "approximate", "points": 150})

# Resample (time buckets)
rates_resampled = get_rates("EURUSD", "H1", 2000, simplify={"mode": "resample", "bucket_seconds": 6*3600})

# Alternative simplification methods
# - RDP (Douglas–Peucker): preserve key bends via epsilon tolerance
rates_rdp = get_rates("EURUSD", "M15", 5000, simplify={"method": "rdp", "epsilon": 0.0008})

# - PLA (Piecewise Linear): segment by max perpendicular error or fixed segments
rates_pla = get_rates("EURUSD", "H1", 3000, simplify={"method": "pla", "max_error": 0.001})
rates_pla_fixed = get_rates("EURUSD", "H1", 3000, simplify={"method": "pla", "points": 150})

# - APCA (Piecewise Constant): stepwise segments by max absolute deviation or fixed segments
ticks_apca = get_ticks("EURUSD", 5000, simplify={"method": "apca", "max_error": 0.0005})

# Threshold auto-tuning
# For RDP/PLA/APCA, if you provide points/max_points/ratio without epsilon/max_error,
# the server auto-tunes epsilon or max_error to hit the requested number of points.

# CLI tips
# - Show JSON instead of CSV (to see simplify meta):
#   python cli.py fetch_candles EURUSD --simplify lttb --simplify-params points=200 --format json
# - Modes:
#   python cli.py fetch_candles EURUSD --simplify lttb --simplify-params "mode=approximate,points=150" --format json
#   python cli.py fetch_candles EURUSD --simplify lttb --simplify-params "mode=resample,bucket_seconds=21600" --format json
# - CSV is default; pass points/ratio to actually reduce rows, e.g.:
#   python cli.py fetch_ticks EURUSD --simplify lttb --simplify-params ratio=0.2
```

### Forecasting

Point forecasts for the next bars using training‑light methods. Discover available methods and params via `get_forecast_methods()`.

- Methods:
  - Baselines: `naive`, `drift`, `seasonal_naive`
  - Fast models: `theta`, `fourier_ols`
  - ETS (statsmodels): `ses`, `holt`, `holt_winters_add`, `holt_winters_mul`
  - ARIMA (statsmodels): `arima`, `sarima`

- Common parameters:
  - `symbol`, `timeframe`, `horizon`
  - `lookback` (optional) or auto lookback per method
  - `params`: method‑specific, e.g., `{K:3}` for `fourier_ols`, `{alpha:0.2}` for `theta`, `{seasonality: m}` for seasonal methods (auto if omitted)
  - `ci_alpha` (default 0.05; set null to disable intervals)
  - `as_of`: forecast anchor in UTC for backtesting (e.g., `2025-08-29 14:30`)
  - `denoise`: apply `get_denoise_methods` to smooth `close` before modeling
  - `target`: `"price"` or `"return"` (log‑returns). When `return`, both return path and recomposed price path are included.

- Output includes:
  - Future `times[]`
  - `forecast_price[]` (and `forecast_return[]` when `target="return"`)

### Denoising

- Defaults:
  - when: `pre_ti` (applies before indicators and simplify)
  - columns: `ohlcv` (all of open, high, low, close, plus volume or tick_volume)
  - keep_original: `false` (overwrites columns in place)
  - indicators: not denoised by default

- Methods: `ema`, `sma`, `median`, `lowpass_fft`, `wavelet`, `emd`, `eemd`, `ceemdan` (availability depends on optional deps)

- Examples:
  - Overwrite all OHLCV with EMA smoothing:
    - `python cli.py fetch_candles EURUSD --denoise ema --limit 200 --format csv`
  - Denoise OHLCV and then reduce rows:
    - `python cli.py fetch_candles EURUSD --denoise emd --simplify lttb --simplify-params points=50 --limit 1000 --format json`
  - Also denoise indicators (post-TI), in place:
    - `python cli.py fetch_candles EURUSD --indicators "rsi(14),ema(20)" --denoise ema --denoise-params apply_to_ti=true --limit 500 --format csv`
  - Customize columns/timing:
    - `--denoise-params "columns=open,high,low,close,volume,when=post_ti,keep_original=true"`
  - Optional `lower_price[]`/`upper_price[]` when `ci_alpha` provided
  - `params_used`, `lookback_used`, `seasonality_period`, `as_of`
  - Training window timestamps: `train_start`, `train_end` (formatted per `client_tz`)
  - Overall forecast trend label: `forecast_trend` (`"up"`, `"down"`, or `"flat"`)

- Dependencies:
  - Base methods require only NumPy/Pandas.
  - ETS/ARIMA require `statsmodels` and `scipy` (declared in `requirements.txt`).

## Intelligent Symbol Search

The `get_symbols` function uses a smart 3-tier search strategy:

Search order:
1. Group name match (e.g., `Majors`, `Forex`)
2. Symbol name match (e.g., `EUR` → EURUSD, JPYEUR, ...)
3. Description match (symbol or group path)

Parameters:
- `search_term`: Optional search term. If omitted, lists visible symbols.
- `limit`: Optional maximum number of symbols to return.

Response format:
- CSV with header `name,group,description` and one row per symbol.

## Date Inputs

Flexible date/time parsing powered by `dateparser`:
- Accepts natural language (e.g., `yesterday 14:00`, `2 days ago`)
- Accepts common formats (e.g., `2025-08-29`, `2025/08/29 14:30`, `2025-08-29 14:30 UTC`)
- Times are interpreted and converted to UTC internally

Usage patterns for `get_rates`:
- Only `start_datetime`: returns bars forward from the start (up to `candles`).
- Only `end_datetime`: returns the last `candles` bars before the end.
- Both `start_datetime` and `end_datetime`: returns bars within the range (ignores `candles`).

## Quote Data Format

Historical rates and tick data are returned in compact CSV format with intelligent column filtering:

### Historical Rates (`get_rates`)
By default returns: `time,close` (using `ohlcv=["C"]`).

Selecting OHLCV subset:
- Use `ohlcv` as a list of letters from `{O,H,L,C,V}`; time is always included (`V` maps to `tick_volume`).
- CLI examples: `--ohlcv O C` → time,open,close; `--ohlcv V` → time,tick_volume

Technical indicators via `ti`:
- Pass a comma-separated list like: `ti=sma(14),rsi(14),ema(50)`
- Supported basics: `sma(length)`, `ema(length)`, `rsi(length)`, `macd(fast,slow,signal)`, `stoch(k,d,smooth)`, `bbands(length,std)`
- Indicator columns are appended to the CSV after the requested OHLCV columns.


MCP structured payload example (preferred for programmatic clients):


- **Meaningful Data Check**: Columns are included only if they have at least one non-zero value OR multiple different values
- **Space Efficiency**: Empty/constant columns are automatically excluded to reduce response size
- **Consistency**: Core OHLC columns (rates) and bid/ask columns (ticks) are always included

### Response Format
Both functions return:
```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "H1",  // rates only
  "candles": 100,
  "csv_data": "time,open,high,low,close,tick_volume\n2025-08-29T14:00:00,1.16945,1.17072,1.16937,1.17017,2690\n..."
}
```

## Timeframes

Supported timeframes for historical data (MetaTrader5):
- Minutes: `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M10`, `M12`, `M15`, `M20`, `M30`
- Hours: `H1`, `H2`, `H3`, `H4`, `H6`, `H8`, `H12`
- Days/Weeks/Months: `D1`, `W1`, `MN1`

## Requirements

- Windows OS (MetaTrader5 requirement)
- MetaTrader5 terminal installed
- Python 3.8+
- Active internet connection for real-time data

## Error Handling

All functions return structured responses with success/error status:

```python
{
    "success": True/False,
    "data": {...},  # on success
    "error": "Error message"  # on failure
}
```

## Security Notes

- Store credentials securely in environment variables
- Use demo accounts for testing
- Never commit credentials to version control
- Consider using read-only API access where possible

## License

MIT License
Optional denoising via `denoise`:
- JSON object: `{ "method": "ema|sma|median|lowpass_fft|none", "columns": ["close"], "when": "pre_ti|post_ti", "params": { ... }, "keep_original": bool, "suffix": "_dn" }`
- Defaults: `method="none"`; if `post_ti`, keeps original columns and appends suffixed denoised columns.
- Examples: pre-TI EMA on close: `{"method":"ema","params":{"span":10},"when":"pre_ti"}`; post-TI median on close and an indicator: `{"method":"median","columns":["close","rsi_14"],"when":"post_ti","params":{"window":7}}`.

Wavelet/EMD options (optional dependencies):
- Enable `method="wavelet"` by installing PyWavelets: `pip install PyWavelets`
  - Params: `wavelet` (e.g., `"db4"`), `level` (auto if omitted), `threshold` (`"auto"` or number), `mode` (`"soft"|"hard"`)
- Enable `method="emd|eemd|ceemdan"` by installing PyEMD: `pip install EMD-signal`
  - Params: `drop_imfs` (e.g., `[0,1]`), `keep_imfs`, `max_imfs` (auto ≈ log2(n), capped [2,10]), `noise_strength` (EEMD/CEEMDAN), `trials`, `random_state`
  - Default behavior: drops the first IMF (highest frequency) and keeps the rest plus the residual trend.
