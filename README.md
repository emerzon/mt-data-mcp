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

1. Install the MetaTrader5 terminal.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

You can configure the server by creating a `.env` file in the project root. This is optional, as the server can connect to an already running and logged-in MetaTrader5 terminal.

To pre-configure credentials, create a `.env` file and add the following:

```
MT5_LOGIN=<your_account_number>
MT5_PASSWORD=<your_password>
MT5_SERVER=<your_broker_server>
```

**Timezone Configuration:**
- If your MT5 server time is not UTC, set `MT5_TIME_OFFSET_MINUTES` in the `.env` file to the offset in minutes (can be negative).
- Example: For UTC+2, use `MT5_TIME_OFFSET_MINUTES=120`. For UTC-4, use `MT5_TIME_OFFSET_MINUTES=-240`.
- All timestamps are normalized to UTC using this offset.

## Usage

### Starting the MCP Server

```bash
python server.py
```

The server exposes tools that can be called programmatically or via the command-line interface (`cli.py`).

### Available Tools (CLI)

The following tools are available via `python cli.py <command>`:

#### Market Data
- `list_symbols [search_term] [--limit N]` - Smart search for trading symbols.
- `list_symbol_groups [search_term] [--limit N]` - Get available symbol groups.
- `describe_symbol <symbol>` - Get detailed symbol information.
- `fetch_candles <symbol> <timeframe> [--limit N]` - Get historical OHLCV data.
- `fetch_ticks <symbol> [--limit N]` - Get tick data.
- `fetch_market_depth <symbol>` - Get market depth (DOM).
- `list_indicators` - List available technical indicators.
- `describe_indicator <name>` - Get detailed information for a specific indicator.
- `list_denoise_methods` - List available denoising methods and their parameters.
- `detect_candlestick_patterns <symbol> <timeframe> [--limit N]` - Detect candlestick patterns.
- `forecast <symbol> <timeframe> ...` - Generate price forecasts.
- `list_forecast_methods` - List available forecasting methods.

*(Note: Programmatic access uses function names like `get_symbols`, `get_rates`, etc.)*

### Example Usage

#### Fetching Data

```bash
# Get all visible symbols
python cli.py list_symbols

# Smart search for symbols containing "EUR"
python cli.py list_symbols EUR --limit 20

# Get EUR/USD hourly data
python cli.py fetch_candles EURUSD --timeframe H1 --limit 100

# Get tick data
python cli.py fetch_ticks EURUSD --limit 50
python cli.py fetch_ticks EURUSD --start "yesterday 14:00" --end "yesterday 15:00" --limit 200
```

#### Data Simplification

The `fetch_candles` and `fetch_ticks` commands support a powerful `--simplify` and `--simplify-params` option to downsample or transform data.

**Modes & Methods:**

**1. SELECT Mode (Default)**
- **`lttb`** (Largest Triangle Three Buckets): General-purpose downsampling preserving chart shape
- **`rdp`** (Ramer-Douglas-Peucker): Error-tolerance based simplification
- **`pla`** (Piecewise Linear Approximation): Linear segment approximation
- **`apca`** (Adaptive Piecewise Constant Approximation): Constant segment approximation

**2. RESAMPLE Mode**
- Buckets data into fixed time intervals
- Parameters: `bucket_seconds` (time window in seconds)

**3. ENCODE Mode**
- **`envelope`**: High/Low bands with Open/Close position encoding
- **`delta`**: Integer or character-encoded deltas of O/H/L/C vs previous close
- Parameters: `bits` (precision), `as_chars` (character output), `alphabet` (custom symbols), `scale` (delta tick size)

**4. SEGMENT Mode (ZigZag)**
- **`zigzag`** (alias: `zz`): Detects price turning points and trend changes
- Parameters: `threshold_pct` (minimum % change), `value_col` (price column to analyze)
- Preserves all original columns at pivot rows and adds: `value`, `direction`, `change_pct`.

**5. SYMBOLIC Mode**
- **`sax`** (Symbolic Aggregate approXimation): Converts time series to symbolic patterns
- Parameters: `paa` (segments), `alphabet` (symbol set), `znorm` (z-normalization)
- Preserves all requested columns via per-segment aggregation (mean for numeric columns). Time is represented by `start_time` and `end_time`.

**6. APPROXIMATE Mode**
- Aggregates data into segments based on selected points

### Simplify Method Comparison

| Method | Best For | Output Size | Key Parameters |
|--------|----------|-------------|-----------------|
| `lttb` | General downsampling | User-controlled | `points`, `ratio` |
| `rdp` | Shape preservation | Varies | `epsilon` (tolerance) |
| `zigzag` | Swing/trend analysis | Varies | `threshold_pct` |
| `sax` | Pattern recognition | Fixed segments | `paa`, `alphabet` |
| `envelope`/`delta` | Compact storage | Fixed | `bits`, `as_chars`, `scale` |
| `resample` | Regular intervals | Time-based | `bucket_seconds` |

Notes:
- Encode preserves all requested columns (e.g., indicators, volumes); it replaces OHLC with envelope positions or deltas.
- When a transform changes columns, the response includes the exact output headers in metadata.

**CLI Examples:**

```bash
# Select 200 points using LTTB algorithm
python cli.py fetch_candles EURUSD --timeframe H1 --limit 2000 --simplify lttb --simplify-params points=200 --format json

# Use ratio instead of fixed points
python cli.py fetch_candles EURUSD --timeframe H1 --limit 1000 --simplify lttb --simplify-params ratio=0.2 --format json

# RDP with error tolerance
python cli.py fetch_candles EURUSD --timeframe H1 --limit 1000 --simplify rdp --simplify-params epsilon=0.001 --format json

# ZigZag trend analysis - sensitive (0.2% threshold)
python cli.py fetch_candles EURUSD --timeframe H1 --limit 500 --simplify segment --simplify-params "algo=zigzag,threshold_pct=0.2" --format json

# ZigZag swing analysis - moderate (0.6% threshold)
python cli.py fetch_candles EURUSD --timeframe H1 --limit 500 --simplify segment --simplify-params "algo=zigzag,threshold_pct=0.6" --format json

# Resample into 6-hour buckets
python cli.py fetch_candles EURUSD --timeframe H1 --limit 2000 --simplify resample --simplify-params bucket_seconds=21600 --format json

# Envelope encoding with numeric positions
python cli.py fetch_candles EURUSD --timeframe H1 --limit 200 --simplify encode --simplify-params "schema=envelope,bits=8" --format json

# Envelope encoding with character output (compact UTF-8)
python cli.py fetch_candles EURUSD --timeframe H1 --limit 200 --simplify encode --simplify-params "schema=envelope,as_chars=true,bits=4" --format json

# Delta encoding (scaled integer or base-N char deltas)
python cli.py fetch_candles EURUSD --timeframe H1 --limit 300 --simplify encode --simplify-params "schema=delta,scale=1e-5,as_chars=true,zero_char=." --format json

# Symbolic representation (SAX)
python cli.py fetch_candles EURUSD --timeframe H1 --limit 500 --simplify symbolic --simplify-params "schema=sax,paa=24,znorm=true" --format json

# Approximate mode with aggregation
python cli.py fetch_candles EURUSD --timeframe H1 --limit 2000 --simplify approximate --simplify-params points=150 --format json

# ZigZag on different price columns
python cli.py fetch_candles EURUSD --timeframe H1 --limit 500 --simplify segment --simplify-params "algo=zigzag,threshold_pct=0.8,value_col=high" --format json
```

### Forecasting

Generate point forecasts for the next `horizon` bars.

- **Methods:** `naive`, `drift`, `seasonal_naive`, `theta`, `fourier_ols`, `ses`, `holt`, `holt_winters_add`, `holt_winters_mul`, `arima`, `sarima`.
- **Discover methods:** `python cli.py list_forecast_methods`

```bash
# Forecast the next 12 hours of EURUSD using the theta model
python cli.py forecast EURUSD --timeframe H1 --method theta --horizon 12 --format json
```

### Denoising

Apply smoothing algorithms to data columns.

- **Methods:** `ema`, `sma`, `median`, `lowpass_fft`, `wavelet`, `emd`, `eemd`, `ceemdan`.
- **Discover methods:** `python cli.py list_denoise_methods`

```bash
# Denoise OHLCV with an EMA before simplifying
python cli.py fetch_candles EURUSD --timeframe H1 --denoise ema --simplify lttb --simplify-params points=50 --limit 1000 --format json

# Denoise the close and an RSI indicator after they are calculated
python cli.py fetch_candles EURUSD --timeframe H1 --indicators "rsi(14)" --denoise ema --denoise-params columns=close,RSI_14 --denoise-params when=post_ti --format json
```

## Intelligent Symbol Search

The `list_symbols` command uses a smart 3-tier search strategy:
1.  **Group name match** (e.g., `Majors`, `Forex`)
2.  **Symbol name match** (e.g., `EUR` → EURUSD, JPYEUR, ...)
3.  **Description match** (symbol or group path)

## Date Inputs

The `start` and `end` parameters for `fetch_candles` use `dateparser` for flexible date/time parsing.
- **Natural language:** `yesterday 14:00`, `2 days ago`
- **Common formats:** `2025-08-29`, `2025/08/29 14:30`

## Requirements

- Windows OS (MetaTrader5 requirement)
- MetaTrader5 terminal installed and running
- Python 3.8+
- Active internet connection

## Error Handling

All functions return a JSON object with a `success` flag. If `success` is `false`, an `error` field will contain the error message.

## Security Notes

- For unattended use, store credentials securely in a `.env` file.
- Do not commit `.env` files to version control.
- Use demo accounts for testing.

## License

MIT License
