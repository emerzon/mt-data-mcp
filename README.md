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
- `symbols_list [--search-term TERM] [--limit N]` - Smart search for trading symbols.
- `symbols_list_groups [--search-term TERM] [--limit N]` - Get available symbol groups.
- `symbols_describe <symbol>` - Get detailed symbol information.
- `data_fetch_candles <symbol> [--timeframe TF] [--limit N]` - Get historical OHLCV data.
- `data_fetch_ticks <symbol> [--limit N]` - Get tick data.
- `market_depth_fetch <symbol>` - Get market depth (DOM).
- `indicators_list` - List available technical indicators. (Use `list_capabilities --sections indicators --include-details true` for JSON details.)
  - CSV: grouped rows (category shown once per group):
    category,name
    Momentum,RSI
    ,MACD
    ,Stoch
  - JSON: grouped by category `{ "categories": { "Momentum": ["RSI", ...], ... } }`
- `indicators_describe <name>` - Get detailed information for a specific indicator.
- `denoise_list_methods` - List available denoising methods and their parameters.
- `patterns_detect_candlesticks <symbol> [--timeframe TF] [--limit N]` - Detect candlestick patterns.
- `pivot_compute_points <symbol> [--timeframe TF]` - Compute pivot point levels across all supported methods.
- `forecast_generate <symbol> [--timeframe TF] [--method METHOD] [--horizon N] ...` - Generate price forecasts.
- `forecast_volatility_estimate <symbol> [--timeframe TF] [--horizon N] [--method METHOD] [--proxy PROXY]` - Forecast volatility using direct estimators, GARCH, or general forecasters on a proxy.
- `report_generate <symbol> [--horizon N] [--template basic|advanced|scalping|intraday|swing|position]` - One-stop consolidated report rendered as Markdown (context, pivots, vol, backtest->best forecast, MC barriers; advanced adds regimes, HAR-RV, conformal). Templates infer timeframes.
  - Default horizons per template: scalping=8 bars, intraday=12, swing=24, position=30, basic/advanced=12. Override with `--horizon` or `--params "horizon=..."` if needed.
  
- `list_capabilities [--sections ...] [--include-details true|false]` - Consolidated features: frameworks, forecast/volatility methods, denoise, indicators, dimred, and pattern_search backends.

*(Note: Programmatic access uses function names like `get_symbols`, `get_rates`, etc.)*

### Example Usage

#### Sample Trade Guides

- Beginner flow: docs/SAMPLE-TRADE.md
- Advanced flow (regimes, HAR-RV, conformal, MC barriers, risk controls): docs/SAMPLE-TRADE-ADVANCED.md

#### One-Stop Consolidated Report

```bash
# Basic report (context, pivots, EWMA vol, backtest->best forecast, MC barrier grid)
python cli.py report_generate EURUSD --template basic

# Advanced report (adds regime summaries, HAR-RV vol, conformal intervals)
python cli.py report_generate EURUSD --template advanced

# Style-specific templates
python cli.py report_generate EURUSD --template scalping
python cli.py report_generate EURUSD --template intraday
python cli.py report_generate EURUSD --template swing
python cli.py report_generate EURUSD --template position

# Optional fine-tuning via params (grid and backtest sizing)
python cli.py report_generate EURUSD --horizon 12 --template basic \
  --params "backtest_steps=25 backtest_spacing=10 tp_min=0.2 tp_max=1.0 tp_steps=5 sl_min=0.2 sl_max=1.0 sl_steps=5 top_k=5"
```

#### Fetching Data

```bash
# Get all visible symbols
python cli.py symbols_list

# Smart search for symbols containing "EUR"
python cli.py symbols_list --search-term EUR --limit 20

# Get EUR/USD hourly data
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 100

# Get tick data
python cli.py data_fetch_ticks EURUSD --limit 50
python cli.py data_fetch_ticks EURUSD --start "yesterday 14:00" --end "yesterday 15:00" --limit 200
```

#### Data Simplification

The `data_fetch_candles` and `data_fetch_ticks` commands support a powerful `--simplify` and `--simplify-params` option to downsample or transform data.

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
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 2000 --simplify lttb --simplify-params points=200 --format json

# Use ratio instead of fixed points
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1000 --simplify lttb --simplify-params ratio=0.2 --format json

# RDP with error tolerance
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 1000 --simplify rdp --simplify-params epsilon=0.001 --format json

# ZigZag trend analysis - sensitive (0.2% threshold)
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 --simplify segment --simplify-params "algo=zigzag,threshold_pct=0.2" --format json

# ZigZag swing analysis - moderate (0.6% threshold)
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 --simplify segment --simplify-params "algo=zigzag,threshold_pct=0.6" --format json

# Resample into 6-hour buckets
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 2000 --simplify resample --simplify-params bucket_seconds=21600 --format json

# Envelope encoding with numeric positions
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 --simplify encode --simplify-params "schema=envelope,bits=8" --format json

# Envelope encoding with character output (compact UTF-8)
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 200 --simplify encode --simplify-params "schema=envelope,as_chars=true,bits=4" --format json

# Delta encoding (scaled integer or base-N char deltas)
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 300 --simplify encode --simplify-params "schema=delta,scale=1e-5,as_chars=true,zero_char=." --format json

# Symbolic representation (SAX)
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 --simplify symbolic --simplify-params "schema=sax,paa=24,znorm=true" --format json

# Approximate mode with aggregation
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 2000 --simplify approximate --simplify-params points=150 --format json

# ZigZag on different price columns
python cli.py data_fetch_candles EURUSD --timeframe H1 --limit 500 --simplify segment --simplify-params "algo=zigzag,threshold_pct=0.8,value_col=high" --format json
```

### Forecasting

Generate point forecasts for the next `horizon` bars. The server supports lightweight classical models, Monte Carlo/HMM simulation, and optional integrations with modern forecasting frameworks.

- Discover methods: `python cli.py list_capabilities --sections forecast`
- Run forecast: `python cli.py forecast_generate <symbol> --timeframe <TF> --method <name> --horizon <N> [--params JSON]`
- Rolling backtest: `python cli.py forecast_backtest_run <symbol> --timeframe <TF> --horizon <N> [--steps S --spacing K --methods ...]`
- Volatility forecasts: use `forecast_volatility_estimate` with methods like `ewma`, `parkinson`, `har_rv`, `garch`, or general `arima`/`ets` with a `--proxy` (e.g., `log_r2`).

Classical example
```bash
python cli.py forecast_generate EURUSD --timeframe H1 --method theta --horizon 12 --format json
```

HAR-RV volatility (daily RV from M5 returns)
```bash
python cli.py forecast_volatility_estimate EURUSD --timeframe H1 --horizon 12 \
  --method har_rv --params "rv_timeframe=M5,days=150,window_w=5,window_m=22" --format json
```

Monte Carlo forecasts (distribution + bands)
```bash
# GBM Monte Carlo
python cli.py forecast_generate EURUSD --timeframe H1 --method mc_gbm --horizon 12 --params "n_sims=2000 seed=7" --format json
# Regime-aware HMM Monte Carlo
python cli.py forecast_generate EURUSD --timeframe H1 --method hmm_mc --horizon 12 --params "n_states=3 n_sims=3000 seed=7" --format json
```

Barrier analytics (TP/SL odds from MC paths)
```bash
# Probability of hitting TP before SL within 12 bars (percent barriers)
python cli.py forecast_barrier_hit_probabilities --symbol EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --tp_pct 0.5 --sl_pct 0.3 --params "n_sims=5000 seed=7" --format json

# Optimize TP/SL grid to maximize edge/Kelly/EV (percent mode)
python cli.py forecast_barrier_optimize --symbol EURUSD --timeframe H1 --horizon 12 \
  --method hmm_mc --mode pct --tp_min 0.2 --tp_max 1.0 --tp_steps 5 --sl_min 0.2 --sl_max 1.0 --sl_steps 5 \
  --params "n_sims=5000 seed=7" --format json
```

For advanced usage, pattern-based signals, Monte Carlo/HMM details, and barrier analytics, see `docs/FORECAST.md`.

### Regime & Change-Points

Detect significant structural changes and market regimes to adapt strategies:

```bash
# Bayesian Online Change-Point Detection (BOCPD) on log-returns
python cli.py regime_detect EURUSD --timeframe H1 --limit 1000 --method bocpd --threshold 0.6 --format json

# HMM-lite regime labeling (Gaussian mixture over returns)
python cli.py regime_detect EURUSD --timeframe H1 --limit 1000 --method hmm --params "n_states=3" --format json

# Markov-Switching AR (statsmodels, if available)
python cli.py regime_detect EURUSD --timeframe H1 --limit 1200 --method ms_ar --params "k_regimes=2 order=1" --format json
```

Use `cp_prob` and `change_points` from BOCPD to reset models/sizes after structural breaks; use `state`/`state_probabilities` from HMM/MS-AR to switch playbooks (trend vs. range) and adjust risk.

### Conformal Prediction (Valid Intervals)

Build statistically valid prediction intervals around any base method:

```bash
python cli.py forecast_conformal_intervals EURUSD --timeframe H1 --method theta --horizon 12 \
  --steps 25 --spacing 10 --alpha 0.1 --format json
```

This runs a rolling backtest to calibrate per-step residual quantiles and then returns point forecast + conformal bands.

### Triple-Barrier Labeling

Create labels (+1 TP first, -1 SL first, 0 neither) for supervised learning and signal analysis:

```bash
python cli.py labels_triple_barrier EURUSD --timeframe H1 --limit 1500 --horizon 12 \
  --tp_pct 0.5 --sl_pct 0.3 --label-on high_low --format json
```

Use these labels to train meta-models or to evaluate rule performance across regimes.

### Closed-Form Barrier (GBM)

Compute single-barrier hit probability using GBM formulas (fast sanity check vs. MC):

```bash
python cli.py forecast_barrier_closed_form EURUSD --timeframe H1 --horizon 12 --direction up --barrier 1.1000 --format json
```

#### Dimensionality Reduction (Pattern Search)

- Discover reducers: `python cli.py list_capabilities --sections dimred`
- Use PCA quickly: `--pca-components 8`
- Flexible: `--dimred-method pca|kpca|umap|isomap|diffusion|dreams_cne|dreams_cne_fast --dimred-params "key=value,..."`
- Notes:
  - Prefer reducers with transform support for online queries (e.g., PCA, KPCA, UMAP, parametric DREAMS-CNE).
  - t-SNE/Laplacian embeddings cannot transform new queries; avoid for pattern_search.
  - DREAMS-CNE requires installation from source; `dreams_cne_fast` uses lighter defaults for speed.

#### Optional Framework Integrations

You can extend forecasting with additional frameworks. Install as needed; methods will appear under `forecast_methods` in `list_capabilities` once available.

- StatsForecast (classical, fast): `pip install statsforecast`
  - Methods: `sf_autoarima`, `sf_theta`, `sf_autoets`, `sf_seasonalnaive`
- MLForecast (tree/GBMs over lags): `pip install mlforecast scikit-learn` (and `lightgbm` for LGBM)
  - Methods: `mlf_rf`, `mlf_lightgbm`
- NeuralForecast (deep learning): `pip install neuralforecast[torch]`
  - Methods: `nhits`, `nbeatsx`, `tft`, `patchtst`
- Foundation models (Transformers/Chronos/TimesFM): `pip install transformers torch accelerate` (plus `chronos-forecasting` or `timesfm` if desired)
  - Methods: `chronos_bolt`, `timesfm`, `lag_llama`

See detailed examples and parameters in `docs/FORECAST.md` under Framework Integrations.


### Denoising

Apply smoothing algorithms to data columns.

- **Methods:** `ema`, `sma`, `median`, `lowpass_fft`, `wavelet`, `emd`, `eemd`, `ceemdan`.
- **Discover methods:** `python cli.py denoise_list_methods`

```bash
# Denoise OHLCV with an EMA before simplifying
python cli.py data_fetch_candles EURUSD --timeframe H1 --denoise ema --simplify lttb --simplify-params points=50 --limit 1000 --format json

# Denoise the close and an RSI indicator after they are calculated
python cli.py data_fetch_candles EURUSD --timeframe H1 --indicators "rsi(14)" --denoise ema --denoise-params "columns=close,RSI_14,when=post_ti" --format json
```

## Intelligent Symbol Search

The `symbols_list` command uses a smart 3-tier search strategy:
1.  **Group name match** (e.g., `Majors`, `Forex`)
2.  **Symbol name match** (e.g., `EUR` → EURUSD, JPYEUR, ...)
3.  **Description match** (symbol or group path)

## Date Inputs

The `start` and `end` parameters for `data_fetch_candles` use `dateparser` for flexible date/time parsing.
- **Natural language:** `yesterday 14:00`, `2 days ago`
- **Common formats:** `2025-08-29`, `2025/08/29 14:30`

## Requirements

- Windows OS (MetaTrader5 requirement)
- MetaTrader5 terminal installed and running
- Python 3.8+
- Active internet connection

Optional dependencies (uncomment in `requirements.txt`):
- Forecast frameworks: `statsforecast`, `mlforecast`, `lightgbm`, `neuralforecast`
- Foundation models: `transformers`, `accelerate`, `torch`, `chronos-forecasting`, `timesfm`
- Pattern search backends: `hnswlib`, `tslearn`, `dtaidistance`
- Dimensionality reduction: `scikit-learn`, `umap-learn`, `pydiffmap`, `pykeops`, and DREAMS-CNE from source

## Error Handling

All functions return a JSON object with a `success` flag. If `success` is `false`, an `error` field will contain the error message.

## Security Notes

- For unattended use, store credentials securely in a `.env` file.
- Do not commit `.env` files to version control.
- Use demo accounts for testing.

## License

MIT License





