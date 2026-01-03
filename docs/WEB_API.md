# Web API Reference

The `mtdata` Web API exposes forecasting, analysis, and data fetching capabilities via REST endpoints. This API powers the Web UI and can be used by other applications.

**Base URL:** `http://localhost:8000` (default)

## Authentication

The API currently runs locally and does not enforce authentication. Ensure the server is only accessible from trusted networks (binds to `127.0.0.1` by default).

## Endpoints

### Health / UI

#### `GET /`
Basic health check.

#### `GET /app`
Serves the built Web UI (if `webui/dist/` exists).

### Market Data

#### `GET /api/instruments`
Search for available trading symbols.

- **Query Params:**
  - `search` (string, optional): Search query for symbol name/description.
  - `limit` (int, optional): Max results to return.

#### `GET /api/timeframes`
Get supported timeframes.

#### `GET /api/history`
Fetch OHLCV candles for a symbol.

- **Query Params:**
  - `symbol` (string, required): e.g., "EURUSD".
  - `timeframe` (string): Default "H1".
  - `limit` (int): Number of bars (default 500).
  - `start`, `end` (string, optional): ISO dates or relative strings.
  - `ohlcv` (string): Column selector (default "ohlc").
  - `include_incomplete` (bool): Include the latest forming candle.
  - `denoise_method` (string, optional): Apply denoising (e.g., "ema").
  - `denoise_params` (string, optional): JSON or "k=v" params for denoising.    

#### `GET /api/tick`
Get the latest real-time tick.

- **Query Params:**
  - `symbol` (string, required).

### Analysis

#### `GET /api/pivots`
Calculate pivot points.

- **Query Params:**
  - `symbol` (string, required).
  - `timeframe` (string): Default "D1".
  - `method` (string): "classic", "fibonacci", "woodie", "camarilla", "demark".

#### `GET /api/support-resistance`
Identify support and resistance levels.

- **Query Params:**
  - `symbol` (string, required).
  - `timeframe` (string): Default "H1".
  - `limit` (int): History depth to analyze.
  - `tolerance_pct` (float): Clustering tolerance (0.0015 = 0.15%).
  - `min_touches` (int): Minimum touches per level (default 2).
  - `max_levels` (int): Max levels per side (default 4).

#### `GET /api/denoise/methods`
List available denoising algorithms and their parameters.

#### `GET /api/denoise/wavelets`
List available wavelet families/names (when PyWavelets is installed).

#### `GET /api/dimred/methods`
List available dimensionality reduction methods (PCA, UMAP, etc.).

### Forecasting

#### `GET /api/methods`
List available forecasting models and their requirements.

#### `GET /api/volatility/methods`
List available volatility models and their requirements.

#### `GET /api/sktime/estimators`
List sktime estimators (when sktime is installed).

#### `POST /api/forecast/price`
Generate price forecasts.

**Body (JSON):**
```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "method": "theta",
  "horizon": 12,
  "lookback": null,
  "as_of": null,
  "params": {},
  "ci_alpha": 0.05,
  "quantity": "price",
  "target": "price",
  "denoise": {
    "method": "ema",
    "params": {"alpha": 0.2}
  },
  "features": null,
  "dimred_method": null,
  "dimred_params": null,
  "target_spec": null
}
```

#### `POST /api/forecast/volatility`
Generate volatility forecasts.

**Body (JSON):**
```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "horizon": 1,
  "method": "ewma",
  "params": {"lambda": 0.94}
}
```

#### `POST /api/backtest`
Run a rolling-origin backtest.

**Body (JSON):**
```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "horizon": 12,
  "steps": 20,
  "spacing": 10,
  "methods": ["theta", "naive"],
  "quantity": "price",
  "target": "price",
  "denoise": null,
  "features": null,
  "dimred_method": null,
  "dimred_params": null
}
```

## Running the Server

Start the API server using the Python launcher:

```bash
python webui.py
```

Or directly via Uvicorn (if installed):
```bash
uvicorn src.mtdata.core.web_api:app --host 127.0.0.1 --port 8000
```

## Configuration

Control the server host and port via environment variables:

- `MTDATA_WEBUI_HOST`: Bind address (default `127.0.0.1`).
- `MTDATA_WEBUI_PORT`: Listen port (default `8000`).
- `MTDATA_WEBUI_RELOAD`: Set to `1` to enable auto-reload (dev only).
