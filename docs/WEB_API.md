# Web API Reference

The `mtdata` Web API exposes forecasting, analysis, and data fetching capabilities via REST endpoints. This API powers the Web UI and can be used by other applications.

**Base URL:** `http://localhost:8000` (default)

Route versioning:
- Every API route below is available under both `/api/...` and `/api/v1/...`.
- Prefer `/api/v1` for new integrations.
- The examples below use `/api` for brevity.

## Authentication

By default the API binds to `127.0.0.1` and permits loopback clients without a token.

If you want remote access, set `WEBAPI_ALLOW_REMOTE=1`, use a non-loopback `WEBAPI_HOST`, and provide `WEBAPI_AUTH_TOKEN`. When a token is configured, clients must send either:

- `Authorization: Bearer <token>`
- `X-API-Key: <token>`

Credentialed CORS requests require explicit origins. `CORS_ORIGINS=*` is rejected.

## Endpoints

### Health / UI

#### `GET /`
Basic health check. Returns JSON, not the SPA.

#### `GET /health`
Same health payload as `/`.

#### `GET /app`
Serves the built Web UI (if `webui/dist/` exists).

#### `GET /api/health`
API health check. Also available at `GET /api/v1/health`.

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
  - `timeframe` (string): Default `"auto"`. `auto` merges levels from `M15`, `H1`, `H4`, and `D1`.
  - `limit` (int): History depth to analyze.
  - `tolerance_pct` (float): Clustering tolerance (0.0015 = 0.15%).
  - `min_touches` (int): Minimum touches per level (default 2).
  - `max_levels` (int): Max levels per side (default 4).
- **Response Notes:**
  - Each level includes a price `zone_low`/`zone_high` envelope rather than only a single line.
  - `status` and `breakout_analysis` expose broken levels and role-reversal confirmations.
  - In `auto` mode, overlapping same-event confirmations across timeframes are deduped instead of fully double-counted.
  - Qualification now uses distinct test `episodes`, while raw `touches` remain available as secondary detail.
  - The response includes both base and effective adaptive settings: `tolerance_pct`/`reaction_bars` are the inputs, while `effective_tolerance_pct`/`effective_reaction_bars` reflect the current ATR regime.

#### `GET /api/denoise/methods`
List available denoising algorithms and their parameters.

#### `GET /api/denoise/wavelets`
List available wavelet families/names (when PyWavelets is installed).

#### `GET /api/dimred/methods`
List available dimensionality reduction methods (PCA, UMAP, t-SNE, etc.) with parameter suggestions.

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
  "proxy": null,
  "params": {"lambda": 0.94},
  "as_of": null,
  "denoise": null
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
  "params_per_method": null,
  "quantity": "price",
  "denoise": null,
  "params": null,
  "features": null,
  "dimred_method": null,
  "dimred_params": null,
  "slippage_bps": 0.0,
  "trade_threshold": 0.0,
  "detail": "compact"
}
```

## Running the Server

Start the API server using the packaged entry point:

```bash
mtdata-webapi
```

Or directly via Uvicorn (if installed):
```bash
uvicorn mtdata.core.web_api:app --host 127.0.0.1 --port 8000
```

## Configuration

Control the server host and port via environment variables:

- `WEBAPI_HOST`: Bind address (default `127.0.0.1`).
- `WEBAPI_PORT`: Listen port (default `8000`).
- `WEBAPI_ALLOW_REMOTE`: Set to `1` to allow non-loopback binds.
- `WEBAPI_AUTH_TOKEN`: Bearer/API key token required for authenticated API access.
- `CORS_ORIGINS`: Comma-separated list of explicit allowed origins.
- `WEBUI_DIST_DIR`: Override the built SPA directory (default `webui/dist`).

---

## See Also

- [SETUP.md](SETUP.md) — Installation and MCP server configuration
- [CLI.md](CLI.md) — CLI command reference
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues
