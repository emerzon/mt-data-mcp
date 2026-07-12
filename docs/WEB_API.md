# Web API Reference

The `mtdata` Web API exposes a focused REST surface for market data, forecasting, analysis, and the bundled Web UI. Use it when you want local HTTP access from dashboards, notebooks, scripts, or another application.

**Base URL:** `http://localhost:8000` (default)

Route versioning:
- Every API route below is available under both `/api/...` and `/api/v1/...`.
- Prefer `/api/v1` for new integrations.
- The examples below use `/api` for brevity.

The Web API is intentionally smaller than the full CLI/MCP tool surface. If an endpoint is not listed here, use `mtdata-cli` or an MCP client for that tool.

## Quick Start

Start the local server:

```bash
mtdata-webapi
```

Check health and fetch a small candle sample:

```bash
curl http://127.0.0.1:8000/api/v1/health
curl "http://127.0.0.1:8000/api/v1/history?symbol=EURUSD&timeframe=H1&limit=50"
```

Open the bundled UI after building `webui/dist/`:

```text
http://127.0.0.1:8000/app
```

## Authentication

By default the API binds to `127.0.0.1` and permits loopback clients without a token.

If you want remote access, set `WEBAPI_ALLOW_REMOTE=1`, use a non-loopback `WEBAPI_HOST`, and provide `WEBAPI_AUTH_TOKEN`. When a token is configured, clients must send either:

- `Authorization: Bearer <token>`
- `X-API-Key: <token>`

Credentialed CORS requests require explicit origins. `CORS_ORIGINS=*` is rejected.

Security checklist for remote access:

- Keep the default local bind (`127.0.0.1`) unless another machine must connect.
- Set `WEBAPI_AUTH_TOKEN` before using `WEBAPI_ALLOW_REMOTE=1`.
- Use explicit `CORS_ORIGINS`; do not rely on browser defaults.
- Treat API access as sensitive because endpoints can expose account, symbol, and market context from the running MT5 terminal.

## Response Style

Responses are JSON. Most endpoints return compact, UI-oriented payloads rather than the full CLI/MCP output contract. For richer historical rows or method diagnostics, prefer the CLI with `--json` and `--extras`.

## Endpoints

### Health / UI

#### `GET /`
Basic health check. Returns JSON, not the SPA.

#### `GET /health`
Same liveness payload as `/`.

#### `GET /ready`
Readiness probe. Returns HTTP 200 when the API can establish an MT5 connection and HTTP 503 when MT5 is unavailable. Also available at `GET /api/ready` and `GET /api/v1/ready`.

#### `GET /app`
Serves the built Web UI (if `webui/dist/` exists).

#### `GET /api/health`
API liveness check. Also available at `GET /api/v1/health`.

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
  - `limit` (int): Number of bars (default 20, matching the data tool default).
  - `start`, `end` (string, optional): ISO dates or relative strings.
  - `ohlcv` (string): Column selector (default "ohlc").
  - `include_spread` (bool): Append the historical candle `spread` field without changing the default row shape.
  - `include_incomplete` (bool): Include the latest forming candle.
  - `timestamp_format` (`epoch` | `iso`): Timestamp encoding for returned rows. Default `epoch`; use `iso` to match the data tool's default.
  - `denoise_method` (string, optional): Apply denoising (e.g., "ema").
  - `denoise_params` (string, optional): JSON or "k=v" params for denoising.    
- **Response Notes:**
  - Both `/api/history` and `/api/v1/history` include modern runtime timezone
    metadata under `meta.runtime.timezone` (`utc`, `server`, and `client` when
    configured). The legacy `used` timezone compatibility field is not emitted.

#### `GET /api/tick`
Get the latest real-time tick.

- **Query Params:**
  - `symbol` (string, required).

### Analysis

#### `GET /api/pivots`
Calculate pivot points.

- **Query Params:**
  - `symbol` (string, required).
  - `timeframe` (string): Default "H1".
  - `method` (string): "classic", "fibonacci", "woodie", "camarilla", "demark".

#### `GET /api/support-resistance`
Identify support and resistance levels, plus Fibonacci retracement/extension levels from the most relevant completed swing.

- **Query Params:**
  - `symbol` (string, required).
  - `timeframe` (string): Default `"H1"`. Pass `auto` to merge levels from `M15`, `H1`, `H4`, and `D1`.
  - `lookback` (int): History depth to analyze (default `200`, matching the support/resistance tool).
  - `tolerance_pct` (float): Clustering tolerance (0.0015 = 0.15%).
  - `min_touches` (int): Minimum touches per level (default 2).
  - `max_levels` (int): Max levels per side (default 4).
  - `max_distance_pct` (float, optional): Percentage distance cap from current price (default `5.0`).
  - `volume_weighting` (`off` | `auto`): Volume weighting mode (default `off`).
  - `reaction_bars` (int): Reaction window used for level qualification (default `6`).
  - `adx_period` (int): ADX period used in scoring (default `14`).
  - `decay_half_life_bars` (int, optional): Half-life for recency decay.
- **Response Notes:**
  - Each level includes a price `zone_low`/`zone_high` envelope rather than only a single line.
  - `status` and `breakout_analysis` expose broken levels and role-reversal confirmations.
  - In `auto` mode, overlapping same-event confirmations across timeframes are deduped instead of fully double-counted.
  - Qualification now uses distinct test `episodes`, while raw `touches` remain available as secondary detail.
  - The response includes both base and effective adaptive settings: `tolerance_pct`/`reaction_bars` are the inputs, while `effective_tolerance_pct`/`effective_reaction_bars` reflect the current ATR regime.
  - A `fibonacci` section exposes retracement levels `23.6%`, `38.2%`, `50%`, `61.8%`, `78.6%` and extensions `127.2%`, `161.8%`, anchored to ATR-filtered historical swings and labeled relative to the latest price.

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
  "library": "native",
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

- `library` supports the same forecast libraries exposed by the forecast tool:
  `native`, `statsforecast`, `sktime`, `mlforecast`, and `pretrained`.

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
  "params": {"lambda_": 0.94},
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
  "extras": null
}
```

Compact response shape is the default. Use `extras` (for example
`["metadata"]` or `"all"`) when you need richer sections such as per-anchor
detail records.

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
