# Web API

Local HTTP access to mtdata for dashboards, notebooks, scripts, and apps — and the **bundled chart workspace** served at `/app`. Peer delivery surface to CLI and MCP (same research stack; **smaller HTTP surface** than full CLI/MCP — if a tool is missing here, use those instead).

**Base URL:** `http://localhost:8000` (default)

| Versioning | Guidance |
|------------|----------|
| `/api/...` and `/api/v1/...` | Both work for every route below |
| New integrations | Prefer `/api/v1` |
| Examples on this page | Use `/api` for brevity |

**Related:** [Setup](SETUP.md) · [Deployment](DEPLOYMENT.md) · [Env vars](ENV_VARS.md) · [Output contract](OUTPUT.md)

## Quick start

### Start API → open UI

Build the production SPA once with Node.js 22.12 or newer (Node is only required for this step, not at runtime):

```bash
cd webui
npm install
npm run build
cd ..
mtdata-webapi
```

Then open the chart workspace:

```text
http://127.0.0.1:8000/app/
```

The Python package does not ship generated `webui/dist/` assets. Without a build, REST stays available and `/app` returns a deliberate enablement page (HTML or JSON) with the same commands — not a silent skip or bare framework 404. Override the dist path with `WEBUI_DIST_DIR` if needed.

### API smoke checks

```bash
curl http://127.0.0.1:8000/api/v1/health
curl "http://127.0.0.1:8000/api/v1/history?symbol=EURUSD&timeframe=H1&limit=50"
```

## Authentication

By default the API binds to `127.0.0.1` and permits loopback clients without a token.

If you want remote access, set `WEBAPI_ALLOW_REMOTE=1`, use a non-loopback `WEBAPI_HOST`, and provide `WEBAPI_AUTH_TOKEN`. When a token is configured, clients must send either:

- `Authorization: Bearer <token>`
- `X-API-Key: <token>`

The bundled Web UI has an **Auth** control in the chart toolbar. Enter the
same token there after the page loads. The token is held only in the current
tab's JavaScript memory, is attached as a Bearer token to API requests, and is
cleared by a page reload or the control's **Clear** action. It is never embedded
in the Vite build or written to browser storage.

Credentialed CORS requests require explicit origins. `CORS_ORIGINS=*` is rejected.

Security checklist for remote access:

- Keep the default local bind (`127.0.0.1`) unless another machine must connect.
- Set `WEBAPI_AUTH_TOKEN` before using `WEBAPI_ALLOW_REMOTE=1`.
- Use explicit `CORS_ORIGINS`; do not rely on browser defaults.
- Treat API access as sensitive because endpoints can expose account, symbol, and market context from the running MT5 terminal.

## Response Style

Responses are JSON. Most endpoints return compact, UI-oriented payloads rather than the full CLI/MCP output contract. Use `detail=full` for richer historical rows or method diagnostics.

Every response includes `X-Request-ID`. Clients may supply a log-safe identifier
in the same request header (1–128 letters, digits, `.`, `_`, `:`, or `-`); the
server otherwise generates one. Error envelopes and request-scoped operation
logs use that same identifier so a failed HTTP call can be traced end to end.

## Endpoints

### Health / UI

#### `GET /`
Basic health check. Returns JSON, not the SPA.

#### `GET /health`
Same liveness payload as `/`.

#### `GET /ready`
Readiness probe. Returns HTTP 200 when the API can establish an MT5 connection and HTTP 503 when MT5 is unavailable. Also available at `GET /api/ready` and `GET /api/v1/ready`.

#### `GET /app` · `GET /app/`
Serves the production chart workspace when `webui/dist/index.html` exists (or `WEBUI_DIST_DIR`). Asset URLs use the `/app/` base. If the dist is missing, responds with HTTP 503 and a professional enablement page (HTML by default; JSON when `Accept: application/json`) describing `npm run build` and restart.

#### `GET /api/health`
API liveness check. Also available at `GET /api/v1/health`.

### Market Data

#### `GET /api/instruments`
Search for available trading symbols.

- **Query Params:**
  - `search` (string, optional): Search query for symbol name/description.
  - `limit` (int, optional): Max results to return.
- **Response:** Items use `symbol`, `group`, and `description`; pass `symbol`
  directly to history, tick, and analysis routes.

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
  - `timestamp_format` (`epoch` | `iso`): Timestamp encoding for returned rows. Default `iso`; explicit epoch responses identify the unit as `unix_seconds_utc`.
  - `detail` (`compact` | `standard` | `summary` | `full`): Use `full` for diagnostics and runtime metadata.
  - `denoise_method` (string, optional): Apply denoising (e.g., "ema").
  - `denoise_params` (string, optional): JSON or comma-separated `k=v` denoising settings. Both forms accept `when`, `causality`, `keep_original`, and `columns`; other keys are method parameters. Use JSON for multiple columns.
- **Response Notes:**
  - Compact responses expose `server_utc_offset_seconds` when available.
    `detail=full` includes the full runtime timezone tree under
    `meta.runtime.timezone`. The legacy `used` compatibility field is not emitted.

#### `GET /api/tick`
Get the latest quote using the same compact schema as `market_ticker`, including
mid/spread, ISO and epoch timestamps, freshness, and live-usability metadata.
Unavailable FX `last` and volume values are omitted rather than represented as zero.

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
  - `detail` (`compact` | `standard` | `summary` | `full`): Response detail level.
- **Response Notes:**
  - The default response is compact: it returns actionable support/resistance lists and omits heavier diagnostics.
  - Pass `detail=full` for the rich shape described below.
  - Rich level rows include a price `zone_low`/`zone_high` envelope rather than only a single line.
  - Rich output includes `status` and `breakout_analysis` for broken levels and role-reversal confirmations.
  - In `auto` mode, overlapping same-event confirmations across timeframes are deduped instead of fully double-counted.
  - Qualification now uses distinct test `episodes`, while raw `touches` remain available as secondary detail.
  - Rich output includes both base and effective adaptive settings: `tolerance_pct`/`reaction_bars` are the inputs, while `effective_tolerance_pct`/`effective_reaction_bars` reflect the current ATR regime.
  - Rich output includes a `fibonacci` section with retracement levels `23.6%`, `38.2%`, `50%`, `61.8%`, `78.6%` and extensions `127.2%`, `161.8%`, anchored to ATR-filtered historical swings and labeled relative to the latest price.

#### `GET /api/denoise/methods`
List available denoising algorithms and their parameters.

#### `GET /api/denoise/wavelets`
List available wavelet families/names (when PyWavelets is installed).

#### `GET /api/dimred/methods`
List available dimensionality reduction methods (PCA, UMAP, t-SNE, etc.) with parameter suggestions.

### Forecasting

#### `GET /api/methods`
List available forecasting models and their requirements.

#### `GET /api/models`
List trained model artifacts currently available in the model store.

- **Query Params:** `method` (optional method-name filter), `detail` (response
  detail level; default `compact`)
- **Default response:** compact model rows plus `count`, `detail`, and
  `success`; request `detail=full` for storage paths, timestamps, TTL, and
  artifact-size diagnostics.

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
  "ci_alpha": 0,
  "quantity": "price",
  "denoise": {
    "method": "ema",
    "params": {"alpha": 0.2}
  },
  "features": null,
  "dimred": null,
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

#### `GET /api/tools`
List registered MCP tools for the Web UI runner (bootstraps the full tool surface).

- **Query Params:** `category`, `search`, `detail` (`compact`|`standard`|`full`), `include_fields` (bool)
- **Response:** tools with `surface` (`dedicated_ui`|`generic_runner`|`intentional_omit`) and `safety` metadata (confirm flags, warnings)

#### `GET /api/tools/{tool_name}`
Return one tool with parameter field descriptors for the form runner.

#### `POST /api/tools/{tool_name}/invoke`
Invoke a registered tool.

```json
{
  "arguments": {
    "symbol": "EURUSD",
    "timeframe": "H1",
    "detail": "full",
    "fields": "symbol,summary"
  },
  "confirm": false
}
```

Generic invocation uses the shared structured-output contract. Output is compact
by default; `detail=full` requests richer sections and adds related-tool
suggestions when the tool defines them. `fields` accepts comma-separated names
or dotted paths and keeps the standard envelope fields alongside each match.

`forecast_tune_genetic` and `forecast_tune_optuna` are cataloged as
`intentional_omit`: tuning can run longer than an HTTP request and the generic
runner has no progress or cancellation contract. Run those tools through CLI or
MCP instead.

Live trade mutations (`trade_place`, `trade_modify`, `trade_close`) and destructive model/task tools require `"confirm": true`. See [WEBUI_TOOL_COVERAGE.md](WEBUI_TOOL_COVERAGE.md).

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
  "dimred": null,
  "slippage_bps": 0.0,
  "trade_threshold": 0.0,
  "detail": "compact"
}
```

Compact response shape is the default. Use `detail=full` for
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

## See also

- [SETUP.md](SETUP.md) — Install and run modes
- [DEPLOYMENT.md](DEPLOYMENT.md) — Long-lived local service
- [CLI.md](CLI.md) — Full tool surface via CLI
- [OUTPUT.md](OUTPUT.md) — Shared payload contract
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues
