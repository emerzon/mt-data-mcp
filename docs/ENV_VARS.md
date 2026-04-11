# Environment Variables Reference

Complete reference for all environment variables recognized by mtdata. Set these in a `.env` file in the project root or export them in your shell.

**Related:**
- [SETUP.md](SETUP.md) — Installation and quick-start configuration
- [WEB_API.md](WEB_API.md) — Web API endpoint details
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common issues

---

## MT5 Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `MT5_LOGIN` | — | MetaTrader 5 account number |
| `MT5_PASSWORD` | — | Account password |
| `MT5_SERVER` | — | Broker server name |
| `MT5_TIMEOUT` | `30` | Connection timeout in seconds |

All four are optional if MT5 is already logged in interactively. Set them for unattended / headless use.

```ini
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Demo
MT5_TIMEOUT=30
```

---

## Timezone

MT5 server clocks vary by broker. Configure one of the two methods below so mtdata can normalize timestamps to UTC correctly.

| Variable | Default | Description |
|----------|---------|-------------|
| `MT5_SERVER_TZ` | — | IANA timezone of the MT5 server (e.g. `Europe/Athens`). Handles DST automatically. |
| `MT5_TIME_OFFSET_MINUTES` | `0` | Fixed offset from UTC in minutes (e.g. `120` for UTC+2). Ignored when `MT5_SERVER_TZ` is set. |
| `MT5_CLIENT_TZ` / `CLIENT_TZ` | auto-detect | IANA timezone of the local machine. `CLIENT_TZ` takes precedence if both are set. |

Prefer `MT5_SERVER_TZ` — it adjusts for DST. Use `MT5_TIME_OFFSET_MINUTES` only when you don't know the server's IANA name.

```ini
# Option A — timezone name (recommended)
MT5_SERVER_TZ=Europe/Athens

# Option B — fixed offset
MT5_TIME_OFFSET_MINUTES=120
```

To detect the offset automatically:

```bash
python scripts/detect_mt5_time_offset.py --symbol EURUSD
```

---

## Broker Time Verification

Optional runtime check that compares the MT5 server clock against known market hours.

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_BROKER_TIME_CHECK` | `false` | Enable broker time verification (`1`, `true`, `yes`, or `on`) |
| `MTDATA_BROKER_TIME_CHECK_TTL_SECONDS` | `60` | Cache TTL for the check result (seconds) |

---

## MCP Server

Control how the MCP server binds and exposes endpoints.

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `sse` | Transport mode: `sse`, `stdio`, or `streamable-http` |
| `FASTMCP_HOST` | `127.0.0.1` | Bind address |
| `FASTMCP_PORT` | `8000` | Listen port |
| `FASTMCP_ALLOW_REMOTE` | `false` | Set to `1` to allow non-loopback binds (e.g. `0.0.0.0`) |
| `FASTMCP_LOG_LEVEL` | `INFO` | Logging level |
| `FASTMCP_MOUNT_PATH` | `/` | Base mount path |
| `FASTMCP_SSE_PATH` | `/sse` | SSE event-stream path |
| `FASTMCP_MESSAGE_PATH` | `/message` | Message endpoint path |

---

## Web API

Settings for the FastAPI server that powers the React Web UI.

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBAPI_HOST` | `127.0.0.1` | Bind address |
| `WEBAPI_PORT` | `8000` | Listen port |
| `WEBAPI_ALLOW_REMOTE` | `false` | Set to `1` to allow non-loopback binds |
| `WEBAPI_AUTH_TOKEN` | — | Bearer / API-key token. **Required** when binding to a non-loopback address. |
| `CORS_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | Comma-separated allowed origins. Wildcard `*` is rejected when credentials are enabled. |
| `WEBUI_DIST_DIR` | `webui/dist` | Path to the built Web UI static files |

```ini
# Expose the Web API on the local network with auth
WEBAPI_ALLOW_REMOTE=1
WEBAPI_HOST=0.0.0.0
WEBAPI_PORT=9000
WEBAPI_AUTH_TOKEN=my-secret-token
CORS_ORIGINS=http://192.168.1.10:5173
```

---

## News Embeddings

Configure the HuggingFace model used to rerank MT5 / external news by relevance.

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_NEWS_EMBEDDINGS_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | HuggingFace model name |
| `MTDATA_NEWS_EMBEDDINGS_TOP_N` | `8` | Number of top-ranked items to keep after reranking |
| `MTDATA_NEWS_EMBEDDINGS_WEIGHT` | `1.0` | Weight for embedding-based reranking (0.0 disables) |
| `MTDATA_NEWS_EMBEDDINGS_TRUNCATE_DIM` | — | Truncate embedding vectors to this dimensionality (model must support Matryoshka) |
| `MTDATA_NEWS_EMBEDDINGS_CACHE_SIZE` | `256` | In-memory embedding vector cache size |
| `MTDATA_NEWS_EMBEDDINGS_HF_TOKEN_ENV_VAR` | `HF_TOKEN` | Name of the env var that holds the HuggingFace token |
| `HF_TOKEN` | — | HuggingFace API token for gated / private models |

---

## Finviz

| Variable | Default | Description |
|----------|---------|-------------|
| `FINVIZ_HTTP_TIMEOUT` | `15.0` | HTTP request timeout in seconds |
| `FINVIZ_SCREENER_MAX_ROWS` | `5000` | Maximum rows returned by a single screener request |
| `FINVIZ_PAGE_LIMIT_MAX` | `500` | Maximum pagination page limit |

---

## Forecasting & GPU

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_NF_ACCEL` | auto-detect | Accelerator for NeuralForecast models: `gpu` or `cpu` |
| `CUDA_VISIBLE_DEVICES` | — | Restrict CUDA to specific GPU device(s). Auto-restricted to the first GPU when multiple are detected. |

---

## Market Depth

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_ENABLE_MARKET_DEPTH_FETCH` | `false` | Enable the `market_depth_fetch` tool (`1`, `true`, `yes`, or `on`). Disabled by default because it requires Level 2 data from the broker. |

---

## Trading

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_ORDER_MAGIC` | `234000` | Magic number stamped on all orders placed by mtdata. Change this to distinguish mtdata orders from orders placed by other EAs or scripts on the same account. |

---

## CLI & Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `MTDATA_CLI_DEBUG` | `false` | Enable verbose debug logging in the CLI (`1`, `true`, `yes`, or `on`) |
| `NO_COLOR` | — | Disable ANSI color output (any non-empty value). Follows the [no-color.org](https://no-color.org) convention. |

---

## Quick `.env` Template

A starter template with all sections. Uncomment and fill in what you need.

```ini
# ── MT5 Connection ──────────────────────────────────────
# MT5_LOGIN=12345678
# MT5_PASSWORD=your_password
# MT5_SERVER=YourBroker-Demo
# MT5_TIMEOUT=30

# ── Timezone (pick one) ────────────────────────────────
# MT5_SERVER_TZ=Europe/Athens
# MT5_TIME_OFFSET_MINUTES=120

# ── Broker Time Check ──────────────────────────────────
# MTDATA_BROKER_TIME_CHECK=false
# MTDATA_BROKER_TIME_CHECK_TTL_SECONDS=60

# ── MCP Server ─────────────────────────────────────────
# MCP_TRANSPORT=sse
# FASTMCP_HOST=127.0.0.1
# FASTMCP_PORT=8000
# FASTMCP_ALLOW_REMOTE=0
# FASTMCP_LOG_LEVEL=INFO

# ── Web API ────────────────────────────────────────────
# WEBAPI_HOST=127.0.0.1
# WEBAPI_PORT=8000
# WEBAPI_ALLOW_REMOTE=0
# WEBAPI_AUTH_TOKEN=
# CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173

# ── News Embeddings ────────────────────────────────────
# MTDATA_NEWS_EMBEDDINGS_MODEL=Qwen/Qwen3-Embedding-0.6B
# MTDATA_NEWS_EMBEDDINGS_TOP_N=8
# MTDATA_NEWS_EMBEDDINGS_WEIGHT=1.0
# MTDATA_NEWS_EMBEDDINGS_CACHE_SIZE=256
# HF_TOKEN=

# ── Finviz ─────────────────────────────────────────────
# FINVIZ_HTTP_TIMEOUT=15.0
# FINVIZ_SCREENER_MAX_ROWS=5000

# ── Forecasting / GPU ──────────────────────────────────
# MTDATA_NF_ACCEL=cpu
# CUDA_VISIBLE_DEVICES=0

# ── Market Depth ───────────────────────────────────────
# MTDATA_ENABLE_MARKET_DEPTH_FETCH=0

# ── Trading ────────────────────────────────────────────
# MTDATA_ORDER_MAGIC=234000

# ── CLI / Debug ────────────────────────────────────────
# MTDATA_CLI_DEBUG=0
# NO_COLOR=
```
