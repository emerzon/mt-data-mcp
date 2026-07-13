# Running mtdata as a local service

Keep the MCP server or Web API running in the background on Windows so agents and apps can reach mtdata without a terminal open. For one-off foreground runs, use [SETUP.md](SETUP.md#running-mtdata) instead.

**Related:** [Setup (run modes)](SETUP.md#running-mtdata) · [Env vars](ENV_VARS.md#mcp-server) · [Web API](WEB_API.md) · [Trading safety](TRADING_SAFETY.md)

> **Safety:** An MCP service exposes the full CLI tool surface, including `trade_*` on the MT5 account that is logged in. The focused Web API currently has no trading routes, but it still exposes account-connected market workflows. Bind either service to loopback, require an auth token, and prefer a demo account until you trust the setup. See [Security hardening](#security-hardening).

---

## The MT5 desktop-session caveat (read first)

The `MetaTrader5` Python package attaches to a **running MT5 terminal**, and the terminal is a desktop GUI application. This shapes how you can host mtdata:

- The mtdata service must run **in the same interactive Windows session and user account** as the MT5 terminal it talks to.
- A classic Windows service running in **Session 0** (the isolated service session) generally **cannot see** an MT5 terminal on the user's desktop. Auto-restart is great, but Session 0 isolation breaks the MT5 connection.
- **Recommended:** run mtdata with **Task Scheduler triggered "At log on"** for the account that runs MT5 (shares the desktop session). Use [NSSM](#option-b-nssm-true-windows-service) only when MT5 runs under an always-logged-on/auto-login account and you have verified the connection works from that context.

Keep MT5 set to start on logon and log in automatically (or store credentials via `MT5_LOGIN`/`MT5_PASSWORD`/`MT5_SERVER`) so both come up together.

---

## Choose what to run

| Run this | Transport / interface | Use when |
|----------|----------------------|----------|
| `mtdata-sse` | MCP over SSE (HTTP) | Remote or browser-based MCP clients connect over the network (default MCP mode) |
| `mtdata-streamable-http` | MCP over streamable HTTP | MCP clients that prefer streamable-HTTP |
| `mtdata-stdio` | MCP over stdio | An IDE/desktop client **spawns** the process itself (Claude Desktop, VS Code). **Not** a network service — do not daemonize it; see [SETUP.md](SETUP.md#mcp-server) |
| `mtdata-webapi` | FastAPI REST + optional built Web UI | Apps, scripts, or the React UI call REST endpoints |

`mtdata-stdio` is launched on demand by its client, so the rest of this guide targets the long-lived network services: **`mtdata-sse`**, **`mtdata-streamable-http`**, and **`mtdata-webapi`**.

---

## Step 1 — Configure the environment

Put configuration in the project `.env` (loaded on startup) or set real environment variables. Minimum for an unattended host:

```ini
# MT5 auto-login so the terminal + service can start together
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Demo
MT5_SERVER_TZ=Europe/Athens        # or MT5_TIME_OFFSET_MINUTES=120

# MCP server (SSE / streamable-HTTP)
MCP_TRANSPORT=sse
FASTMCP_HOST=127.0.0.1             # loopback only by default
FASTMCP_PORT=8000
MCP_AUTH_TOKEN=change-me           # required for non-loopback binds; recommended always

# Web API (only if you run mtdata-webapi)
WEBAPI_HOST=127.0.0.1
WEBAPI_PORT=8001                   # use a different port if MCP already uses 8000
WEBAPI_AUTH_TOKEN=change-me
CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

See [ENV_VARS.md](ENV_VARS.md#mcp-server) for every variable and its default. If you run the MCP server **and** the Web API at the same time, give them **different ports** (both default to `8000`).

---

## Step 2 — Smoke-test in the foreground

Confirm it works interactively before turning it into a service:

```powershell
# In your activated environment, with MT5 running and logged in:
mtdata-sse                 # or: mtdata-streamable-http / mtdata-webapi
```

Then, from another terminal:

```powershell
# Web API readiness (returns {"service":"mtdata-webui","status":"ok"})
curl http://127.0.0.1:8001/health

# MCP SSE endpoint is reachable (streams events; Ctrl+C to stop)
curl -N http://127.0.0.1:8000/sse
```

Stop it with `Ctrl+C` once you've confirmed the endpoints respond.

---

## Step 3 — Run it persistently

### Locate your entry-point path

Task Scheduler and NSSM need an absolute command. If you installed into a conda env, the console scripts live in that env's `Scripts` directory:

```powershell
# Resolve the absolute path to the entry point in the active environment
(Get-Command mtdata-sse).Source
# e.g. C:\Users\you\miniconda3\envs\mtdata\Scripts\mtdata-sse.exe
```

You can also invoke through conda without activating first:

```powershell
conda run -n mtdata mtdata-sse
```

### Option A: Task Scheduler (recommended)

Task Scheduler can start mtdata **at logon in your desktop session**, which satisfies the [MT5 desktop-session caveat](#the-mt5-desktop-session-caveat-read-first). Create the task from an elevated PowerShell prompt:

```powershell
$exe = (Get-Command mtdata-sse).Source
$action  = New-ScheduledTaskAction -Execute $exe -WorkingDirectory 'C:\Users\you\Documents\Code\mtdata'
$trigger = New-ScheduledTaskTrigger -AtLogOn
# Restart if it exits, and keep it running:
$settings = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName 'mtdata-sse' -Action $action -Trigger $trigger -Settings $settings `
    -RunLevel Limited -Description 'mtdata MCP SSE server'
```

Notes:
- Use **"Run only when user is logged on"** (the default for an interactive trigger) so the task shares the desktop session with MT5.
- Set `WorkingDirectory` to the repo root so the project `.env` is picked up.
- To start it immediately without logging off: `Start-ScheduledTask -TaskName 'mtdata-sse'`.
- Repeat with `mtdata-webapi` (and a different port) if you also need the REST API.

### Option B: NSSM (true Windows service)

[NSSM](https://nssm.cc/) wraps the process as a real service with automatic restart. Only use this when MT5 runs under an **always-logged-on / auto-login** account — configure the service to run **as that user**, not `LocalSystem`, or it will not see the MT5 terminal.

```powershell
# Install (adjust the path to the resolved entry point)
nssm install mtdata-sse "C:\Users\you\miniconda3\envs\mtdata\Scripts\mtdata-sse.exe"
nssm set mtdata-sse AppDirectory "C:\Users\you\Documents\Code\mtdata"

# CRITICAL: run as the interactive user that owns the MT5 terminal (not LocalSystem)
nssm set mtdata-sse ObjectName ".\your-windows-user" "your-windows-password"

# Log stdout/stderr to files
nssm set mtdata-sse AppStdout "C:\Users\you\Documents\Code\mtdata\logs\mtdata-sse.out.log"
nssm set mtdata-sse AppStderr "C:\Users\you\Documents\Code\mtdata\logs\mtdata-sse.err.log"

nssm start mtdata-sse
```

Remove later with `nssm stop mtdata-sse` then `nssm remove mtdata-sse confirm`.

---

## Health checks

| Service | Check | Healthy response |
|---------|-------|------------------|
| `mtdata-webapi` | `GET http://<host>:<port>/health` | `{"service":"mtdata-webui","status":"ok"}` |
| `mtdata-webapi` | `GET http://<host>:<port>/ready` | `200` when MT5 is reachable; non-`200` otherwise |
| `mtdata-sse` | `GET http://<host>:<port>/sse` | An open, streaming SSE connection |

`/health` and `/ready` are also served under `/api` and `/api/v1`. Use `/ready` (not `/health`) if you want the probe to fail when the MT5 terminal is down. Point your task/service monitor or an external uptime check at these URLs.

---

## Security hardening

- **Stay on loopback** (`127.0.0.1`) unless you truly need remote access. Non-loopback binds require `FASTMCP_ALLOW_REMOTE=1` / `WEBAPI_ALLOW_REMOTE=1` **and** an auth token.
- **Always set a token** — `MCP_AUTH_TOKEN` (SSE/streamable-HTTP) and `WEBAPI_AUTH_TOKEN` (Web API). Clients then send `Authorization: Bearer <token>` or `X-API-Key: <token>`.
- **Pin CORS** — set `CORS_ORIGINS` to explicit origins. A wildcard `*` is rejected when credentials are enabled.
- **Firewall** — if you expose a non-loopback port, restrict it to trusted source IPs.
- **Demo first** — the service can execute live trades. Validate on a demo account before pointing it at anything real.

---

## Logging

- The servers log to stdout/stderr; capture it via the scheduler action, NSSM (`AppStdout`/`AppStderr`), or a redirect wrapper (`mtdata-sse *> logs\mtdata-sse.log`).
- Control MCP verbosity with `FASTMCP_LOG_LEVEL` (e.g. `DEBUG`, `INFO`).
- Background training/task state persists under `~/.mtdata/` (see [ENV_VARS.md](ENV_VARS.md#async-training--model-store)) and survives restarts.

---

## Updating and restarting

1. Stop the task/service (`Stop-ScheduledTask` or `nssm stop`).
2. Pull changes and reinstall if dependencies changed (`pip install -e .` or `pip install -r requirements.txt`).
3. Start it again. mtdata loads tools and settings fresh on each start.

---

## Common pitfalls

- **Port already in use** — the MCP server and Web API both default to `8000`. Give each its own port.
- **Service can't reach MT5** — almost always Session 0 isolation (Option B run as `LocalSystem`) or MT5 not logged in. Prefer [Option A](#option-a-task-scheduler-recommended), or run NSSM as the interactive user.
- **Entry point not found** — the scheduler/service runs without your shell's `PATH`; use the absolute `Scripts\*.exe` path or `conda run -n <env> ...`.
- **`.env` ignored** — set the working directory to the repo root, or use real environment variables in the task/service definition.

---

## See Also

- [SETUP.md](SETUP.md#running-mtdata) — Interactive/foreground runs and MCP client (stdio) config
- [ENV_VARS.md](ENV_VARS.md) — Full environment-variable reference
- [WEB_API.md](WEB_API.md) — REST endpoints, authentication, and response style
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Connection and startup issues
