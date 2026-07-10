# Timestamp & Timezone Policy

mtdata touches up to **four different clocks**, and mixing them up is the most common source of "my candles look shifted" confusion. This page explains each clock, how mtdata normalizes MT5 timestamps, which timezone appears in output, and how to make it deterministic.

| Clock | Where it comes from |
|-------|---------------------|
| **Broker server time** | The wall-clock the MT5 terminal reports on every candle/tick. Varies by broker (often UTC+2/+3). |
| **UTC** | The canonical internal basis mtdata normalizes to before shaping output. |
| **Client-local time** | The display timezone — your machine's timezone by default. |
| **External provider time** | Finviz, news, and options sources report in their own timezones (usually US market time). |

---

## How MT5 timestamps are normalized

MT5 returns timestamps as epoch seconds expressed in **broker server local wall-clock time**, not UTC. mtdata converts them in two stages:

**Stage 1 — server time → UTC** (controlled by your `.env`):

| Configuration | Behavior |
|---------------|----------|
| `MT5_TIME_OFFSET_MINUTES` set (non-zero) | Subtract the fixed offset. **Overrides** `MT5_SERVER_TZ`. |
| `MT5_SERVER_TZ` set (IANA name) | DST-aware conversion from server time to UTC. **Preferred.** |
| Neither set | ⚠️ Timestamps are left as **raw broker server epoch** — wall-clock values will be wrong. |

> **Always configure one of these.** Without a server timezone or offset, mtdata cannot know the broker's UTC offset, so every displayed time (and any downstream local conversion) is off by the broker's offset.

**Stage 2 — UTC → display timezone** (for the `time` values you see):

1. If `CLIENT_TZ` / `MT5_CLIENT_TZ` is set, display in that timezone.
2. Otherwise, display in the **auto-detected local machine timezone**.
3. If detection is unavailable, fall back to **UTC**.

Because of step 2, **default output is in your local machine timezone, not UTC.** Every payload includes a `timezone` field that states exactly which zone the displayed times are in, and internal processing retains the UTC epoch, so precision is never lost.

```text
MT5 server epoch ──(MT5_SERVER_TZ / MT5_TIME_OFFSET_MINUTES)──▶ UTC ──(CLIENT_TZ / local)──▶ displayed time
                                                                      └── payload "timezone" labels this
```

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `MT5_SERVER_TZ` | — | IANA timezone of the MT5 server (e.g. `Europe/Athens`). DST-aware. **Recommended.** |
| `MT5_TIME_OFFSET_MINUTES` | `0` | Fixed server offset from UTC in minutes. Overrides `MT5_SERVER_TZ` when non-zero. |
| `MT5_CLIENT_TZ` / `CLIENT_TZ` | auto-detect | Display timezone for output. `CLIENT_TZ` wins if both are set. |
| `MTDATA_BROKER_TIME_CHECK` | `false` | Optionally verify the server clock against known market hours. |

See [ENV_VARS.md § Timezone](ENV_VARS.md#timezone) for the full reference.

### Make output deterministic

Local-timezone defaults are convenient interactively but risky for stored/shared data (the same command yields different `time` strings on different machines). For reproducible pipelines, **pin the display timezone to UTC**:

```ini
MT5_SERVER_TZ=Europe/Athens   # normalize broker time correctly
CLIENT_TZ=UTC                 # force UTC display everywhere
```

Then keep the `timezone` field alongside any saved results so the basis is unambiguous.

---

## Timezone in output

- **Candles / ticks** (`data_fetch_candles`, `data_fetch_ticks`): a `time` column plus a top-level `timezone` field (`"UTC"`, a client tz name like `America/New_York`, or `"local"`).
- **`market_snapshot`**: three explicit timestamps —
  - `as_of` — the latest quote time when available,
  - `quote_as_of` — the normalized quote timestamp, stated explicitly,
  - `assembled_at` — when the snapshot payload was built.
- **Forecasts** (`forecast_generate`): forecast time fields are normalized consistently with the candle basis and labeled with the effective timezone.
- **Time-normalization metadata**: request the `metadata` extra to see how timestamps were interpreted:
  ```bash
  mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 5 --extras metadata --json
  ```
  The metadata includes a `time_basis` (`utc_normalized` / `raw_mt5_server_epoch`), a `time_normalization` mode (`dst_aware_server_timezone`, `static_utc_offset`, `live_auto_alignment`, or `unconfigured`), and a human-readable `timezone_note`.

---

## External provider time

External data sources do **not** use MT5 server time and are normalized separately:

- **Finviz** — publish times and the economic/earnings calendar are normalized to sensible absolute times (US market context). Treat these as provider time, independent of your broker.
- **News** (`news`) — relative time filters you pass are parsed against your **client** timezone; ranked items carry their own published timestamps.
- **Options** — expirations and quotes follow the provider's convention (Yahoo/Tradier).

When correlating external events with MT5 candles, compare in **UTC** to avoid cross-source drift.

---

## Verifying the broker offset

If candle times look shifted, verify the broker's offset:

```bash
# Detect the server offset empirically
python scripts/detect_mt5_time_offset.py --symbol EURUSD
```

- Set `MTDATA_BROKER_TIME_CHECK=1` to enable a runtime sanity check that compares the server clock against known market hours (cached for `MTDATA_BROKER_TIME_CHECK_TTL_SECONDS`).
- Then set `MT5_SERVER_TZ` (preferred) or `MT5_TIME_OFFSET_MINUTES` accordingly and re-run your query.

---

## Pitfalls

- **Unset server timezone** → times are raw broker epoch, so everything is off by the broker offset. Fix with `MT5_SERVER_TZ`.
- **Assuming UTC output** → the default is *local machine time*. Check the `timezone` field, or set `CLIENT_TZ=UTC`.
- **Comparing MT5 and provider times directly** → convert both to UTC first.
- **Double-normalizing** → mtdata normalizes MT5 epochs to UTC exactly once at the data layer; don't re-apply an offset downstream.

---

## See Also

- [ENV_VARS.md § Timezone](ENV_VARS.md#timezone) — All timezone variables
- [ENV_VARS.md § Broker Time Verification](ENV_VARS.md#broker-time-verification) — The runtime clock check
- [OUTPUT.md](OUTPUT.md) — Response envelope and the `metadata` extra
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Timezone-related symptoms and fixes
