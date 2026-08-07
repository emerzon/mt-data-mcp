# Timestamps and timezones

MetaTrader5 documents UTC request datetimes and UTC Unix epochs. Most terminals
follow that contract, but some broker terminals expose Unix-shaped values on
their server-clock axis. When a broker offset is configured, mtdata verifies
that variant from a fresh live tick and normalizes it at the MT5 boundary. See MetaQuotes' [`copy_rates_from`](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py)
and [`copy_ticks_range`](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py)
documentation for the upstream UTC contract.

**Related:** [Setup](SETUP.md) · [Env vars](ENV_VARS.md) · [Output contract](OUTPUT.md)

---

## MT5 timestamp contract

The data path is:

```text
UTC request instant ──▶ MT5 adapter ──▶ terminal clock axis ──▶ UTC epoch
                                                                  │
                                                                  ▼
                                                        client display timezone
```

- Pass timezone-aware UTC datetimes to the MT5 Python API. mtdata converts
  parsed request times to that form.
- Native terminals pass UTC request bounds and returned epochs through.
- When a fresh tick is close to the configured broker offset rather than wall
  UTC, the adapter converts request bounds to the server-clock axis and
  returned `time`/`time_msc` values back to UTC exactly once.
- During a closed market, a configured positive broker offset is also applied
  when the raw tick is implausibly ahead of wall UTC but offset normalization
  places the last tick within the preceding four days. This keeps weekend
  snapshots deterministic without treating an ordinary Friday close as a
  future quote.
- Callers must not apply another broker offset to normalized payloads.
- `CLIENT_TZ` / `MT5_CLIENT_TZ` controls presentation. If neither is set,
  mtdata uses the local machine timezone when it can detect it, otherwise UTC.

Every timestamped payload includes a `timezone` field for displayed values.
Internal filtering and range comparisons stay on the UTC epoch axis.

---

## Broker session configuration

Broker wall-clock configuration is optional and is used only where a market
session, trading day, or calendar boundary needs broker context.

| Variable | Default | Purpose |
|----------|---------|---------|
| `MT5_SERVER_TZ` | — | Broker IANA timezone, such as `Europe/Athens`; required to recognize and normalize server-clock epochs with DST-aware offsets, and used for session/calendar calculations. |
| `MT5_TIME_OFFSET_MINUTES` | `0` | Fixed broker offset from UTC. A non-zero value overrides `MT5_SERVER_TZ`, including server-clock recognition and conversion. |
| `CLIENT_TZ` / `MT5_CLIENT_TZ` | auto-detect | Display timezone; `CLIENT_TZ` wins if both are set. |
| `MTDATA_BROKER_TIME_CHECK` | `false` | Optionally perform additional live tick/bar freshness verification. |

For deterministic stored output, pin the display timezone:

```ini
CLIENT_TZ=UTC
```

Add `MT5_SERVER_TZ` when broker-local session boundaries matter or when a
terminal exposes broker server-clock epochs. Without a broker offset, mtdata
uses the upstream native-UTC contract rather than guessing an offset from a
possibly stale tick:

```ini
MT5_SERVER_TZ=Europe/Athens
```

---

## Time metadata

Compact candle responses retain the thin time contract (`time_basis` and
`timestamp_mode`) plus `limit_satisfied`, so callers can detect a short
response without requesting verbose diagnostics. Request the `metadata` extra
to inspect the full normalization contract:

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 5 --extras metadata --json
```

With no configured broker offset, payloads use the upstream native-UTC
contract and report `raw_time_basis=mt5_utc_epoch`,
`time_basis=utc`, `time_normalization=mt5_utc_native`, and
`timestamp_mode=native_utc`. A detected server-clock terminal instead reports
`raw_time_basis=mt5_server_clock_epoch`,
`time_normalization=server_clock_to_utc`, and `timestamp_mode=server_clock`.
The public timestamp values are UTC in both cases.

---

## External providers

External sources do not use MT5 server time and are normalized separately:

- Finviz publish times and calendars use their provider/US-market context.
- News relative filters use the client timezone; results carry publication times.
- Options expirations and quotes follow the selected provider's convention.

Compare sources using UTC absolute instants, and retain the `timezone` or source
metadata alongside saved results.

---

## Troubleshooting

If candles appear shifted:

1. Inspect the payload's `timezone`; presentation may be client-local.
2. Set `CLIENT_TZ=UTC` and rerun the same absolute range.
3. Confirm the input included an explicit offset or `Z` when it was intended as
   an absolute instant.
4. Request `--extras metadata` and inspect `timestamp_mode`,
   `raw_time_basis`, and `time_normalization`.
5. Configure `MT5_SERVER_TZ` (preferred) or `MT5_TIME_OFFSET_MINUTES` to match
   the broker before relying on a terminal that exposes server-clock epochs.
6. Enable `MTDATA_BROKER_TIME_CHECK=1` for additional live freshness checks.

Do not manually shift public payload epochs. The configured broker offset is
applied inside the adapter only after server-clock mode is detected; applying it
again double-shifts the data.

---

## See also

- [ENV_VARS.md § Timezone](ENV_VARS.md#timezone)
- [OUTPUT.md](OUTPUT.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
