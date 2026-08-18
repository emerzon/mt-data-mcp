# Timestamps and timezones

**Audience:** Operator

**Plain English for everyone else:** set `CLIENT_TZ=UTC` in `.env` so saved
results look the same tomorrow. That is enough for the first week. The rest of
this page explains how MetaTrader 5 clocks are normalized when a broker does
not follow the usual UTC contract.

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

Live quote freshness is anchored to the wall clock after quote acquisition.
Broker ticks less than 10 seconds ahead are retained as live but disclose a
floored `data_age_seconds=0`, `timestamp_ahead_of_wall_clock=true`, and the
measured lead in `timestamp_skew_seconds`. A lead of 10 seconds or more is unsafe and sets
`timestamp_in_future=true`. Quote-reading tools reconcile the cached
`symbol_info_tick` snapshot with the latest tick stream before applying this
single policy.

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

Compact candle responses retain the thin public time contract (`time_basis`,
`timestamp_mode=utc`, and `public_timestamp_mode=utc`). Latest-N queries expose `limit_satisfied`; historical
ranges expose `range_complete`, `limit_reached`, and a `query_applied` block
that states whether the limit was anchored at the start or end. An omitted
range limit uses the 100,000-bar range safety cap and is reported as
`default_limit`, not as a user-requested count. Latest-N queries still default
to 20 rows. Timestamp ends use `end_filter=bar_close`; only bars closed by the
requested instant are returned.
Request full detail
to inspect the full normalization contract:

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 5 --detail full --json
```

With no configured broker offset, full payloads report
`raw_time_basis=mt5_utc_epoch`, `raw_timestamp_mode=native_utc`, and
`time_normalization=mt5_utc_native`. A detected server-clock terminal instead
reports `raw_time_basis=mt5_server_clock_epoch`,
`raw_timestamp_mode=server_clock`, and
`time_normalization=server_clock_to_utc`. Public candle payloads use
`timestamp_mode=utc`; compact server-clock payloads retain
`time_normalization=server_clock_to_utc` without exposing the raw mode as the
public timestamp axis.
The public timestamp values are UTC in both cases.

MT5 stamps candles at bar open. Daily, weekly, and monthly candle rows also
include `broker_session_date`; D1 rows include `broker_trading_day`. These
labels use the configured broker timezone and disambiguate sessions whose UTC
open falls on the preceding calendar date.

Latest-N requests exclude a forming candle by default. If that forming candle
starts after a broker session break, the response still retains the observed
discontinuity in `session_gaps` and `gap_after_last_bar`. In that case,
`bar_spacing.status=session_gaps_detected`; `spacing_matches_timeframe` may
remain true because it describes the dominant interval, while
`spacing_complete=false` describes the missing session interval.

For completed-bar analytics, freshness ages are measured from bar close, even
though `data_as_of` and row timestamps remain bar-open anchors. Forecast and
volatility payloads identify this as
`latest_completed_bar_close_age_seconds`. Point-in-time `as_of` cutoffs must
not be in the future; an unfulfillable future cutoff is rejected instead of
falling back to current data.

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
4. Request `--detail full` and inspect `timestamp_mode`,
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
