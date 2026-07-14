# Timestamps and timezones

MetaTrader5 Python requests use UTC datetimes and MT5 returns Unix epochs in
UTC. mtdata preserves those absolute instants. Broker timezone settings affect
session/calendar interpretation only; they never shift request bounds or data
epochs. See MetaQuotes' [`copy_rates_from`](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrom_py)
and [`copy_ticks_range`](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py)
documentation for the upstream UTC contract.

**Related:** [Setup](SETUP.md) · [Env vars](ENV_VARS.md) · [Output contract](OUTPUT.md)

---

## MT5 timestamp contract

The data path is:

```text
UTC request instant ──▶ MetaTrader5 ──▶ UTC epoch ──▶ client display timezone
                                                    └── payload "timezone" labels this
```

- Pass timezone-aware UTC datetimes to the MT5 Python API. mtdata converts
  parsed request times to that form.
- Treat returned `time`, `time_msc`, order, position, and deal epochs as UTC.
- Do not add or subtract the broker's wall-clock offset from an epoch.
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
| `MT5_SERVER_TZ` | — | Broker IANA timezone, such as `Europe/Athens`; preferred for DST-aware session/calendar calculations. |
| `MT5_TIME_OFFSET_MINUTES` | `0` | Fixed broker-session offset from UTC. A non-zero value overrides `MT5_SERVER_TZ`. |
| `CLIENT_TZ` / `MT5_CLIENT_TZ` | auto-detect | Display timezone; `CLIENT_TZ` wins if both are set. |
| `MTDATA_BROKER_TIME_CHECK` | `false` | Optionally check live MT5 tick/bar freshness and reject implausible future timestamps. |

For deterministic stored output, pin the display timezone:

```ini
CLIENT_TZ=UTC
```

Add `MT5_SERVER_TZ` only when broker-local session boundaries matter:

```ini
MT5_SERVER_TZ=Europe/Athens
```

---

## Time metadata

Request the `metadata` extra to inspect the contract:

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 5 --extras metadata --json
```

MT5 payloads report `raw_time_basis=mt5_utc_epoch`, `time_basis=utc`, and
`time_normalization=mt5_utc_native`. When broker session configuration exists,
the metadata also identifies it without implying that epochs were shifted.

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
4. Enable `MTDATA_BROKER_TIME_CHECK=1` if you need live tick/bar freshness
   diagnostics.

Changing `MT5_SERVER_TZ` is not a timestamp correction. Applying a broker offset
to an MT5 epoch double-shifts the data.

---

## See also

- [ENV_VARS.md § Timezone](ENV_VARS.md#timezone)
- [OUTPUT.md](OUTPUT.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
