# Options and QuantLib

Fetch **US equity options chains** and price **barrier / exotic** structures with [QuantLib](https://www.quantlib.org/). Chains default to Yahoo Finance (Tradier optional). Pure QuantLib pricing does not need a live chain provider.

For *MT5 path* TP/SL hit probabilities on underlyings, see [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — related idea, different stack.

**Dense terms:** [QuantLib](GLOSSARY.md#quantlib) · [Heston](GLOSSARY.md#heston-model) · [Barrier option](GLOSSARY.md#barrier-option) · [Implied vol](GLOSSARY.md#volatility)

**Related:** [Barriers (MT5)](BARRIER_FUNCTIONS.md) · [Finviz](FINVIZ.md) · [CLI](CLI.md) · [Glossary](GLOSSARY.md)

---

> **Dependencies:** Options chain data uses Yahoo Finance by default and may fail with unauthenticated 401/429 responses. When `MTDATA_OPTIONS_PROVIDER=tradier` or `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured, but reliable chain data still requires `MTDATA_OPTIONS_API_KEY`. QuantLib tools require `pip install QuantLib` and are independent of both MT5 and chain-provider access.

---

## Options Data

The options data tools are external-data helpers. Yahoo Finance is the default, but it is an **unauthenticated fallback** that can reject requests (401/429). If you select `tradier` (or `auto` with a Tradier token), mtdata retries Yahoo once when Tradier is unavailable or misconfigured. To use reliable authenticated chains, add these values to `.env`:

```bash
MTDATA_OPTIONS_PROVIDER=tradier
MTDATA_OPTIONS_API_KEY=your_tradier_token
```

Run `options_provider_status` to see the configured vs. effective provider and whether mtdata is using authenticated or best-effort fallback access:

```bash
mtdata-cli options_provider_status --json
```

`options_barrier_price` is a local QuantLib calculator and still works without options-chain provider access when you supply spot, strike, barrier, maturity, and volatility.

### `options_expirations`

List available option expiration dates for a US stock.

```bash
mtdata-cli options_expirations AAPL --json
```

**Returns:** List of expiration dates available for the symbol.

### `options_chain`

Fetch an options chain snapshot with filtering.

```bash
# Full chain (calls + puts)
mtdata-cli options_chain AAPL --json

# Calls only for a specific expiration
mtdata-cli options_chain AAPL --expiration 2026-04-17 --option-type call --json

# Filter by liquidity
mtdata-cli options_chain TSLA --min-open-interest 100 --min-volume 50 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Stock ticker |
| `--expiration` | (nearest) | Specific expiration date `YYYY-MM-DD` |
| `--option-type` | `both` | `call`, `put`, or `both` |
| `--min-open-interest` | 0 | Minimum open interest filter |
| `--min-volume` | 0 | Minimum volume filter |
| `--limit` | 200 | Maximum contracts to return |

---

## QuantLib Barrier Option Pricing

### `options_barrier_price`

Price a barrier option using QuantLib's numerical engine.

By default, QuantLib pricing assumes the `UnitedStates.NYSE` calendar and interprets `maturity_days` as calendar days. Override `--calendar` and `--maturity-basis` for non-US or non-equity workflows.

```bash
# Down-and-out call (knock-out if price falls to barrier)
mtdata-cli options_barrier_price \
  --spot 150 --strike 155 --barrier 140 --maturity-days 30 \
  --option-type call --barrier-type down_out --volatility 0.25 --json

# Up-and-in put (activates if price rises to barrier)
mtdata-cli options_barrier_price \
  --spot 150 --strike 145 --barrier 160 --maturity-days 60 \
  --option-type put --barrier-type up_in --volatility 0.3 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--spot` | (required) | Current spot price |
| `--strike` | (required) | Strike price |
| `--barrier` | (required) | Barrier level |
| `--maturity-days` | (required) | Time to maturity in calendar days |
| `--option-type` | `call` | `call` or `put` |
| `--barrier-type` | `up_out` | `up_in`, `up_out`, `down_in`, `down_out` |
| `--risk-free-rate` | 0.02 | Risk-free rate (decimal) |
| `--dividend-yield` | 0.0 | Dividend yield (decimal) |
| `--volatility` | 0.2 | Implied volatility (decimal, e.g., 0.2 = 20%) |
| `--rebate` | 0.0 | Rebate paid at barrier touch |
| `--calendar` | `UnitedStates.NYSE` | QuantLib calendar name (for example `UnitedStates.NYSE` or `NullCalendar`) |
| `--maturity-basis` | `calendar_days` | Interpret `--maturity-days` as `calendar_days` or `business_days` in the selected calendar |

**Barrier types explained:**

| Type | Meaning |
|------|---------|
| `up_in` | Option activates when price rises through barrier |
| `up_out` | Option deactivates when price rises through barrier |
| `down_in` | Option activates when price falls through barrier |
| `down_out` | Option deactivates when price falls through barrier |

**Returns:** Option price and Greeks (delta, gamma, vega).

---

## Heston Model Calibration

### `options_heston_calibrate`

Calibrate the Heston stochastic volatility model from live options data. The Heston model captures volatility clustering and the volatility smile/skew.

```bash
# Calibrate from call options
mtdata-cli options_heston_calibrate AAPL --option-type call --json

# Calibrate from a specific expiration with liquidity filters
mtdata-cli options_heston_calibrate TSLA \
  --expiration 2026-04-17 --option-type both \
  --min-open-interest 50 --min-volume 10 --max-contracts 30 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Stock ticker |
| `--expiration` | (nearest) | Specific expiration to calibrate against |
| `--option-type` | `call` | `call`, `put`, or `both` |
| `--risk-free-rate` | 0.02 | Risk-free rate (decimal) |
| `--dividend-yield` | 0.0 | Dividend yield (decimal) |
| `--min-open-interest` | 0 | Min open interest for contract selection |
| `--min-volume` | 0 | Min volume for contract selection |
| `--max-contracts` | 25 | Max contracts used in calibration |
| `--calendar` | `UnitedStates.NYSE` | QuantLib calendar name used for maturity assumptions |
| `--maturity-basis` | `calendar_days` | Interpret days-to-expiry as `calendar_days` or `business_days` in the selected calendar |

**Heston parameters returned:**

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `v0` | v₀ | Initial variance |
| `kappa` | κ | Mean reversion speed |
| `theta` | θ | Long-term average variance |
| `sigma` | σ | Volatility of volatility ("vol of vol") |
| `rho` | ρ | Correlation between asset and volatility processes |

**Use cases:**
- More accurate barrier option pricing (using calibrated vol surface instead of flat vol)
- Volatility smile/skew analysis
- Exotic option valuation inputs

---

## Quick Reference

| Task | Command |
|------|---------|
| List expirations | `mtdata-cli options_expirations AAPL` |
| Options chain | `mtdata-cli options_chain AAPL --option-type call` |
| Barrier option price | `mtdata-cli options_barrier_price --spot 150 --strike 155 --barrier 140 --maturity-days 30` |
| Heston calibration | `mtdata-cli options_heston_calibrate AAPL` |

---

## See Also

- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — MT5-based barrier probability analysis
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation methods
- [FINVIZ.md](FINVIZ.md) — Fundamental data
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
