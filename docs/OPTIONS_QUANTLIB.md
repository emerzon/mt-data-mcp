# Options & QuantLib Tools

mtdata includes tools for fetching US equity options data and pricing exotic options using [QuantLib](https://www.quantlib.org/).

**Related:**
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — Barrier probability analysis (MT5-based)
- [FINVIZ.md](FINVIZ.md) — Fundamental data for equities
- [CLI.md](CLI.md) — Command usage and output formats

---

> **Dependencies:** Options chain data requires the options service (internet access). QuantLib tools require `pip install QuantLib`. These features are independent of MT5.

---

## Options Data

### `forecast_options_expirations`

List available option expiration dates for a US stock.

```bash
mtdata-cli forecast_options_expirations AAPL --json
```

**Returns:** List of expiration dates available for the symbol.

### `forecast_options_chain`

Fetch an options chain snapshot with filtering.

```bash
# Full chain (calls + puts)
mtdata-cli forecast_options_chain AAPL --json

# Calls only for a specific expiration
mtdata-cli forecast_options_chain AAPL --expiration 2026-04-17 --option-type call --json

# Filter by liquidity
mtdata-cli forecast_options_chain TSLA --min-open-interest 100 --min-volume 50 --json
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

### `forecast_quantlib_barrier_price`

Price a barrier option using QuantLib's numerical engine.

```bash
# Down-and-out call (knock-out if price falls to barrier)
mtdata-cli forecast_quantlib_barrier_price \
  --spot 150 --strike 155 --barrier 140 --maturity-days 30 \
  --option-type call --barrier-type down_out --volatility 0.25 --json

# Up-and-in put (activates if price rises to barrier)
mtdata-cli forecast_quantlib_barrier_price \
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

**Barrier types explained:**

| Type | Meaning |
|------|---------|
| `up_in` | Option activates when price rises through barrier |
| `up_out` | Option deactivates when price rises through barrier |
| `down_in` | Option activates when price falls through barrier |
| `down_out` | Option deactivates when price falls through barrier |

**Returns:** Option price and Greeks (delta, gamma, theta, vega, rho).

---

## Heston Model Calibration

### `forecast_quantlib_heston_calibrate`

Calibrate the Heston stochastic volatility model from live options data. The Heston model captures volatility clustering and the volatility smile/skew.

```bash
# Calibrate from call options
mtdata-cli forecast_quantlib_heston_calibrate AAPL --option-type call --json

# Calibrate from a specific expiration with liquidity filters
mtdata-cli forecast_quantlib_heston_calibrate TSLA \
  --expiration 2026-04-17 --option-type both \
  --min-open-interest 50 --min-volume 10 --max-contracts 30 --json
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `symbol` | (required) | Stock ticker |
| `--expiration` | (latest) | Specific expiration to calibrate against |
| `--option-type` | `call` | `call`, `put`, or `both` |
| `--risk-free-rate` | 0.02 | Risk-free rate (decimal) |
| `--dividend-yield` | 0.0 | Dividend yield (decimal) |
| `--min-open-interest` | 0 | Min open interest for contract selection |
| `--min-volume` | 0 | Min volume for contract selection |
| `--max-contracts` | 25 | Max contracts used in calibration |

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
| List expirations | `mtdata-cli forecast_options_expirations AAPL` |
| Options chain | `mtdata-cli forecast_options_chain AAPL --option-type call` |
| Barrier option price | `mtdata-cli forecast_quantlib_barrier_price --spot 150 --strike 155 --barrier 140 --maturity-days 30` |
| Heston calibration | `mtdata-cli forecast_quantlib_heston_calibrate AAPL` |

---

## See Also

- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) — MT5-based barrier probability analysis
- [forecast/VOLATILITY.md](forecast/VOLATILITY.md) — Volatility estimation methods
- [FINVIZ.md](FINVIZ.md) — Fundamental data
- [GLOSSARY.md](GLOSSARY.md) — Term definitions
