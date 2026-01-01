# `forecast_generate`: Price Forecasts

**Related documentation:**
- [../FORECAST.md](../FORECAST.md) - Forecasting overview (start here)
- [../TECHNICAL_INDICATORS.md](../TECHNICAL_INDICATORS.md) - Indicators as inputs/targets
- [../DENOISING.md](../DENOISING.md) - Denoising as preprocessing
- [../BARRIER_FUNCTIONS.md](../BARRIER_FUNCTIONS.md) - Using forecasts for TP/SL sizing

`forecast_generate` produces a forecast for the next `horizon` bars. In plain terms: “given the recent history, what do the next N candles look like?”

## Quick start

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model theta --format json
```

## Key ideas (non-quant friendly)

- A forecast is an *estimate*, not a promise. Always combine it with risk controls.
- `horizon` is “how far into the future” (in bars).
- `lookback` is “how much past data” the model uses.
- Forecasts are often more useful for *risk sizing* (how big can moves be?) than for pinpointing an exact price.

## Picking a model

To see what’s available on your machine:

```bash
python cli.py forecast_list_library_models native --format json
python cli.py forecast_list_library_models statsforecast --format json
python cli.py forecast_list_library_models sktime --format json
python cli.py forecast_list_library_models pretrained --format json
```

If you want a single consolidated list with availability + parameter docs:

```bash
python cli.py forecast_list_methods --format json
```

## Common parameters

- `--timeframe`: candle timeframe (e.g., `H1`, `M15`, `D1`)
- `--horizon`: number of future bars to predict
- `--lookback`: how many historical bars to use (optional; defaults are chosen for you)
- `--ci-alpha`: request confidence intervals where supported (0.05 = 95%)
- `--denoise` / `--denoise-params`: optional smoothing (see [../DENOISING.md](../DENOISING.md))

## Examples

### Baseline classical forecast (Theta)

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library native --model theta --format json
```

### Monte Carlo simulation forecast (distributional)

Useful when you care about the range of outcomes (risk sizing), not only the point forecast.

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model mc_gbm --model-params "n_sims=2000 seed=7" --format json
```

### StatsForecast (fast classical models)

StatsForecast uses its own model names (class-like names). Example:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library statsforecast --model AutoARIMA --format json
```

Tip: if you prefer method-style names, the “native” library also exposes wrappers like `sf_autoarima`:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library native --model sf_autoarima --format json
```

### MLForecast (tree/GBM over lag features)

You can use native wrappers (`mlf_rf`, `mlf_lightgbm`) or provide a regressor class via `--library mlforecast`.

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 12 \
  --library native --model mlf_lightgbm --format json
```

## Using sktime forecasters (SKTIME adapter)

sktime is a large forecasting library. The adapter lets you run many sktime forecasters through `forecast_generate`.

You can pass:

- An alias name (best-effort match), e.g. `ThetaForecaster`
- Or a dotted class path, e.g. `sktime.forecasting.theta.ThetaForecaster`

Example:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 16 \
  --library sktime --model ThetaForecaster --format json
```

Seasonality (when supported by the estimator) is typically `sp`:

```bash
python cli.py forecast_generate EURUSD --timeframe H1 --horizon 24 \
  --library sktime --model sktime.forecasting.naive.NaiveForecaster \
  --model-params "strategy=last sp=24" --format json
```

Tip: Run `python cli.py forecast_list_library_models sktime --format json` for the latest usage notes.

Notes:

- If `--model` is omitted with `--library sktime`, the adapter defaults to `ThetaForecaster`.
- `--model-params` values are passed through as forecaster constructor kwargs.
- If you provide exogenous features via `--features`, they are forwarded to `fit(X=...)` / `predict(X=...)` when the estimator supports it.
