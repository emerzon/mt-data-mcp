Sktime Adapter
==============

The forecasting stack includes a generic adapter for sktime forecasters, exposed via the `sktime` library.

**Related Documentation:**
- [FORECAST.md](FORECAST.md) - General forecasting guide and framework integrations
- [BARRIER_FUNCTIONS.md](BARRIER_FUNCTIONS.md) - Barrier analytics methods
- [EXAMPLE.md](EXAMPLE.md) - End-to-end workflow examples

**Overview:**
This adapter allows you to use any sktime forecaster within the mtdata forecasting pipeline. See [FORECAST.md](FORECAST.md) for general forecasting concepts and other framework integrations.

Requirements
- `sktime` (installed via `pip install -r requirements.txt`)

Usage
- Library: `sktime`
- Model: alias (e.g., `ThetaForecaster`) or dotted class path
- Model params:
  - `estimator_params` (object): constructor kwargs
  - `seasonality` (int, optional): injected as `sp` if supported by the estimator

Examples
```bash
# ThetaForecaster via sktime
python cli.py forecast_generate EURUSD --timeframe H1 --library sktime --model ThetaForecaster --horizon 16

# NaiveForecaster (last) with seasonal period 24
python cli.py forecast_generate EURUSD --timeframe H1 --library sktime --model sktime.forecasting.naive.NaiveForecaster --horizon 24 \
  --model-params '{"estimator_params":{"strategy":"last"},"seasonality":24}'

# AutoETS with additive errors
python cli.py forecast_generate EURUSD --timeframe H1 --library sktime --model sktime.forecasting.ets.AutoETS --horizon 24 \
  --model-params '{"estimator_params":{"error":"add"}}'
```

Notes
- If `model` is omitted, the adapter defaults to `ThetaForecaster`.
- When `seasonality` is provided and the estimator accepts `sp`, the adapter injects it automatically.
- If you pass exogenous features via the existing `features` mechanism, they are forwarded to `fit(X=...)` and `predict(X=...)` when available.
