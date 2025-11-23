Sktime Adapter
==============

The forecasting stack includes a generic adapter for sktime forecasters, exposed as the `sktime` method.

Requirements
- `sktime` (installed via `pip install -r requirements.txt`)

Usage
- Method name: `sktime`
- Params:
  - `estimator` (str): fully qualified class path, e.g., `sktime.forecasting.theta.ThetaForecaster`
  - `estimator_params` (object): constructor kwargs
  - `seasonality` (int, optional): injected as `sp` if supported by the estimator

Examples
```bash
# ThetaForecaster via sktime
python cli.py forecast_generate EURUSD --timeframe H1 --method sktime --horizon 16 \
  --params '{"estimator":"sktime.forecasting.theta.ThetaForecaster"}'

# NaiveForecaster (last) with seasonal period 24
python cli.py forecast_generate EURUSD --timeframe H1 --method sktime --horizon 24 \
  --params '{"estimator":"sktime.forecasting.naive.NaiveForecaster","estimator_params":{"strategy":"last"},"seasonality":24}'

# AutoETS with additive errors
python cli.py forecast_generate EURUSD --timeframe H1 --method sktime --horizon 24 \
  --params '{"estimator":"sktime.forecasting.ets.AutoETS","estimator_params":{"error":"add"}}'
```

Notes
- If `estimator` is omitted, the adapter defaults to `NaiveForecaster(strategy="last")`.
- When `seasonality` is provided and the estimator accepts `sp`, the adapter injects it automatically.
- If you pass exogenous features via the existing `features` mechanism, they are forwarded to `fit(X=...)` and `predict(X=...)` when available.

