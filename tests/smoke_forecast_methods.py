"""Smoke test for forecast methods on synthetic data.

Runs a small subset of representative forecast methods against a synthetic series.
This is intended to quickly catch import/runtime regressions without needing MT5.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from mtdata.forecast.registry import ForecastRegistry

# Ensure method modules are imported so they self-register with ForecastRegistry.
for _mod in (
    "mtdata.forecast.methods.classical",
    "mtdata.forecast.methods.ets_arima",
    "mtdata.forecast.methods.statsforecast",
    "mtdata.forecast.methods.mlforecast",
    "mtdata.forecast.methods.pretrained",
    "mtdata.forecast.methods.sktime",
    "mtdata.forecast.methods.monte_carlo",
    "mtdata.forecast.methods.analog",
):
    try:
        __import__(_mod)
    except Exception:
        pass


def _run_method(method: str, series: pd.Series, horizon: int, seasonality: int) -> Tuple[bool, str]:
    try:
        params = {}
        if method == "statsforecast":
            params = {"model_name": "Theta"}
        elif method == "sktime":
            params = {"estimator": "sktime.forecasting.theta.ThetaForecaster"}

        res = ForecastRegistry.get(method).forecast(
            series,
            horizon=horizon,
            seasonality=seasonality,
            params=params,
            ci_alpha=0.1,
        )
        first = float(res.forecast[0]) if getattr(res, "forecast", None) is not None and len(res.forecast) else float("nan")
        return True, f"{method}: ok (first={first:.6g})"
    except Exception as ex:
        return False, f"{method}: err ({type(ex).__name__}) {ex}"


def main() -> int:
    # Keep series strictly positive so methods that assume positive data
    # (e.g., some multiplicative seasonal components) don't fail.
    series = pd.Series(2.0 + np.sin(np.arange(800, dtype=float) / 20.0))
    horizon = 12
    seasonality = 24

    methods: List[str] = [
        # Classical / statsmodels
        "theta",
        "arima",
        # StatsForecast generic wrapper
        "statsforecast",
        # sktime generic wrapper
        "sktime",
        # MLForecast
        "mlf_rf",
        # Monte Carlo
        "mc_gbm",
        "hmm_mc",
        # Foundation model (local weights, no HF download required)
        "timesfm",
    ]

    results: Dict[str, bool] = {}
    for m in methods:
        ok, msg = _run_method(m, series, horizon, seasonality)
        results[m] = ok
        print(msg)

    failed = [m for m, ok in results.items() if not ok]
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
