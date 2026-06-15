import pytest
from pydantic import ValidationError

from mtdata.forecast.requests import (
    ForecastBacktestRequest,
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
    ForecastConformalIntervalsRequest,
    ForecastGenerateRequest,
    ForecastOptimizeHintsRequest,
    ForecastTuneGeneticRequest,
    ForecastTuneOptunaRequest,
    ForecastVolatilityEstimateRequest,
)


@pytest.mark.parametrize(
    "model",
    [
        ForecastGenerateRequest,
        ForecastBacktestRequest,
        ForecastConformalIntervalsRequest,
        ForecastTuneGeneticRequest,
        ForecastTuneOptunaRequest,
        ForecastBarrierProbRequest,
        ForecastOptimizeHintsRequest,
        ForecastBarrierOptimizeRequest,
        ForecastVolatilityEstimateRequest,
    ],
)
def test_forecast_requests_reject_extreme_horizons(model) -> None:
    with pytest.raises(ValidationError):
        model(symbol="EURUSD", horizon=501)


@pytest.mark.parametrize(
    "model",
    [
        ForecastBacktestRequest,
        ForecastConformalIntervalsRequest,
        ForecastTuneGeneticRequest,
        ForecastTuneOptunaRequest,
        ForecastOptimizeHintsRequest,
    ],
)
def test_forecast_requests_reject_extreme_backtest_windows(model) -> None:
    with pytest.raises(ValidationError):
        model(symbol="EURUSD", steps=201)
    with pytest.raises(ValidationError):
        model(symbol="EURUSD", spacing=10_001)
