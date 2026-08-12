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
    StrategyBacktestRequest,
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


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_backtest_requests_reject_invalid_slippage(value) -> None:
    with pytest.raises(ValidationError, match="slippage_bps"):
        ForecastBacktestRequest(symbol="EURUSD", slippage_bps=value)
    with pytest.raises(ValidationError, match="slippage_bps"):
        StrategyBacktestRequest(symbol="EURUSD", slippage_bps=value)
