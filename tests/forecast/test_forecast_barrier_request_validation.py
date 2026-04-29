import pytest
from pydantic import ValidationError

from mtdata.forecast.requests import ForecastBarrierProbRequest


def test_forecast_barrier_prob_request_rejects_multiple_tp_unit_families():
    with pytest.raises(
        ValidationError,
        match="Provide only one take-profit unit family: tp_abs, tp_pct, tp_ticks",
    ):
        ForecastBarrierProbRequest(symbol="EURUSD", tp_abs=1.11, tp_pct=0.5)


def test_forecast_barrier_prob_request_rejects_multiple_sl_unit_families():
    with pytest.raises(
        ValidationError,
        match="Provide only one stop-loss unit family: sl_abs, sl_pct, sl_ticks",
    ):
        ForecastBarrierProbRequest(symbol="EURUSD", sl_abs=1.09, sl_ticks=15.0)


def test_forecast_barrier_prob_request_allows_one_unit_family_per_side():
    request = ForecastBarrierProbRequest(symbol="EURUSD", tp_pct=0.5, sl_ticks=15.0)

    assert request.tp_pct == 0.5
    assert request.sl_ticks == 15.0


def test_forecast_barrier_prob_request_uses_tick_fields_as_canonical_names():
    request = ForecastBarrierProbRequest(symbol="EURUSD", tp_ticks=12.0, sl_ticks=9.0)

    assert request.tp_ticks == 12.0
    assert request.sl_ticks == 9.0
    assert "tp_ticks" in request.model_dump()
