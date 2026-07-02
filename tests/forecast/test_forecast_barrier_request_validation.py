import pytest
from pydantic import ValidationError

from mtdata.forecast.requests import (
    ForecastBarrierOptimizeRequest,
    ForecastBarrierProbRequest,
)


def test_forecast_barrier_prob_request_rejects_multiple_tp_unit_families():
    with pytest.raises(
        ValidationError,
        match="Use one TP/SL barrier unit family",
    ):
        ForecastBarrierProbRequest(symbol="EURUSD", tp_abs=1.11, tp_pct=0.5)


def test_forecast_barrier_prob_request_rejects_multiple_sl_unit_families():
    with pytest.raises(
        ValidationError,
        match="Use one TP/SL barrier unit family",
    ):
        ForecastBarrierProbRequest(symbol="EURUSD", sl_abs=1.09, sl_ticks=15.0)


def test_forecast_barrier_prob_request_rejects_mixed_unit_families():
    with pytest.raises(
        ValidationError,
        match="Use one TP/SL barrier unit family",
    ):
        ForecastBarrierProbRequest(symbol="EURUSD", tp_pct=0.5, sl_ticks=15.0)


def test_forecast_barrier_prob_request_allows_single_shared_unit_family():
    request = ForecastBarrierProbRequest(symbol="EURUSD", tp_pct=0.5, sl_pct=0.25)

    assert request.tp_pct == 0.5
    assert request.sl_pct == 0.25


def test_forecast_barrier_prob_request_uses_tick_fields_as_canonical_names():
    request = ForecastBarrierProbRequest(symbol="EURUSD", tp_ticks=12.0, sl_ticks=9.0)

    assert request.tp_ticks == 12.0
    assert request.sl_ticks == 9.0
    assert "tp_ticks" in request.model_dump()


def test_forecast_barrier_prob_request_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ForecastBarrierProbRequest(symbol="EURUSD", tp_percent=0.5)


def test_forecast_barrier_optimize_request_keeps_ticks_mode_canonical():
    request = ForecastBarrierOptimizeRequest(symbol="EURUSD", mode="ticks")

    assert request.mode == "ticks"
