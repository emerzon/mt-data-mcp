from __future__ import annotations

from mtdata.forecast.requests import ForecastConformalIntervalsRequest
from mtdata.forecast.use_cases import _apply_conformal_intervals_detail


def test_conformal_full_detail_keeps_instrument_identity() -> None:
    request = ForecastConformalIntervalsRequest(
        symbol="EURUSD",
        timeframe="H1",
        detail="full",
    )

    result = _apply_conformal_intervals_detail(
        {
            "success": True,
            "method": "theta",
            "horizon": 12,
            "forecast_time": ["2026-08-20T21:00:00Z"],
            "forecast_price": [1.17],
        },
        request,
    )

    assert result["symbol"] == "EURUSD"
    assert result["timeframe"] == "H1"
    assert result["detail"] == "full"
