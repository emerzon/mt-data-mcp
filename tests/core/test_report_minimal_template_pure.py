from unittest.mock import patch


def _make_context_bars() -> list[dict]:
    bars = []
    for index in range(6):
        close = 1.1000 + (index * 0.0010)
        bars.append(
            {
                "time": f"2026-03-29 10:0{index}",
                "open": close - 0.0005,
                "high": close + 0.0008,
                "low": close - 0.0007,
                "close": close,
                "EMA_20": close - 0.0003,
                "EMA_50": close - 0.0005,
                "RSI_14": 55.0 + index,
                "tick_volume": 1000 + index,
            }
        )
    return bars


def test_template_minimal_builds_fast_path_without_basic_template() -> None:
    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        if func_name == "data_fetch_candles":
            return {"bars": _make_context_bars()}
        if func_name == "forecast_generate":
            assert kwargs["method"] == "arima"
            return {
                "forecast_price": [1.1070, 1.1080, 1.1090],
                "lower_price": [1.1040, 1.1050, 1.1060],
                "upper_price": [1.1100, 1.1110, 1.1120],
                "trend": "up",
                "ci_alpha": 0.05,
            }
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal("EURUSD", 12, None, {"methods": ["arima", "theta"]})

    assert report["meta"]["template"] == "minimal"
    assert report["meta"]["fast_path"] is True
    assert "backtest" in report["meta"]["skipped_sections"]
    assert "barriers" in report["meta"]["skipped_sections"]
    assert list(report["sections"].keys()) == ["context", "forecast"]
    assert report["sections"]["forecast"]["method"] == "arima"
    assert report["sections"]["forecast"]["selection_mode"] == "direct"
    assert "skips backtest ranking" in report["sections"]["forecast"]["selection_note"]
    assert "trend_compact" in report["sections"]["context"]


def test_template_minimal_reports_direct_forecast_error() -> None:
    def _fake_get_raw_result(func, *args, **kwargs):
        func_name = getattr(func, "__name__", "")
        if func_name == "data_fetch_candles":
            return {"bars": _make_context_bars()}
        if func_name == "forecast_generate":
            return {"error": "forecast failed"}
        raise AssertionError(f"Unexpected tool call: {func_name}")

    with patch(
        "mtdata.core.report_templates.minimal._get_raw_result",
        side_effect=_fake_get_raw_result,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal("EURUSD", 12, None, None)

    assert report["sections"]["forecast"]["error"] == "forecast failed"
    assert report["sections"]["forecast"]["method"] == "theta"
    assert report["sections"]["forecast"]["selection_mode"] == "direct"
