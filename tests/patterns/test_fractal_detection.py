import pandas as pd

from mtdata.patterns.fractal import FractalDetectorConfig, detect_fractal_patterns


def _fractal_breakout_frame(close_breakout: bool = True) -> pd.DataFrame:
    bearish_break_close = 12.8 if close_breakout else 11.8
    return pd.DataFrame(
        {
            "time": [float(i) for i in range(11)],
            "open": [8.4, 9.0, 10.5, 9.2, 7.8, 6.2, 7.0, 8.0, 10.8, 12.2, 4.8],
            "high": [9.0, 10.0, 12.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 13.0, 12.0],
            "low": [8.0, 8.5, 9.0, 8.5, 7.0, 5.0, 6.0, 7.0, 8.0, 9.0, 4.0],
            "close": [8.5, 9.5, 11.0, 9.0, 7.5, 6.0, 7.0, 8.0, 11.5, bearish_break_close, 4.5],
        }
    )


def _fractal_active_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [float(i) for i in range(9)],
            "open": [8.4, 9.0, 10.5, 9.2, 7.8, 6.2, 7.0, 8.0, 8.3],
            "high": [9.0, 10.0, 12.0, 10.0, 9.0, 8.0, 9.0, 10.0, 9.5],
            "low": [8.0, 8.5, 9.0, 8.5, 7.0, 5.0, 6.0, 7.0, 6.4],
            "close": [8.5, 9.5, 11.0, 9.0, 7.5, 6.0, 7.0, 8.0, 8.8],
        }
    )


def test_detect_fractal_patterns_marks_broken_levels_and_breakout_direction():
    results = detect_fractal_patterns(_fractal_breakout_frame(), FractalDetectorConfig())

    by_direction = {result.direction: result for result in results}
    bearish = by_direction["bearish"]
    bullish = by_direction["bullish"]

    assert bearish.name == "Bearish Fractal"
    assert bearish.status == "completed"
    assert bearish.details["breakout_direction"] == "bullish"
    assert bearish.details["level_state"] == "broken"
    assert bearish.end_index == bearish.details["breakout_index"]

    assert bullish.name == "Bullish Fractal"
    assert bullish.status == "completed"
    assert bullish.details["breakout_direction"] == "bearish"
    assert bullish.details["level_state"] == "broken"
    assert bullish.end_index == bullish.details["breakout_index"]


def test_detect_fractal_patterns_keeps_unbroken_levels_forming():
    results = detect_fractal_patterns(_fractal_active_frame(), FractalDetectorConfig())

    by_direction = {result.direction: result for result in results}
    bearish = by_direction["bearish"]
    bullish = by_direction["bullish"]

    assert bearish.status == "forming"
    assert bearish.details["level_state"] == "active"
    assert "breakout_direction" not in bearish.details

    assert bullish.status == "forming"
    assert bullish.details["level_state"] == "active"
    assert "breakout_direction" not in bullish.details


def test_detect_fractal_patterns_supports_high_low_breakout_basis():
    frame = _fractal_breakout_frame(close_breakout=False)

    close_results = detect_fractal_patterns(frame, FractalDetectorConfig())
    high_low_results = detect_fractal_patterns(
        frame,
        FractalDetectorConfig(breakout_basis="high_low"),
    )

    close_bearish = next(result for result in close_results if result.direction == "bearish")
    high_low_bearish = next(
        result for result in high_low_results if result.direction == "bearish"
    )

    assert close_bearish.status == "forming"
    assert high_low_bearish.status == "completed"
    assert high_low_bearish.details["breakout_direction"] == "bullish"
