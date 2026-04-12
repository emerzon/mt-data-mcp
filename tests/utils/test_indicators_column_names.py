import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


INDICATORS_PATH = SRC / "mtdata" / "utils" / "indicators.py"
spec = importlib.util.spec_from_file_location("mtdata.utils.indicators", INDICATORS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load indicators module from {INDICATORS_PATH}")
indicators = importlib.util.module_from_spec(spec)
sys.modules["mtdata.utils.indicators"] = indicators
spec.loader.exec_module(indicators)
_apply_ta_indicators = indicators._apply_ta_indicators


def _sample_df(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)


@pytest.mark.parametrize(
    ("ti_spec", "expected_cols"),
    [
        (
            "ema(20),rsi(14),macd(12,26,9)",
            ["EMA_20", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
        ),
        (
            "ema(20.0),rsi(14.0),macd(12.0,26.0,9.0)",
            ["EMA_20", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
        ),
        ("ema21", ["EMA_21"]),
    ],
)
def test_ti_column_names_use_ints_for_integer_like_params(
    ti_spec: str, expected_cols: list[str]
) -> None:
    df = _sample_df()
    _apply_ta_indicators(df, ti_spec)

    created = [c for c in df.columns if c != "close"]
    for col in expected_cols:
        assert col in created

    assert all(".0" not in c for c in created)


def test_apply_ta_indicators_raises_for_missing_required_columns() -> None:
    df = _sample_df()

    with pytest.raises(ValueError, match=r"Indicator 'atr' requires columns: high, low, close"):
        _apply_ta_indicators(df, "atr(14)")


def test_apply_ta_indicators_restores_original_index_on_value_error() -> None:
    df = pd.DataFrame(
        {
            "time": np.arange(1_700_000_000, 1_700_000_010),
            "close": np.linspace(1.0, 2.0, 10),
        }
    )
    original_index = df.index.copy()

    with pytest.raises(ValueError, match=r"Indicator 'atr' requires columns: high, low, close"):
        _apply_ta_indicators(df, "atr(14)")

    assert df.index.equals(original_index)


@pytest.mark.parametrize("volume_col", ["tick_volume", "real_volume"])
def test_apply_ta_indicators_accepts_volume_alias_columns(volume_col: str) -> None:
    df = _sample_df()
    df[volume_col] = np.arange(1, len(df) + 1, dtype=float)

    added = _apply_ta_indicators(df, "obv")

    assert any(str(col).upper().startswith("OBV") for col in added)


def test_apply_ta_indicators_logs_warning_when_all_fallbacks_fail(monkeypatch, caplog) -> None:
    df = _sample_df()

    def _broken_indicator(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(indicators.pta, "ema", _broken_indicator, raising=False)

    with caplog.at_level(logging.WARNING, logger=indicators.logger.name):
        added = _apply_ta_indicators(df, "ema(20)")

    assert added == []
    assert any("Indicator ema failed after all call fallbacks" in record.message for record in caplog.records)
