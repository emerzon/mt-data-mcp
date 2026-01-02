import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
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
