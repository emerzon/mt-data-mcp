from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.modules.setdefault("MetaTrader5", MagicMock())

from mtdata.services import data_service


def test_trim_df_to_target_uses_utc_epoch_seconds() -> None:
    df = pd.DataFrame({"__epoch": [100.0, 200.0, 300.0], "close": [1.0, 2.0, 3.0]})
    with patch("mtdata.services.data_service._parse_start_datetime") as mock_parse, patch(
        "mtdata.services.data_service._utc_epoch_seconds"
    ) as mock_epoch:
        mock_parse.side_effect = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        mock_epoch.side_effect = [150.0, 250.0]
        out = data_service._trim_df_to_target(df, "2025-01-01 00:00", "2025-01-01 01:00", candles=100)

    assert mock_epoch.call_count == 2
    assert out["__epoch"].tolist() == [200.0]


def test_fetch_rates_with_warmup_uses_utc_epoch_seconds_for_end_ts() -> None:
    rates = [{"time": 1000.0}]
    with patch("mtdata.services.data_service._parse_start_datetime") as mock_parse, patch(
        "mtdata.services.data_service._utc_epoch_seconds", return_value=1000.0
    ) as mock_epoch, patch("mtdata.services.data_service._mt5_copy_rates_range", return_value=rates):
        mock_parse.side_effect = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        out_rates, out_err = data_service._fetch_rates_with_warmup(
            symbol="EURUSD",
            mt5_timeframe=1,
            timeframe="H1",
            candles=10,
            warmup_bars=2,
            start_datetime="2025-01-01 00:00",
            end_datetime="2025-01-01 01:00",
            retry=False,
            sanity_check=True,
        )

    assert out_err is None
    assert out_rates == rates
    assert mock_epoch.called
