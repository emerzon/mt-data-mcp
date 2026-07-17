from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from mtdata.services import data_service


def test_trim_df_to_target_uses_utc_epoch_seconds() -> None:
    df = pd.DataFrame({"__epoch": [100.0, 200.0, 250.0, 300.0], "close": [1.0, 2.0, 2.5, 3.0]})
    with patch("mtdata.services.data_service._parse_start_datetime") as mock_parse, patch(
        "mtdata.services.data_service._utc_epoch_seconds"
    ) as mock_epoch:
        mock_parse.side_effect = [datetime(2025, 1, 1, 0, 0), datetime(2025, 1, 1, 1, 0)]
        mock_epoch.side_effect = [150.0, 250.0]
        out = data_service._trim_df_to_target(df, "2025-01-01 00:00", "2025-01-01 01:00", candles=100)

    assert mock_epoch.call_count == 2
    # End bound is inclusive: epochs in [150, 250] are kept.
    assert out["__epoch"].tolist() == [200.0, 250.0]


def test_trim_df_to_target_includes_entire_date_only_end() -> None:
    epochs = [
        datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp(),
        datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp(),
        datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc).timestamp(),
    ]
    df = pd.DataFrame({"__epoch": epochs, "close": [1.0, 2.0, 3.0]})

    out = data_service._trim_df_to_target(
        df,
        "2025-01-01",
        "2025-01-01",
        candles=100,
    )

    assert out["close"].tolist() == [1.0, 2.0]


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


def test_fetch_candles_exposes_time_normalization_metadata() -> None:
    rates = [
        {"time": 1_700_000_000.0, "open": 1.10, "high": 1.12, "low": 1.09, "close": 1.11},
        {"time": 1_700_003_600.0, "open": 1.11, "high": 1.13, "low": 1.10, "close": 1.12},
    ]

    @contextmanager
    def _guard(*args, **kwargs):
        yield None, MagicMock(digits=5)

    def _fake_fetch(*args, diagnostics=None, **kwargs):
        return rates, None

    with patch("mtdata.services.data_service.get_symbol_info_cached", return_value=MagicMock(digits=5)), patch(
        "mtdata.services.data_service._symbol_ready_guard",
        _guard,
    ), patch(
        "mtdata.services.data_service._fetch_rates_with_warmup",
        side_effect=_fake_fetch,
    ), patch(
        "mtdata.services.data_service._resolve_client_tz",
        return_value=None,
    ), patch(
        "mtdata.services.data_service.mt5_config.server_tz_name",
        "Europe/Nicosia",
    ), patch(
        "mtdata.services.data_service.mt5_config.time_offset_minutes",
        0,
    ):
        result = data_service.fetch_candles(
            "EURUSD",
            timeframe="H1",
            limit=2,
            include_incomplete=True,
        )

    assert result["time_basis"] == "utc"
    assert result["raw_time_basis"] == "mt5_utc_epoch"
    assert result["time_normalization"] == "mt5_utc_native"
    assert result["broker_server_tz"] == "Europe/Nicosia"
    assert "request bounds and returned epochs use native UTC" in result["timezone_note"]
    assert (
        result["meta"]["diagnostics"]["time_normalization"]["broker_server_tz"]
        == "Europe/Nicosia"
    )
    assert "timezone_note" in result["meta"]["diagnostics"]["time_normalization"]
