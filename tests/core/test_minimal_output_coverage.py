"""Tests for utils/minimal_output.py — TOON formatting helpers."""
import pytest

from mtdata.utils.minimal_output import (
    _build_forecast_meta,
    _compact_forecast_ci,
    _encode_expanded_array,
    _encode_inline_array,
    _format_complex_value,
    _format_to_toon,
    _is_empty_value,
    _is_scalar_value,
    _normalize_forecast_payload,
    _normalize_market_status_payload,
    _normalize_market_ticker_payload,
    _normalize_trade_payload,
    _normalize_trade_table_payload,
    _normalize_triple_barrier_payload,
    _stringify_cell,
    _stringify_scalar,
    format_result_minimal,
    format_table_toon,
)


class TestIsScalarValue:
    def test_string(self):
        assert _is_scalar_value("hello")

    def test_int(self):
        assert _is_scalar_value(42)

    def test_float(self):
        assert _is_scalar_value(3.14)

    def test_bool(self):
        assert _is_scalar_value(True)

    def test_none(self):
        assert _is_scalar_value(None)

    def test_list_not_scalar(self):
        assert not _is_scalar_value([1, 2])

    def test_dict_not_scalar(self):
        assert not _is_scalar_value({"a": 1})


class TestIsEmptyValue:
    def test_none(self):
        assert _is_empty_value(None)

    def test_empty_string(self):
        assert _is_empty_value("")
        assert _is_empty_value("  ")

    def test_nonempty_string(self):
        assert not _is_empty_value("hello")

    def test_empty_list(self):
        assert _is_empty_value([])
        assert _is_empty_value([None, None])

    def test_nonempty_list(self):
        assert not _is_empty_value([1])

    def test_empty_dict(self):
        assert _is_empty_value({})
        assert _is_empty_value({"a": None})

    def test_nonempty_dict(self):
        assert not _is_empty_value({"a": 1})

    def test_number_not_empty(self):
        assert not _is_empty_value(0)
        assert not _is_empty_value(0.0)


class TestStringifyScalar:
    def test_none(self):
        assert _stringify_scalar(None) == "null"

    def test_int(self):
        result = _stringify_scalar(42)
        assert "42" in result

    def test_float(self):
        result = _stringify_scalar(3.14)
        assert "3.14" in result

    def test_string(self):
        assert _stringify_scalar("hello") == "hello"


class TestStringifyCell:
    def test_scalar(self):
        assert _stringify_cell("hello") == "hello"

    def test_list_of_scalars(self):
        result = _stringify_cell([1, 2, 3])
        assert "|" in result or "," in result

    def test_dict(self):
        result = _stringify_cell({"a": 1, "b": 2})
        assert "a=" in result
        assert "b=" in result

    def test_empty_list(self):
        assert _stringify_cell([None]) == ""

    def test_nested_list(self):
        result = _stringify_cell([[1, 2], [3, 4]])
        assert len(result) > 0


class TestNormalizeForecastPayload:
    def test_basic_forecast(self):
        payload = {
            "success": True,
            "times": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "forecast_price": [100.0, 101.0, 102.0],
            "symbol": "EURUSD",
            "method": "arima",
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["success"] is True
        assert "forecast" in result
        assert len(result["forecast"]) == 3
        assert result["forecast"][0]["time"] == "2024-01-01"
        assert result["forecast"][0]["forecast"] == 100.0

    def test_with_ci_bounds(self):
        payload = {
            "times": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "lower_price": [98.0, 99.0],
            "upper_price": [102.0, 103.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert "lower" in result["forecast"][0]
        assert "upper" in result["forecast"][0]

    def test_with_short_ci_bounds_fills_missing_cells(self):
        payload = {
            "times": ["t1", "t2", "t3"],
            "forecast_price": [100.0, 101.0, 102.0],
            "lower_price": [98.0, 99.0, 100.0],
            "upper_price": [102.0, 103.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][2]["lower"] == 100.0
        assert "upper" in result["forecast"][2]
        assert result["forecast"][2]["upper"] is None

    def test_with_digits(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [1.23456],
            "digits": 2,
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][0]["forecast"] == "1.23"

    def test_with_quantiles(self):
        payload = {
            "times": ["t1", "t2"],
            "forecast_price": [100.0, 101.0],
            "forecast_quantiles": {
                "0.1": [95.0, 96.0],
                "0.9": [105.0, 106.0],
            },
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert "q0.1" in result["forecast"][0]
        assert "q0.9" in result["forecast"][0]

    def test_with_short_quantiles_fills_missing_cells(self):
        payload = {
            "times": ["t1", "t2", "t3"],
            "forecast_price": [100.0, 101.0, 102.0],
            "forecast_quantiles": {
                "0.1": [95.0, 96.0, 97.0],
                "0.9": [105.0, 106.0],
                "bad": "skip-me",
            },
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][2]["q0.1"] == 97.0
        assert "q0.9" in result["forecast"][2]
        assert result["forecast"][2]["q0.9"] is None
        assert "qbad" not in result["forecast"][0]

    def test_no_times_returns_none(self):
        assert _normalize_forecast_payload({"forecast_price": [1.0]}) is None

    def test_no_forecast_returns_none(self):
        assert _normalize_forecast_payload({"times": ["t1"]}) is None

    def test_forecast_return_key(self):
        payload = {
            "times": ["t1"],
            "forecast_return": [0.01],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None

    def test_verbose_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "EURUSD",
            "timeframe": "H1",
            "method": "arima",
            "horizon": 5,
            "timezone": "UTC",
            "last_price": 101.25,
            "last_price_close": 101.0,
            "last_price_source": "live_tick_mid",
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert "meta" in result
        assert result["meta"]["domain"]["symbol"] == "EURUSD"
        assert result["meta"]["domain"]["timezone"] == "UTC"
        assert result["meta"]["domain"]["last_price"] == 101.25
        assert result["meta"]["domain"]["last_price_close"] == 101.0
        assert result["meta"]["domain"]["last_price_source"] == "live_tick_mid"

    def test_non_verbose_no_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "EURUSD",
            "method": "arima",
            "quantity": "price",
            "detail": "compact",
            "last_price": 101.0,
            "last_price_source": "candle_close",
            "forecast_vs_last_price": {"first_forecast_delta": -1.0},
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert "meta" not in result
        assert result["symbol"] == "EURUSD"
        assert result["method"] == "arima"
        assert result["quantity"] == "price"
        assert result["detail"] == "compact"
        assert result["last_price"] == 101.0
        assert result["last_price_source"] == "candle_close"
        assert result["forecast_vs_last_price"] == {"first_forecast_delta": -1.0}

    def test_q50_dedup(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "forecast_quantiles": {
                "0.1": [95.0],
                "0.5": [100.0],
                "0.9": [105.0],
            },
        }
        result = _normalize_forecast_payload(payload)
        headers_present = set(result["forecast"][0].keys())
        # q0.5 should be deduplicated since it matches forecast_price
        assert "q0.5" not in headers_present

    def test_forecast_epoch_fallback(self):
        payload = {
            "forecast_epoch": [1000, 2000],
            "forecast_price": [100.0, 101.0],
        }
        result = _normalize_forecast_payload(payload)
        assert result is not None
        assert result["forecast"][0]["time"] == 1000

    def test_denoise_in_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "denoise_used": {"method": "wavelet"},
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["domain"]["denoise"] == "wavelet"

    def test_denoise_applied_flag(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "denoise_applied": True,
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["domain"]["denoise"] == "applied"

    def test_timezone_moves_under_meta_runtime(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "method": "analog",
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "utc": {
                            "tz": "UTC",
                            "now": "2026-03-08T16:10:00+00:00",
                        },
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {
                            "tz": "US/Central",
                            "now": "2026-03-08T10:10:00-06:00",
                        },
                    },
                },
            },
        }
        result = _normalize_forecast_payload(payload, verbose=True)
        assert result["meta"]["tool"] == "forecast_generate"
        assert result["meta"]["runtime"]["timezone"]["utc"]["tz"] == "UTC"
        assert result["meta"]["runtime"]["timezone"]["server"]["tz"] == "Europe/Nicosia"
        assert result["meta"]["runtime"]["timezone"]["client"]["tz"] == "US/Central"
        assert result["meta"]["runtime"]["timezone"]["utc"]["now"] == "2026-03-08T16:10:00+00:00"
        assert result["meta"]["runtime"]["timezone"]["server"]["now"] == "2026-03-08T18:10:00+02:00"
        assert result["meta"]["runtime"]["timezone"]["client"]["now"] == "2026-03-08T10:10:00-06:00"

    def test_ci_warnings_suppressed_in_non_verbose_output(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "method": "theta",
            "ci_status": "unavailable",
            "ci_alpha": 0.05,
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable."
            ],
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert "meta" not in result
        assert result["ci"] == {
            "status": "unavailable",
            "ci_alpha": 0.05,
            "confidence_level": 0.95,
            "hint": (
                "theta produces point forecasts only. "
                "Use forecast_conformal_intervals for uncertainty bands."
            ),
        }
        # CI unavailable is already conveyed structurally via ci.status, so
        # the warning text is only surfaced when the user opts into --verbose.
        assert "warnings" not in result
        verbose_result = _normalize_forecast_payload(payload, verbose=True)
        assert verbose_result["ci"] == {
            "status": "unavailable",
            "ci_alpha": 0.05,
            "confidence_level": 0.95,
            "hint": "theta produces point forecasts only. Use forecast_conformal_intervals for uncertainty bands.",
        }
        assert verbose_result["warnings"][0].startswith("Point forecast only")

    def test_ci_diag_omitted_when_bounds_already_rendered(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "lower_price": [98.0],
            "upper_price": [102.0],
            "ci_status": "available",
            "ci_alpha": 0.05,
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert result["ci"] == {"confidence_level": 0.95}

    def test_interval_summary_rendered_when_bounds_are_compacted_away(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "method": "arima",
            "quantity": "price",
            "detail": "compact",
            "ci_status": "available",
            "ci_alpha": 0.05,
            "interval_summary": {
                "first_low": 98.0,
                "first_high": 102.0,
                "median_width": 4.0,
            },
        }
        result = _normalize_forecast_payload(payload, verbose=False)
        assert result["method"] == "arima"
        assert result["quantity"] == "price"
        assert result["detail"] == "compact"
        assert result["ci"] == {
            "status": "available",
            "ci_alpha": 0.05,
            "confidence_level": 0.95,
            "interval_summary": {
                "first_low": 98.0,
                "first_high": 102.0,
                "median_width": 4.0,
            },
        }


class TestNormalizeTripleBarrierPayload:
    def test_columnar_payload_becomes_label_rows(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 3,
            "entries": ["2026-03-17 00:00", "2026-03-17 01:00"],
            "labels": [1, 0],
            "outcomes": ["tp", "neutral"],
            "holding_bars": [2, 3],
            "tp_time": ["2026-03-17 02:00", None],
            "sl_time": [None, None],
            "summary": {
                "lookback": 2,
                "counts": {"tp": 1, "sl": 0, "neutral": 1},
            },
            "label_legend": {
                "1": {"label": "tp_first"},
                "-1": {"label": "sl_first"},
                "0": {"label": "hold"},
            },
            "sample_size": 2,
            "sample_note": "entries, labels, and timing arrays show the most recent 2 observations.",
        }
        result = _normalize_triple_barrier_payload(payload)
        assert result == {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 3,
            "labels": [
                {
                    "entry": "2026-03-17 00:00",
                    "label": 1,
                    "outcome": "tp",
                    "holding_bars": 2,
                    "tp_time": "2026-03-17 02:00",
                    "sl_time": None,
                },
                {
                    "entry": "2026-03-17 01:00",
                    "label": 0,
                    "outcome": "neutral",
                    "holding_bars": 3,
                    "tp_time": None,
                    "sl_time": None,
                },
            ],
            "summary": {
                "lookback": 2,
                "counts": {"tp": 1, "sl": 0, "neutral": 1},
            },
            "label_legend": {
                "1": {"label": "tp_first"},
                "-1": {"label": "sl_first"},
                "0": {"label": "hold"},
            },
            "sample_size": 2,
            "sample_note": "entries, labels, and timing arrays show the most recent 2 observations.",
        }


class TestNormalizeTradePayload:
    def test_results_branch_does_not_fallback_to_raw_rows(self):
        payload = {
            "success": True,
            "results": [
                {
                    "ticket": 0,
                    "message": " ",
                    "internal_only": "secret",
                }
            ],
            "internal_root": "secret",
        }
        result = _normalize_trade_payload(payload, verbose=False, tool_name="trade_close")
        assert result == {"success": True}

    def test_main_trade_path_does_not_fallback_to_raw_payload(self):
        payload = {
            "ticket": 0,
            "message": "",
            "internal_only": "secret",
        }
        result = _normalize_trade_payload(payload, verbose=False, tool_name="trade_close")
        assert result == {}

    def test_trade_place_compact_hides_duplicate_order_when_ticket_matches(self):
        payload = {
            "success": True,
            "retcode_name": "TRADE_RETCODE_DONE",
            "order": 4392901844,
            "position_ticket": 4392901844,
            "deal": 0,
            "volume": 0.01,
        }
        result = _normalize_trade_payload(payload, verbose=False, tool_name="trade_place")
        assert result["ticket"] == 4392901844
        assert "order" not in result


class TestNormalizeTradeTablePayload:
    def test_trade_history_default_view_hides_low_signal_order_columns(self):
        payload = [
            {
                "ticket": 1,
                "time_setup": "2026-03-29 10:00",
                "time_done": "2026-03-29 10:05",
                "type": "Buy Limit",
                "state": "Canceled",
                "reason": "Client",
                "volume_initial": 0.01,
                "price_open": 64500.0,
                "sl": 64000.0,
                "tp": 67200.0,
                "symbol": "BTCUSD",
                "comment": "scan2",
                "time_setup_msc": 1770000000000,
                "time_done_msc": 1770000300000,
                "type_time": "GTC",
                "type_filling": "Return",
                "position_by_id": 0,
                "price_stoplimit": 0.0,
                "external_id": "4392901617-01",
            }
        ]

        result = _normalize_trade_table_payload(
            payload,
            verbose=False,
            tool_name="trade_history",
        )

        row = result[0]
        assert row["ticket"] == 1
        assert row["time_setup"] == "2026-03-29 10:00"
        assert row["time_done"] == "2026-03-29 10:05"
        assert row["type"] == "Buy Limit"
        assert "time_setup_msc" not in row
        assert "time_done_msc" not in row
        assert "type_time" not in row
        assert "type_filling" not in row
        assert "position_by_id" not in row
        assert "price_stoplimit" not in row
        assert "external_id" not in row


class TestCompactForecastCi:
    def test_keeps_available_ci_confidence_when_bounds_exist(self):
        payload = {
            "ci_status": "available",
            "ci_alpha": 0.05,
        }
        assert _compact_forecast_ci(payload, lower=[1.0], upper=[2.0]) == {
            "confidence_level": 0.95
        }

    def test_compacts_unavailable_ci_to_status_and_ci_alpha(self):
        payload = {
            "method": "theta",
            "ci_status": "unavailable",
            "ci_alpha": 0.1,
            "warnings": [
                "Point forecast only for method 'theta'; confidence intervals are unavailable."
            ],
        }
        assert _compact_forecast_ci(payload, lower=[], upper=[]) == {
            "status": "unavailable",
            "ci_alpha": 0.1,
            "confidence_level": 0.9,
            "hint": "theta produces point forecasts only. Use forecast_conformal_intervals for uncertainty bands.",
        }


class TestBuildForecastMeta:
    def test_groups_domain_and_common_metadata(self):
        payload = {
            "method": "analog",
            "horizon": 12,
            "params_used": {"window_size": 64},
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "utc": {
                            "tz": "UTC",
                            "now": "2026-03-08T16:10:00+00:00",
                        },
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {
                            "tz": "US/Central",
                            "now": "2026-03-08T10:10:00-06:00",
                        },
                    },
                },
            },
        }
        assert _build_forecast_meta(payload) == {
            "tool": "forecast_generate",
            "domain": {
                "method": "analog",
                "horizon": 12,
                "params": {"window_size": 64},
            },
            "runtime": {
                "timezone": {
                    "utc": {
                        "tz": "UTC",
                        "now": "2026-03-08T16:10:00+00:00",
                    },
                    "server": {
                        "source": "MT5_SERVER_TZ",
                        "tz": "Europe/Nicosia",
                        "now": "2026-03-08T18:10:00+02:00",
                    },
                    "client": {
                        "tz": "US/Central",
                        "now": "2026-03-08T10:10:00-06:00",
                    },
                },
            },
        }

    def test_flattens_nested_timezone_dicts_in_common_meta(self):
        payload = {
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "server": {
                            "tz": {
                                "configured": "Europe/Nicosia",
                                "resolved": "Europe/Nicosia",
                            },
                        },
                        "client": {
                            "tz": {
                                "configured": "US/Central",
                                "resolved": "US/Central",
                            },
                        },
                    },
                },
            },
        }
        result = _build_forecast_meta(payload)
        assert result["runtime"]["timezone"]["server"]["tz"] == "Europe/Nicosia"
        assert result["runtime"]["timezone"]["client"]["tz"] == "US/Central"

    def test_prefers_resolved_timezone_when_it_differs(self):
        payload = {
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "server": {
                            "tz": {
                                "configured": "EET",
                                "resolved": "Europe/Nicosia",
                            },
                        },
                    },
                },
            },
        }
        result = _build_forecast_meta(payload)
        assert result["runtime"]["timezone"]["server"]["tz"] == "Europe/Nicosia"


class TestFormatTableToon:
    def test_basic(self):
        lines = format_table_toon(["name", "value"], [["a", 1], ["b", 2]])
        assert len(lines) > 0

    def test_empty(self):
        assert format_table_toon([], []) == []


class TestEncodeInlineArray:
    def test_basic(self):
        result = _encode_inline_array("prices", [1.0, 2.0, 3.0])
        assert "prices" in result
        assert "3" in result  # length indicator


class TestEncodeExpandedArray:
    def test_basic(self):
        result = _encode_expanded_array("items", [{"a": 1}, {"a": 2}])
        assert "items" in result
        assert "2" in result  # length indicator


class TestFormatToToon:
    def test_scalar(self):
        result = _format_to_toon(42, key="count")
        assert "42" in result

    def test_string(self):
        result = _format_to_toon("hello", key="msg")
        assert "hello" in result

    def test_dict(self):
        result = _format_to_toon({"a": 1, "b": 2}, key="data")
        assert "a" in result

    def test_tick_mid_stats_preserve_quote_precision(self):
        result = _format_to_toon(
            {
                "stats": {
                    "mid": {
                        "first": 1.172295,
                        "mean": 1.1722825,
                        "median": 1.17229,
                        "q25": 1.17227,
                        "q75": 1.1723,
                    },
                    "bid": {
                        "q25": 1.17224,
                        "q75": 1.17225,
                    },
                }
            }
        )

        assert "first: 1.172295" in result
        assert "mean: 1.1722825" in result
        assert "median: 1.17229" in result
        assert "q25: 1.17227" in result
        assert "q75: 1.1723" in result
        assert "q25: 1.17224" in result
        assert "q75: 1.17225" in result

    def test_list(self):
        result = _format_to_toon([1, 2, 3], key="items")
        assert len(result) > 0

    def test_none(self):
        result = _format_to_toon(None, key="empty")
        assert result == "" or "null" in result.lower() or result is None or not result

    def test_single_nested_value_collapses_to_dotted_key(self):
        result = _format_to_toon({"tz": "US/Central"}, key="client")
        assert result == "client.tz: US/Central"

    def test_root_dict_keys_remain_top_level(self):
        result = _format_to_toon({
            "meta": {
                "tool": "forecast_generate",
                "domain": {"method": "analog"},
                "runtime": {
                    "timezone": {
                        "utc": {"tz": "UTC", "now": "2026-03-08T16:10:00+00:00"},
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {"tz": "US/Central", "now": "2026-03-08T10:10:00-06:00"},
                    },
                },
            },
            "forecast": [{"time": "t1", "forecast": 1.0}],
        })
        lines = result.splitlines()
        assert lines[0] == "meta:"
        assert lines[1] == "  tool: forecast_generate"
        assert "  domain.method: analog" in lines
        assert "  runtime.timezone:" in lines
        assert "    utc:" in lines
        assert "      tz: UTC" in lines
        assert "      now: \"2026-03-08T16:10:00+00:00\"" in lines
        assert "    server:" in lines
        assert "      source: MT5_SERVER_TZ" in lines
        assert "      tz: Europe/Nicosia" in lines
        assert "      now: \"2026-03-08T18:10:00+02:00\"" in lines
        assert "    client:" in lines
        assert "      tz: US/Central" in lines
        assert "      now: \"2026-03-08T10:10:00-06:00\"" in lines
        assert "forecast[1]{time,forecast}:" in lines

    def test_price_like_fields_preserve_precision_in_compact_text(self):
        result = _format_to_toon(
            {
                "current_price": 1.17221,
                "best": {"tp_price": 1.19045, "sl_price": 1.16987},
                "active_levels": {
                    "bullish": {
                        "level_price": 1.17005,
                        "reference_price": 1.17221,
                    },
                },
                "levels": {"PP": 1.16893, "R1": 1.17456, "S1": 1.16321},
                "nearest": {"support": {"value": 1.16893}},
                "interval_summary": {"first_low": 1.16987, "first_high": 1.17453},
            }
        )

        assert "current_price: 1.17221" in result
        assert "tp_price: 1.19045" in result
        assert "sl_price: 1.16987" in result
        assert "level_price: 1.17005" in result
        assert "reference_price: 1.17221" in result
        assert "PP: 1.16893" in result
        assert "R1: 1.17456" in result
        assert "S1: 1.16321" in result
        assert "value: 1.16893" in result
        assert "first_low: 1.16987" in result


class TestFormatResultMinimal:
    def test_market_ticker_verbose_uses_display_time_and_keeps_epoch_field(self):
        payload = {
            "success": True,
            "symbol": "BTCUSD",
            "time": 1700000000,
            "time_display": "2023-11-14 22:13",
            "meta": {"tool": "market_ticker"},
        }
        result = _normalize_market_ticker_payload(payload, verbose=True, tool_name="market_ticker")
        assert result["time"] == "2023-11-14 22:13"
        assert result["time_epoch"] == 1700000000
        assert "time_display" not in result
        assert result["meta"]["tool"] == "market_ticker"

    def test_market_ticker_minimal_keeps_spread_pricing_basis(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "type": "ticker",
            "spread_usd": 9.0,
            "spread_currency": "USD",
            "pricing_basis": "per_1_lot_estimate",
        }

        result = _normalize_market_ticker_payload(
            payload,
            verbose=False,
            tool_name="market_ticker",
        )

        assert result["spread_usd"] == 9.0
        assert result["spread_currency"] == "USD"
        assert result["pricing_basis"] == "per_1_lot_estimate"

    def test_market_ticker_text_uses_symbol_price_precision(self):
        payload = {
            "success": True,
            "symbol": "US500",
            "type": "ticker",
            "price_precision": 2,
            "bid": 7175,
            "ask": 7175.5,
            "spread": 0.5,
            "meta": {"tool": "market_ticker"},
        }

        result = format_result_minimal(payload, verbose=False, tool_name="market_ticker")

        assert "bid: 7175.00" in result
        assert "ask: 7175.50" in result
        assert "spread: 0.50" in result

    def test_market_ticker_price_field_text_includes_field_and_price(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "type": "price",
            "field": "bid",
            "price": 1.17088,
            "price_precision": 5,
            "meta": {"tool": "market_ticker"},
        }

        result = format_result_minimal(payload, verbose=False, tool_name="market_ticker")

        assert "field: bid" in result
        assert "price: 1.17088" in result

    def test_wait_event_text_uses_symbol_price_precision_without_echoing_it(self):
        payload = {
            "success": True,
            "status": "boundary_reached",
            "symbol": "BTCUSD",
            "price_precision": 2,
            "bid": 76864.08,
            "ask": 76876.08,
            "boundary_event": {
                "type": "candle_close",
                "timeframe": "M15",
                "closed_candle": {
                    "symbol": "BTCUSD",
                    "timeframe": "M15",
                    "open": 76876.8,
                    "high": 76886.32,
                    "low": 76848.9,
                    "close": 76864.08,
                },
            },
        }

        result = format_result_minimal(payload, verbose=False, tool_name="wait_event")

        assert "price_precision" not in result
        assert "bid: 76864.08" in result
        assert "ask: 76876.08" in result
        assert "open: 76876.80" in result
        assert "low: 76848.90" in result

    def test_market_status_hides_upcoming_holidays_by_default(self):
        payload = {
            "success": True,
            "markets": [],
            "upcoming_holidays": [
                {
                    "date": "2031-01-01",
                    "holiday": "New Year's Day",
                    "markets_affected": ["NYSE", "NASDAQ"],
                    "impact": "closed",
                    "days_away": 2,
                },
                {
                    "date": "2031-01-20",
                    "holiday": "Martin Luther King Jr. Day",
                    "markets_affected": ["NYSE"],
                    "impact": "closed",
                    "days_away": 21,
                },
            ],
        }
        compact = _normalize_market_status_payload(
            payload, verbose=False, tool_name="market_status"
        )
        assert compact is not None
        assert "upcoming_holidays" not in compact
        assert "upcoming_holidays_count" not in compact
        assert "upcoming_holidays_summary" not in compact
        assert "show_all_hint" not in compact
        # Verbose mode leaves the payload untouched.
        assert _normalize_market_status_payload(
            payload, verbose=True, tool_name="market_status"
        ) is None
        # Other tools are not affected.
        assert _normalize_market_status_payload(
            payload, verbose=False, tool_name="market_ticker"
        ) is None

    def test_verbose_forecast_sections_are_not_nested_under_meta(self):
        payload = {
            "times": ["t1"],
            "forecast_price": [100.0],
            "symbol": "BTCUSD",
            "method": "analog",
            "horizon": 12,
            "params_used": {"window_size": 64},
            "meta": {
                "tool": "forecast_generate",
                "runtime": {
                    "timezone": {
                        "utc": {"tz": "UTC", "now": "2026-03-08T16:10:00+00:00"},
                        "server": {
                            "source": "MT5_SERVER_TZ",
                            "tz": "Europe/Nicosia",
                            "now": "2026-03-08T18:10:00+02:00",
                        },
                        "client": {"tz": "US/Central", "now": "2026-03-08T10:10:00-06:00"},
                    },
                },
            },
        }
        result = format_result_minimal(payload, verbose=True)
        lines = result.splitlines()
        assert lines[0] == "meta:"
        assert "  tool: forecast_generate" in lines
        assert "  domain:" in lines
        assert "    symbol: BTCUSD" in lines
        assert "    params.window_size: 64" in lines
        assert "  runtime.timezone:" in lines
        assert "    utc:" in lines
        assert "      tz: UTC" in lines
        assert "      now: \"2026-03-08T16:10:00+00:00\"" in lines
        assert "      source: MT5_SERVER_TZ" in lines
        assert "      tz: Europe/Nicosia" in lines
        assert "      now: \"2026-03-08T18:10:00+02:00\"" in lines
        assert "    client:" in lines
        assert "      tz: US/Central" in lines
        assert "      now: \"2026-03-08T10:10:00-06:00\"" in lines
        assert "forecast[1]{time,forecast}:" in lines
        assert lines.index("  runtime.timezone:") > lines.index("  domain:")
        assert lines.index("forecast[1]{time,forecast}:") > lines.index("  runtime.timezone:")

    def test_compact_collection_output_suppresses_duplicate_data_alias(self):
        table_payload = {
            "data": [{"name": "EURUSD"}, {"name": "GBPUSD"}],
            "rows": [{"name": "EURUSD"}, {"name": "GBPUSD"}],
            "success": True,
            "count": 2,
            "collection_kind": "table",
            "collection_contract_version": "collection.v1",
        }
        series_payload = {
            "data": [{"time": "t1", "close": 1.17221}],
            "series": [{"time": "t1", "close": 1.17221}],
            "success": True,
            "count": 1,
            "collection_kind": "time_series",
            "collection_contract_version": "collection.v1",
        }

        table_compact = format_result_minimal(table_payload, verbose=False)
        series_compact = format_result_minimal(series_payload, verbose=False)
        table_verbose = format_result_minimal(table_payload, verbose=True)

        assert "rows[2]{name}:" in table_compact
        assert "data[2]{name}:" not in table_compact
        assert "collection_kind:" not in table_compact
        assert "collection_contract_version:" not in table_compact
        assert "series[1]{time,close}:" in series_compact
        assert "data[1]{time,close}:" not in series_compact
        assert "collection_kind:" not in series_compact
        assert "collection_contract_version:" not in series_compact
        assert "data[2]{name}:" in table_verbose
        assert "rows[2]{name}:" in table_verbose
        assert "collection_kind: table" in table_verbose
        assert "collection_contract_version: collection.v1" in table_verbose

    def test_compact_trade_risk_output_hides_intermediate_sizing_fields(self):
        payload = {
            "success": True,
            "account": {"equity": 10000.0, "currency": "USD"},
            "portfolio_risk": {
                "overall_risk_status": "defined",
                "quantified_risk_level": "low",
                "total_risk_currency": 100.0,
                "total_risk_pct": 1.0,
                "positions_count": 1,
                "notional_exposure": 100000.0,
            },
            "position_sizing": {
                "symbol": "EURUSD",
                "direction": "long",
                "suggested_volume": 0.04,
                "requested_risk_currency": 100.0,
                "risk_currency": 99.5,
                "risk_pct": 0.99,
                "rr_ratio": 2.0,
                "raw_volume": 0.04123456,
                "volume_step": 0.01,
                "volume_rounding": "rounded_down_to_step",
            },
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="trade_risk_analyze",
        )
        verbose = format_result_minimal(
            payload,
            verbose=True,
            tool_name="trade_risk_analyze",
        )

        assert "suggested_volume: 0.04" in compact
        assert "rr_ratio: 2" in compact
        assert "raw_volume" not in compact
        assert "volume_step" not in compact
        assert "notional_exposure" not in compact
        assert "raw_volume" in verbose

    def test_compact_barrier_probability_output_hides_confidence_bands(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "tp_price": 1.19,
            "sl_price": 1.17,
            "prob_tp_first": 0.52,
            "prob_sl_first": 0.48,
            "edge": 0.04,
            "confidence": {
                "prob_tp_first_ci95": {"low": 0.48, "high": 0.56},
            },
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="forecast_barrier_prob",
        )

        assert "prob_tp_first: 0.52" in compact
        assert "prob_sl_first: 0.48" in compact
        assert "confidence" not in compact
        assert "ci95" not in compact

    def test_compact_barrier_optimize_output_keeps_best_only(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "viable": True,
            "best": {
                "tp": 1.0,
                "sl": 0.5,
                "tp_price": 1.19,
                "sl_price": 1.17,
                "prob_win": 0.55,
                "prob_no_hit": 0.1,
                "prob_resolve": 0.9,
                "edge": -0.05,
                "breakeven_win_rate": 0.33,
            },
            "results": [{"tp": 1.0, "sl": 0.5}],
            "actionability_flags": ["ok"],
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="forecast_barrier_optimize",
        )

        assert "best:" in compact
        assert "viable: true" in compact
        assert "results[" not in compact
        assert "actionability_flags" not in compact
        assert "edge: -0.05" in compact
        assert "prob_no_hit: 0.1" in compact
        assert "prob_resolve: 0.9" in compact
        assert "breakeven_win_rate" not in compact

    def test_compact_patterns_output_prefers_highlights(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "mode": "all",
            "total_patterns": 12,
            "highlights": [{"pattern": "triangle", "bias": "bullish"}],
            "classic": {"patterns": [{"pattern": "triangle", "details": {"x": 1}}]},
            "fractal": {"patterns": [{"pattern": "breakout"}]},
        }

        compact = format_result_minimal(
            payload,
            verbose=False,
            tool_name="patterns_detect",
        )

        assert "highlights[1]{pattern,bias}:" in compact
        assert "classic:" not in compact
        assert "fractal:" not in compact

    def test_triple_barrier_output_renders_as_single_table(self):
        payload = {
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "H1",
            "horizon": 3,
            "entries": ["2026-03-17 00:00", "2026-03-17 01:00"],
            "labels": [1, 0],
            "outcomes": ["tp", "neutral"],
            "holding_bars": [2, 3],
            "tp_time": ["2026-03-17 02:00", None],
            "sl_time": [None, None],
        }
        result = format_result_minimal(payload, verbose=True)
        lines = result.splitlines()
        assert "labels[2]{entry,label,outcome,holding_bars,tp_time,sl_time}:" in lines
        assert "  \"2026-03-17 00:00\",1,tp,2,\"2026-03-17 02:00\",null" in lines
        assert not any(line.startswith("entries[") for line in lines)
        assert not any(line.startswith("holding_bars[") for line in lines)
        assert not any(line.startswith("tp_time[") for line in lines)

    def test_trade_place_default_view_hides_comment_diagnostics(self):
        payload = {
            "success": True,
            "retcode": 10009,
            "retcode_name": "TRADE_RETCODE_DONE",
            "deal": 0,
            "order": 4384151941,
            "volume": 0.01,
            "price": 0,
            "bid": 0.8634,
            "ask": 0.8635,
            "requested_price": 0.8634,
            "requested_sl": 0.8652,
            "requested_tp": 0.8615,
            "comment": "Request executed",
            "request_id": 3109214586,
            "type_filling_used": 1,
            "comment_sanitization": {
                "requested": "Auto: Bearish setup, S/T breakdown",
                "applied": "Auto Bearish setup S T breakdow",
            },
            "comment_truncation": {
                "requested": "Auto: Bearish setup, S/T breakdown",
                "applied": "Auto Bearish setup S T breakdow",
                "max_length": 31,
            },
            "comment_fallback": {
                "used": True,
                "strategy": "minimal",
            },
            "fill_mode_attempts": [
                {"type_filling": 1, "retcode": 10009, "retcode_name": "TRADE_RETCODE_DONE"},
            ],
            "warnings": [
                "Comment sanitized for broker compatibility: 'Auto Bearish setup S T breakdow'",
                "Comment truncated to 31 characters: 'Auto Bearish setup S T breakdow'",
                "Broker rejected the comment field; pending order was retried with a minimal MT5-safe comment.",
            ],
        }
        result = format_result_minimal(
            payload,
            verbose=False,
            tool_name="trade_place",
            simplify_numbers=False,
        )
        lines = result.splitlines()
        assert "success: true" in lines
        assert "retcode_name: TRADE_RETCODE_DONE" in lines
        assert "order: 4384151941" in lines
        assert "price: 0.8634" in lines
        assert "requested_sl: 0.8652" in lines
        assert "requested_tp: 0.8615" in lines
        assert not any("comment_sanitization" in line for line in lines)
        assert not any("comment_truncation" in line for line in lines)
        assert not any("comment_fallback" in line for line in lines)
        assert not any("fill_mode_attempts" in line for line in lines)
        assert not any("request_id" in line for line in lines)
        assert not any("type_filling_used" in line for line in lines)
        assert not any("bid:" in line for line in lines)
        assert not any("ask:" in line for line in lines)
        assert not any("warnings" in line for line in lines)

    def test_trade_place_default_view_keeps_only_actionable_warning(self):
        payload = {
            "retcode_name": "TRADE_RETCODE_DONE",
            "order": 123,
            "position_ticket": 456,
            "protection_status": "unprotected_position",
            "sl_tp_result": {
                "status": "failed",
                "requested": {"sl": 64000.0, "tp": 68000.0},
                "error": "Failed to set TP/SL",
            },
            "warnings": [
                "Comment sanitized for broker compatibility: 'AUTO CLOSE'",
                "CRITICAL: Order executed without applied TP/SL protection. Run trade_modify 456 now, or close the position.",
                "Broker rejected the comment field; order was retried with a minimal MT5-safe comment.",
            ],
        }
        result = format_result_minimal(payload, verbose=False, tool_name="trade_place")
        assert "CRITICAL: Order executed without applied TP/SL protection." in result
        assert "Comment sanitized" not in result
        assert "Broker rejected the comment field" not in result

    def test_trade_place_dry_run_default_view_surfaces_preview_fields(self):
        payload = {
            "success": True,
            "dry_run": True,
            "no_action": True,
            "trade_gate_passed": False,
            "actionability": "preview_only",
            "symbol": "BTCUSD",
            "order_type": "BUY_LIMIT",
            "pending": True,
            "action": "place_pending_order",
            "volume": 0.01,
            "requested_price": 64500.0,
            "requested_sl": 64000.0,
            "requested_tp": 67200.0,
            "expiration": "GTC",
            "validation_scope": "request_routing_only",
            "preview_scope_summary": "Routing and local request checks only.",
            "message": "Dry run only. No order was sent to MT5.",
            "actionability_reason": "Dry run did not execute MT5 or broker-side validation. Use this preview for request routing only.",
            "warnings": [
                "Dry run only. Routing and local safety checks passed; MT5/broker validation was not executed.",
                "Not validated in dry run: broker acceptance, live price-distance rules, margin/funds, fillability, and SL/TP attachment.",
            ],
        }
        result = format_result_minimal(payload, verbose=False, tool_name="trade_place")
        assert "dry_run: true" in result
        assert "trade_gate_passed: false" in result
        assert "actionability: preview_only" in result
        assert "symbol: BTCUSD" in result
        assert "order_type: BUY_LIMIT" in result
        assert "pending: true" in result
        assert "action: place_pending_order" in result
        assert "price: 64500" in result
        assert "validation_scope: request_routing_only" in result
        assert "preview_scope_summary: Routing and local request checks only." in result
        assert "message: Dry run only. No order was sent to MT5." in result
        assert "actionability_reason: Dry run did not execute MT5 or broker-side validation. Use this preview for request routing only." in result
        assert "warnings[1]:" in result
        assert "MT5/broker validation was not executed" in result


class TestFormatComplexValue:
    def test_dict(self):
        result = _format_complex_value({"key": "val"})
        assert "key" in result

    def test_list(self):
        result = _format_complex_value([1, 2, 3])
        assert len(result) > 0

    def test_scalar(self):
        result = _format_complex_value(42)
        assert "42" in result
