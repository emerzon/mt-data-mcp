"""Comprehensive tests for pure functions across multiple modules.

Covers:
  - mtdata.utils.indicators  (TA indicator helpers)
  - mtdata.core.indicators    (thin wrappers / delegates)
  - mtdata.utils.utils        (utility pure functions)
  - mtdata.core.schema        (schema validation/parsing)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import pytest
from typing_extensions import TypedDict

from mtdata.core.schema import (
    _DENOISE_METHODS,
    _PIVOT_METHODS,
    _SIMPLIFY_METHODS,
    _SIMPLIFY_MODES,
    PARAM_HINTS,
    _allow_null,
    _ensure_defs,
    _is_typed_dict_type,
    _load_indicator_doc_choices,
    _parameters_obj,
    _type_hint_to_schema,
    apply_param_hints,
    apply_timeframe_ref,
    build_minimal_schema,
    complex_defs,
    enrich_schema_with_shared_defs,
    get_function_info,
    get_shared_enum_lists,
    shared_defs,
)

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from mtdata.utils.indicators import (
    _estimate_warmup_bars,
    _find_unknown_ta_indicators,
    _parse_ti_number,
    _parse_ti_specs,
    _try_number,
    clean_help_text,
    infer_defaults_from_doc,
    list_ta_indicators,
)
from mtdata.utils.utils import (
    _coerce_scalar,
    _format_float,
    _format_numeric_rows_from_df,
    _format_time_minimal,
    _maybe_strip_year,
    _normalize_limit,
    _normalize_ohlcv_arg,
    _parse_start_datetime,
    _style_time_format,
    _table_from_rows,
    _time_format_from_epochs,
    _utc_epoch_seconds,
    align_finite,
    parse_kv_or_json,
    to_float_np,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RS = np.random.RandomState(42)


def _make_ohlcv_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with deterministic data."""
    close = 100.0 + RS.standard_normal(n).cumsum()
    high = close + RS.uniform(0.5, 2.0, n)
    low = close - RS.uniform(0.5, 2.0, n)
    open_ = close + RS.uniform(-1.0, 1.0, n)
    volume = RS.randint(100, 10000, n).astype(float)
    time_vals = np.arange(n) * 3600 + 1_600_000_000
    return pd.DataFrame({
        "time": time_vals,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ===================================================================
# 1. mtdata.utils.indicators
# ===================================================================
class TestCleanHelpText:
    def test_empty_string(self):
        assert clean_help_text("") == ""

    def test_non_string_returns_empty(self):
        assert clean_help_text(None) == ""  # type: ignore[arg-type]
        assert clean_help_text(123) == ""  # type: ignore[arg-type]

    def test_strips_backspace_overstrikes(self):
        # "A\x08B" should become "B" after overstrike removal
        result = clean_help_text("A\x08B")
        assert "\x08" not in result

    def test_with_func_name_finds_signature(self):
        text = "some preamble\nema(close, length=10)\n  description here"
        result = clean_help_text(text, func_name="ema")
        assert result.startswith("ema(")

    def test_removes_method_of_suffix(self):
        text = "sma(close) method of pandas_ta.overlap\n  desc"
        result = clean_help_text(text, func_name="sma")
        assert "method of" not in result.split("\n")[0]

    def test_removes_method_of_on_second_line(self):
        text = "sma(close)\n  method of pandas_ta.core\n  desc"
        result = clean_help_text(text, func_name="sma")
        assert "method of" not in result


class TestTryNumber:
    def test_integer(self):
        assert _try_number("42") == 42
        assert isinstance(_try_number("42"), int)

    def test_float(self):
        assert _try_number("3.14") == pytest.approx(3.14)

    def test_non_numeric(self):
        assert _try_number("hello") is None

    def test_negative(self):
        assert _try_number("-5") == -5


class TestParseTiNumber:
    def test_integer_float_normalized(self):
        assert _parse_ti_number("20.0") == 20
        assert isinstance(_parse_ti_number("20.0"), int)

    def test_true_float(self):
        assert _parse_ti_number("0.5") == pytest.approx(0.5)

    def test_non_numeric(self):
        assert _parse_ti_number("abc") is None

    def test_int_string(self):
        assert _parse_ti_number("14") == 14


class TestInferDefaultsFromDoc:
    def test_from_signature_line(self):
        params = [{"name": "length"}, {"name": "offset"}]
        doc = "ema(close, length=20, offset=0)\n  desc"
        infer_defaults_from_doc("ema", doc, params)
        assert params[0].get("default") == 20
        assert params[1].get("default") == 0

    def test_from_body_default_keyword(self):
        params = [{"name": "length"}]
        doc = "sma(close)\n  length: Default: 14"
        infer_defaults_from_doc("sma", doc, params)
        assert params[0].get("default") == 14

    def test_empty_doc(self):
        params = [{"name": "length"}]
        infer_defaults_from_doc("sma", "", params)
        assert "default" not in params[0]

    def test_existing_default_not_overwritten(self):
        params = [{"name": "length", "default": 99}]
        doc = "ema(close, length=20)\n  desc"
        infer_defaults_from_doc("ema", doc, params)
        assert params[0]["default"] == 99


class TestParseTiSpecs:
    def test_empty_string(self):
        assert _parse_ti_specs("") == []

    def test_simple_name(self):
        specs = _parse_ti_specs("rsi")
        assert len(specs) == 1
        assert specs[0][0] == "rsi"

    def test_name_with_args(self):
        specs = _parse_ti_specs("ema(20)")
        assert specs[0][0] == "ema"
        # 20 is parsed as a positional arg (not kwargs since it's inside parens)
        assert 20 in specs[0][1]

    def test_name_with_kwargs(self):
        specs = _parse_ti_specs("macd(fast=12,slow=26)")
        name, args, kwargs = specs[0]
        assert name == "macd"
        assert kwargs["fast"] == 12
        assert kwargs["slow"] == 26

    def test_multiple_specs(self):
        specs = _parse_ti_specs("rsi(14),ema(20),macd")
        assert len(specs) == 3

    def test_trailing_number_in_name(self):
        specs = _parse_ti_specs("EMA21")
        name, args, kwargs = specs[0]
        assert name == "ema"
        assert kwargs.get("length") == 21

    def test_cdl_name_with_trailing_digits_is_not_rewritten(self):
        specs = _parse_ti_specs("CDL_FAKE12")
        name, args, kwargs = specs[0]
        assert name == "cdl_fake12"
        assert args == []
        assert kwargs == {}

    def test_cdl_name_with_trailing_digits_is_reported_as_unknown(self):
        assert _find_unknown_ta_indicators("CDL_FAKE12") == ["cdl_fake12"]

    def test_nested_parens_not_split(self):
        specs = _parse_ti_specs("rsi(14),macd(12,26,9)")
        assert len(specs) == 2

    def test_name_with_positional_args(self):
        specs = _parse_ti_specs("stoch(14,3,3)")
        name, args, kwargs = specs[0]
        assert name == "stoch"
        assert 14 in args or kwargs.get("length") == 14

    def test_bollinger_alias_normalizes_to_bbands(self):
        specs = _parse_ti_specs("bb(20),bollinger_bands(20)")
        assert [name for name, _args, _kwargs in specs] == ["bbands", "bbands"]

    def test_bollinger_aliases_are_not_reported_as_unknown(self):
        assert _find_unknown_ta_indicators("bb(20),bollinger_bands(20)") == []

    def test_indicator_lookup_results_are_cached(self, monkeypatch):
        import mtdata.utils.indicators as indicators_mod

        class _ProxyPta:
            def __init__(self) -> None:
                self.lookups = 0

            def __getattr__(self, name: str):
                self.lookups += 1
                if name == "ema":
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

        proxy = _ProxyPta()
        monkeypatch.setattr(indicators_mod, "pta", proxy)
        indicators_mod._is_available_ta_indicator.cache_clear()

        assert indicators_mod._find_unknown_ta_indicators("ema(20),ema(50)") == []
        assert indicators_mod._find_unknown_ta_indicators("ema(20)") == []
        assert proxy.lookups == 1

        indicators_mod._is_available_ta_indicator.cache_clear()

    def test_list_ta_indicators_includes_shadowed_volatility_category(self):
        items = list_ta_indicators(detailed=False)
        by_name = {str(item.get("name")): item for item in items}

        for name in ("atr", "natr", "bbands", "kc", "donchian"):
            assert name in by_name
            assert by_name[name]["category"] == "volatility"


class TestEstimateWarmupBars:
    def test_empty_spec(self):
        assert _estimate_warmup_bars("") == 0
        assert _estimate_warmup_bars(None) == 0

    def test_sma(self):
        result = _estimate_warmup_bars("sma(20)")
        assert result >= 50  # min(max_warmup*3, 50)

    def test_rsi(self):
        result = _estimate_warmup_bars("rsi(14)")
        assert result >= 42  # 14*3

    def test_macd(self):
        result = _estimate_warmup_bars("macd(12,26,9)")
        assert result == 105

    def test_macd_uses_default_signal_when_missing(self):
        result = _estimate_warmup_bars("macd(fast=12,slow=26)")
        assert result == 105

    def test_bbands(self):
        result = _estimate_warmup_bars("bbands(20)")
        assert result >= 50

    def test_stoch(self):
        result = _estimate_warmup_bars("stoch(14,3,3)")
        assert result >= 50

    def test_unknown_indicator(self):
        result = _estimate_warmup_bars("someunknown")
        assert result >= 50  # default warmup 50 * 3

    def test_multiple_indicators_takes_max(self):
        result = _estimate_warmup_bars("sma(50),rsi(14)")
        assert result >= 150  # sma(50) -> 50*3=150


# ===================================================================
# 2. mtdata.core.indicators  (thin wrappers)
# ===================================================================
class TestCoreIndicatorsWrappers:
    def test_try_number_delegation(self):
        """core.indicators._try_number delegates to indicators_docs; verify import path."""
        from mtdata.core.indicators import _try_number as core_try
        try:
            assert core_try("42") == 42
            assert core_try("3.14") == pytest.approx(3.14)
            assert core_try("bad") is None
        except ImportError:
            pytest.skip("core.indicators_docs not exposing _try_number")

    def test_clean_help_text_delegation(self):
        """core.indicators._clean_help_text delegates to indicators_docs."""
        from mtdata.core.indicators import _clean_help_text as core_clean
        try:
            assert core_clean("") == ""
            assert core_clean(None) == ""  # type: ignore[arg-type]
        except ImportError:
            pytest.skip("core.indicators_docs not exposing clean_help_text")

    def test_indicators_describe_returns_structured_documentation(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        sample_doc = """rsi(close, length=14) method of pandas_ta.momentum
Relative Strength Index for momentum.
Sources:
https://example.com/rsi
Calculation:
1) Compute average gains and losses over length.
Args:
length (int): Window length.
scalar (float): Optional scalar multiplier.
Interpretation:
Values above 70 often indicate overbought conditions.
"""
        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [{"name": "length", "default": 14}, {"name": "scalar"}],
                    "description": sample_doc,
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        out = raw_describe("rsi")
        assert out["success"] is True
        indicator = out["indicator"]
        assert "method of" not in indicator["description"].lower()
        docs = indicator["documentation"]
        assert docs["calculation"]
        assert docs["interpretation"]
        assert docs["sources"] == ["https://example.com/rsi"]
        assert "parameters" not in docs
        params = {p["name"]: p for p in indicator["params"]}
        assert params["length"]["description"] == "Window length."
        assert params["scalar"]["description"] == "Optional scalar multiplier."
        assert "usage_examples" not in indicator

    def test_indicators_describe_cleans_signature_and_preserves_multiline_docs(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        sample_doc = """Python Library Documentation: function rsi in module pandas_ta_classic.momentum.rsi

rsi(
close: Series,
length: Optional[int] = None,
talib: Optional[bool] = None
) -> Optional[Series]
Relative Strength Index
Tracks momentum across the selected lookback window.
Useful for overbought and oversold analysis.
Parameters:
talib (bool): If TA Lib is installed and talib is True,
    Returns the TA Lib version. Default: True
Interpretation:
Values above 70 often indicate overbought conditions.
Values below 30 often indicate oversold conditions.
"""
        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [{"name": "talib"}],
                    "description": sample_doc,
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        out = raw_describe("rsi")
        indicator = out["indicator"]
        docs = indicator["documentation"]
        assert "parameters" not in docs
        params = {p["name"]: p for p in indicator["params"]}

        assert "pandas_ta_classic" not in indicator["description"]
        assert "rsi(" not in indicator["description"]
        assert "Useful for overbought and oversold analysis." in indicator["description"]
        assert docs["interpretation"] == (
            "Values above 70 often indicate overbought conditions.\n"
            "Values below 30 often indicate oversold conditions."
        )
        assert params["talib"]["description"] == (
            "If TA Lib is installed and talib is True, Returns the TA Lib version. Default: True"
        )

    def test_indicators_describe_logs_finish_event(self, monkeypatch, caplog):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [],
                    "description": "Relative Strength Index.",
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        with caplog.at_level(logging.DEBUG, logger=core_indicators.logger.name):
            out = raw_describe("rsi")

        assert out["success"] is True
        assert any(
            "event=finish" in record.message and "operation=indicators_describe" in record.message
            for record in caplog.records
        )

    def test_indicators_list_reports_when_results_are_truncated(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {
                    "name": f"ind_{i:02d}",
                    "category": "momentum",
                    "description": "",
                    "params": [{"name": "length", "default": 14}],
                }
                for i in range(30)
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(category="momentum", limit=25)

        assert out["success"] is True
        assert out["count"] == 25
        assert out["data"][0]["params_count"] == 1
        assert out["data"][0]["params"] == "length=14"
        assert out["total_count"] == 30
        assert out["more_available"] == 5
        assert out["truncated"] is True
        assert out["show_all_hint"] == "Set limit to a higher value to view more matching indicators."

    def test_indicators_list_full_detail_includes_aliases_and_descriptions(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {
                    "name": "bbands",
                    "category": "volatility",
                    "description": "Bollinger Bands volatility envelope.",
                    "aliases": ["bb", "bollinger_bands"],
                    "params": [{"name": "length", "default": 20}],
                }
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(search_term="bb", detail="full")

        assert out["success"] is True
        assert out["detail"] == "full"
        assert out["data"][0]["summary"] == "Bollinger Bands volatility envelope."
        assert out["data"][0]["params_count"] == 1
        assert out["data"][0]["params"] == [{"name": "length", "default": 20}]
        assert out["data"][0]["aliases"] == "bb, bollinger_bands"
        assert "Bollinger Bands" in out["data"][0]["description"]

    def test_indicators_list_full_detail_strips_signature_and_doc_sections(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        sample_doc = (
            "adx(high, low, close, length=14) -> pandas.core.frame.DataFrame\n"
            "Average Directional Movement\n\n"
            "This indicator attempts to quantify trend strength.\n\n"
            "Parameters:\n"
            "    length (int): Window length.\n"
            "Sources:\n"
            "    * https://example.com/adx\n"
        )

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {
                    "name": "adx",
                    "category": "trend",
                    "description": sample_doc,
                    "aliases": [],
                    "params": [{"name": "length", "default": 14}],
                }
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(search_term="adx", detail="full")

        assert out["success"] is True
        description = out["data"][0]["description"]
        assert "Average Directional Movement" in description
        assert "quantify trend strength" in description
        assert "adx(" not in description
        assert "Parameters:" not in description
        assert "Sources:" not in description
        assert out["data"][0]["params"] == [
            {"name": "length", "default": 14, "description": "Window length."}
        ]

    def test_indicators_list_search_matches_names_aliases_and_categories(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {"name": "rsi", "category": "momentum", "description": "Relative Strength Index", "aliases": []},
                {"name": "lrsi", "category": "momentum", "description": "Laguerre RSI", "aliases": []},
                {"name": "stochrsi", "category": "momentum", "description": "Stochastic RSI", "aliases": []},
                {"name": "obv", "category": "volume", "description": "Volume tool that references RSI in docs", "aliases": []},
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(search_term="rsi", detail="full")

        assert out["success"] is True
        assert [row["name"] for row in out["data"]] == ["rsi", "lrsi", "stochrsi"]

        category_out = raw_list(search_term="momentum", detail="full")
        assert category_out["success"] is True
        assert [row["name"] for row in category_out["data"]] == ["lrsi", "rsi", "stochrsi"]

    def test_indicators_describe_accepts_aliases(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "bbands",
                    "category": "volatility",
                    "params": [{"name": "length", "default": 20}],
                    "description": "Bollinger Bands volatility envelope.",
                    "aliases": ["bb", "bollinger_bands"],
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        out = raw_describe("bb")

        assert out["success"] is True
        assert out["indicator"]["name"] == "bbands"
        assert "usage_examples" not in out["indicator"]


# ===================================================================
# 3. mtdata.utils.utils
# ===================================================================
class TestCoerceScalar:
    def test_integer_string(self):
        assert _coerce_scalar("42") == 42

    def test_negative_integer(self):
        assert _coerce_scalar("-7") == -7

    def test_float_string(self):
        assert _coerce_scalar("3.14") == pytest.approx(3.14)

    def test_non_numeric_string(self):
        assert _coerce_scalar("hello") == "hello"

    def test_none(self):
        assert _coerce_scalar(None) is None

    def test_empty_string(self):
        assert _coerce_scalar("") == ""

    def test_whitespace(self):
        assert _coerce_scalar("  42  ") == 42


class TestNormalizeOhlcvArg:
    def test_none(self):
        assert _normalize_ohlcv_arg(None) is None

    def test_empty(self):
        assert _normalize_ohlcv_arg("") is None

    def test_all(self):
        assert _normalize_ohlcv_arg("all") == {"O", "H", "L", "C", "V"}

    def test_ohlcv(self):
        assert _normalize_ohlcv_arg("ohlcv") == {"O", "H", "L", "C", "V"}

    def test_ohlc(self):
        assert _normalize_ohlcv_arg("ohlc") == {"O", "H", "L", "C"}

    def test_price(self):
        assert _normalize_ohlcv_arg("price") == {"C"}

    def test_close(self):
        assert _normalize_ohlcv_arg("close") == {"C"}

    def test_compact_letters(self):
        assert _normalize_ohlcv_arg("hl") == {"H", "L"}

    def test_comma_separated_names(self):
        result = _normalize_ohlcv_arg("open,high,volume")
        assert result == {"O", "H", "V"}

    def test_semicolon_separated(self):
        result = _normalize_ohlcv_arg("open;close")
        assert result == {"O", "C"}

    def test_unrecognized_returns_none(self):
        assert _normalize_ohlcv_arg("unknown_field") is None


class TestNormalizeLimit:
    def test_none(self):
        assert _normalize_limit(None) is None

    def test_positive_int(self):
        assert _normalize_limit(10) == 10

    def test_zero(self):
        assert _normalize_limit(0) is None

    def test_negative(self):
        assert _normalize_limit(-5) is None

    def test_string(self):
        assert _normalize_limit("25") == 25

    def test_float_string(self):
        assert _normalize_limit("10.9") == 10

    def test_empty_string(self):
        assert _normalize_limit("") is None

    def test_non_numeric_string(self):
        assert _normalize_limit("abc") is None


class TestTableFromRows:
    def test_basic(self):
        result = _table_from_rows(["a", "b"], [[1, 2], [3, 4]])
        assert result["success"] is True
        assert result["count"] == 2
        assert result["data"][0] == {"a": 1, "b": 2}

    def test_empty_rows(self):
        result = _table_from_rows(["a"], [])
        assert result["count"] == 0
        assert result["data"] == []

    def test_short_row_pads_none(self):
        result = _table_from_rows(["a", "b", "c"], [[1]])
        assert result["data"][0]["c"] is None

    def test_none_rows(self):
        result = _table_from_rows(["a"], None)
        assert result["count"] == 0


class TestFormatTimeMinimal:
    def test_epoch_zero(self):
        result = _format_time_minimal(0)
        assert result == "1970-01-01 00:00"

    def test_known_epoch(self):
        # 2020-01-01 00:00 UTC = 1577836800
        result = _format_time_minimal(1577836800)
        assert result == "2020-01-01 00:00"


class TestStyleTimeFormat:
    def test_replaces_T(self):
        assert _style_time_format("%Y-%m-%dT%H:%M") == "%Y-%m-%d %H:%M"

    def test_no_T(self):
        fmt = "%Y-%m-%d %H:%M"
        assert _style_time_format(fmt) == fmt


class TestTimeFormatFromEpochs:
    def test_returns_constant(self):
        assert _time_format_from_epochs([0, 100]) == "%Y-%m-%d %H:%M"


class TestMaybeStripYear:
    def test_noop(self):
        fmt = "%Y-%m-%d %H:%M"
        assert _maybe_strip_year(fmt, [0]) == fmt


class TestParseKvOrJson:
    def test_none(self):
        assert parse_kv_or_json(None) == {}

    def test_dict_passthrough(self):
        assert parse_kv_or_json({"a": 1}) == {"a": 1}

    def test_empty_string(self):
        assert parse_kv_or_json("") == {}

    def test_json_object(self):
        result = parse_kv_or_json('{"key": "val"}')
        assert result == {"key": "val"}

    def test_json_list_of_pairs(self):
        result = parse_kv_or_json('[["k1","v1"],["k2","v2"]]')
        assert result == {"k1": "v1", "k2": "v2"}

    def test_kv_equals(self):
        result = parse_kv_or_json("alpha=0.05 beta=0.1")
        assert result["alpha"] == "0.05"
        assert result["beta"] == "0.1"

    def test_kv_colon(self):
        result = parse_kv_or_json("alpha:0.05")
        assert result["alpha"] == "0.05"

    def test_windows_path_not_parsed(self):
        result = parse_kv_or_json("C:\\Users\\foo")
        assert "C" not in result or not result.get("C", "").startswith("\\")

    def test_non_string_non_dict(self):
        assert parse_kv_or_json(42) == {}

    def test_malformed_json_braces_fallback(self):
        result = parse_kv_or_json("{a=1, b=2}")
        assert "a" in result or "b" in result


class TestFormatFloat:
    def test_basic(self):
        assert _format_float(3.14, 2) == "3.14"

    def test_trailing_zeros_trimmed(self):
        assert _format_float(3.10, 2) == "3.1"

    def test_integer_value(self):
        assert _format_float(5.0, 2) == "5"

    def test_zero(self):
        assert _format_float(0.0, 4) == "0"


class TestFormatNumericRowsFromDf:
    def test_basic_formatting(self):
        df = pd.DataFrame({"time": [0], "close": [1.23456789]})
        rows = _format_numeric_rows_from_df(df, ["time", "close"])
        assert len(rows) == 1
        assert len(rows[0]) == 2

    def test_with_none_values(self):
        df = pd.DataFrame({"time": [0], "val": [None]})
        rows = _format_numeric_rows_from_df(df, ["time", "val"])
        assert rows[0][1] == "null"

    def test_bool_formatting(self):
        df = pd.DataFrame({"time": [0], "flag": [True]})
        rows = _format_numeric_rows_from_df(df, ["time", "flag"])
        assert rows[0][1] == "true"


class TestToFloatNp:
    def test_list_input(self):
        arr = to_float_np([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_series_input(self):
        s = pd.Series([1.5, 2.5])
        arr = to_float_np(s)
        np.testing.assert_array_almost_equal(arr, [1.5, 2.5])

    def test_coerce_non_numeric(self):
        arr = to_float_np(["1", "bad", "3"], coerce=True)
        assert len(arr) == 3
        assert np.isnan(arr[1])

    def test_drop_na(self):
        arr = to_float_np([1.0, np.nan, 3.0], drop_na=True)
        np.testing.assert_array_equal(arr, [1.0, 3.0])

    def test_finite_only(self):
        arr = to_float_np([1.0, np.inf, 3.0], finite_only=True)
        np.testing.assert_array_equal(arr, [1.0, 3.0])

    def test_return_mask(self):
        arr, mask = to_float_np([1.0, np.nan, 3.0], drop_na=True, return_mask=True)
        assert len(arr) == 2
        assert mask.sum() == 2

    def test_return_mask_no_filter(self):
        arr, mask = to_float_np([1.0, 2.0], return_mask=True)
        assert len(arr) == 2
        assert mask.all()

    def test_empty_input(self):
        arr = to_float_np([])
        assert len(arr) == 0


class TestAlignFinite:
    def test_basic(self):
        a, b = align_finite([1.0, np.nan, 3.0], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(a, [1.0, 3.0])
        np.testing.assert_array_equal(b, [4.0, 6.0])

    def test_all_finite(self):
        a, b = align_finite([1.0, 2.0], [3.0, 4.0])
        assert len(a) == 2
        assert len(b) == 2

    def test_empty_call(self):
        result = align_finite()
        assert result == ()


class TestParseStartDatetime:
    def test_valid_date(self):
        dt = _parse_start_datetime("2023-01-15")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 1

    def test_empty_string(self):
        assert _parse_start_datetime("") is None

    def test_result_is_naive_utc(self):
        dt = _parse_start_datetime("2023-06-01 12:00 UTC")
        assert dt is not None
        assert dt.tzinfo is None


class TestUtcEpochSeconds:
    def test_naive_as_utc(self):
        dt = datetime(2020, 1, 1, 0, 0, 0)
        epoch = _utc_epoch_seconds(dt)
        assert epoch == pytest.approx(1577836800.0)

    def test_aware_utc(self):
        dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        epoch = _utc_epoch_seconds(dt)
        assert epoch == pytest.approx(1577836800.0)


# ===================================================================
# 4. mtdata.core.schema
# ===================================================================
class TestSharedDefs:
    def test_returns_timeframe_spec(self):
        defs = shared_defs()
        assert "TimeframeSpec" in defs
        assert defs["TimeframeSpec"]["type"] == "string"
        assert "enum" in defs["TimeframeSpec"]

    def test_timeframe_enum_sorted(self):
        defs = shared_defs()
        vals = defs["TimeframeSpec"]["enum"]
        assert vals == sorted(vals)

    def test_load_indicator_doc_choices_uses_single_loader_call(self):
        calls = []

        def fake_loader(*, detailed=False):
            calls.append(detailed)
            return [
                {"category": "trend", "name": "ema"},
                {"category": "momentum", "name": "rsi"},
                {"category": "trend", "name": "ema"},
            ]

        categories, names = _load_indicator_doc_choices(fake_loader)

        assert calls == [False]
        assert categories == ["momentum", "trend"]
        assert names == []

    def test_load_indicator_doc_choices_falls_back_on_loader_error(self):
        categories, names = _load_indicator_doc_choices(
            lambda *, detailed=False: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        assert categories == []
        assert names == []

    def test_load_indicator_doc_choices_logs_loader_error(self, caplog):
        with caplog.at_level(logging.WARNING):
            categories, names = _load_indicator_doc_choices(
                lambda *, detailed=False: (_ for _ in ()).throw(RuntimeError("boom"))
            )

        assert categories == []
        assert names == []
        assert any(
            "indicator metadata loading failed" in record.message.lower()
            for record in caplog.records
        )


class TestComplexDefs:
    def test_has_expected_keys(self):
        defs = complex_defs()
        assert "IndicatorSpec" in defs
        assert "DenoiseSpec" in defs
        assert "SimplifySpec" in defs

    def test_indicator_spec_structure(self):
        defs = complex_defs()
        spec = defs["IndicatorSpec"]
        assert spec["type"] == "object"
        assert "name" in spec["properties"]


class TestEnsureDefs:
    def test_adds_defs_if_missing(self):
        schema = {}
        _ensure_defs(schema)
        assert "$defs" in schema
        assert "TimeframeSpec" in schema["$defs"]

    def test_does_not_overwrite_existing(self):
        schema = {"$defs": {"Custom": {"type": "string"}}}
        _ensure_defs(schema)
        assert "Custom" in schema["$defs"]
        assert "TimeframeSpec" in schema["$defs"]


class TestParametersObj:
    def test_creates_if_missing(self):
        schema = {}
        params = _parameters_obj(schema)
        assert params["type"] == "object"
        assert "properties" in params

    def test_returns_existing(self):
        schema = {"parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}
        params = _parameters_obj(schema)
        assert "a" in params["properties"]


class TestApplyParamHints:
    def test_adds_description(self):
        schema = {"parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"},
        }}}
        apply_param_hints(schema)
        assert schema["parameters"]["properties"]["symbol"]["description"] == PARAM_HINTS["symbol"]

    def test_no_overwrite_existing_description(self):
        schema = {"parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "custom"},
        }}}
        apply_param_hints(schema)
        assert schema["parameters"]["properties"]["symbol"]["description"] == "custom"


class TestApplyTimeframeRef:
    def test_replaces_timeframe_prop(self):
        schema = {"parameters": {"type": "object", "properties": {
            "timeframe": {"type": "string"},
        }}}
        apply_timeframe_ref(schema)
        assert schema["parameters"]["properties"]["timeframe"] == {"$ref": "#/$defs/TimeframeSpec"}

    def test_ignores_non_timeframe(self):
        schema = {"parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"},
        }}}
        apply_timeframe_ref(schema)
        assert schema["parameters"]["properties"]["symbol"] == {"type": "string"}


class TestAllowNull:
    def test_simple_type(self):
        result = _allow_null({"type": "string"})
        assert result["type"] == ["string", "null"]

    def test_already_null(self):
        result = _allow_null({"type": "null"})
        assert result["type"] == "null"

    def test_list_type(self):
        result = _allow_null({"type": ["string", "integer"]})
        assert "null" in result["type"]

    def test_list_type_already_has_null(self):
        result = _allow_null({"type": ["string", "null"]})
        assert result["type"].count("null") == 1

    def test_no_type_key(self):
        result = _allow_null({"enum": ["a", "b"]})
        assert "type" not in result


class TestTypeHintToSchema:
    def test_none(self):
        assert _type_hint_to_schema(None) == {"type": "string"}

    def test_any(self):
        assert _type_hint_to_schema(Any) == {}

    def test_str(self):
        assert _type_hint_to_schema(str) == {"type": "string"}

    def test_int(self):
        assert _type_hint_to_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _type_hint_to_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _type_hint_to_schema(bool) == {"type": "boolean"}

    def test_bare_dict(self):
        result = _type_hint_to_schema(dict)
        assert result["type"] == "object"

    def test_bare_list(self):
        result = _type_hint_to_schema(list)
        assert result["type"] == "array"

    def test_list_of_str(self):
        result = _type_hint_to_schema(List[str])
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_dict_str_int(self):
        result = _type_hint_to_schema(Dict[str, int])
        assert result["type"] == "object"
        assert result["additionalProperties"] == {"type": "integer"}

    def test_optional_str(self):
        result = _type_hint_to_schema(Optional[str])
        assert "null" in str(result.get("type", ""))

    def test_literal_strings(self):
        result = _type_hint_to_schema(Literal["a", "b"])
        assert result["type"] == "string"
        assert set(result["enum"]) == {"a", "b"}

    def test_literal_ints(self):
        result = _type_hint_to_schema(Literal[1, 2, 3])
        assert result["type"] == "integer"
        assert result["enum"] == [1, 2, 3]

    def test_literal_bools(self):
        result = _type_hint_to_schema(Literal[True, False])
        assert result["type"] == "boolean"

    def test_union_type(self):
        result = _type_hint_to_schema(Union[str, int])
        assert "oneOf" in result


class TestIsTypedDictType:
    def test_regular_class_false(self):
        class Foo:
            pass
        assert _is_typed_dict_type(Foo) is False

    def test_typed_dict_true(self):
        class Bar(TypedDict):
            name: str
        assert _is_typed_dict_type(Bar) is True


class TestBuildMinimalSchema:
    def test_basic(self):
        func_info = {
            "params": [
                {"name": "symbol", "required": True, "type": str, "default": None},
                {"name": "limit", "required": False, "type": int, "default": 25},
            ]
        }
        schema = build_minimal_schema(func_info)
        props = schema["parameters"]["properties"]
        assert "symbol" in props
        assert "limit" in props
        assert props["limit"].get("default") == 25
        assert "symbol" in schema["parameters"]["required"]
        assert "$defs" in schema

    def test_empty_params(self):
        schema = build_minimal_schema({"params": []})
        assert schema["parameters"]["properties"] == {}


class TestEnrichSchemaWithSharedDefs:
    def test_empty_schema_builds_from_func_info(self):
        func_info = {"params": [{"name": "x", "required": True, "type": str, "default": None}]}
        schema = enrich_schema_with_shared_defs({}, func_info)
        assert "$defs" in schema
        assert "x" in schema["parameters"]["properties"]

    def test_existing_schema_gets_defs(self):
        schema = {"parameters": {"type": "object", "properties": {"timeframe": {"type": "string"}}}}
        enriched = enrich_schema_with_shared_defs(schema, {"params": []})
        assert "$defs" in enriched
        tf_prop = enriched["parameters"]["properties"]["timeframe"]
        assert tf_prop["$ref"] == "#/$defs/TimeframeSpec"


class TestGetSharedEnumLists:
    def test_has_expected_keys(self):
        enums = get_shared_enum_lists()
        assert "DENOISE_METHODS" in enums
        assert "SIMPLIFY_MODES" in enums
        assert "SIMPLIFY_METHODS" in enums
        assert "PIVOT_METHODS" in enums
        assert "FORECAST_METHODS" in enums

    def test_values_match_tuples(self):
        enums = get_shared_enum_lists()
        assert enums["DENOISE_METHODS"] == list(_DENOISE_METHODS)
        assert enums["SIMPLIFY_MODES"] == list(_SIMPLIFY_MODES)
        assert enums["SIMPLIFY_METHODS"] == list(_SIMPLIFY_METHODS)
        assert enums["PIVOT_METHODS"] == list(_PIVOT_METHODS)


class TestGetFunctionInfo:
    def test_simple_function(self):
        def example(x: int, y: str = "hello") -> bool:
            """A test function."""
            return True

        info = get_function_info(example)
        assert info["name"] == "example"
        assert info["doc"] == "A test function."
        params = info["params"]
        assert len(params) == 2
        assert params[0]["name"] == "x"
        assert params[0]["required"] is True
        assert params[1]["name"] == "y"
        assert params[1]["required"] is False
        assert params[1]["default"] == "hello"

    def test_no_params(self):
        def noop():
            pass
        info = get_function_info(noop)
        assert info["params"] == []

    def test_skips_self(self):
        class Foo:
            def bar(self, x: int) -> None:
                pass
        info = get_function_info(Foo.bar)
        names = [p["name"] for p in info["params"]]
        assert "self" not in names
        assert "x" in names


class TestSchemaConstants:
    """Smoke-test that schema-level constants are well-formed."""

    def test_denoise_methods_are_strings(self):
        assert all(isinstance(m, str) for m in _DENOISE_METHODS)
        assert "ema" in _DENOISE_METHODS

    def test_simplify_modes(self):
        assert "select" in _SIMPLIFY_MODES
        assert "resample" in _SIMPLIFY_MODES

    def test_pivot_methods(self):
        assert "classic" in _PIVOT_METHODS
        assert "fibonacci" in _PIVOT_METHODS
