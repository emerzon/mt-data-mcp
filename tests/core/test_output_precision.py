import pytest

from mtdata.shared.output_precision import (
    normalize_precision_mode,
    resolve_output_precision,
)
from mtdata.utils.minimal_output import format_result_minimal


def test_precision_aliases_normalize_to_canonical_modes():
    assert normalize_precision_mode("display") == "compact"
    assert normalize_precision_mode("raw") == "full"
    assert normalize_precision_mode("auto") == "auto"


def test_invalid_precision_controls_raise_clear_errors():
    with pytest.raises(ValueError, match="precision"):
        normalize_precision_mode("lossy")


def test_json_auto_precision_keeps_full_numbers():
    policy = resolve_output_precision(
        None,
        tool_name="data_fetch_candles",
        fmt="json",
        precision="auto",
    )

    assert policy.simplify_numbers is False


def test_auto_precision_compacts_large_tables_but_not_trading_tools():
    compact = resolve_output_precision(None, tool_name="data_fetch_candles")
    trading = resolve_output_precision(None, tool_name="trade_positions")

    assert compact.simplify_numbers is True
    assert trading.simplify_numbers is False


def test_full_precision_rendering_does_not_display_round_price_fields():
    result = format_result_minimal(
        {"data": [{"close": 1.234567891234}]},
        verbose=False,
        tool_name="data_fetch_candles",
        precision="full",
    )

    assert "1.234567891234" in result



