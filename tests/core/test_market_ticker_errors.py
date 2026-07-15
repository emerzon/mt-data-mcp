from mtdata.core.market_depth import _describe_symbol_select_error


def test_symbol_selection_does_not_report_success_as_an_error_detail() -> None:
    assert _describe_symbol_select_error("NOTREAL", (1, "Success")) == (
        "Symbol 'NOTREAL' was not found or is not available in MT5."
    )
