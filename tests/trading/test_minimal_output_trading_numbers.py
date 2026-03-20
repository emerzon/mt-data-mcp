from src.mtdata.utils.minimal_output import format_result_minimal


def test_trading_numbers_no_simplify_when_mixed_scales():
    rows = [
        {
            "Symbol": "BTCUSD",
            "Open Price": 91100.0,
            "SL": 90500.0,
            "TP": 92100.0,
            "Current Price": 92454.0,
            "Volume": 0.02,
        },
        {
            "Symbol": "USDCAD",
            "Open Price": 1.37473,
            "SL": 1.37001,
            "TP": 1.38001,
            "Current Price": 1.37362,
            "Volume": 0.15,
        },
    ]

    simplified = format_result_minimal(rows, simplify_numbers=True)
    assert "1.37473" not in simplified

    fixed = format_result_minimal(rows, simplify_numbers=False)
    assert "1.37473" in fixed
    assert "0.15" in fixed

