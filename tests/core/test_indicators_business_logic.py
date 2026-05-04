from mtdata.core import indicators as core_indicators


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_indicators_list_full_uses_cleaned_summary(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {
                "name": "rsi",
                "category": "momentum",
                "description": (
                    "Python Library Documentation: function rsi in module pandas_ta\n"
                    "rsi(close, length=14)\n"
                    "Relative Strength Index (RSI)\n"
                    "Measures momentum by comparing recent average gains and losses."
                ),
                "params": [{"name": "length", "default": 14}],
                "aliases": [],
            }
        ],
    )

    raw = _unwrap(core_indicators.indicators_list)
    result = raw(search_term="rsi", detail="full")

    row = result["data"][0]
    assert row["summary"] == "Relative Strength Index (RSI)"
    assert "Python Library Documentation" not in row["summary"]
    assert "Python Library Documentation" not in row["description"]
