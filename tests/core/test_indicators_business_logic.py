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


def test_indicators_list_default_prioritizes_common_indicators(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {"name": "cdl_doji", "category": "candles", "description": "", "params": []},
            {"name": "ema", "category": "overlap", "description": "", "params": []},
            {"name": "rsi", "category": "momentum", "description": "", "params": []},
            {"name": "zscore", "category": "statistics", "description": "", "params": []},
        ],
    )

    raw = _unwrap(core_indicators.indicators_list)

    default = raw()
    filtered = raw(category="candles")

    assert [row["name"] for row in default["data"]] == ["rsi", "ema", "cdl_doji", "zscore"]
    assert [row["name"] for row in filtered["data"]] == ["cdl_doji"]


def test_indicators_list_discloses_trading_style_filter_basis(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {
                "name": "rsi",
                "category": "momentum",
                "description": "Relative Strength Index.",
                "params": [],
            },
            {
                "name": "coppock",
                "category": "momentum",
                "description": "Designed for use on a monthly time scale.",
                "params": [],
            },
            {
                "name": "sma",
                "category": "overlap",
                "description": "Simple moving average.",
                "params": [],
            },
        ],
    )

    raw = _unwrap(core_indicators.indicators_list)
    result = raw(trading_style="intraday", detail="full", limit=20)

    assert result["trading_style_filter"] == {
        "requested": "intraday",
        "semantics": "broad_workflow_tag_not_performance_recommendation",
        "curated_indicator_matches": 1,
        "category_heuristic_matches": 1,
        "unknown_basis_matches": 0,
    }
    assert "1 match(es) inherit" in result["warnings"][0]
    rows = {row["name"]: row for row in result["data"]}
    assert rows["rsi"]["trading_context"]["trading_styles_basis"] == (
        "curated_indicator"
    )
    assert rows["coppock"]["trading_context"]["trading_styles_basis"] == (
        "category_heuristic"
    )
    assert "not an indicator-specific recommendation" in rows["coppock"][
        "trading_context"
    ]["trading_styles_note"]
