from unittest.mock import patch


def test_template_minimal_keeps_only_core_sections() -> None:
    base_report = {
        "meta": {"symbol": "EURUSD", "template": "basic", "timeframe": "H1"},
        "sections": {
            "context": {"close": 1.1},
            "pivot": {"levels": []},
            "forecast": {"method": "ema"},
            "barriers": {"long": {"best": {"edge": 0.1}}},
            "patterns": {"recent": []},
        },
        "diagnostics": {"warnings": ["x"]},
    }

    with patch(
        "mtdata.core.report_templates.minimal.template_basic",
        return_value=base_report,
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal("EURUSD", 12, None, None)

    assert report["meta"]["template"] == "minimal"
    assert list(report["sections"].keys()) == ["context", "forecast", "barriers"]
    assert report["sections"]["forecast"]["method"] == "ema"
    assert report["diagnostics"] == {"warnings": ["x"]}


def test_template_minimal_handles_basic_string_error() -> None:
    with patch(
        "mtdata.core.report_templates.minimal.template_basic",
        return_value="bad payload",
    ):
        from mtdata.core.report_templates.minimal import template_minimal

        report = template_minimal("EURUSD", 12, None, None)

    assert report["error"] == "template_basic returned string: bad payload"
