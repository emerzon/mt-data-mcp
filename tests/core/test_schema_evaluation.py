from __future__ import annotations

from mtdata.core.schema_evaluation import (
    SchemaEvaluationReport,
    SchemaFinding,
    _evaluate_tool,
    format_schema_evaluation,
)


def test_schema_evaluation_report_is_sorted_and_fails_only_on_errors() -> None:
    warning = SchemaFinding(
        "warning",
        "candidate",
        "sample_tool",
        "limit",
        "Potential simplification.",
    )
    report = SchemaEvaluationReport(
        tool_count=91,
        expected_tool_count=91,
        findings=(warning,),
    )

    assert report.ok is True
    assert report.errors == ()
    assert report.warnings == (warning,)
    assert report.to_dict()["warning_count"] == 1
    assert format_schema_evaluation(report).startswith(
        "Schema evaluation PASS: 91/91 tools, 0 errors, 1 warnings"
    )


def test_schema_evaluation_report_error_is_machine_readable() -> None:
    error = SchemaFinding(
        "error",
        "signature_mismatch",
        "sample_tool",
        "symbol",
        "Schema and runtime differ.",
    )
    report = SchemaEvaluationReport(
        tool_count=90,
        expected_tool_count=91,
        findings=(error,),
    )

    assert report.ok is False
    assert report.to_dict()["error_count"] == 1
    assert "sample_tool.symbol" in format_schema_evaluation(report)


def test_schema_evaluation_rejects_generated_placeholder_descriptions() -> None:
    def sample_tool(symbol: str, json: bool = False, output_fields: list[str] | None = None):
        return symbol, json, output_fields

    findings: list[SchemaFinding] = []
    _evaluate_tool(
        "sample_tool",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["symbol"],
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Value for symbol.",
                },
                "json": {
                    "type": "boolean",
                    "default": False,
                    "description": "Return JSON.",
                },
                "output_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Fields to return.",
                },
            },
        },
        sample_tool,
        findings,
    )

    assert any(
        finding.code == "placeholder_description"
        and finding.parameter == "symbol"
        for finding in findings
    )
