from __future__ import annotations

from mtdata.core.schema_evaluation import (
    SchemaEvaluationReport,
    SchemaFinding,
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
