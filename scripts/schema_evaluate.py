"""Run the public MCP schema evaluation gate."""

from __future__ import annotations

import argparse
import json

from mtdata.core.schema_evaluation import (
    evaluate_public_tool_schemas,
    format_schema_evaluation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare final public tool schemas with runtime signatures and intent rules."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a deterministic JSON report instead of the human-readable report.",
    )
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Enable and include the gated market_depth_fetch tool (fresh process required).",
    )
    args = parser.parse_args()

    report = evaluate_public_tool_schemas(include_gated=args.include_gated)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_schema_evaluation(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
