# Response and output contract

CLI and [MCP](GLOSSARY.md#mcp-model-context-protocol) expose the canonical tool
payloads described here. The Web API preserves the same domain semantics for
the endpoints it exposes, but is a focused, UI-oriented subset and may return
a more compact representation; see [WEB_API.md](WEB_API.md). This page is the
reference for the canonical model — success/error envelope, `detail`,
`output_fields`, and pagination.

Presentation flags and exit codes: [CLI.md](CLI.md#output-contract).

---

## The response envelope

Successful tool responses are JSON objects that carry a `success` flag plus the tool's data:

```json
{
  "success": true,
  "symbol": "EURUSD",
  "timeframe": "H1",
  "data": [ ... ]
}
```

- `success` — `true` on success, `false` on failure. Always present on failures; most tools also set it on success.
- The remaining keys are tool-specific (`data`, `rows`, `levels`, `forecast`, etc.).
- List-style tools include a pagination block (see [Pagination](#pagination)).

> **Scripting tip:** branch on `success` first, then read the tool-specific fields. On the CLI, also check the [exit code](CLI.md#exit-codes).

### Broker data provenance

Successful MT5 market-data, analytics, and forecast envelopes expose a root
`source` object:

```json
{
  "provider": "mt5",
  "broker_company": "Broker Co",
  "server": "Broker-Demo",
  "source_context_id": "2d86b49c6e6c8b9e",
  "context_available": true
}
```

The context id is a stable digest of the non-secret broker company/server pair,
so detached results can be compared without exposing an account login or
credentials. When account context is unavailable, `provider` remains `mt5` and
`context_available` is false. Method-level lineage such as candle price basis or
tick retrieval method remains in its tool-specific field.

---

## Detail levels (detail)

`detail` controls **how much field-level verbosity** a response carries. The accepted values are:

| Value | Meaning |
|-------|---------|
| `compact` | **Default.** Essential fields only — the slim, token-efficient shared shape used for TOON output. |
| `standard` | Shared stripping is the same as `compact`; an individual tool may provide a distinct standard shape. |
| `summary` | Shared stripping is the same as `compact`; an individual tool may provide a distinct summary shape. |
| `full` | Retains runtime metadata and verbose-only sections that the shared compact strip removes. |

Notes:
- `detail` changes verbosity **within** the sections a tool already returns; it does **not** add new analysis. (For example, `market_snapshot` uses a separate `sections` parameter to choose analysis modules. Its compact/summary envelope reports `sections_summarized`, while standard/full reports `sections_embedded`.)
- The shared output layer has two retention modes: `full`, and the compact strip used by `compact`, `standard`, and `summary`. Tools can independently distinguish the accepted values in their own payloads.
- Use `detail=full` when runtime metadata, diagnostics, request context, or raw supporting rows are needed.

---

## Richer output

Compact output is implicit. Set `detail=full` to retain the richer metadata and
diagnostic sections a tool produces. Tools with a meaningful intermediate or
summary representation expose only those detail values in their schema.

```bash
# Full market-session diagnostics
mtdata-cli market_status --detail full

# Full forecast context
mtdata-cli forecast_generate EURUSD --horizon 12 --detail full --json
```

## Field selection (output_fields)

`output_fields` narrows a response to specific top-level keys or dotted paths.
Combine it with `--json` for token-lean machine parsing:

```bash
mtdata-cli symbols_describe EURUSD --output-fields symbol,digits,point --json
```

Bare names select top-level keys only. Use a dotted path such as
`general_news.title` to select nested values. Any requested path that is not
present is returned in `unresolved_output_fields`; projection never silently
searches unrelated nested objects for a matching key.

`json` and `output_fields` are the shared output-shaping parameters available
across tools. A domain-specific parameter named `fields` (currently used by
Finviz fundamentals) selects source data and is not response projection.

---

## Pagination

List-style tools return a normalized pagination block so you can page deterministically:

```json
{
  "total": 420,
  "returned": 50,
  "offset": 0,
  "limit": 50,
  "has_more": true,
  "more_available": 370
}
```

| Field | Meaning |
|-------|---------|
| `total` | Exact rows available before paging, or `null` when the provider cannot determine it |
| `total_lower_bound` | Present only when `total` is unknown; minimum rows known to exist |
| `returned` | Rows in this response |
| `offset` | Zero-based start index of this page |
| `limit` | Page size requested (`null` when unbounded) |
| `has_more` | `true` when more rows remain after this page |
| `more_available` | Exact count of rows remaining, or `null` when `total` is unknown |

When a bounded provider can only prove that another row exists, `total` and
`more_available` remain `null`; `total_lower_bound` and `has_more` carry the
available evidence without presenting a page-size-dependent estimate as an
exact universe count.

The `pagination` object is authoritative and is the only pagination
representation in canonical payloads. Root-level `total_count`, `offset`,
`limit`, `page`, `pages`, `has_more`, and `more_available` aliases are not
emitted. A root `count` may still describe the size of the returned collection.
Tools that accept a one-based `page` input convert it to the zero-based
`pagination.offset` value.

Page through results with `--offset` and `--limit`:

```bash
mtdata-cli tools_list --category forecast --limit 20 --offset 0 --json
mtdata-cli tools_list --category forecast --limit 20 --offset 20 --json
```

---

## Error envelope

Failures return a **structured** payload (not just a string) so callers can react programmatically:

```json
{
  "success": false,
  "error": "Symbol NOTAREALPAIR not found.",
  "error_code": "symbol_not_found",
  "request_id": "b0f3…",
  "operation": "symbols_describe",
  "remediation": "Use symbols_list to browse available broker symbols.",
  "related_tools": ["symbols_list"],
  "valid_values": { ... },
  "example": "mtdata-cli symbols_describe EURUSD",
  "documentation": "docs/CLI.md",
  "details": { ... }
}
```

| Field | Always present | Meaning |
|-------|:---:|---------|
| `success` | ✅ | Always `false` on errors |
| `error` | ✅ | Human-readable message |
| `error_code` | ✅ | Stable lowercase machine-readable code (e.g. `symbol_not_found`, `mt5_connection_error`) |
| `request_id` | ✅ | Correlation id for logs |
| `operation` | | The tool that failed |
| `remediation` | | Suggested fix |
| `related_tools` | | Tools that can help |
| `valid_values` | | Accepted values when the failure was a bad argument |
| `example` | | A corrected example invocation |
| `documentation` | | Relevant doc pointer |
| `details` | | Structured, tool-specific context |

Prefer `error_code` over string-matching `error` when you need to branch on failure type. On the CLI, tool/provider failures share [exit code `1`](CLI.md#exit-codes), so parse `error_code` to distinguish them.

For `symbol_not_found`, market-data tools consistently include
`details.did_you_mean`, an ordered array of broker catalog candidates with
`symbol` and optional `description`/`group` fields. The field is present as an
empty array when no candidate matches, so callers do not need to parse names
from the human-readable error string.

---

## Freshness and execution readiness

`usable_for_live_trading` is reserved for execution-oriented quote and session
outputs and is accompanied by `usable_for_live_trading_basis`:

- `quote_age_and_market_session` is an execution-quote check. Its default age
  threshold is 30 seconds, matching pre-trade validation.
- Historical bars, forecasts, volatility estimates, and research backtests do
  not publish this execution-sounding boolean. Use `history_policy_ok`,
  `signal_status`, and `usage` for their respective contracts, then obtain a
  current quote before execution.
- Combined forecast outputs may require both model-history and reference-quote
  readiness and expose `execution_blockers` when either input fails.

---

## TOON vs JSON

The canonical payload above is what you get with `--json`. Without `--json`, the CLI renders the same payload as compact **TOON** text and applies `--precision auto`. Format and precision are presentation-only and never change the underlying values. See [CLI.md](CLI.md#output-contract) for details, and set `MTDATA_OUTPUT_FORMAT=json` to default all output to JSON.

---

## See Also

- [CLI.md](CLI.md#output-contract) — TOON/JSON, `--precision`, exit codes
- [ENV_VARS.md](ENV_VARS.md) — `MTDATA_OUTPUT_FORMAT` and related settings
- [WEB_API.md](WEB_API.md) — how the same payloads are served over REST
