# Response and output contract

One of mtdata’s strengths: **CLI, [MCP](GLOSSARY.md#mcp-model-context-protocol), and Web API share the same payload meaning**. Transports only change presentation ([TOON](GLOSSARY.md#toon) vs JSON, HTTP status), not the underlying data model. This page is the reference for that model — success/error envelope, `detail`, `extras`, field selection, and pagination.

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

---

## Detail levels (detail)

`detail` controls **how much field-level verbosity** a response carries. The accepted values are:

| Value | Meaning |
|-------|---------|
| `compact` | **Default.** Essential fields only — the slim, token-efficient shape used for TOON output. |
| `standard` | Adds moderate context to each row/section (e.g. `tools_list` includes per-parameter summaries). |
| `summary` | A condensed summary form for tools that support it. |
| `full` | Everything `compact` returns **plus** runtime metadata and verbose-only sections. |

Notes:
- `detail` changes verbosity **within** the sections a tool already returns; it does **not** add new analysis. (For example, `market_snapshot` uses a separate `sections` parameter to choose analysis modules — `detail` only controls verbosity inside them.)
- Not every tool distinguishes all four levels; unsupported values resolve to the nearest supported shape (`compact` or `full`).
- Requesting any [extras](#richer-sections-extras) automatically shapes the response as `full`.

---

## Richer sections (extras)

Compact output is implicit. To opt into heavier, optional sections, pass `extras`.
The canonical tokens are listed below. Support is tool-specific: a token preserves
or enables that section when the selected tool produces it; it does not synthesize
metadata or diagnostics that the tool cannot provide. Use `tools_list` to inspect a
tool's parameters and documentation.

| Token | Adds |
|-------|------|
| `metadata` | Runtime/context metadata (service, tool, timing, time-normalization notes) |
| `diagnostics` | Diagnostic detail about how the result was produced |
| `request` | The echoed, resolved request context (parameters the tool actually used) |
| `raw` | Raw/unshaped payload values |
| `raw_rows` | Raw underlying rows behind a summarized table |
| `method_docs` | Inline documentation for the selected method/indicator |
| `guidance` | Suggested next steps and related tools |

The alias `all` requests every canonical section; the response includes the sections
supported by that tool.

```bash
# Just metadata + diagnostics
mtdata-cli market_status --extras metadata,diagnostics

# Everything
mtdata-cli forecast_generate EURUSD --horizon 12 --extras all --json
```

`detail` and `extras` are complementary: `detail` tunes verbosity of the sections a
tool already returns, while `extras` asks the tool to include supported optional
sections. Any non-empty `extras` request also preserves the full response shape.

---

## Field selection (fields)

`fields` narrows a response to a specific set of top-level keys (or row columns), which is useful for token-lean scripting. Combine it with `--json` for machine parsing:

```bash
mtdata-cli symbols_describe EURUSD --fields symbol,digits,point --json
```

`json`, `extras`, and `fields` are the shared output-shaping parameters available across tools.

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
| `total` | Total rows available before paging |
| `returned` | Rows in this response |
| `offset` | Zero-based start index of this page |
| `limit` | Page size requested (`null` when unbounded) |
| `has_more` | `true` when more rows remain after this page |
| `more_available` | Count of rows remaining after this page |

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
  "error_code": "SYMBOL_NOT_FOUND",
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
| `error_code` | ✅ | Stable machine-readable code (e.g. `SYMBOL_NOT_FOUND`, `MT5_CONNECTION`) |
| `request_id` | ✅ | Correlation id for logs |
| `operation` | | The tool that failed |
| `remediation` | | Suggested fix |
| `related_tools` | | Tools that can help |
| `valid_values` | | Accepted values when the failure was a bad argument |
| `example` | | A corrected example invocation |
| `documentation` | | Relevant doc pointer |
| `details` | | Structured, tool-specific context |

Prefer `error_code` over string-matching `error` when you need to branch on failure type. On the CLI, tool/provider failures share [exit code `1`](CLI.md#exit-codes), so parse `error_code` to distinguish them.

---

## TOON vs JSON

The canonical payload above is what you get with `--json`. Without `--json`, the CLI renders the same payload as compact **TOON** text and applies `--precision auto`. Format and precision are presentation-only and never change the underlying values. See [CLI.md](CLI.md#output-contract) for details, and set `MTDATA_OUTPUT_FORMAT=json` to default all output to JSON.

---

## See Also

- [CLI.md](CLI.md#output-contract) — TOON/JSON, `--precision`, exit codes
- [ENV_VARS.md](ENV_VARS.md) — `MTDATA_OUTPUT_FORMAT` and related settings
- [WEB_API.md](WEB_API.md) — how the same payloads are served over REST
