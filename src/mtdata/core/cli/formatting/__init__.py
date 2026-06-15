import json
import os
import time
from typing import Any, Callable, Dict, Optional

from ....shared.output_precision import resolve_output_precision
from ....utils.minimal_output import _is_empty_value
from ....utils.minimal_output import format_result_minimal as _shared_minimal
from ...mt5_gateway import create_mt5_gateway
from ...output_contract import apply_output_verbosity
from ...output_serialization import json_default as _json_default
from ...output_serialization import sanitize_json as _sanitize_json
from ...runtime_metadata import build_runtime_timezone_meta

CLI_FORMAT_TOON = "toon"
CLI_FORMAT_JSON = "json"


def _format_result_minimal(result: Any, verbose: bool = True) -> str:
    try:
        return _shared_minimal(result, verbose=verbose)
    except Exception:
        return str(result) if result is not None else ""


def _normalize_cli_formatter(fmt: Any) -> str:
    raw = str(fmt or CLI_FORMAT_TOON).strip().lower()
    if raw == CLI_FORMAT_JSON:
        return CLI_FORMAT_JSON
    return CLI_FORMAT_TOON


def _resolve_cli_formatter(args: Any) -> str:
    if bool(getattr(args, "json", False)):
        return CLI_FORMAT_JSON
    env_format = os.environ.get("MTDATA_OUTPUT_FORMAT")
    if env_format is not None:
        return _normalize_cli_formatter(env_format)
    return CLI_FORMAT_TOON


def _format_result_for_cli(
    result: Any,
    *,
    fmt: str,
    verbose: bool,
    cmd_name: str,
    precision: Any = None,
) -> str:
    fmt_s = _normalize_cli_formatter(fmt)
    precision_policy = resolve_output_precision(
        None,
        tool_name=cmd_name,
        fmt=fmt_s,
        precision=precision,
    )
    prepared = _prepare_cli_payload(
        result,
        fmt=fmt_s,
        verbose=verbose,
        cmd_name=cmd_name,
        precision=precision_policy.mode,
    )
    if fmt_s == CLI_FORMAT_JSON:
        payload = {"text": prepared} if isinstance(prepared, str) else prepared
        payload = _sanitize_json(
            payload,
            compact_numbers=precision_policy.simplify_numbers,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
    if isinstance(prepared, str):
        return prepared
    try:
        return _shared_minimal(
            prepared,
            verbose=verbose,
            precision=precision_policy.mode,
            tool_name=cmd_name,
        )
    except TypeError:
        return _format_result_minimal(prepared, verbose=verbose)


def _build_cli_timezone_meta(result: Any) -> Dict[str, Any]:
    return build_runtime_timezone_meta(result)


def _build_cli_timezone_meta_brief(result: Any) -> Dict[str, Any]:
    full = _build_cli_timezone_meta(result)
    out: Dict[str, Any] = {}
    utc_meta = full.get("utc")
    if isinstance(utc_meta, dict):
        out["utc"] = {"tz": utc_meta.get("tz"), "now": utc_meta.get("now")}
    server_meta = full.get("server")
    if isinstance(server_meta, dict):
        out["server"] = {
            "source": server_meta.get("source"),
            "tz": server_meta.get("tz"),
            "offset_seconds": server_meta.get("offset_seconds"),
            "now": server_meta.get("now"),
        }
    client_meta = full.get("client")
    if isinstance(client_meta, dict):
        out["client"] = {"tz": client_meta.get("tz"), "now": client_meta.get("now")}
    return out


def _build_candle_cli_verbose_meta(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    out: Dict[str, Any] = {}
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    if isinstance(diagnostics, dict):
        for key in ("query", "indicators", "denoise", "session_gaps", "simplify"):
            value = diagnostics.get(key)
            if isinstance(value, dict):
                out[key] = dict(value)
    warnings = result.get("warnings")
    if isinstance(warnings, list) and warnings:
        out["warnings"] = [str(w) for w in warnings]
    session_gaps = result.get("session_gaps")
    if isinstance(session_gaps, list) and session_gaps:
        out["session_gaps_preview"] = session_gaps[:3]
    return out


def _build_market_ticker_cli_verbose_meta(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    out: Dict[str, Any] = {}
    meta = result.get("meta")
    diagnostics = meta.get("diagnostics") if isinstance(meta, dict) else None
    if not isinstance(diagnostics, dict):
        diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in (
            "source",
            "cache_used",
            "query_latency_ms",
            "data_freshness_seconds",
            "data_freshness_anchor",
            "data_freshness_metric",
        ):
            if key in diagnostics:
                out[key] = diagnostics.get(key)
    tick_epoch = result.get("time")
    if isinstance(tick_epoch, (int, float)):
        out["tick_time_epoch"] = float(tick_epoch)
        if "data_freshness_seconds" not in out:
            try:
                out["data_freshness_seconds"] = max(
                    0.0,
                    time.time() - float(tick_epoch),
                )
                out["data_freshness_anchor"] = "wall_clock"
                out["data_freshness_metric"] = "last_tick_age_seconds"
            except Exception:
                pass
    for field in ("bid", "ask", "spread", "spread_points", "spread_cost_per_lot"):
        if field in result:
            out[field] = result.get(field)
    try:
        mt5 = create_mt5_gateway()
        terminal = mt5.terminal_info() if hasattr(mt5, "terminal_info") else None
        if terminal is not None:
            out["terminal"] = {
                "connected": bool(getattr(terminal, "connected", False)),
                "trade_allowed": bool(getattr(terminal, "trade_allowed", False)),
                "ping_last": getattr(terminal, "ping_last", None),
            }
    except Exception:
        pass
    return out


def _prune_compact_runtime_meta(result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    meta_in = result.get("meta")
    if not isinstance(meta_in, dict):
        return result

    out = dict(result)
    meta = dict(meta_in)
    runtime_in = meta.get("runtime")
    if isinstance(runtime_in, dict):
        runtime = dict(runtime_in)
        timezone_in = runtime.get("timezone")
        if isinstance(timezone_in, dict):
            used_in = timezone_in.get("used")
            if isinstance(used_in, dict) and not _is_empty_value(used_in):
                runtime["timezone"] = {"used": dict(used_in)}
            else:
                runtime.pop("timezone", None)
        else:
            runtime.pop("timezone", None)
        if runtime:
            meta["runtime"] = runtime
        else:
            meta.pop("runtime", None)

    if set(meta.keys()) <= {"tool"}:
        out.pop("meta", None)
        return out

    if meta:
        out["meta"] = meta
    else:
        out.pop("meta", None)
    return out


def _round_cli_float(value: Any, *, digits: int) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    try:
        return round(float(value), int(digits))
    except Exception:
        return value


def _price_precision_from_cli_quote(quote: Any) -> Optional[int]:
    if not isinstance(quote, dict):
        return None
    for key in ("price_precision", "digits"):
        try:
            raw = quote.get(key)
            if raw is not None:
                return max(0, int(raw))
        except Exception:
            continue
    return None


def _fixed_cli_quote_price(value: Any, *, digits: int) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    try:
        return f"{float(value):.{max(0, int(digits))}f}"
    except Exception:
        return value


def _normalize_market_ticker_cli_payload(
    result: Any,
    *,
    verbose: bool,
    compact_numbers: bool = False,
) -> Any:
    if not isinstance(result, dict):
        return result

    out = dict(result)
    display_time = out.get("time_display")
    raw_epoch = out.get("time_epoch")
    if _is_empty_value(raw_epoch):
        epoch_candidate = out.get("time")
        if isinstance(epoch_candidate, (int, float)):
            raw_epoch = epoch_candidate

    canonical_time = display_time
    if _is_empty_value(canonical_time):
        canonical_time = out.get("time")

    if not _is_empty_value(canonical_time):
        out["time"] = canonical_time
    else:
        out.pop("time", None)

    if compact_numbers:
        for field, digits in (
            ("spread_points", 4),
            ("spread_pips", 4),
            ("spread_pct", 6),
            ("spread_cost_per_lot", 6),
        ):
            if field in out:
                out[field] = _round_cli_float(out.get(field), digits=digits)

    out.pop("time_display", None)
    out.pop("spread_display", None)
    out.pop("spread_pct_display", None)
    out.pop("data_age_hours", None)
    if not verbose:
        primary_spread_key = next(
            (
                key
                for key in ("spread_pips", "spread_points", "spread")
                if not _is_empty_value(out.get(key))
            ),
            None,
        )
        for field in ("spread", "spread_points", "spread_pips", "spread_pct"):
            if field != primary_spread_key:
                out.pop(field, None)
    if verbose and not _is_empty_value(raw_epoch):
        out["time_epoch"] = raw_epoch
    else:
        out.pop("time_epoch", None)
    return out


def _compact_trade_session_items(
    section: Any,
    *,
    field_map: tuple[tuple[str, ...], ...],
) -> Optional[list[Dict[str, Any]]]:
    if not isinstance(section, dict):
        return None
    items = section.get("items")
    if not isinstance(items, list) or not items:
        return None

    rows: list[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact: Dict[str, Any] = {}
        for out_key, *input_keys in field_map:
            for input_key in input_keys:
                if input_key in item and not _is_empty_value(item.get(input_key)):
                    compact[out_key] = item.get(input_key)
                    break
        if compact:
            rows.append(compact)
    return rows or None


def _normalize_trade_session_context_cli_payload(
    result: Any,
    *,
    verbose: bool,
    compact_numbers: bool = False,
    format_quote_prices: bool = False,
) -> Any:
    if not isinstance(result, dict):
        return result

    out = dict(result)
    quote_in = out.get("quote")
    if isinstance(quote_in, dict):
        quote_norm = _normalize_market_ticker_cli_payload(
            quote_in,
            verbose=verbose,
            compact_numbers=compact_numbers,
        )
        if verbose:
            out["quote"] = quote_norm
        else:
            compact_quote = {
                key: quote_norm.get(key)
                for key in (
                    "bid",
                    "ask",
                    "last",
                    "spread",
                    "spread_points",
                    "spread_pips",
                    "spread_pct",
                    "spread_cost_per_lot",
                    "spread_cost_currency",
                    "time",
                    "timezone",
                    "data_age_seconds",
                    "data_age",
                    "data_stale",
                    "stale_warning",
                    "warning",
                )
                if key in quote_norm and not _is_empty_value(quote_norm.get(key))
            }
            price_precision = _price_precision_from_cli_quote(quote_norm)
            if format_quote_prices and price_precision is not None:
                for key in ("bid", "ask", "last", "spread"):
                    if key in compact_quote:
                        compact_quote[key] = _fixed_cli_quote_price(
                            compact_quote.get(key),
                            digits=price_precision,
                        )
            if "error" in quote_norm and not _is_empty_value(quote_norm.get("error")):
                compact_quote = {"error": quote_norm.get("error")}
            if compact_quote:
                out["quote"] = compact_quote
            else:
                out.pop("quote", None)

    if verbose:
        return out

    compact_out: Dict[str, Any] = {}
    for key in (
        "success",
        "symbol",
        "state",
        "state_scope",
        "portfolio_positions_count",
        "other_positions_count",
        "partial_failure",
    ):
        value = out.get(key)
        if not _is_empty_value(value):
            compact_out[key] = value

    account_in = out.get("account")
    if isinstance(account_in, dict):
        if "error" in account_in and not _is_empty_value(account_in.get("error")):
            compact_out["account"] = {"error": account_in.get("error")}
        else:
            account_out = {
                key: account_in.get(key)
                for key in ("balance", "equity", "profit", "margin_level")
                if key in account_in and not _is_empty_value(account_in.get(key))
            }
            execution_ready = account_in.get("execution_ready")
            if execution_ready is not None and not bool(execution_ready):
                account_out["execution_ready"] = False
            if account_out:
                compact_out["account"] = account_out

    if "quote" in out:
        compact_out["quote"] = out["quote"]

    volume_units: Dict[str, str] = {}
    open_positions_in = out.get("open_positions")
    if isinstance(open_positions_in, list):
        compact_rows = [
            row for row in open_positions_in if isinstance(row, dict) and row
        ]
        compact_out["open_positions"] = compact_rows
        compact_out["open_positions_count"] = len(compact_rows)
        if compact_rows:
            volume_units["volume"] = "lots"
    if isinstance(open_positions_in, dict):
        if "error" in open_positions_in and not _is_empty_value(
            open_positions_in.get("error")
        ):
            compact_out["open_positions"] = {"error": open_positions_in.get("error")}
            if not _is_empty_value(open_positions_in.get("count")):
                compact_out["open_positions_count"] = open_positions_in.get("count")
        else:
            compact_rows = _compact_trade_session_items(
                open_positions_in,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("type", "type", "Type"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("profit", "profit", "Profit"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            compact_out["open_positions"] = compact_rows or []
            compact_out["open_positions_count"] = int(
                open_positions_in.get("count") or 0
            )
            if compact_out["open_positions_count"] > 0:
                volume_units["volume"] = "lots"

    pending_orders_in = out.get("pending_orders")
    if isinstance(pending_orders_in, list):
        compact_rows = [
            row for row in pending_orders_in if isinstance(row, dict) and row
        ]
        compact_out["pending_orders"] = compact_rows
        compact_out["pending_orders_count"] = len(compact_rows)
        if compact_rows:
            volume_units["volume"] = "lots"
    if isinstance(pending_orders_in, dict):
        if "error" in pending_orders_in and not _is_empty_value(
            pending_orders_in.get("error")
        ):
            compact_out["pending_orders"] = {"error": pending_orders_in.get("error")}
            if not _is_empty_value(pending_orders_in.get("count")):
                compact_out["pending_orders_count"] = pending_orders_in.get("count")
        else:
            compact_rows = _compact_trade_session_items(
                pending_orders_in,
                field_map=(
                    ("symbol", "symbol", "Symbol"),
                    ("ticket", "ticket", "Ticket"),
                    ("time", "time", "Time"),
                    ("expiration", "expiration", "Expiration"),
                    ("type", "type", "Type"),
                    ("volume", "volume", "Volume"),
                    ("price_open", "price_open", "open_price", "Open Price"),
                    (
                        "price_current",
                        "price_current",
                        "current_price",
                        "Current Price",
                    ),
                    ("sl", "sl", "SL"),
                    ("tp", "tp", "TP"),
                    ("timezone", "timezone", "Timezone"),
                ),
            )
            compact_out["pending_orders"] = compact_rows or []
            compact_out["pending_orders_count"] = int(
                pending_orders_in.get("count") or 0
            )
            if compact_out["pending_orders_count"] > 0:
                volume_units["volume"] = "lots"

    if volume_units:
        compact_out["units"] = volume_units

    show_all_hint = out.get("show_all_hint")
    if not _is_empty_value(show_all_hint):
        compact_out["show_all_hint"] = show_all_hint
    return compact_out


def _normalize_symbols_describe_cli_payload(result: Any, *, verbose: bool) -> Any:
    if not isinstance(result, dict):
        return result

    details_in = result.get("details")
    if isinstance(details_in, dict):
        out = dict(result)
        details = dict(details_in)
        if not verbose:
            details.pop("time_epoch", None)
            out.pop("meta", None)
        out["details"] = details
        return out

    symbol_in = result.get("symbol")
    if not isinstance(symbol_in, dict):
        return result

    out = dict(result)
    symbol = dict(symbol_in)
    if not verbose:
        symbol.pop("time_epoch", None)
        out.pop("meta", None)
    out["symbol"] = symbol
    return out


def _normalize_market_scan_cli_payload(result: Any, *, verbose: bool) -> Any:
    if not isinstance(result, dict) or verbose:
        return result
    if "error" in result:
        return result

    compact_out: Dict[str, Any] = {}
    for key in (
        "success",
        "count",
        "returned_count",
        "total_count",
        "requested_limit",
        "offset",
        "has_more",
        "rank_by",
        "rank_order",
        "ranking",
        "price_change_basis",
        "headers",
        "data",
        "freshness",
        "stale_rows",
        "data_as_of",
        "session_status",
        "units",
        "summary",
        "no_action",
        "message",
    ):
        value = result.get(key)
        if not _is_empty_value(value):
            compact_out[key] = value
    return compact_out or result


def _rows_to_table(rows: Any) -> Any:
    if not isinstance(rows, list):
        return rows
    columns: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            col = str(key)
            if col not in columns:
                columns.append(col)
    if not columns:
        return {"columns": [], "rows": []}
    return {
        "columns": columns,
        "rows": [
            [row.get(col) if isinstance(row, dict) else None for col in columns]
            for row in rows
        ],
    }


def _normalize_candle_cli_payload(result: Any, *, fmt: str) -> Any:
    if not isinstance(result, dict):
        return result
    out = dict(result)
    if fmt == CLI_FORMAT_TOON:
        meta_in = out.get("meta")
        if isinstance(meta_in, dict):
            meta: Dict[str, Any] = {}
            used_tz = None
            runtime_in = meta_in.get("runtime")
            if isinstance(runtime_in, dict):
                timezone_in = runtime_in.get("timezone")
                if isinstance(timezone_in, dict):
                    used_in = timezone_in.get("used")
                    if isinstance(used_in, dict):
                        used_tz = used_in.get("tz")
            for key, value in meta_in.items():
                if key in {"runtime", "tool"} or _is_empty_value(value):
                    continue
                meta[key] = dict(value) if isinstance(value, dict) else value
            if used_tz:
                meta["timezone"] = used_tz
            if meta:
                out["meta"] = meta
            else:
                out.pop("meta", None)
    return out


def _render_mt5_news_cli_compact(result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    news_rows = result.get("news")
    if not isinstance(news_rows, list):
        return result

    lines: list[str] = []
    success = result.get("success")
    if isinstance(success, bool):
        lines.append(f"success: {str(success).lower()}")

    for key in ("count", "total_records", "database_path"):
        value = result.get(key)
        if _is_empty_value(value):
            continue
        lines.append(f"{key}: {value}")

    lines.append(f"news[{len(news_rows)}]:")
    for row in news_rows:
        if not isinstance(row, dict):
            continue
        relative_time = str(row.get("relative_time") or "").strip()
        subject = str(row.get("subject") or "").strip()
        if not relative_time and not subject:
            continue
        if relative_time and subject:
            lines.append(f"  {relative_time}  {subject}")
        else:
            lines.append(f"  {relative_time or subject}")

    return "\n".join(lines)


_CLI_PRESENTATION_ADAPTERS: Dict[str, Callable[..., Any]] = {
    "market_ticker": _normalize_market_ticker_cli_payload,
    "trade_session_context": _normalize_trade_session_context_cli_payload,
    "symbols_describe": _normalize_symbols_describe_cli_payload,
    "market_scan": _normalize_market_scan_cli_payload,
    "data_fetch_candles": _normalize_candle_cli_payload,
}


def _prepare_cli_payload(
    result: Any,
    *,
    fmt: str,
    verbose: bool,
    cmd_name: str,
    precision: Any = None,
) -> Any:
    prepared = result
    compact_numbers = resolve_output_precision(
        None,
        tool_name=cmd_name,
        fmt=fmt,
        precision=precision,
    ).simplify_numbers
    adapter = _CLI_PRESENTATION_ADAPTERS.get(cmd_name)
    if adapter is _normalize_market_ticker_cli_payload:
        prepared = adapter(
            prepared,
            verbose=verbose,
            compact_numbers=compact_numbers,
        )
    elif adapter is _normalize_trade_session_context_cli_payload:
        prepared = adapter(
            prepared,
            verbose=verbose,
            compact_numbers=compact_numbers,
            format_quote_prices=fmt == CLI_FORMAT_TOON,
        )
    elif adapter is _normalize_symbols_describe_cli_payload:
        prepared = adapter(prepared, verbose=verbose)
    elif adapter is _normalize_market_scan_cli_payload and fmt == CLI_FORMAT_TOON:
        prepared = adapter(prepared, verbose=verbose)

    if fmt == CLI_FORMAT_TOON and cmd_name == "mt5_news" and not verbose:
        return _render_mt5_news_cli_compact(prepared)

    if adapter is _normalize_candle_cli_payload:
        prepared = adapter(prepared, fmt=fmt)

    if fmt == CLI_FORMAT_TOON and not verbose:
        prepared = _prune_compact_runtime_meta(prepared)

    return prepared


def _attach_cli_meta(result: Any, *, cmd_name: str, verbose: bool) -> Any:
    detail = "full" if verbose else "compact"
    if cmd_name == "news" and isinstance(result, dict):
        from ...news import normalize_news_output

        result = normalize_news_output(result, detail=detail)
    return apply_output_verbosity(result, tool_name=cmd_name, detail=detail)
