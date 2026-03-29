import json
import math
import os
import types
from datetime import datetime
from typing import Any, Dict, Optional

from .output_contract import ensure_common_meta
from .runtime_metadata import build_runtime_timezone_meta, _safe_tz_name as _runtime_safe_tz_name
from ..utils.minimal_output import (
    format_result_minimal as _shared_minimal,
    _is_empty_value,
)
from .mt5_gateway import get_default_mt5_gateway

CLI_FORMAT_TOON = "toon"
CLI_FORMAT_JSON = "json"
_JSON_UNSET = object()


def _format_result_minimal(result: Any, verbose: bool = True) -> str:
    try:
        return _shared_minimal(result, verbose=verbose)
    except Exception:
        return str(result) if result is not None else ""


def _json_default(value: Any) -> Any:
    return _sanitize_json_compat(value)


def _json_special_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return str(value)

    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:
            pass

    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return [_sanitize_json_compat(v) for v in value.tolist()]
        if isinstance(value, np.integer):
            return int(value.item())
        if isinstance(value, np.bool_):
            return bool(value.item())
        if isinstance(value, np.floating):
            fv = float(value.item())
            return fv if math.isfinite(fv) else None
    except Exception:
        pass

    return _JSON_UNSET


def _sanitize_json_compat(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _sanitize_json_compat(v) for k, v in value.items()}
    asdict = getattr(value, "_asdict", None)
    if callable(asdict):
        try:
            return _sanitize_json_compat(asdict())
        except Exception:
            pass
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_json_compat(v) for v in value]
    if isinstance(value, types.GeneratorType):
        return [_sanitize_json_compat(v) for v in value]
    if isinstance(value, range):
        return [_sanitize_json_compat(v) for v in value]
    special_value = _json_special_value(value)
    if special_value is not _JSON_UNSET:
        return special_value

    return str(value)


def _normalize_cli_formatter(fmt: Any) -> str:
    raw = str(fmt or CLI_FORMAT_TOON).strip().lower()
    if raw == CLI_FORMAT_JSON:
        return CLI_FORMAT_JSON
    return CLI_FORMAT_TOON


def _resolve_cli_formatter(args: Any) -> str:
    if bool(getattr(args, "json", False)):
        return CLI_FORMAT_JSON
    return CLI_FORMAT_TOON


def _format_result_for_cli(result: Any, *, fmt: str, verbose: bool, cmd_name: str) -> str:
    fmt_s = _normalize_cli_formatter(fmt)
    prepared = _prepare_cli_payload(
        result,
        fmt=fmt_s,
        verbose=verbose,
        cmd_name=cmd_name,
    )
    if fmt_s == CLI_FORMAT_JSON:
        payload = {"text": prepared} if isinstance(prepared, str) else prepared
        payload = _sanitize_json_compat(payload)
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
    if isinstance(prepared, str):
        return prepared
    simplify_numbers = not str(cmd_name or "").startswith("trade_")
    try:
        return _shared_minimal(
            prepared,
            verbose=verbose,
            simplify_numbers=simplify_numbers,
            tool_name=cmd_name,
        )
    except TypeError:
        return _format_result_minimal(prepared, verbose=verbose)


def _safe_tz_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = _runtime_safe_tz_name(value)
    if name:
        return name
    text = str(value).strip()
    return text or None


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
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in ("source", "cache_used", "query_latency_ms", "data_freshness_seconds"):
            if key in diagnostics:
                out[key] = diagnostics.get(key)
    tick_epoch = result.get("time")
    if isinstance(tick_epoch, (int, float)):
        out["tick_time_epoch"] = float(tick_epoch)
        if "data_freshness_seconds" not in out:
            try:
                out["data_freshness_seconds"] = max(0.0, datetime.now().timestamp() - float(tick_epoch))
            except Exception:
                pass
    for field in ("bid", "ask", "spread", "spread_points", "spread_usd"):
        if field in result:
            out[field] = result.get(field)
    try:
        mt5 = get_default_mt5_gateway()
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


def _merge_meta_dict(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        existing = out.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            out[str(key)] = _merge_meta_dict(existing, value)
            continue
        out[str(key)] = value
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


def _normalize_market_ticker_cli_payload(result: Any, *, verbose: bool) -> Any:
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

    out.pop("time_display", None)
    if verbose and not _is_empty_value(raw_epoch):
        out["time_epoch"] = raw_epoch
    else:
        out.pop("time_epoch", None)
    return out


def _normalize_symbols_describe_cli_payload(result: Any, *, verbose: bool) -> Any:
    if not isinstance(result, dict):
        return result

    symbol_in = result.get("symbol")
    if not isinstance(symbol_in, dict):
        return result

    out = dict(result)
    symbol = dict(symbol_in)
    if not verbose:
        symbol.pop("time_epoch", None)
    out["symbol"] = symbol
    return out


def _normalize_candle_json_payload(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    if "bars" in result or not isinstance(result.get("data"), list):
        return result
    out = dict(result)
    out["bars"] = out.pop("data")
    return out


def _prepare_cli_payload(result: Any, *, fmt: str, verbose: bool, cmd_name: str) -> Any:
    prepared = result
    if cmd_name == "market_ticker":
        prepared = _normalize_market_ticker_cli_payload(prepared, verbose=verbose)
    elif cmd_name == "symbols_describe":
        prepared = _normalize_symbols_describe_cli_payload(prepared, verbose=verbose)

    if fmt == CLI_FORMAT_JSON and cmd_name == "data_fetch_candles":
        prepared = _normalize_candle_json_payload(prepared)

    if fmt == CLI_FORMAT_TOON and not verbose:
        prepared = _prune_compact_runtime_meta(prepared)

    return prepared


def _attach_cli_meta(result: Any, *, cmd_name: str, verbose: bool) -> Any:
    return ensure_common_meta(result, tool_name=cmd_name)
