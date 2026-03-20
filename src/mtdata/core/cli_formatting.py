import json
import math
import types
from datetime import datetime
from typing import Any, Dict, Optional

from .runtime_metadata import build_runtime_timezone_meta
from ..utils.minimal_output import format_result_minimal as _shared_minimal
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


def _format_result_for_cli(
    result: Any, *, fmt: str, verbose: bool, cmd_name: str
) -> str:
    fmt_s = _normalize_cli_formatter(fmt)
    if fmt_s == CLI_FORMAT_JSON:
        payload = {"text": result} if isinstance(result, str) else result
        payload = _sanitize_json_compat(payload)
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
    if isinstance(result, str):
        return result
    simplify_numbers = not str(cmd_name or "").startswith("trade_")
    try:
        return _shared_minimal(
            result,
            verbose=verbose,
            simplify_numbers=simplify_numbers,
            tool_name=cmd_name,
        )
    except TypeError:
        return _format_result_minimal(result, verbose=verbose)


def _safe_tz_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return getattr(value, "zone", None) or str(value)


def _build_cli_timezone_meta(result: Any) -> Dict[str, Any]:
    return build_runtime_timezone_meta(result)


def _build_cli_timezone_meta_brief(result: Any) -> Dict[str, Any]:
    full = _build_cli_timezone_meta(result)
    return {
        "output": {
            "tz": {
                "value": (
                    (((full.get("output") or {}).get("tz")) or {}).get("value")
                    if isinstance((full.get("output") or {}).get("tz"), dict)
                    else None
                ),
                "hint": (
                    (((full.get("output") or {}).get("tz")) or {}).get("hint")
                    if isinstance((full.get("output") or {}).get("tz"), dict)
                    else None
                ),
            },
        },
        "server": {
            "tz": {
                "value": (
                    ((((full.get("server") or {}).get("tz")) or {}).get("resolved"))
                    or (
                        (((full.get("server") or {}).get("tz")) or {}).get("configured")
                    )
                )
                if isinstance((full.get("server") or {}).get("tz"), dict)
                else None,
            },
            "now": (full.get("server") or {}).get("now")
            if isinstance(full.get("server"), dict)
            else None,
        },
        "client": {
            "tz": {
                "value": (
                    ((((full.get("client") or {}).get("tz")) or {}).get("resolved"))
                    or (
                        (((full.get("client") or {}).get("tz")) or {}).get("configured")
                    )
                )
                if isinstance((full.get("client") or {}).get("tz"), dict)
                else None,
            },
            "now": (full.get("client") or {}).get("now")
            if isinstance(full.get("client"), dict)
            else None,
        },
        "utc": {
            "now": (full.get("utc") or {}).get("now")
            if isinstance(full.get("utc"), dict)
            else None,
        },
    }


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
        for key in (
            "source",
            "cache_used",
            "query_latency_ms",
            "data_freshness_seconds",
        ):
            if key in diagnostics:
                out[key] = diagnostics.get(key)
    tick_epoch = result.get("time")
    if isinstance(tick_epoch, (int, float)):
        out["tick_time_epoch"] = float(tick_epoch)
        if "data_freshness_seconds" not in out:
            try:
                out["data_freshness_seconds"] = max(
                    0.0, datetime.now().timestamp() - float(tick_epoch)
                )
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


def _attach_cli_meta(result: Any, *, cmd_name: str, verbose: bool) -> Any:
    if not isinstance(result, dict):
        return result
    out = dict(result)
    meta = out.get("cli_meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("command", cmd_name or "")
    meta["timezone"] = (
        _build_cli_timezone_meta(result)
        if bool(verbose)
        else _build_cli_timezone_meta_brief(result)
    )
    cmd_diag: Dict[str, Any] = {}
    if bool(verbose) and str(cmd_name or "").strip() == "data_fetch_candles":
        details = _build_candle_cli_verbose_meta(out)
        if details:
            cmd_diag["data_fetch_candles"] = details
    if bool(verbose) and str(cmd_name or "").strip() == "market_ticker":
        details = _build_market_ticker_cli_verbose_meta(out)
        if details:
            cmd_diag["market_ticker"] = details
    if cmd_diag:
        existing = meta.get("command_diagnostics")
        if not isinstance(existing, dict):
            existing = {}
        existing.update(cmd_diag)
        meta["command_diagnostics"] = existing
    out["cli_meta"] = meta
    return out
