from __future__ import annotations

from typing import Any, Iterator, Optional

from ..utils.utils import _UNPARSED_BOOL, _parse_bool_like
from .runtime_metadata import build_runtime_timezone_meta

_MISSING = object()
_VERBOSE_ONLY_KEYS = frozenset({"meta", "diagnostics", "debug", "debug_info"})
_VERBOSE_DETAIL_LEVELS = frozenset({"full"})
_COMPACT_DETAIL_LEVELS = frozenset({"compact", "summary", "summary_only"})


def _strip_verbose_only_fields(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, subvalue in value.items():
            if str(key) in _VERBOSE_ONLY_KEYS:
                continue
            out[key] = _strip_verbose_only_fields(subvalue)
        return out
    if isinstance(value, list):
        return [_strip_verbose_only_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_verbose_only_fields(item) for item in value)
    return value


def _iter_verbosity_sources(source: Any) -> Iterator[Any]:
    yield source
    if not isinstance(source, dict):
        return
    for candidate in source.values():
        if any(hasattr(candidate, field) for field in ("verbose", "detail", "concise")):
            yield candidate


def _read_verbosity_field(source: Any, field: str) -> Any:
    if isinstance(source, dict):
        return source.get(field, _MISSING)
    return getattr(source, field, _MISSING)


def _coerce_optional_verbose_flag(value: Any) -> Optional[bool]:
    if value is _MISSING or value is None:
        return None
    parsed = _parse_bool_like(value, allow_none=True)
    if parsed is _UNPARSED_BOOL:
        return bool(value)
    if parsed is None:
        return None
    return bool(parsed)


def resolve_requested_output_verbosity(source: Any, *, default: bool = False) -> bool:
    """Resolve shared output verbosity from explicit and legacy detail controls."""
    saw_explicit_false = False
    for candidate in _iter_verbosity_sources(source):
        verbose = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "verbose"))
        if verbose is True:
            return True
        if verbose is False:
            saw_explicit_false = True

    for candidate in _iter_verbosity_sources(source):
        detail_value = _read_verbosity_field(candidate, "detail")
        if detail_value is _MISSING or detail_value is None:
            continue
        normalized = str(detail_value).strip().lower()
        if normalized in _VERBOSE_DETAIL_LEVELS:
            return True
        if normalized in _COMPACT_DETAIL_LEVELS:
            return False

    for candidate in _iter_verbosity_sources(source):
        concise = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "concise"))
        if concise is True:
            return False

    if saw_explicit_false:
        return False

    return bool(default)


def ensure_common_meta(
    result: Any,
    *,
    tool_name: Optional[str] = None,
    mt5_config: Any = None,
) -> Any:
    """Attach the shared output contract metadata without interface-specific wrappers."""
    if not isinstance(result, dict):
        return result

    out = dict(result)
    out.pop("cli_meta", None)

    meta_in = out.get("meta")
    meta = dict(meta_in) if isinstance(meta_in, dict) else {}

    normalized_tool = str(tool_name or "").strip()
    if normalized_tool and not str(meta.get("tool") or "").strip():
        meta["tool"] = normalized_tool

    runtime_in = meta.get("runtime")
    runtime = dict(runtime_in) if isinstance(runtime_in, dict) else {}
    if not isinstance(runtime.get("timezone"), dict):
        runtime["timezone"] = build_runtime_timezone_meta(
            out,
            mt5_config=mt5_config,
            include_local=False,
            include_now=False,
        )
    if runtime:
        meta["runtime"] = runtime

    if meta:
        out["meta"] = meta
    return out


def apply_output_verbosity(
    result: Any,
    *,
    verbose: bool = False,
    tool_name: Optional[str] = None,
    mt5_config: Any = None,
) -> Any:
    """Normalize shared verbose-only output sections across transports."""
    if not isinstance(result, dict):
        return result

    out = dict(result)
    out.pop("cli_meta", None)

    if not verbose:
        return _strip_verbose_only_fields(out)

    return ensure_common_meta(
        out,
        tool_name=tool_name,
        mt5_config=mt5_config,
    )
