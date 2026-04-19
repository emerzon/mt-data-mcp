from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

from ..utils.utils import _UNPARSED_BOOL, _parse_bool_like
from .runtime_metadata import build_runtime_timezone_meta

_MISSING = object()
_VERBOSE_ONLY_KEYS = frozenset({"meta", "diagnostics", "debug", "debug_info"})
_DETAIL_COMPACT_ALIASES = {"summary": "compact", "summary_only": "compact"}
_VERBOSE_DETAIL_LEVELS = frozenset({"full"})
_COMPACT_DETAIL_LEVELS = frozenset({"compact"})


@dataclass(frozen=True)
class OutputContractState:
    """Resolved shared output contract state.

    detail:
        Normalized requested detail value, preserving tool-specific aliases when
        the caller provides them via ``aliases``.
    shape_detail:
        Shared compact/full shape resolved from detail plus explicit transport
        verbosity.
    verbose:
        Legacy compatibility verbosity that still treats ``detail=full`` as a
        verbose request for existing wrappers.
    transport_verbose:
        Explicit transport verbosity resolved from verbose/concise only.
    """

    detail: str
    shape_detail: str
    verbose: bool
    transport_verbose: bool


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


def _resolve_requested_detail_value(source: Any, *, detail: Any = _MISSING) -> Any:
    if detail is not _MISSING and detail is not None:
        return detail
    for candidate in _iter_verbosity_sources(source):
        detail_value = _read_verbosity_field(candidate, "detail")
        if detail_value is _MISSING or detail_value is None:
            continue
        return detail_value
    return None


def _iter_contract_sources(
    source: Any,
    *,
    detail: Any = _MISSING,
    verbose: Any = _MISSING,
    concise: Any = _MISSING,
) -> tuple[Any, ...]:
    explicit: dict[str, Any] = {}
    if detail is not _MISSING:
        explicit["detail"] = detail
    if verbose is not _MISSING:
        explicit["verbose"] = verbose
    if concise is not _MISSING:
        explicit["concise"] = concise

    candidates: list[Any] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(_iter_verbosity_sources(source))
    return tuple(candidates)


def normalize_output_detail(
    value: Any,
    *,
    default: str = "compact",
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    """Normalize detail-like values while preserving tool-specific legacy modes."""
    normalized = str(default if value is None else value).strip().lower()
    if not aliases:
        return normalized
    return str(aliases.get(normalized, normalized))


def normalize_output_verbosity_detail(value: Any, *, default: str = "compact") -> str:
    """Map legacy detail aliases onto the shared compact/full verbosity contract."""
    return normalize_output_detail(
        value,
        default=default,
        aliases=_DETAIL_COMPACT_ALIASES,
    )


def _resolve_output_detail_value(
    *,
    detail: Any = None,
    verbose: Any = None,
    default: str = "compact",
) -> str:
    normalized = normalize_output_verbosity_detail(detail, default=default)
    if normalized in _VERBOSE_DETAIL_LEVELS:
        return "full"
    if _coerce_optional_verbose_flag(verbose) is True:
        return "full"
    return "compact"


def resolve_output_detail(
    *,
    detail: Any = None,
    verbose: Any = None,
    default: str = "compact",
) -> str:
    """Resolve the shared compact/full detail mode from detail/verbose inputs."""
    return _resolve_output_detail_value(detail=detail, verbose=verbose, default=default)


def _resolve_transport_output_verbosity(
    source: Any,
    *,
    verbose: Any = _MISSING,
    concise: Any = _MISSING,
    default: bool = False,
) -> bool:
    saw_explicit_false = False
    for candidate in _iter_contract_sources(
        source,
        verbose=verbose,
        concise=concise,
    ):
        verbose_value = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "verbose"))
        if verbose_value is True:
            return True
        if verbose_value is False:
            saw_explicit_false = True

    for candidate in _iter_contract_sources(
        source,
        verbose=verbose,
        concise=concise,
    ):
        concise_value = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "concise"))
        if concise_value is True:
            return False

    if saw_explicit_false:
        return False

    return bool(default)


def _resolve_requested_output_verbosity_value(
    source: Any,
    *,
    detail: Any = _MISSING,
    verbose: Any = _MISSING,
    concise: Any = _MISSING,
    default: bool = False,
) -> bool:
    saw_explicit_false = False
    candidates = _iter_contract_sources(
        source,
        detail=detail,
        verbose=verbose,
        concise=concise,
    )
    for candidate in candidates:
        verbose_value = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "verbose"))
        if verbose_value is True:
            return True
        if verbose_value is False:
            saw_explicit_false = True

    for candidate in candidates:
        detail_value = _read_verbosity_field(candidate, "detail")
        if detail_value is _MISSING or detail_value is None:
            continue
        normalized = normalize_output_verbosity_detail(detail_value)
        if normalized in _VERBOSE_DETAIL_LEVELS:
            return True
        if normalized in _COMPACT_DETAIL_LEVELS:
            return False

    for candidate in candidates:
        concise_value = _coerce_optional_verbose_flag(_read_verbosity_field(candidate, "concise"))
        if concise_value is True:
            return False

    if saw_explicit_false:
        return False

    return bool(default)


def resolve_output_contract(
    source: Any = _MISSING,
    *,
    detail: Any = _MISSING,
    verbose: Any = _MISSING,
    concise: Any = _MISSING,
    default_detail: str = "compact",
    default_verbose: bool = False,
    aliases: Optional[Mapping[str, str]] = None,
) -> OutputContractState:
    """Resolve shared detail and verbosity state without dropping legacy helpers."""
    requested_detail_value = _resolve_requested_detail_value(source, detail=detail)
    explicit_transport_verbose = _resolve_transport_output_verbosity(
        source,
        verbose=verbose,
        concise=concise,
        default=default_verbose,
    )
    return OutputContractState(
        detail=normalize_output_detail(
            requested_detail_value,
            default=default_detail,
            aliases=aliases,
        ),
        shape_detail=_resolve_output_detail_value(
            detail=requested_detail_value,
            verbose=explicit_transport_verbose,
            default=default_detail,
        ),
        verbose=_resolve_requested_output_verbosity_value(
            source,
            detail=detail,
            verbose=verbose,
            concise=concise,
            default=default_verbose,
        ),
        transport_verbose=explicit_transport_verbose,
    )


def resolve_requested_output_verbosity(source: Any, *, default: bool = False) -> bool:
    """Resolve shared output verbosity from explicit and legacy detail controls."""
    return _resolve_requested_output_verbosity_value(source, default=default)


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
