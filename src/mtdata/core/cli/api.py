"""
Dynamic CLI wrapper for testing MetaTrader5 MCP server functions
Automatically discovers function parameters and creates CLI arguments
"""

import argparse
import difflib
import json
import os
import shlex
import sys
import types
import warnings
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    get_args,
    get_origin,
    is_typeddict,
)

from pydantic import BaseModel

from ...bootstrap.settings import load_environment
from ...bootstrap.tools import bootstrap_tools, cli_tool_module_names
from ...forecast.requests import ForecastGenerateRequest
from .._mcp_instance import mcp
from .._mcp_tools import _get_pydantic_model_fields, _select_output_fields
from .._mcp_tools import get_tool_registry as get_registered_tools
from ..output_contract import resolve_output_contract
from .formatting import (
    _attach_cli_meta,
    _format_result_for_cli,
    _json_default,
    _resolve_cli_formatter,
)
from .parsing.discovery import (
    _COMMAND_PARAM_CHOICE_OVERRIDES,
)
from .parsing.discovery import (
    add_dynamic_arguments as _add_dynamic_arguments_impl,
)
from .parsing.discovery import (
    apply_schema_overrides as _apply_schema_overrides_impl,
)
from .parsing.discovery import (
    discover_tools as _discover_tools_impl,
)
from .parsing.discovery import (
    extract_function_from_tool_obj as _extract_function_from_tool_obj_impl,
)
from .parsing.discovery import (
    extract_metadata_from_tool_obj as _extract_metadata_from_tool_obj_impl,
)
from .parsing.discovery import (
    get_function_info as _get_function_info_impl,
)
from .parsing.discovery import (
    resolve_param_kwargs as _resolve_param_kwargs_impl,
)
from .parsing.discovery import (
    should_expose_cli_param as _should_expose_cli_param_impl,
)
from .runtime import (
    _argparse_color_enabled,
    _capture_runtime_warnings,
    _configure_cli_logging,
    _debug,
    _debug_enabled,
    _suppress_cli_side_output,
)
from .runtime.commands import LIVE_TRADE_MUTATION_TOOLS, LIVE_TRADE_MUTATION_WARNING
from .runtime.commands import (
    coerce_cli_scalar as _coerce_cli_scalar_impl,
)
from .runtime.commands import (
    create_command_function as _create_command_function_impl,
)
from .runtime.commands import (
    merge_dict as _merge_dict_impl,
)
from .runtime.commands import (
    normalize_cli_list_value as _normalize_cli_list_value_impl,
)
from .runtime.commands import (
    parse_kv_string as _parse_kv_string_impl,
)
from .runtime.commands import (
    parse_set_overrides as _parse_set_overrides_impl,
)


class _CLIHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Preserve command descriptions while showing effective defaults."""

    def _format_args(self, action: argparse.Action, default_metavar: str) -> str:
        if getattr(action, "_cli_logically_required", False):
            return self._metavar_formatter(action, default_metavar)(1)[0]
        return super()._format_args(action, default_metavar)

def _is_typed_dict_type(value: Any) -> bool:
    try:
        if is_typeddict(value):
            return True
    except Exception:
        pass
    annotations = getattr(value, "__annotations__", None)
    return isinstance(annotations, dict) and (
        getattr(value, "__required_keys__", None) is not None
        or getattr(value, "__optional_keys__", None) is not None
    )


def _invoke_cli_tool_function(
    func: Any, *, args: Any, cmd_name: str, kwargs: Dict[str, Any]
) -> Any:
    del cmd_name
    with _capture_runtime_warnings() as warning_records:
        with _suppress_cli_side_output(enabled=True):
            result = func(**kwargs)

    warning_texts: List[str] = []
    seen: set[str] = set()
    for record in warning_records:
        category = getattr(record, "category", Warning)
        if isinstance(category, type) and issubclass(
            category,
            (DeprecationWarning, PendingDeprecationWarning, ImportWarning),
        ):
            continue
        if isinstance(category, type) and issubclass(category, ResourceWarning):
            continue
        if isinstance(category, type) and issubclass(category, FutureWarning):
            filename = os.path.normcase(str(getattr(record, "filename", "") or ""))
            if "site-packages" in filename or "dist-packages" in filename:
                continue
        try:
            text = warnings.formatwarning(
                message=record.message,
                category=category,
                filename=record.filename,
                lineno=record.lineno,
                line=record.line,
            ).strip()
        except Exception:
            text = str(getattr(record, "message", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        warning_texts.append(text)

    if not warning_texts:
        return result
    if isinstance(result, dict):
        out = dict(result)
        combined: List[str] = []
        existing = out.get("warnings")
        if isinstance(existing, list):
            for item in existing:
                item_text = str(item).strip()
                if item_text and item_text not in combined:
                    combined.append(item_text)
        elif isinstance(existing, str):
            existing_text = existing.strip()
            if existing_text:
                combined.append(existing_text)
        for item in warning_texts:
            if item not in combined:
                combined.append(item)
        out["warnings"] = combined
        return out
    return {"success": True, "data": result, "warnings": warning_texts}


from ...shared.schema import PARAM_HINTS as _PARAM_HINTS
from ...shared.schema import enrich_schema_with_shared_defs
from ...shared.schema import get_function_info as _schema_get_function_info
from .._mcp_tools import get_mcp_registry
from ..unified_params import add_global_args_to_parser

# Types for discovered metadata
ToolInfo = Dict[str, Any]

CLI_PROGRAM = "mtdata-cli"
PACKAGE_NAME = "mtdata"


def _read_local_project_version() -> Optional[str]:
    pyproject_path = Path(__file__).resolve().parents[4] / "pyproject.toml"
    try:
        for line in pyproject_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version"):
                _, raw_value = line.split("=", 1)
                return raw_value.strip().strip('"').strip("'") or None
    except Exception:
        return None
    return None


def _cli_version() -> str:
    try:
        return importlib_metadata.version(PACKAGE_NAME)
    except importlib_metadata.PackageNotFoundError:
        return _read_local_project_version() or "unknown"


def _is_pydantic_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def _iter_request_model_params(model_type: type[BaseModel]) -> List[Dict[str, Any]]:
    fields, modern_fields = _get_pydantic_model_fields(model_type)
    if modern_fields:
        params: List[Dict[str, Any]] = []
        for name, field in fields.items():
            required = (
                bool(field.is_required())
                if callable(getattr(field, "is_required", None))
                else False
            )
            default = None if required else getattr(field, "default", None)
            default_class = getattr(default, "__class__", None)
            if (
                default_class is not None
                and getattr(default_class, "__name__", "") == "PydanticUndefinedType"
            ):
                default = None
            params.append(
                {
                    "name": name,
                    "required": required,
                    "default": default,
                    "type": getattr(field, "annotation", Any) or Any,
                }
            )
        return params

    if fields:
        return [
            {
                "name": name,
                "required": bool(getattr(field, "required", False)),
                "default": None
                if getattr(field, "required", False)
                else getattr(field, "default", None),
                "type": getattr(field, "outer_type_", Any) or Any,
            }
            for name, field in fields.items()
        ]

    return []


def _flatten_request_model_param(info: Dict[str, Any]) -> Dict[str, Any]:
    params = info.get("params") or []
    if len(params) != 1:
        return info
    request_param = params[0]
    request_model = request_param.get("type")
    if not _is_pydantic_model_type(request_model):
        return info
    info["request_model"] = request_model
    info["request_param_name"] = request_param["name"]
    info["params"] = _iter_request_model_params(request_model)
    return info


def _argv_option_present_after_command(
    argv: List[str], command: str, option: str
) -> bool:
    cmd_index = _find_command_index(argv, command)
    if cmd_index is None:
        return False
    for token in argv[cmd_index + 1 :]:
        if token == option or token.startswith(f"{option}="):
            return True
    return False


def _command_variants(command: str) -> Tuple[str, ...]:
    text = str(command or "").strip()
    if not text:
        return ()
    variants = [text]
    hyphen = text.replace("_", "-")
    underscore = text.replace("-", "_")
    if hyphen not in variants:
        variants.append(hyphen)
    if underscore not in variants:
        variants.append(underscore)
    return tuple(variants)


def _find_command_index(argv: List[str], command: str) -> Optional[int]:
    for candidate in _command_variants(command):
        try:
            return argv.index(candidate)
        except ValueError:
            continue
    return None


def _command_aliases(command: str) -> List[str]:
    text = str(command or "").strip()
    if not text or "_" not in text:
        return []
    alias = text.replace("_", "-")
    return [alias] if alias != text else []


def _normalize_cli_argv_aliases(
    argv: List[str], functions: Dict[str, ToolInfo]
) -> List[str]:
    normalized = list(argv)
    alias_map: Dict[str, str] = {}
    for command in functions.keys():
        canonical = str(command or "").strip()
        if not canonical:
            continue
        for alias in _command_aliases(canonical):
            alias_map[alias] = canonical
    for index, token in enumerate(normalized):
        token_text = str(token)
        canonical = alias_map.get(token_text)
        if canonical:
            normalized[index] = canonical
            break
        if token_text in functions:
            break
    return normalized


def _apply_global_cli_overrides(args: Any, argv: List[str]) -> Any:
    command = getattr(args, "command", None)
    if not isinstance(command, str) or not command:
        return args
    command = command.replace("-", "_")
    args.command = command
    global_timeframe = getattr(args, "_global_timeframe", None)
    if global_timeframe is not None and not _argv_option_present_after_command(
        argv, command, "--timeframe"
    ):
        args.timeframe = global_timeframe
    trade_days = getattr(args, "_trade_days", None)
    if command.startswith("trade_") and trade_days is not None:
        if not (
            _argv_option_present_after_command(argv, command, "--minutes-back")
            or _argv_option_present_after_command(argv, command, "--minutes_back")
        ):
            try:
                args.minutes_back = int(round(float(trade_days) * 1440.0))
            except Exception:
                args.minutes_back = trade_days
    return args


def _argv_param_present_after_command(
    argv: List[str], command: str, param_name: str
) -> bool:
    flags = (
        f"--{param_name.replace('_', '-')}",
        f"--{param_name}",
    )
    return any(
        _argv_option_present_after_command(argv, command, flag) for flag in flags
    )


def _literal_choices_for_cli_param(
    param: Dict[str, Any],
    *,
    cmd_name: Optional[str] = None,
) -> Optional[List[str]]:
    choice_override = _COMMAND_PARAM_CHOICE_OVERRIDES.get(
        (str(cmd_name or ""), str(param.get("name") or "")),
    )
    if choice_override:
        return list(choice_override)
    try:
        ptype = param.get("type")
        base_type, origin = _unwrap_optional_type(ptype)
    except Exception:
        return None
    if not _is_literal_origin(origin):
        return None
    choices = [str(value) for value in get_args(base_type) if value is not None]
    return choices or None


def _default_cli_compact_choice(
    choices: List[str],
) -> Optional[str]:
    by_lower = {
        str(choice).strip().lower(): str(choice)
        for choice in choices
        if str(choice).strip()
    }
    if "full" not in by_lower:
        return None
    if "compact" in by_lower:
        return by_lower["compact"]
    if "summary" in by_lower:
        return by_lower["summary"]
    return None


def _apply_cli_output_mode_defaults(
    args: Any, argv: List[str], functions: Dict[str, ToolInfo]
) -> Any:
    command = getattr(args, "command", None)
    if not isinstance(command, str) or not command:
        return args

    tool = functions.get(command)
    if not isinstance(tool, dict):
        return args

    func = tool.get("func")
    if func is None:
        return args

    func_info = tool.setdefault("_cli_func_info", get_function_info(func))
    _apply_schema_overrides(tool, func_info)
    for param in func_info.get("params") or []:
        if not isinstance(param, dict):
            continue
        param_name = str(param.get("name") or "").strip()
        if param_name != "detail":
            continue
        if _argv_param_present_after_command(argv, command, param_name):
            continue
        choices = _literal_choices_for_cli_param(param, cmd_name=command)
        if not choices:
            continue
        selected = _default_cli_compact_choice(choices)
        if selected is None:
            continue
        setattr(args, param_name, selected)
    return args


def _normalize_console_text(text: str) -> str:
    normalized = str(text)
    for src, dst in {
        "\u2192": "->",
        "\u2190": "<-",
        "\u2026": "...",
    }.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _should_force_utf8_stream(target: Any) -> bool:
    buffer = getattr(target, "buffer", None)
    if buffer is None or not hasattr(buffer, "write"):
        return False
    try:
        return not bool(target.isatty())
    except Exception:
        return False


def _write_cli_text(text: str, *, stream: Any = None) -> None:
    target = stream if stream is not None else sys.stdout
    payload = str(text)
    rendered = payload if payload.endswith("\n") else f"{payload}\n"
    if _should_force_utf8_stream(target):
        buffer = getattr(target, "buffer", None)
        if buffer is not None and hasattr(buffer, "write"):
            try:
                buffer.write(rendered.encode("utf-8"))
                if hasattr(target, "flush"):
                    try:
                        target.flush()
                    except Exception:
                        pass
                return
            except Exception:
                pass
    try:
        target.write(rendered)
    except UnicodeEncodeError:
        safe_text = _normalize_console_text(payload)
        safe_rendered = safe_text if safe_text.endswith("\n") else f"{safe_text}\n"
        encoding = getattr(target, "encoding", None) or "utf-8"
        encoded = safe_rendered.encode(encoding, errors="replace")
        buffer = getattr(target, "buffer", None)
        if buffer is not None and hasattr(buffer, "write"):
            buffer.write(encoded)
        else:
            target.write(encoded.decode(encoding, errors="replace"))
    if hasattr(target, "flush"):
        try:
            target.flush()
        except Exception:
            pass


def _render_cli_result(result: Any, *, args: Any, cmd_name: str) -> Any:
    verbose = resolve_output_contract(args).verbose
    result = _attach_cli_meta(result, cmd_name=cmd_name, verbose=verbose)
    result = _select_output_fields(result, getattr(args, "fields", None))
    output = _format_result_for_cli(
        result,
        fmt=_resolve_cli_formatter(args),
        verbose=verbose,
        cmd_name=cmd_name,
        precision=getattr(args, "precision", None),
    )
    if output:
        _write_cli_text(output)
    return result


def _result_has_tool_error(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("success") is False:
            return True
        if bool(result.get("no_action", False)) and result.get("success") is not True:
            return True
        err = result.get("error")
        if isinstance(err, str):
            return bool(err.strip())
        return err not in (None, False)
    if isinstance(result, str):
        return result.strip().lower().startswith("error:")
    return False


def _render_cli_result_status(result: Any, *, args: Any, cmd_name: str) -> int:
    rendered_result = _render_cli_result(result, args=args, cmd_name=cmd_name)
    return int(_result_has_tool_error(rendered_result))


def _json_parse_errors_requested() -> bool:
    if "--json" in sys.argv[1:]:
        return True
    return str(os.getenv("MTDATA_OUTPUT_FORMAT") or "").strip().lower() == "json"


class _CLIArgumentParser(argparse.ArgumentParser):
    """Emit parse failures in the selected CLI transport format."""

    def error(self, message: str) -> None:
        market_depth_disabled = (
            "market_depth_fetch" in str(message)
            and str(os.getenv("MTDATA_ENABLE_MARKET_DEPTH_FETCH") or "")
            .strip()
            .lower()
            not in {"1", "true", "yes", "on"}
        )
        if market_depth_disabled:
            message = (
                "market_depth_fetch is disabled; set "
                "MTDATA_ENABLE_MARKET_DEPTH_FETCH=1 before starting the CLI. "
                "The broker must also provide Level 2/DOM data."
            )
        if _json_parse_errors_requested():
            payload = {
                "success": False,
                "error": str(message),
                "error_code": (
                    "feature_disabled" if market_depth_disabled else "cli_invalid_arguments"
                ),
                "remediation": (
                    "Set MTDATA_ENABLE_MARKET_DEPTH_FETCH=1 and restart the process."
                    if market_depth_disabled
                    else f"Run '{self.prog} --help' to inspect valid arguments."
                ),
            }
            if market_depth_disabled:
                payload["details"] = {
                    "feature": "market_depth_fetch",
                    "enable_env": "MTDATA_ENABLE_MARKET_DEPTH_FETCH",
                    "broker_prerequisite": "Level 2/DOM market data",
                }
            _write_cli_text(json.dumps(payload, ensure_ascii=False, indent=2))
            self.exit(2)
        super().error(message)


def _resolve_cli_output_contract_or_error(parser: argparse.ArgumentParser, args: Any):
    try:
        return resolve_output_contract(args)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))


def get_function_info(func):
    """Thin wrapper around schema.get_function_info that attaches the callable.

    This avoids duplicating introspection logic while preserving the CLI's
    expectation that the returned dict contains a 'func' key for invocation.
    """
    return _get_function_info_impl(
        func,
        schema_get_function_info=_schema_get_function_info,
        flatten_request_model_param=_flatten_request_model_param,
    )


def _apply_schema_overrides(
    tool: ToolInfo, func_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply schema metadata to the introspected CLI param info."""
    return _apply_schema_overrides_impl(
        tool,
        func_info,
        enrich_schema_with_shared_defs=enrich_schema_with_shared_defs,
    )


_extract_function_from_tool_obj = _extract_function_from_tool_obj_impl


_extract_metadata_from_tool_obj = _extract_metadata_from_tool_obj_impl


_DISCOVERY_ERRORS: List[str] = []


def _is_union_origin(origin: Any) -> bool:
    return origin in (Union, types.UnionType) or str(origin) in {
        "typing.Union",
        "<class 'typing.Union'>",
    }


def _is_literal_origin(origin: Any) -> bool:
    return origin is Literal or str(origin) in {
        "typing.Literal",
        "<class 'typing.Literal'>",
    }


def discover_tools(module_names: Optional[Tuple[str, ...]] = None):
    """Discover MCP tools from the shared bootstrap registry.

    Priority:
    1) Use the shared tool registry after bootstrap
    2) Use the MCP registry if available
    3) Fallback to scanning bootstrapped tool modules
    """
    _DISCOVERY_ERRORS.clear()
    return _discover_tools_impl(
        bootstrap_tools=lambda: bootstrap_tools(module_names),
        get_registered_tools=get_registered_tools,
        mcp=mcp,
        get_mcp_registry=get_mcp_registry,
        debug=_debug,
        extract_function_from_tool_obj=_extract_function_from_tool_obj,
        extract_metadata_from_tool_obj=_extract_metadata_from_tool_obj,
        errors=_DISCOVERY_ERRORS,
    )


def _resolve_param_kwargs(
    param: Dict[str, Any],
    param_docs: Optional[Dict[str, str]],
    cmd_name: Optional[str] = None,
    param_names: Optional[set] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Resolve CLI argument kwargs and determine if parameter is a mapping type."""
    return _resolve_param_kwargs_impl(
        param,
        param_docs,
        cmd_name=cmd_name,
        param_names=param_names,
        param_hints=_PARAM_HINTS,
        debug=_debug,
        is_literal_origin=_is_literal_origin,
        unwrap_optional_type=_unwrap_optional_type,
        is_typed_dict_type=_is_typed_dict_type,
        get_origin=get_origin,
        get_args=get_args,
    )


def add_dynamic_arguments(
    parser,
    param_info,
    param_docs: Optional[Dict[str, str]] = None,
    cmd_name: Optional[str] = None,
):
    """Add arguments to parser based on parameter info.

    Adds both hyphen and underscore long-option aliases and sets dest to the
    original param name (snake_case) so downstream mapping works.
    Also casts Optional[int|float|bool] to their base types for argparse.
    """
    _add_dynamic_arguments_impl(
        parser,
        param_info,
        resolve_param_kwargs=_resolve_param_kwargs,
        param_docs=param_docs,
        cmd_name=cmd_name,
    )


def _parse_kv_string(s: str) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' (commas or spaces) into a dict. Delegates to utils implementation."""
    return _parse_kv_string_impl(s, debug=_debug)


def _unwrap_optional_type(ptype: Any) -> Tuple[Any, Any]:
    """Unwrap Optional[T] to (T, origin(T))."""
    origin = get_origin(ptype)
    if _is_union_origin(origin):
        args_t = [a for a in get_args(ptype) if a is not type(None)]
        if len(args_t) == 1:
            ptype = args_t[0]
            origin = get_origin(ptype)
    return ptype, origin


_normalize_cli_list_value = _normalize_cli_list_value_impl


_coerce_cli_scalar = _coerce_cli_scalar_impl


def _parse_set_overrides(items: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """Parse repeated --set entries like 'method.sp=24' into nested dicts."""
    return _parse_set_overrides_impl(items, coerce_cli_scalar=_coerce_cli_scalar)


_merge_dict = _merge_dict_impl


_FORECAST_TYPED_ARG_SPECS: Dict[str, Dict[str, Any]] = {
    "params": {
        "flag": "--params",
        "section": "method",
        "metavar": "JSON|k=v",
        "help": "Method params as JSON or key=value pairs.",
        "examples": [
            '--params "window_size=64 top_k=20"',
            '--params \'{"window_size":64,"top_k":20}\'',
            "--params --set method.window_size=64 --set method.top_k=20",
        ],
    },
    "denoise": {
        "flag": "--denoise",
        "section": "denoise",
        "metavar": "PRESET|JSON",
        "help": "Denoise preset name or JSON spec.",
        "examples": [
            "--denoise ema",
            '--denoise \'{"method":"ema","params":{"span":10}}\'',
            "--denoise --set denoise.method=ema",
        ],
    },
    "features": {
        "flag": "--features",
        "section": "features",
        "metavar": "JSON|k=v",
        "help": "Feature spec as JSON or key=value pairs.",
        "examples": [
            '--features "include=open,high future_covariates=hour,dow"',
            '--features \'{"include":["open","high"],"future_covariates":["hour","dow"]}\'',
            "--features --set features.include=open,high",
        ],
    },
    "dimred_params": {
        "flag": "--dimred-params",
        "section": "dimred",
        "metavar": "JSON|k=v",
        "help": "Dimred params as JSON or key=value pairs.",
        "examples": [
            '--dimred-params "n_components=4"',
            "--dimred-params '{\"n_components\":4}'",
            "--dimred-params --set dimred.n_components=4",
        ],
    },
    "target_spec": {
        "flag": "--target-spec",
        "section": "target",
        "metavar": "JSON|k=v",
        "help": "Target spec as JSON or key=value pairs.",
        "examples": [
            '--target-spec "column=close transform=log"',
            '--target-spec \'{"column":"close","transform":"log"}\'',
            "--target-spec --set target.column=close --set target.transform=log",
        ],
    },
}


def _add_forecast_typed_arg(
    group: argparse._ArgumentGroup,
    flag: str,
    *,
    dest: str,
    metavar: str,
    help_text: str,
) -> None:
    group.add_argument(
        flag,
        dest=dest,
        type=str,
        nargs="?",
        const="__PRESENT__",
        default=None,
        metavar=metavar,
        help=help_text,
    )


def _forecast_generate_typed_value_epilog() -> str:
    lines = ["Typed Value Formats:"]
    for key in ("denoise", "params", "features", "dimred_params", "target_spec"):
        spec = _FORECAST_TYPED_ARG_SPECS[key]
        lines.append(f"  {spec['flag']} {spec['metavar']}")
        for example in spec["examples"]:
            lines.append(
                f"    Example: {CLI_PROGRAM} forecast_generate SYMBOL {example}"
            )
    lines.append("  --set SECTION.KEY=VALUE")
    lines.append(
        f"    Example: {CLI_PROGRAM} forecast_generate SYMBOL --set method.window_size=64"
    )
    return "\n".join(lines)


def _parse_cli_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _resolve_forecast_typed_cli_value(
    raw_value: Any,
    *,
    key: str,
    overrides: Dict[str, Dict[str, Any]],
    parser: argparse.ArgumentParser,
) -> Any:
    if raw_value != "__PRESENT__":
        return raw_value
    spec = _FORECAST_TYPED_ARG_SPECS[key]
    if overrides.get(spec["section"]):
        return {}
    examples = "; ".join(spec["examples"][:2])
    parser.error(f"{spec['flag']} expects a value. Examples: {examples}")


def _forecast_method_help() -> str:
    base = (
        "Method name within the selected library. Registered built-in methods: "
    )
    try:
        from mtdata.forecast.forecast_methods import get_forecast_method_names

        names = sorted(set(get_forecast_method_names()))
    except Exception:
        names = []
    if names:
        return (
            base
            + ", ".join(names)
            + ". Dotted class paths are also accepted for supported libraries; "
            "use forecast_list_methods for details."
        )
    return (
        "Method name within the selected library. Use forecast_list_methods to "
        "browse registered methods; dotted class paths are also accepted for "
        "supported libraries."
    )


def _add_forecast_generate_args(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.description = (
        "Generate forecasts with an optional preprocessing pipeline."
    )
    cmd_parser.epilog = _forecast_generate_typed_value_epilog()

    cmd_parser.add_argument(
        "symbol",
        nargs="?",
        default=argparse.SUPPRESS,
        help=_PARAM_HINTS["symbol"],
    )
    cmd_parser.add_argument(
        "--symbol", dest="symbol", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )

    group_method = cmd_parser.add_argument_group("Method")
    group_method.add_argument(
        "--library",
        dest="library",
        type=str,
        choices=["native", "statsforecast", "sktime", "mlforecast", "pretrained"],
        default="native",
        help="Method library.",
    )
    group_method.add_argument(
        "--method",
        dest="method",
        type=str,
        default="theta",
        help=_forecast_method_help(),
    )
    group_method.add_argument(
        "--params",
        dest="params",
        type=str,
        nargs="?",
        const="__PRESENT__",
        default=None,
        metavar=_FORECAST_TYPED_ARG_SPECS["params"]["metavar"],
        help=_FORECAST_TYPED_ARG_SPECS["params"]["help"],
    )

    group_window = cmd_parser.add_argument_group("Window")
    group_window.add_argument(
        "--timeframe",
        type=str,
        default="H1",
        help=_PARAM_HINTS["timeframe"],
    )
    group_window.add_argument(
        "--horizon", type=int, default=12, help="Forecast horizon in bars."
    )
    group_window.add_argument(
        "--lookback", type=int, default=None, help="Historical bars to use."
    )
    group_window.add_argument(
        "--as-of", dest="as_of", type=str, default=None, help="Reference time override."
    )
    group_window.add_argument(
        "--start",
        dest="start",
        type=str,
        default=None,
        help="Start of the historical training window.",
    )
    group_window.add_argument(
        "--end",
        dest="end",
        type=str,
        default=None,
        help="End of the historical training window.",
    )

    group_target = cmd_parser.add_argument_group("Target")
    group_target.add_argument(
        "--quantity",
        choices=["price", "return", "volatility"],
        default="price",
        help="Target quantity.",
    )
    group_target.add_argument(
        "--proxy",
        choices=["squared_return", "abs_return", "log_r2"],
        default=None,
        help="Volatility proxy when quantity=volatility.",
    )

    group_uncertainty = cmd_parser.add_argument_group("Uncertainty")
    group_uncertainty.add_argument(
        "--ci-alpha",
        dest="ci_alpha",
        type=float,
        default=None,
        help="Request a confidence interval with this alpha (0.05 => 95%%).",
    )
    group_uncertainty.add_argument(
        "--detail",
        choices=["compact", "standard", "summary", "full"],
        default="compact",
        help="Output detail level.",
    )

    group_pipe = cmd_parser.add_argument_group("Pipeline")
    _add_forecast_typed_arg(
        group_pipe,
        "--denoise",
        dest="denoise",
        metavar=_FORECAST_TYPED_ARG_SPECS["denoise"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["denoise"]["help"],
    )
    _add_forecast_typed_arg(
        group_pipe,
        "--features",
        dest="features",
        metavar=_FORECAST_TYPED_ARG_SPECS["features"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["features"]["help"],
    )
    group_pipe.add_argument(
        "--dimred-method",
        dest="dimred_method",
        type=str,
        default=None,
        help="Dimred method.",
    )
    _add_forecast_typed_arg(
        group_pipe,
        "--dimred-params",
        dest="dimred_params",
        metavar=_FORECAST_TYPED_ARG_SPECS["dimred_params"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["dimred_params"]["help"],
    )
    _add_forecast_typed_arg(
        group_pipe,
        "--target-spec",
        dest="target_spec",
        metavar=_FORECAST_TYPED_ARG_SPECS["target_spec"]["metavar"],
        help_text=_FORECAST_TYPED_ARG_SPECS["target_spec"]["help"],
    )

    group_overrides = cmd_parser.add_argument_group("Overrides")
    group_overrides.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="SECTION.KEY=VALUE",
        help="Override nested params (method, denoise, features, dimred, target).",
    )

    group_exec = cmd_parser.add_argument_group("Execution")
    group_exec.add_argument(
        "--async-mode",
        dest="async_mode",
        type=_parse_cli_bool,
        nargs="?",
        const=True,
        default=False,
        metavar="BOOL",
        help="Submit heavy model training in the background when supported.",
    )
    group_exec.add_argument(
        "--model-id",
        dest="model_id",
        type=str,
        default=None,
        help="Use a trained-model params_hash from the model store.",
    )

    group_dbg = cmd_parser.add_argument_group("Debug")
    group_dbg.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the resolved forecast config and exit.",
    )


def create_command_function(
    func_info, cmd_name: str = "", cmd_parser: Optional[argparse.ArgumentParser] = None
):
    """Create a command function that calls the MCP function dynamically"""
    return _create_command_function_impl(
        func_info,
        cmd_name=cmd_name,
        render_cli_result=_render_cli_result,
        result_has_tool_error=_result_has_tool_error,
        normalize_cli_list_value=_normalize_cli_list_value,
        parse_kv_string=_parse_kv_string,
        unwrap_optional_type=_unwrap_optional_type,
        is_typed_dict_type=_is_typed_dict_type,
        invoke_tool_function=_invoke_cli_tool_function,
    )


def _type_name(t):
    try:
        return t.__name__
    except Exception:
        return str(t)


def _first_line(text: Optional[str]) -> str:
    if not text:
        return ""
    for line in str(text).splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _should_expose_cli_param(*, cmd_name: str, param_name: str) -> bool:
    return _should_expose_cli_param_impl(cmd_name=cmd_name, param_name=param_name)


def _format_epilog_param_usage(
    param: Dict[str, Any], *, cmd_name: str, index: int
) -> Optional[str]:
    name = str(param.get("name") or "").strip()
    if not name or not _should_expose_cli_param(cmd_name=cmd_name, param_name=name):
        return None
    choices = _literal_choices_for_cli_param(param, cmd_name=cmd_name)
    if choices:
        type_token = "{" + ",".join(choices) + "}"
    else:
        try:
            base_type, _ = _unwrap_optional_type(param.get("type"))
        except Exception:
            base_type = param.get("type")
        type_token = f"<{_type_name(base_type or str)}>"

    if bool(param.get("required")):
        if index == 0:
            return f"{name}{type_token}"
        return f"--{name.replace('_', '-')}{type_token}"

    default = param.get("default")
    return f"--{name.replace('_', '-')}{type_token}=[{default}]"


_COMMAND_HELP_CATEGORY_ORDER = (
    "DATA ACCESS",
    "FORECASTING",
    "TRADING",
    "PATTERNS & LEVELS",
    "MARKET CONTEXT",
    "ANALYTICS",
    "NEWS & FUNDAMENTALS",
    "REPORTS & TOOLS",
    "OTHER TOOLS",
)


def _command_help_category(command: str) -> str:
    name = str(command or "").strip().lower()
    if name.startswith("data_") or name.startswith("market_depth"):
        return "DATA ACCESS"
    if name.startswith("forecast_") or name == "strategy_backtest":
        return "FORECASTING"
    if name.startswith("trade_"):
        return "TRADING"
    if (
        name.startswith("patterns_")
        or name.startswith("pivot_")
        or name in {"confluence_levels", "support_resistance_levels", "volume_profile"}
    ):
        return "PATTERNS & LEVELS"
    if name.startswith("market_") or name.startswith("symbols_") or name.startswith("options_"):
        return "MARKET CONTEXT"
    if (
        name.startswith("regime_")
        or name.startswith("indicators_")
        or name.startswith("denoise_")
        or name.startswith("temporal_")
        or name.startswith("causal_")
        or name.startswith("labels_")
        or name
        in {
            "cointegration_test",
            "correlation_matrix",
            "cross_correlation",
            "outliers_detect",
            "seasonality_detect",
            "stationarity_test",
        }
    ):
        return "ANALYTICS"
    if name.startswith("finviz_") or name.startswith("news_"):
        return "NEWS & FUNDAMENTALS"
    if name.startswith("report_") or name.startswith("tools_") or name.startswith("diagnostics_"):
        return "REPORTS & TOOLS"
    return "OTHER TOOLS"


_CLI_DESCRIPTION = (
    "Dynamic CLI for MetaTrader5 MCP tools "
    "(TOON by default; set MTDATA_OUTPUT_FORMAT=json for JSON). "
    "One-shot commands initialize the requested tool family; for repeated local calls "
    "use `mtdata-cli shell`, and for agents use a long-lived stdio or HTTP server."
)


def _sort_subparser_help_choices(subparsers: argparse._SubParsersAction) -> None:
    """Keep custom command parsers in the alphabetical help listing."""
    subparsers._choices_actions.sort(key=lambda action: action.dest)


def _build_epilog(functions: Dict[str, ToolInfo]) -> str:
    lines = []
    lines.append("Commands and Arguments by Category:")
    grouped: Dict[str, List[Tuple[str, ToolInfo]]] = {
        category: [] for category in _COMMAND_HELP_CATEGORY_ORDER
    }
    for cmd_name, tool in sorted(functions.items()):
        grouped.setdefault(_command_help_category(cmd_name), []).append((cmd_name, tool))
    for category in _COMMAND_HELP_CATEGORY_ORDER:
        rows = grouped.get(category) or []
        if not rows:
            continue
        lines.append("")
        lines.append(f"{category}:")
        for cmd_name, tool in rows:
            func = tool["func"]
            func_info = tool.setdefault("_cli_func_info", get_function_info(func))
            _apply_schema_overrides(tool, func_info)
            arg_strs = []
            for index, param in enumerate(func_info["params"]):
                rendered = _format_epilog_param_usage(param, cmd_name=cmd_name, index=index)
                if rendered:
                    arg_strs.append(rendered)
            meta = tool.get("meta") or {}
            desc = meta.get("description") or _first_line(func_info.get("doc"))
            lines.append(f"- {cmd_name}: {' '.join(arg_strs) if arg_strs else '(no args)'}")
            if desc:
                lines.append(f"  {desc}")
    lines.append("")
    lines.append("Tip: Use `--help <keyword>` to search commands and examples.")
    lines.append(
        "Aliases: commands also accept kebab-case spellings (e.g. market-ticker)."
    )
    lines.append("Type Conventions:")
    lines.append("  - int: integer")
    lines.append("  - str: string")
    lines.append("  - bool: pass true|false (e.g., --flag true)")
    lines.append("")
    lines.append("General Examples:")
    lines.append("  # Basic forecast with a native method")
    lines.append(
        f"  {CLI_PROGRAM} forecast_generate EURUSD --library native --method theta --timeframe H1 --horizon 24"
    )
    lines.append("")
    lines.append("  # Foundation model (Chronos-2) with covariates")
    lines.append(
        f"  {CLI_PROGRAM} forecast_generate BTCUSD --library pretrained --method chronos2 --timeframe H1 --horizon 12 \\"
    )
    lines.append(
        '    --features "include=open,high future_covariates=hour,dow,is_holiday" \\'
    )
    lines.append("    --json")
    lines.append("")
    lines.append("  # Rolling backtest for accuracy check")
    lines.append(
        f"  {CLI_PROGRAM} forecast_backtest_run EURUSD --timeframe H1 --methods theta,seasonal_naive \\"
    )
    lines.append("    --steps 5 --horizon 12")
    return "\n".join(lines)


_EXTENDED_HELP_EXAMPLE_HINTS: Dict[str, Any] = {
    "symbol": "EURUSD",
    "timeframe": "H1",
    "method": "nhits",
    "library": "native",
    "methods": "theta nhits",
    "horizon": "8",
    "lookback": "200",
    "steps": "5",
    "spacing": "20",
    "quantity": "price",
    "ci_alpha": "0.1",
    "params": '"max_epochs=20"',
    "features": '"include=open,high future_covariates=hour,dow"',
    "as_of": "2025-09-01T12:00:00Z",
    "population": "16",
    "generations": "5",
    "seed": "42",
}

_COMMAND_USAGE_EXAMPLES: Dict[str, Tuple[str, Optional[str]]] = {
    "patterns_detect": (
        f"{CLI_PROGRAM} patterns_detect BTCUSD --timeframe H1 --mode candlestick",
        f"{CLI_PROGRAM} patterns_detect BTCUSD --timeframe H1 --mode fractal --limit 300",
    ),
    "pivot_compute_points": (
        f"{CLI_PROGRAM} pivot_compute_points BTCUSD --timeframe D1",
        None,
    ),
    "confluence_levels": (
        f"{CLI_PROGRAM} confluence_levels EURUSD --pivot-timeframe D1 --sr-timeframe auto",
        f"{CLI_PROGRAM} confluence_levels EURUSD --min-source-families 2 --detail standard --json",
    ),
    "regime_detect": (
        f"{CLI_PROGRAM} regime_detect BTCUSD --timeframe H1 --method hmm",
        f"{CLI_PROGRAM} regime_detect BTCUSD --timeframe H1 --method hmm --extras metadata",
    ),
    "trade_risk_analyze": (
        f"{CLI_PROGRAM} trade_risk_analyze --symbol BTCUSD --direction long --desired-risk-pct 1 --entry 66317 --stop-loss 65000",
        f"{CLI_PROGRAM} trade_risk_analyze --symbol BTCUSD --direction long --desired-risk-pct 1 --entry 66317 --stop-loss 65000 --take-profit 69000",
    ),
    "trade_modify": (
        f"{CLI_PROGRAM} trade_modify 123456789 --price 61000",
        f"{CLI_PROGRAM} trade_modify 123456789 --stop-loss 60500 --take-profit 62500",
    ),
    "trade_place": (
        f"{CLI_PROGRAM} trade_place BTCUSD --volume 0.01 --order-type SELL --stop-loss 68521 --take-profit 67071",
        f'{CLI_PROGRAM} trade_place BTCUSD --volume 0.01 --order-type BUY --stop-loss 64500 --take-profit 67200 --comment "swing long"',
    ),
    "trade_close": (
        f"{CLI_PROGRAM} trade_close --ticket 123456789",
        f"{CLI_PROGRAM} trade_close --ticket 123456789 --volume 0.05",
    ),
}

_TIMEFRAMELESS_GLOBAL_COMMANDS: set[str] = {
    "indicators_describe",
    "indicators_list",
    "market_ticker",
    "data_fetch_ticks",
    "options_barrier_price",
    "options_chain",
    "options_expirations",
    "options_heston_calibrate",
    "symbols_describe",
    "symbols_list",
    "tools_list",
    "trade_account_info",
    "trade_close",
    "trade_history",
    "trade_modify",
    "trade_risk_analyze",
}


def _format_cli_literal(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        dumped = json.dumps(value, allow_nan=False, default=_json_default)
    except Exception:
        return str(value)
    if dumped.startswith('"') and dumped.endswith('"'):
        try:
            unquoted = json.loads(dumped)
        except Exception:
            return dumped
        if isinstance(unquoted, str):
            return unquoted
    return dumped


def _quote_cli_value(text: str) -> str:
    if text == "":
        return '""'
    if any(ch.isspace() for ch in text):
        if text.startswith('"') and text.endswith('"'):
            return text
        return f'"{text}"'
    return text


def _example_value(param: Dict[str, Any], *, prefer_default: bool) -> str:
    name = param["name"]
    default_text = _format_cli_literal(param.get("default"))
    if not prefer_default:
        hint = _EXTENDED_HELP_EXAMPLE_HINTS.get(name)
        if callable(hint):
            try:
                return str(hint(param))
            except Exception:
                pass
        if isinstance(hint, str):
            return hint
    if prefer_default and default_text is not None:
        return default_text
    if not prefer_default and default_text is not None:
        return default_text
    ptype = param.get("type")
    if ptype is int:
        return "10"
    if ptype is float:
        return "0.1"
    if ptype is bool:
        return "true"
    if ptype in (list, tuple):
        return "a,b"
    return f"<{name}>"


def _build_usage_examples(
    cmd_name: str, func_info: Dict[str, Any]
) -> Tuple[str, Optional[str]]:
    override = _COMMAND_USAGE_EXAMPLES.get(cmd_name)
    if override:
        return override
    required_tokens: List[str] = []
    optional_tokens: List[str] = []
    for index, param in enumerate(func_info["params"]):
        if param["required"]:
            value = _quote_cli_value(_example_value(param, prefer_default=True))
            if index == 0:
                required_tokens.append(value)
            else:
                required_tokens.append(f"--{param['name'].replace('_', '-')} {value}")
        else:
            value = _example_value(param, prefer_default=False)
            default_text = _format_cli_literal(param.get("default"))
            if value is None:
                continue
            if default_text is not None and value == default_text:
                continue
            optional_tokens.append(
                f"--{param['name'].replace('_', '-')} {_quote_cli_value(value)}"
            )
    base_parts = [cmd_name]
    base_parts.extend(required_tokens)
    base = CLI_PROGRAM + " " + " ".join(base_parts)
    advanced = None
    if optional_tokens:
        adv_parts = base_parts + optional_tokens[:2]
        advanced = CLI_PROGRAM + " " + " ".join(adv_parts)
    return base, advanced


def _match_commands(
    functions: Dict[str, ToolInfo], query: str
) -> List[Tuple[str, ToolInfo, Dict[str, Any]]]:
    tokens = [tok for tok in query.lower().split() if tok]
    if not tokens:
        return []
    matches: List[Tuple[str, ToolInfo, Dict[str, Any]]] = []
    for name, tool in sorted(functions.items()):
        func = tool["func"]
        func_info = tool.setdefault("_cli_func_info", get_function_info(func))
        _apply_schema_overrides(tool, func_info)
        meta = tool.get("meta") or {}
        haystack = " ".join(
            [
                name.lower(),
                str(meta.get("description") or func_info.get("doc") or "").lower(),
            ]
        )
        if all(tok in haystack for tok in tokens):
            matches.append((name, tool, func_info))
    return matches


def _suggest_commands(
    functions: Dict[str, ToolInfo], query: str, *, limit: int = 3
) -> List[str]:
    needle = str(query or "").strip().lower()
    if not needle:
        return []
    name_map = {
        str(name).strip().lower(): str(name)
        for name in functions.keys()
        if str(name).strip()
    }
    if not name_map:
        return []
    matches = difflib.get_close_matches(
        needle, list(name_map.keys()), n=max(1, int(limit)), cutoff=0.45
    )
    return [name_map[name] for name in matches]


def _extract_help_query(argv: List[str]) -> Optional[str]:
    for flag in ("--help", "-h"):
        if flag in argv:
            idx = argv.index(flag)
            query_tokens: List[str] = []
            for token in argv[idx + 1 :]:
                if token.startswith("-"):
                    break
                query_tokens.append(token)
            if query_tokens:
                return " ".join(query_tokens)
    return None


_GLOBAL_FLAG_HELP: Dict[str, str] = {
    "precision": (
        "--precision {auto,compact,display,full,raw}: TOON numeric display precision "
        "(auto compacts most tools but keeps full for forecast/trade analytics; JSON is "
        "always full precision)."
    ),
    "extras": (
        "--extras EXTRA[,EXTRA...]: include richer TOON output sections (e.g. diagnostics, "
        "metadata) that are omitted from compact output by default."
    ),
    "fields": (
        "--fields FIELD[,FIELD...]: return only selected output fields plus envelope metadata."
    ),
    "json": (
        "--json: emit machine-readable JSON instead of TOON (always full precision)."
    ),
    "timeframe": (
        "--timeframe TF: default timeframe; may be supplied before the command for "
        "one-shot sessionless use."
    ),
}


def _match_global_flags(query: str) -> List[tuple[str, str]]:
    token = str(query or "").strip().lower().lstrip("-")
    if not token:
        return []
    return [
        (name, doc)
        for name, doc in _GLOBAL_FLAG_HELP.items()
        if token == name or token in name or name.startswith(token)
    ]


def _print_extended_help(functions: Dict[str, ToolInfo], query: str) -> None:
    def _format_optional_param(param: Dict[str, Any]) -> str:
        name = param["name"]
        default_text = _format_cli_literal(param.get("default"))
        if default_text is None:
            return name
        return f"{name}={default_text}"

    matches = _match_commands(functions, query)
    global_matches = _match_global_flags(query)
    if not matches:
        if global_matches:
            print(f"Global options matching '{query}':")
            for _name, doc in global_matches:
                print(f"  {doc}")
            print(f"\nThese apply to every command. Run `{CLI_PROGRAM} --help` for the full list.")
            return
        print(f"No commands match '{query}'.")
        suggestions = _suggest_commands(functions, query)
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}")
        print("Available commands:")
        for name in sorted(functions.keys()):
            print(f"  {name}")
        print(f"\nTip: run `{CLI_PROGRAM} --help` to view the full list.")
        return
    print(f"Extended help for query: {query}")
    print("")
    for name, tool, func_info in matches:
        meta = tool.get("meta") or {}
        summary = meta.get("description") or _first_line(func_info.get("doc"))
        required = [p["name"] for p in func_info["params"] if p["required"]]
        optional = [
            _format_optional_param(p) for p in func_info["params"] if not p["required"]
        ]
        base_example, advanced_example = _build_usage_examples(name, func_info)
        print(name)
        if summary:
            print(f"  Summary: {summary}")
        if required:
            print(f"  Required: {', '.join(required)}")
        if optional:
            print(f"  Optional: {', '.join(optional)}")
        if name == "trade_place":
            print(
                "  Safety: market orders default to require_sl_tp=true; add both stop_loss and take_profit or explicitly set --require-sl-tp false."
            )
            print(
                "  Recovery: auto_close_on_sl_tp_fail defaults true; set --auto-close-on-sl-tp-fail false only if you will handle unprotected fills manually."
            )
            print(
                "  Preview: dry_run=true is the default; set --dry-run false explicitly to send an order to MT5."
            )
        print(f"  Example: {base_example}")
        if advanced_example and advanced_example != base_example:
            print(f"  Example+: {advanced_example}")
        print(f"  More: {CLI_PROGRAM} {name} --help")
        print("")


def main():
    """Main CLI entry point with dynamic parameter discovery"""
    raw_argv = sys.argv[1:]
    if raw_argv in (["--version"], ["-V"]):
        print(f"{CLI_PROGRAM} {_cli_version()}")
        return 0

    load_environment()
    # Discover only the requested command family for one-shot execution. Root
    # help, search, tools_list, and unknown commands retain full discovery.
    _DISCOVERY_ERRORS.clear()
    raw_command = raw_argv[0] if raw_argv and not raw_argv[0].startswith("-") else ""
    selective_modules = cli_tool_module_names(raw_command)
    functions = (
        discover_tools(selective_modules)
        if selective_modules is not None
        else discover_tools()
    )
    if not functions:
        print("No tools discovered from server module.", file=sys.stderr)
        if _DISCOVERY_ERRORS:
            print(f"Discovery error: {_DISCOVERY_ERRORS[0]}", file=sys.stderr)
            print(
                "Set MTDATA_CLI_DEBUG=1 and rerun for full diagnostics.",
                file=sys.stderr,
            )
        return 1
    argv = _normalize_cli_argv_aliases(sys.argv[1:], functions)
    help_query = _extract_help_query(argv)
    if help_query:
        _print_extended_help(functions, help_query)
        return 0

    parser_prog = os.path.basename(str(sys.argv[0] or "")) or CLI_PROGRAM

    parser = _CLIArgumentParser(
        prog=parser_prog,
        description=_CLI_DESCRIPTION,
        formatter_class=_CLIHelpFormatter,
        epilog=_build_epilog(functions),
        allow_abbrev=False,
        suggest_on_error=True,
        color=_argparse_color_enabled(),
    )
    # Add unified global parameters
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"{CLI_PROGRAM} {_cli_version()}",
        help="Show installed mtdata version and exit.",
    )
    add_global_args_to_parser(parser, exclude_params=["timeframe"])
    parser.add_argument(
        "--timeframe",
        dest="_global_timeframe",
        default=argparse.SUPPRESS,
        metavar="TIMEFRAME",
        help=(
            "Default MT5 timeframe for commands with a timeframe parameter; "
            "command-level --timeframe overrides it."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="<command>"
    )

    shell_parser = subparsers.add_parser(
        "shell",
        help="Run repeated CLI commands in one warm Python process",
        description=(
            "Run an interactive mtdata-cli session. Enter ordinary command lines "
            "without the mtdata-cli prefix; use exit or quit to stop."
        ),
        formatter_class=_CLIHelpFormatter,
        allow_abbrev=False,
    )
    shell_parser.set_defaults(func=lambda _args: run_shell())

    # Dynamically create subparsers for each function, except forecast_generate
    forecast_tool = None
    forecast_tool_info = None
    for cmd_name, tool in sorted(functions.items()):
        func = tool["func"]
        func_info = tool.setdefault("_cli_func_info", get_function_info(func))
        _apply_schema_overrides(tool, func_info)
        meta = tool.get("meta") or {}
        if cmd_name == "forecast_generate":
            forecast_tool = tool
            forecast_tool_info = func_info
            continue

        # Create subparser
        cmd_parser = subparsers.add_parser(
            cmd_name,
            help=(
                (
                    meta.get("description")
                    or (
                        func_info["doc"].split("\n")[0]
                        if func_info["doc"]
                        else f"Execute {cmd_name}"
                    )
                ).replace("%", "%%")
            ),
            formatter_class=_CLIHelpFormatter,
            allow_abbrev=False,
            suggest_on_error=True,
            color=_argparse_color_enabled(),
        )
        if cmd_name in LIVE_TRADE_MUTATION_TOOLS:
            summary = meta.get("description") or (
                func_info["doc"].split("\n")[0]
                if func_info["doc"]
                else f"Execute {cmd_name}"
            )
            cmd_parser.description = (
                f"{str(summary).replace('%', '%%')}\n\n{LIVE_TRADE_MUTATION_WARNING}"
            )

        # Add global parameters to each subparser, excluding any that conflict with function params
        existing_param_names = [p["name"] for p in func_info["params"]]
        exclude_globals = list(existing_param_names)
        if "timeframe" not in existing_param_names:
            exclude_globals.append("timeframe")
        if cmd_name == "report_generate":
            exclude_globals.append("timeframe")
        # Finviz tools don't use MT5 timeframe
        if cmd_name.startswith("finviz_"):
            exclude_globals.append("timeframe")
        if cmd_name in _TIMEFRAMELESS_GLOBAL_COMMANDS:
            exclude_globals.append("timeframe")
        add_global_args_to_parser(
            cmd_parser, exclude_params=exclude_globals, suppress_defaults=True
        )

        # Add dynamic arguments
        add_dynamic_arguments(
            cmd_parser, func_info, meta.get("param_docs"), cmd_name=cmd_name
        )

        # Set the command function
        cmd_parser.set_defaults(
            func=create_command_function(func_info, cmd_name, cmd_parser=cmd_parser)
        )

    # Custom forecast_generate parser (grouped UX)
    if forecast_tool is not None:
        cmd_name = "forecast_generate"
        func = forecast_tool["func"]
        func_info = forecast_tool_info or get_function_info(func)
        meta = forecast_tool.get("meta") or {}
        cmd_parser = subparsers.add_parser(
            cmd_name,
            help=(
                (
                    meta.get("description") or func_info["doc"].split("\n")[0]
                    if func_info["doc"]
                    else f"Execute {cmd_name}"
                ).replace("%", "%%")
            ),
            formatter_class=_CLIHelpFormatter,
            allow_abbrev=False,
            suggest_on_error=True,
            color=_argparse_color_enabled(),
        )
        # Add global parameters to each subparser, excluding any that conflict
        exclude_globals = ["symbol", "timeframe"]
        add_global_args_to_parser(
            cmd_parser, exclude_params=exclude_globals, suppress_defaults=True
        )
        _add_forecast_generate_args(cmd_parser)

        def _forecast_generate_cmd(args):
            try:
                overrides = _parse_set_overrides(args.set_overrides)
            except ValueError as exc:
                cmd_parser.error(str(exc))

            symbol = getattr(args, "symbol", None)
            if symbol in (None, ""):
                _render_cli_result(
                    {"error": "Missing required argument(s): symbol."},
                    args=args,
                    cmd_name="forecast_generate",
                )
                return 1

            params_raw = _resolve_forecast_typed_cli_value(
                args.params,
                key="params",
                overrides=overrides,
                parser=cmd_parser,
            )
            denoise_raw = _resolve_forecast_typed_cli_value(
                args.denoise,
                key="denoise",
                overrides=overrides,
                parser=cmd_parser,
            )
            features_raw = _resolve_forecast_typed_cli_value(
                args.features,
                key="features",
                overrides=overrides,
                parser=cmd_parser,
            )
            dimred_params_raw = _resolve_forecast_typed_cli_value(
                args.dimred_params,
                key="dimred_params",
                overrides=overrides,
                parser=cmd_parser,
            )
            target_spec_raw = _resolve_forecast_typed_cli_value(
                args.target_spec,
                key="target_spec",
                overrides=overrides,
                parser=cmd_parser,
            )

            params = (
                _parse_kv_string(params_raw)
                if isinstance(params_raw, str)
                else params_raw
            )

            denoise = None
            if isinstance(denoise_raw, dict):
                denoise = dict(denoise_raw)
            elif denoise_raw:
                denoise = {"method": str(denoise_raw).strip()}
                if str(denoise_raw).strip().startswith("{"):
                    parsed = _parse_kv_string(str(denoise_raw))
                    denoise = parsed if parsed is not None else denoise

            features = (
                _parse_kv_string(features_raw)
                if isinstance(features_raw, str)
                else features_raw
            )
            dimred_params = (
                _parse_kv_string(dimred_params_raw)
                if isinstance(dimred_params_raw, str)
                else dimred_params_raw
            )
            target_spec = (
                _parse_kv_string(target_spec_raw)
                if isinstance(target_spec_raw, str)
                else target_spec_raw
            )

            # --set overrides (sections: method/denoise/features/dimred/target)
            params = _merge_dict(params, overrides.get("method"))
            denoise = _merge_dict(denoise, overrides.get("denoise"))
            features = _merge_dict(features, overrides.get("features"))
            dimred_params = _merge_dict(dimred_params, overrides.get("dimred"))
            target_spec = _merge_dict(target_spec, overrides.get("target"))

            request = ForecastGenerateRequest(
                symbol=symbol,
                timeframe=args.timeframe,
                library=args.library,
                method=args.method,
                horizon=int(args.horizon),
                lookback=args.lookback,
                as_of=args.as_of,
                start=args.start,
                end=args.end,
                params=params,
                ci_alpha=args.ci_alpha,
                quantity=args.quantity,
                proxy=args.proxy,
                denoise=cast(Any, denoise or None),
                features=features or None,
                dimred_method=args.dimred_method,
                dimred_params=dimred_params or None,
                target_spec=target_spec or None,
                async_mode=bool(args.async_mode),
                model_id=args.model_id,
                detail=resolve_output_contract(args).detail,
            )

            if getattr(args, "print_config", False):
                config_output = _format_result_for_cli(
                    {"forecast_generate": request.model_dump(mode="json")},
                    fmt=_resolve_cli_formatter(args),
                    verbose=True,
                    cmd_name="forecast_generate",
                    precision=getattr(args, "precision", None),
                )
                if config_output:
                    _write_cli_text(config_output)
                return 0

            out = _invoke_cli_tool_function(
                func,
                args=args,
                cmd_name="forecast_generate",
                kwargs={
                    "request": request,
                    "extras": getattr(args, "extras", None),
                    "fields": getattr(args, "fields", None),
                    "__cli_raw": True,
                },
            )
            return _render_cli_result_status(
                out,
                args=args,
                cmd_name="forecast_generate",
            )

        cmd_parser.set_defaults(func=_forecast_generate_cmd)

        # forecast_generate uses a custom parser, but belongs in the same
        # alphabetical top-level command list as dynamically generated tools.
        _sort_subparser_help_choices(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)
    args = _apply_global_cli_overrides(args, argv)
    args = _apply_cli_output_mode_defaults(args, argv, functions)

    if not args.command:
        parser.print_help()
        return 1

    output_contract = _resolve_cli_output_contract_or_error(parser, args)
    _configure_cli_logging(verbose=output_contract.verbose)

    try:
        status = args.func(args)
        if isinstance(status, int):
            return int(status)
        return 0
    except KeyboardInterrupt:
        print("\nAborted by user", file=sys.stderr)
        return 1
    except Exception as e:
        if _debug_enabled():
            import traceback

            traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run_shell() -> int:
    """Run repeated CLI commands while reusing the initialized Python process."""
    print("mtdata-cli shell (type 'exit' or 'quit' to stop)")
    original_argv = list(sys.argv)
    try:
        while True:
            try:
                line = input("mtdata> ")
            except EOFError:
                print("")
                return 0
            except KeyboardInterrupt:
                print("")
                continue
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower() in {"exit", "quit"}:
                return 0
            try:
                command_argv = shlex.split(stripped, posix=False)
            except ValueError as exc:
                print(f"Invalid command line: {exc}", file=sys.stderr)
                continue
            if command_argv and command_argv[0].lower() == "shell":
                print("A shell session is already active.", file=sys.stderr)
                continue
            sys.argv = [original_argv[0], *command_argv]
            main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
