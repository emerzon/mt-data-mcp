#!/usr/bin/env python3
"""
Dynamic CLI wrapper for testing MetaTrader5 MCP server functions
Automatically discovers function parameters and creates CLI arguments
"""

import argparse
import sys
import os
import types
import math
import json
from typing import get_origin, get_args, Optional, Dict, Any, List, Tuple, Literal, Union, is_typeddict
from pydantic import BaseModel
from .config import load_environment

from ..bootstrap.tools import bootstrap_tools
from ..forecast.requests import ForecastGenerateRequest
from ..utils.minimal_output import format_result_minimal as _shared_minimal
from ._mcp_instance import mcp
from ._mcp_tools import get_tool_registry as get_registered_tools
from .cli_formatting import (
    CLI_FORMAT_JSON,
    CLI_FORMAT_TOON,
    _attach_cli_meta,
    _build_cli_timezone_meta,
    _build_cli_timezone_meta_brief,
    _format_result_for_cli,
    _format_result_minimal,
    _json_default,
    _normalize_cli_formatter,
    _resolve_cli_formatter,
    _safe_tz_name,
    _sanitize_json_compat,
)
from .cli_discovery import (
    add_dynamic_arguments as _add_dynamic_arguments_impl,
    apply_schema_overrides as _apply_schema_overrides_impl,
    discover_tools as _discover_tools_impl,
    extract_function_from_tool_obj as _extract_function_from_tool_obj_impl,
    extract_metadata_from_tool_obj as _extract_metadata_from_tool_obj_impl,
    get_function_info as _get_function_info_impl,
    resolve_param_kwargs as _resolve_param_kwargs_impl,
)
from .cli_runtime import (
    coerce_cli_scalar as _coerce_cli_scalar_impl,
    create_command_function as _create_command_function_impl,
    merge_dict as _merge_dict_impl,
    normalize_cli_list_value as _normalize_cli_list_value_impl,
    parse_kv_string as _parse_kv_string_impl,
    parse_set_overrides as _parse_set_overrides_impl,
)

# Simple debug logging controlled by env var MTDATA_CLI_DEBUG
def _debug_enabled() -> bool:
    try:
        v = os.environ.get("MTDATA_CLI_DEBUG", "").strip().lower()
        return v not in ("", "0", "false", "no")
    except Exception:
        return False


def _debug(msg: str) -> None:
    if _debug_enabled():
        try:
            print(f"[cli-debug] {msg}", file=sys.stderr)
        except Exception:
            pass


def _argparse_color_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


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

from .unified_params import add_global_args_to_parser
from .server_utils import get_mcp_registry
from .schema import enrich_schema_with_shared_defs, get_function_info as _schema_get_function_info, PARAM_HINTS as _PARAM_HINTS

# Types for discovered metadata
ToolInfo = Dict[str, Any]

CLI_PROGRAM = "mtdata-cli"


def _is_pydantic_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


def _iter_request_model_params(model_type: type[BaseModel]) -> List[Dict[str, Any]]:
    fields = getattr(model_type, "model_fields", None)
    if isinstance(fields, dict):
        params: List[Dict[str, Any]] = []
        for name, field in fields.items():
            required = bool(field.is_required()) if callable(getattr(field, "is_required", None)) else False
            default = None if required else getattr(field, "default", None)
            if getattr(default, "__class__", None).__name__ == "PydanticUndefinedType":
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

    legacy_fields = getattr(model_type, "__fields__", None)
    if isinstance(legacy_fields, dict):
        return [
            {
                "name": name,
                "required": bool(getattr(field, "required", False)),
                "default": None if getattr(field, "required", False) else getattr(field, "default", None),
                "type": getattr(field, "outer_type_", Any) or Any,
            }
            for name, field in legacy_fields.items()
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


def _model_dump_compat(model: BaseModel) -> Dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if callable(dump):
        return dump()
    return model.dict()


def _argv_option_present_after_command(argv: List[str], command: str, option: str) -> bool:
    try:
        cmd_index = argv.index(command)
    except ValueError:
        return False
    for token in argv[cmd_index + 1 :]:
        if token == option or token.startswith(f"{option}="):
            return True
    return False


def _apply_global_cli_overrides(args: Any, argv: List[str]) -> Any:
    command = getattr(args, "command", None)
    if not isinstance(command, str) or not command:
        return args
    global_timeframe = getattr(args, "_global_timeframe", None)
    if global_timeframe is not None and not _argv_option_present_after_command(argv, command, "--timeframe"):
        setattr(args, "timeframe", global_timeframe)
    if command == "trade_history":
        history_days = getattr(args, "_trade_history_days", None)
        if history_days is not None and not (
            _argv_option_present_after_command(argv, command, "--minutes-back")
            or _argv_option_present_after_command(argv, command, "--minutes_back")
        ):
            try:
                setattr(args, "minutes_back", int(round(float(history_days) * 1440.0)))
            except Exception:
                setattr(args, "minutes_back", history_days)
    return args


def _format_result_minimal(result: Any, verbose: bool = True) -> str:
    try:
        return _shared_minimal(result, verbose=verbose)
    except Exception:
        return str(result) if result is not None else ""


def _prepare_result_for_cli_display(result: Any, *, cmd_name: str, verbose: bool) -> Any:
    if bool(verbose):
        return result

    cmd = str(cmd_name or "").strip()

    def _drop_keys_from_rows(value: Any, keys: set[str]) -> Any:
        if not isinstance(value, list):
            return value
        rows_out: List[Any] = []
        for row in value:
            if isinstance(row, dict):
                rows_out.append({k: v for k, v in row.items() if k not in keys})
            else:
                rows_out.append(row)
        return rows_out

    def _drop_keys_from_dict(value: Any, keys: set[str]) -> Any:
        if not isinstance(value, dict):
            return value
        return {k: v for k, v in value.items() if k not in keys}

    if isinstance(result, list):
        if cmd == "trade_get_open":
            return _drop_keys_from_rows(
                result,
                {
                    "Comment Length",
                    "Comment Limit",
                    "Comment May Be Truncated",
                },
            )
        if cmd == "trade_history":
            return _drop_keys_from_rows(
                result,
                {
                    "comment_visible_length",
                    "comment_max_length",
                    "comment_may_be_truncated",
                    "type_code",
                    "entry_code",
                    "reason_code",
                    "time_msc",
                },
            )
        if cmd == "trade_get_pending":
            if (
                len(result) == 1
                and isinstance(result[0], dict)
                and set(result[0].keys()) == {"message"}
            ):
                return {"count": 0, "message": result[0]["message"]}
            return _drop_keys_from_rows(
                result,
                {
                    "Comment Length",
                    "Comment Limit",
                    "Comment May Be Truncated",
                },
            )
        return result

    if not isinstance(result, dict):
        return result

    out = dict(result)

    if cmd == "data_fetch_candles":
        meta = out.get("meta")
        if isinstance(meta, dict):
            meta_out = dict(meta)
            diagnostics = meta_out.get("diagnostics")
            if isinstance(diagnostics, dict):
                diagnostics_out = dict(diagnostics)
                query = diagnostics_out.get("query")
                if isinstance(query, dict):
                    query_out = dict(query)
                    query_out.pop("warmup_bars", None)
                    warmup_retry = query_out.get("warmup_retry")
                    if isinstance(warmup_retry, dict):
                        retry_out = dict(warmup_retry)
                        retry_out.pop("warmup_bars", None)
                        if retry_out:
                            query_out["warmup_retry"] = retry_out
                        else:
                            query_out.pop("warmup_retry", None)
                    diagnostics_out["query"] = query_out
                meta_out["diagnostics"] = diagnostics_out
            out["meta"] = meta_out

    if cmd == "forecast_volatility_estimate":
        out.pop("params_explained", None)

    if cmd == "forecast_barrier_prob":
        out.pop("tp_hit_prob_by_t", None)
        out.pop("sl_hit_prob_by_t", None)

    if cmd == "data_fetch_ticks":
        out.pop("start_epoch", None)
        out.pop("end_epoch", None)
        out.pop("stats_display", None)

    if cmd == "market_ticker" and out.get("time_display") is not None:
        out.pop("time", None)

    if cmd == "market_depth_fetch":
        data = out.get("data")
        if isinstance(data, dict) and data.get("time_display") is not None:
            data_out = dict(data)
            data_out.pop("time", None)
            out["data"] = data_out

    if cmd == "causal_discover_signals":
        data = out.get("data")
        if isinstance(data, dict):
            data_out = dict(data)
            data_out.pop("summary_text", None)
            out["data"] = data_out

    if cmd == "regime_detect":
        params_used = out.get("params_used")
        if isinstance(params_used, dict):
            params_out = dict(params_used)

            auto_cal = params_out.get("auto_calibration")
            if isinstance(auto_cal, dict):
                auto_cal_out = _drop_keys_from_dict(
                    auto_cal,
                    {
                        "sigma",
                        "kurtosis_excess",
                        "jump_share_abs_z_ge_2_5",
                        "trend_strength",
                        "move_zscore",
                        "vol_norm",
                        "kurt_norm",
                        "jump_norm",
                        "trend_norm",
                        "move_sig_norm",
                        "sensitivity",
                        "hazard_floor",
                        "hazard_cap",
                    },
                )
                params_out["auto_calibration"] = auto_cal_out

            threshold_cal = params_out.get("cp_threshold_calibration")
            if isinstance(threshold_cal, dict):
                threshold_cal_out = _drop_keys_from_dict(
                    threshold_cal,
                    {
                        "base_threshold",
                        "window",
                        "step",
                        "windows_used",
                        "bootstrap_runs",
                        "null_scores_count",
                        "null_max_quantile",
                        "quantile",
                        "threshold_delta",
                        "error",
                    },
                )
                params_out["cp_threshold_calibration"] = threshold_cal_out

            out["params_used"] = params_out

    return out


def _format_result_for_cli(result: Any, *, fmt: str, verbose: bool, cmd_name: str) -> str:
    fmt_s = _normalize_cli_formatter(fmt)
    if fmt_s == CLI_FORMAT_JSON:
        payload = {"text": result} if isinstance(result, str) else result
        payload = _sanitize_json_compat(payload)
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
    result = _prepare_result_for_cli_display(result, cmd_name=cmd_name, verbose=verbose)
    if isinstance(result, str):
        return result
    simplify_numbers = not str(cmd_name or "").startswith("trade_")
    try:
        return _shared_minimal(result, verbose=verbose, simplify_numbers=simplify_numbers)
    except TypeError:
        return _format_result_minimal(result, verbose=verbose)


def _normalize_console_text(text: str) -> str:
    normalized = str(text)
    for src, dst in {
        "\u2192": "->",
        "\u2190": "<-",
        "\u2026": "...",
    }.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _write_cli_text(text: str, *, stream: Any = None) -> None:
    target = stream if stream is not None else sys.stdout
    payload = str(text)
    rendered = payload if payload.endswith("\n") else f"{payload}\n"
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


def _render_cli_result(result: Any, *, args: Any, cmd_name: str) -> None:
    verbose = bool(getattr(args, "verbose", False))
    result = _attach_cli_meta(result, cmd_name=cmd_name, verbose=verbose)
    output = _format_result_for_cli(
        result,
        fmt=_resolve_cli_formatter(args),
        verbose=verbose,
        cmd_name=cmd_name,
    )
    if output:
        _write_cli_text(output)


def _result_has_tool_error(result: Any) -> bool:
    if isinstance(result, dict):
        if bool(result.get("no_action", False)):
            return True
        err = result.get("error")
        if isinstance(err, str):
            return bool(err.strip())
        return err not in (None, False)
    if isinstance(result, str):
        return result.strip().lower().startswith("error:")
    return False


def _safe_argument_parser(*args: Any, **kwargs: Any) -> argparse.ArgumentParser:
    try:
        return argparse.ArgumentParser(*args, **kwargs)
    except TypeError:
        fallback = dict(kwargs)
        fallback.pop("suggest_on_error", None)
        fallback.pop("color", None)
        return argparse.ArgumentParser(*args, **fallback)


def _safe_add_subparser(subparsers: Any, name: str, **kwargs: Any) -> argparse.ArgumentParser:
    try:
        return subparsers.add_parser(name, **kwargs)
    except TypeError:
        fallback = dict(kwargs)
        fallback.pop("suggest_on_error", None)
        fallback.pop("color", None)
        return subparsers.add_parser(name, **fallback)

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

def _apply_schema_overrides(tool: ToolInfo, func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Apply schema metadata to the introspected CLI param info."""
    return _apply_schema_overrides_impl(
        tool,
        func_info,
        enrich_schema_with_shared_defs=enrich_schema_with_shared_defs,
    )


def _extract_function_from_tool_obj(tool_obj):
    """Best-effort extraction of the underlying function from an MCP tool object."""
    return _extract_function_from_tool_obj_impl(tool_obj)

def _extract_metadata_from_tool_obj(tool_obj) -> Dict[str, Any]:
    """Attempt to extract description and parameter docs from an MCP tool object.

    Returns a dict with keys:
    - description: Optional[str]
    - param_docs: Dict[str, str]
    """
    return _extract_metadata_from_tool_obj_impl(tool_obj)


def _is_union_origin(origin: Any) -> bool:
    return origin in (Union, types.UnionType) or str(origin) in {"typing.Union", "<class 'typing.Union'>"}


def _is_literal_origin(origin: Any) -> bool:
    return origin is Literal or str(origin) in {"typing.Literal", "<class 'typing.Literal'>"}

def discover_tools():
    """Discover MCP tools from the shared bootstrap registry.

    Priority:
    1) Use the shared tool registry after bootstrap
    2) Use the MCP registry if available
    3) Fallback to scanning bootstrapped tool modules
    """
    return _discover_tools_impl(
        bootstrap_tools=bootstrap_tools,
        get_registered_tools=get_registered_tools,
        mcp=mcp,
        get_mcp_registry=get_mcp_registry,
        debug=_debug,
        extract_function_from_tool_obj=_extract_function_from_tool_obj,
        extract_metadata_from_tool_obj=_extract_metadata_from_tool_obj,
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

def add_dynamic_arguments(parser, param_info, param_docs: Optional[Dict[str, str]] = None, cmd_name: Optional[str] = None):
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


def _normalize_cli_list_value(value: Any) -> Any:
    """Normalize CLI list values from comma/space/JSON forms into a flat list."""
    return _normalize_cli_list_value_impl(value)


def _coerce_cli_scalar(v: str) -> Any:
    return _coerce_cli_scalar_impl(v)


def _parse_set_overrides(items: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """Parse repeated --set entries like 'model.sp=24' into nested dicts."""
    return _parse_set_overrides_impl(items, coerce_cli_scalar=_coerce_cli_scalar)


def _merge_dict(dst: Optional[Dict[str, Any]], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _merge_dict_impl(dst, src)


_FORECAST_TYPED_ARG_SPECS: Dict[str, Dict[str, Any]] = {
    "model_params": {
        "flag": "--model-params",
        "section": "model",
        "metavar": "JSON|k=v",
        "help": "Model params as JSON or key=value pairs.",
        "examples": [
            '--model-params "window_size=64 top_k=20"',
            '--model-params \'{"window_size":64,"top_k":20}\'',
            '--model-params --set model.window_size=64 --set model.top_k=20',
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
            '--denoise --set denoise.method=ema',
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
            '--features --set features.include=open,high',
        ],
    },
    "dimred_params": {
        "flag": "--dimred-params",
        "section": "dimred",
        "metavar": "JSON|k=v",
        "help": "Dimred params as JSON or key=value pairs.",
        "examples": [
            '--dimred-params "n_components=4"',
            '--dimred-params \'{"n_components":4}\'',
            '--dimred-params --set dimred.n_components=4',
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
            '--target-spec --set target.column=close --set target.transform=log',
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
    for key in ("denoise", "model_params", "features", "dimred_params", "target_spec"):
        spec = _FORECAST_TYPED_ARG_SPECS[key]
        lines.append(f"  {spec['flag']} {spec['metavar']}")
        for example in spec["examples"]:
            lines.append(f"    Example: {CLI_PROGRAM} forecast_generate SYMBOL {example}")
    lines.append("  --set SECTION.KEY=VALUE")
    lines.append(f"    Example: {CLI_PROGRAM} forecast_generate SYMBOL --set model.window_size=64")
    return "\n".join(lines)


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


def _add_forecast_generate_args(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.description = "Generate forecasts with an optional preprocessing pipeline."
    cmd_parser.epilog = _forecast_generate_typed_value_epilog()

    cmd_parser.add_argument("symbol", help="Trading symbol.")

    group_model = cmd_parser.add_argument_group("Model")
    group_model.add_argument(
        "--library",
        dest="library",
        type=str,
        choices=["native", "statsforecast", "sktime", "mlforecast", "pretrained"],
        default="native",
        help="Model library.",
    )
    group_model.add_argument(
        "--model",
        "--method",
        dest="model",
        type=str,
        default="theta",
        help="Model name. The legacy --method alias is also accepted.",
    )
    group_model.add_argument(
        "--model-params",
        dest="model_params",
        type=str,
        nargs="?",
        const="__PRESENT__",
        default=None,
        metavar=_FORECAST_TYPED_ARG_SPECS["model_params"]["metavar"],
        help=_FORECAST_TYPED_ARG_SPECS["model_params"]["help"],
    )

    group_window = cmd_parser.add_argument_group("Window")
    group_window.add_argument("--timeframe", type=str, default="H1", help="MT5 timeframe.")
    group_window.add_argument("--horizon", type=int, default=12, help="Forecast horizon in bars.")
    group_window.add_argument("--lookback", type=int, default=None, help="Historical bars to use.")
    group_window.add_argument("--as-of", dest="as_of", type=str, default=None, help="Reference time override.")

    group_target = cmd_parser.add_argument_group("Target")
    group_target.add_argument(
        "--quantity",
        choices=["price", "return", "volatility"],
        default="price",
        help="Target quantity.",
    )

    group_uncertainty = cmd_parser.add_argument_group("Uncertainty")
    group_uncertainty.add_argument("--ci-alpha", dest="ci_alpha", type=float, default=0.05, help="CI alpha (0.05 => 95%%).")

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
    group_pipe.add_argument("--dimred-method", dest="dimred_method", type=str, default=None, help="Dimred method.")
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
        help="Override nested params (model, denoise, features, dimred, target).",
    )

    group_dbg = cmd_parser.add_argument_group("Debug")
    group_dbg.add_argument("--verbose", action="store_true", default=False, help="Show detailed metadata in output.")
    group_dbg.add_argument(
        "--print-config",
        action="store_true",
        default=False,
        help="Print the resolved forecast config and exit.",
    )


def create_command_function(func_info, cmd_name: str = "", cmd_parser: Optional[argparse.ArgumentParser] = None):
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

def _build_epilog(functions: Dict[str, ToolInfo]) -> str:
    lines = []
    lines.append("Commands and Arguments:")
    for cmd_name, tool in sorted(functions.items()):
        func = tool['func']
        func_info = tool.setdefault('_cli_func_info', get_function_info(func))
        _apply_schema_overrides(tool, func_info)
        arg_strs = []
        for index, param in enumerate(func_info['params']):
            tname = _type_name(param['type']) if param['type'] else 'str'
            if param['required']:
                if index == 0:
                    arg_strs.append(f"{param['name']}<{tname}>")
                else:
                    arg_strs.append(f"--{param['name'].replace('_','-')}<{tname}>")
            else:
                default = param.get('default')
                arg_strs.append(
                    f"--{param['name'].replace('_','-')}<{tname}>=[{default}]"
                )
        meta = tool.get('meta') or {}
        desc = meta.get('description') or _first_line(func_info.get('doc'))
        lines.append(f"  {cmd_name}: {' '.join(arg_strs) if arg_strs else '(no args)'}")
        if desc:
            lines.append(f"    - {desc}")
    lines.append("")
    lines.append("Tip: Use `--help <keyword>` to search commands and examples.")
    lines.append("Type Conventions:")
    lines.append("  - int: integer")
    lines.append("  - str: string")
    lines.append("  - bool: pass true|false (e.g., --flag true)")
    lines.append("")
    lines.append("General Examples:")
    lines.append("  # Basic forecast with a native model")
    lines.append(f"  {CLI_PROGRAM} forecast_generate EURUSD --library native --model theta --timeframe H1 --horizon 24")
    lines.append("")
    lines.append("  # Foundation model (Chronos-2) with covariates")
    lines.append(f"  {CLI_PROGRAM} forecast_generate BTCUSD --library pretrained --model chronos2 --timeframe H1 --horizon 12 \\")
    lines.append("    --features \"include=open,high future_covariates=hour,dow,is_holiday\" \\")
    lines.append("    --verbose")
    lines.append("")
    lines.append("  # Rolling backtest for accuracy check")
    lines.append(f"  {CLI_PROGRAM} forecast_backtest_run EURUSD --timeframe H1 --methods theta,seasonal_naive \\")
    lines.append("    --steps 5 --horizon 12")
    return "\n".join(lines)



_EXTENDED_HELP_EXAMPLE_HINTS: Dict[str, Any] = {
    'symbol': 'EURUSD',
    'timeframe': 'H1',
    'method': 'nhits',
    'library': 'native',
    'model': 'theta',
    'model_params': '"sp=24"',
    'methods': 'theta nhits',
    'horizon': '8',
    'lookback': '200',
    'steps': '5',
    'spacing': '20',
    'quantity': 'return',
    'target': 'price',
    'ci_alpha': '0.1',
    'params': '"max_epochs=20"',
    'features': '"include=open,high future_covariates=hour,dow"',
    'as_of': '2025-09-01T12:00:00Z',
    'population': '16',
    'generations': '5',
    'seed': '42',
}

_COMMAND_USAGE_EXAMPLES: Dict[str, Tuple[str, Optional[str]]] = {
    "patterns_detect": (
        f"{CLI_PROGRAM} patterns_detect BTCUSD --timeframe H1 --mode candlestick",
        f"{CLI_PROGRAM} patterns_detect BTCUSD --timeframe H1 --mode classic --detail full --limit 300",
    ),
    "pivot_compute_points": (
        f"{CLI_PROGRAM} pivot_compute_points BTCUSD --timeframe D1",
        None,
    ),
    "regime_detect": (
        f"{CLI_PROGRAM} regime_detect BTCUSD --timeframe H1 --method hmm",
        f"{CLI_PROGRAM} regime_detect BTCUSD --timeframe H1 --method hmm --output full --verbose",
    ),
    "trade_risk_analyze": (
        f"{CLI_PROGRAM} trade_risk_analyze --symbol BTCUSD --direction long --desired-risk-pct 1 --proposed-entry 66317 --proposed-sl 65000",
        f"{CLI_PROGRAM} trade_risk_analyze --symbol BTCUSD --direction long --desired-risk-pct 1 --proposed-entry 66317 --proposed-sl 65000 --proposed-tp 69000",
    ),
    "trade_modify": (
        f"{CLI_PROGRAM} trade_modify 123456789 --price 61000",
        f"{CLI_PROGRAM} trade_modify 123456789 --stop-loss 60500 --take-profit 62500",
    ),
    "trade_place": (
        f"{CLI_PROGRAM} trade_place BTCUSD --volume 0.01 --order-type SELL --stop-loss 68521 --take-profit 67071",
        f"{CLI_PROGRAM} trade_place BTCUSD --volume 0.01 --order-type BUY --stop-loss 64500 --take-profit 67200 --comment \"swing long\"",
    ),
    "trade_close": (
        f"{CLI_PROGRAM} trade_close --ticket 123456789",
        f"{CLI_PROGRAM} trade_close --ticket 123456789 --volume 0.05",
    ),
}

_TIMEFRAMELESS_GLOBAL_COMMANDS: set[str] = {
    "forecast_options_chain",
    "forecast_options_expirations",
    "forecast_quantlib_barrier_price",
    "forecast_quantlib_heston_calibrate",
    "indicators_describe",
    "indicators_list",
    "market_ticker",
    "symbols_describe",
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
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def _quote_cli_value(text: str) -> str:
    if text == "":
        return '""'
    if any(ch.isspace() for ch in text):
        if text.startswith('"') and text.endswith('"'):
            return text
        return f'"{text}"'
    return text


def _example_value(param: Dict[str, Any], *, prefer_default: bool) -> str:
    name = param['name']
    default_text = _format_cli_literal(param.get('default'))
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
    ptype = param.get('type')
    if ptype == int:
        return '10'
    if ptype == float:
        return '0.1'
    if ptype == bool:
        return 'true'
    if ptype in (list, tuple):
        return 'a,b'
    return f'<{name}>'


def _build_usage_examples(cmd_name: str, func_info: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    override = _COMMAND_USAGE_EXAMPLES.get(cmd_name)
    if override:
        return override
    required_tokens: List[str] = []
    optional_tokens: List[str] = []
    for index, param in enumerate(func_info['params']):
        if param['required']:
            value = _quote_cli_value(_example_value(param, prefer_default=True))
            if index == 0:
                required_tokens.append(value)
            else:
                required_tokens.append(f"--{param['name'].replace('_','-')} {value}")
        else:
            value = _example_value(param, prefer_default=False)
            default_text = _format_cli_literal(param.get('default'))
            if value is None:
                continue
            if default_text is not None and value == default_text:
                continue
            optional_tokens.append(f"--{param['name'].replace('_','-')} {_quote_cli_value(value)}")
    base_parts = [cmd_name]
    base_parts.extend(required_tokens)
    base = CLI_PROGRAM + " " + " ".join(base_parts)
    advanced = None
    if optional_tokens:
        adv_parts = base_parts + optional_tokens[:2]
        advanced = CLI_PROGRAM + " " + " ".join(adv_parts)
    return base, advanced


def _match_commands(functions: Dict[str, ToolInfo], query: str) -> List[Tuple[str, ToolInfo, Dict[str, Any]]]:
    tokens = [tok for tok in query.lower().split() if tok]
    if not tokens:
        return []
    matches: List[Tuple[str, ToolInfo, Dict[str, Any]]] = []
    for name, tool in sorted(functions.items()):
        func = tool['func']
        func_info = tool.setdefault('_cli_func_info', get_function_info(func))
        _apply_schema_overrides(tool, func_info)
        meta = tool.get('meta') or {}
        haystack = ' '.join([
            name.lower(),
            str(meta.get('description') or func_info.get('doc') or '').lower(),
        ])
        if all(tok in haystack for tok in tokens):
            matches.append((name, tool, func_info))
    return matches


def _extract_help_query(argv: List[str]) -> Optional[str]:
    for flag in ('--help', '-h'):
        if flag in argv:
            idx = argv.index(flag)
            query_tokens: List[str] = []
            for token in argv[idx + 1:]:
                if token.startswith('-'):
                    break
                query_tokens.append(token)
            if query_tokens:
                return ' '.join(query_tokens)
    return None


def _print_extended_help(functions: Dict[str, ToolInfo], query: str) -> None:
    def _format_optional_param(param: Dict[str, Any]) -> str:
        name = param["name"]
        default_text = _format_cli_literal(param.get("default"))
        if default_text is None:
            return name
        return f"{name}={default_text}"

    matches = _match_commands(functions, query)
    if not matches:
        print(f"No commands match '{query}'.")
        print("Available commands:")
        for name in sorted(functions.keys()):
            print(f"  {name}")
        print(f"\nTip: run `{CLI_PROGRAM} --help` to view the full list.")
        return
    print(f"Extended help for query: {query}")
    print("")
    for name, tool, func_info in matches:
        meta = tool.get('meta') or {}
        summary = meta.get('description') or _first_line(func_info.get('doc'))
        required = [p['name'] for p in func_info['params'] if p['required']]
        optional = [_format_optional_param(p) for p in func_info['params'] if not p['required']]
        base_example, advanced_example = _build_usage_examples(name, func_info)
        print(name)
        if summary:
            print(f"  Summary: {summary}")
        if required:
            print(f"  Required: {', '.join(required)}")
        if optional:
            print(f"  Optional: {', '.join(optional)}")
        if name == "trade_place":
            print("  Safety: market orders default to require_sl_tp=true; add both stop_loss and take_profit or explicitly set --require-sl-tp false.")
            print("  Recovery: set --auto-close-on-sl-tp-fail true to try to close a filled order if TP/SL attachment fails.")
        print(f"  Example: {base_example}")
        if advanced_example and advanced_example != base_example:
            print(f"  Example+: {advanced_example}")
        print(f"  More: {CLI_PROGRAM} {name} --help")
        print("")
def main():
    """Main CLI entry point with dynamic parameter discovery"""
    load_environment()
    # Discover functions to expose dynamically
    functions = discover_tools()
    if not functions:
        print("No tools discovered from server module.", file=sys.stderr)
        return 1
    help_query = _extract_help_query(sys.argv[1:])
    if help_query:
        _print_extended_help(functions, help_query)
        return 0

    
    parser = _safe_argument_parser(
        description="Dynamic CLI for MetaTrader5 MCP tools (formatted text output by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(functions),
        suggest_on_error=True,
        color=_argparse_color_enabled(),
    )
    # Add unified global parameters
    add_global_args_to_parser(parser, exclude_params=["timeframe"])
    parser.add_argument(
        "--timeframe",
        dest="_global_timeframe",
        default=argparse.SUPPRESS,
        metavar="TIMEFRAME",
        help="Timeframe for market data (H1, M30, D1, etc.)",
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dynamically create subparsers for each function, except forecast_generate
    forecast_tool = None
    forecast_tool_info = None
    for cmd_name, tool in sorted(functions.items()):
        func = tool['func']
        func_info = tool.setdefault('_cli_func_info', get_function_info(func))
        _apply_schema_overrides(tool, func_info)
        meta = tool.get('meta') or {}
        if cmd_name == "forecast_generate":
            forecast_tool = tool
            forecast_tool_info = func_info
            continue

        # Create subparser
        cmd_parser = _safe_add_subparser(
            subparsers,
            cmd_name, 
            help=((meta.get('description') or func_info['doc'].split('\n')[0] if func_info['doc'] else f"Execute {cmd_name}").replace('%', '%%')),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            suggest_on_error=True,
            color=_argparse_color_enabled(),
        )
        
        # Add global parameters to each subparser, excluding any that conflict with function params
        existing_param_names = [p['name'] for p in func_info['params']]
        exclude_globals = list(existing_param_names)
        if cmd_name == 'report_generate':
            exclude_globals.append('timeframe')
        # Finviz tools don't use MT5 timeframe
        if cmd_name.startswith('finviz_'):
            exclude_globals.append('timeframe')
        if cmd_name in _TIMEFRAMELESS_GLOBAL_COMMANDS:
            exclude_globals.append('timeframe')
        add_global_args_to_parser(cmd_parser, exclude_params=exclude_globals, suppress_defaults=True)
        
        # Add dynamic arguments
        add_dynamic_arguments(cmd_parser, func_info, meta.get('param_docs'), cmd_name=cmd_name)
        
        # Set the command function
        cmd_parser.set_defaults(func=create_command_function(func_info, cmd_name, cmd_parser=cmd_parser))

    # Custom forecast_generate parser (grouped UX)
    if forecast_tool is not None:
        cmd_name = "forecast_generate"
        func = forecast_tool["func"]
        func_info = forecast_tool_info or get_function_info(func)
        meta = forecast_tool.get("meta") or {}
        cmd_parser = _safe_add_subparser(
            subparsers,
            cmd_name,
            help=((meta.get('description') or func_info['doc'].split('\n')[0] if func_info['doc'] else f"Execute {cmd_name}").replace('%', '%%')),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            suggest_on_error=True,
            color=_argparse_color_enabled(),
        )
        # Add global parameters to each subparser, excluding any that conflict
        exclude_globals = ["symbol", "timeframe", "verbose"]  # handled manually
        add_global_args_to_parser(cmd_parser, exclude_params=exclude_globals, suppress_defaults=True)
        _add_forecast_generate_args(cmd_parser)

        def _forecast_generate_cmd(args):
            try:
                overrides = _parse_set_overrides(args.set_overrides)
            except ValueError as exc:
                cmd_parser.error(str(exc))

            model_params_raw = _resolve_forecast_typed_cli_value(
                args.model_params,
                key="model_params",
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

            model_params = _parse_kv_string(model_params_raw) if isinstance(model_params_raw, str) else model_params_raw

            denoise = None
            if isinstance(denoise_raw, dict):
                denoise = dict(denoise_raw)
            elif denoise_raw:
                denoise = {"method": str(denoise_raw).strip()}
                if str(denoise_raw).strip().startswith("{"):
                    parsed = _parse_kv_string(str(denoise_raw))
                    denoise = parsed if parsed is not None else denoise

            features = _parse_kv_string(features_raw) if isinstance(features_raw, str) else features_raw
            if isinstance(features_raw, str) and not features_raw.strip().startswith("{"):
                # Accept shorthand like "include=close,volume" (already handled by parse_kv_or_json)
                pass

            dimred_params = _parse_kv_string(dimred_params_raw) if isinstance(dimred_params_raw, str) else dimred_params_raw
            target_spec = _parse_kv_string(target_spec_raw) if isinstance(target_spec_raw, str) else target_spec_raw

            # --set overrides (sections: model/denoise/features/dimred/target)
            model_params = _merge_dict(model_params, overrides.get("model"))
            denoise = _merge_dict(denoise, overrides.get("denoise"))
            features = _merge_dict(features, overrides.get("features"))
            dimred_params = _merge_dict(dimred_params, overrides.get("dimred"))
            target_spec = _merge_dict(target_spec, overrides.get("target"))

            request = ForecastGenerateRequest(
                symbol=args.symbol,
                timeframe=args.timeframe,
                library=args.library,
                model=args.model,
                horizon=int(args.horizon),
                lookback=args.lookback,
                as_of=args.as_of,
                model_params=model_params,
                ci_alpha=args.ci_alpha,
                quantity=args.quantity,
                denoise=denoise or None,
                features=features or None,
                dimred_method=args.dimred_method,
                dimred_params=dimred_params or None,
                target_spec=target_spec or None,
            )

            if getattr(args, "print_config", False):
                print(_format_result_minimal({"forecast_generate": _model_dump_compat(request)}, verbose=True))
                return 0

            out = func(request=request, __cli_raw=True)
            _render_cli_result(out, args=args, cmd_name="forecast_generate")
            return 1 if _result_has_tool_error(out) else 0

        cmd_parser.set_defaults(func=_forecast_generate_cmd)
    
    # Parse arguments
    args = parser.parse_args()
    args = _apply_global_cli_overrides(args, sys.argv[1:])
    
    if not args.command:
        parser.print_help()
        return 1
    
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

if __name__ == "__main__":
    sys.exit(main())

