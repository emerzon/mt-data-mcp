#!/usr/bin/env python3
"""
Dynamic CLI wrapper for testing MetaTrader5 MCP server functions
Automatically discovers function parameters and creates CLI arguments
"""

import argparse
import sys
import inspect
import os
from typing import get_type_hints, get_origin, get_args, Optional, Dict, Any, List, Tuple, Literal
import json
from ..utils.minimal_output import format_result_minimal as _shared_minimal

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


# Import server module and attempt to discover tools dynamically
try:
    # Ensure .env is loaded for CLI runs too (redundant with server/config, but robust)
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    _env_path = find_dotenv()
    if _env_path:
        load_dotenv(_env_path)
    else:
        load_dotenv()
except Exception as e:
    _debug(f"dotenv load failed: {e}")
from . import server
from .unified_params import add_global_args_to_parser
from .schema import enrich_schema_with_shared_defs, get_function_info as _schema_get_function_info, PARAM_HINTS as _PARAM_HINTS

# Types for discovered metadata
ToolInfo = Dict[str, Any]





# Import string formatting utilities from utils to avoid duplication
from ..utils.minimal_output import (
    _is_scalar_value, _is_empty_value, _minify_number, _stringify_scalar,
    _stringify_cell, _indent_text, _list_of_dicts_to_csv, _format_complex_value
)


def _format_meta_block(meta: Dict[str, Any]) -> str:
    """Delegate to shared format_complex_value for consistency."""
    return _format_complex_value(meta)


def _format_result_minimal(result: Any, verbose: bool = True) -> str:
    # Delegate to shared formatter used by the server so CLI output matches API output exactly
    try:
        return _shared_minimal(result, verbose=verbose)
    except Exception:
        return str(result) if result is not None else ""

def get_function_info(func):
    """Thin wrapper around schema.get_function_info that attaches the callable.

    This avoids duplicating introspection logic while preserving the CLI's
    expectation that the returned dict contains a 'func' key for invocation.
    """
    info = _schema_get_function_info(func)
    info['func'] = func
    # Ensure a minimal doc for CLI help if missing
    if not info.get('doc'):
        info['doc'] = f"Execute {info.get('name') or getattr(func, '__name__', 'function')}"
    # Backfill type defaults to str for any missing types to keep CLI robust
    for p in info.get('params', []):
        if p.get('type') is None:
            p['type'] = str
        if 'required' not in p:
            # Default required based on availability of a default value
            p['required'] = p.get('default') is None
    return info

def _apply_schema_overrides(tool: ToolInfo, func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Apply schema metadata to the introspected CLI param info."""
    meta = tool.setdefault('meta', {})
    schema = meta.get('schema') or {}
    schema = enrich_schema_with_shared_defs(schema, func_info)
    meta['schema'] = schema
    params_obj = schema.get('parameters') if isinstance(schema.get('parameters'), dict) else schema
    schema_props = params_obj.get('properties') if isinstance(params_obj, dict) else {}
    schema_required = set(params_obj.get('required', [])) if isinstance(params_obj, dict) else set()
    for param in func_info.get('params', []):
        prop = schema_props.get(param['name']) if isinstance(schema_props, dict) else None
        if isinstance(prop, dict) and 'default' in prop and param.get('default') is None:
            param['default'] = prop['default']
        if param['name'] in schema_required:
            param['required'] = True
    return schema


def _extract_function_from_tool_obj(tool_obj):
    """Best-effort extraction of the underlying function from an MCP tool object."""
    # Common attributes we might find in registry entries
    for attr in ("func", "function", "callable", "handler", "wrapped", "_func"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    # Some registries may store the function directly
    if callable(tool_obj):
        return tool_obj
    return None

def _extract_metadata_from_tool_obj(tool_obj) -> Dict[str, Any]:
    """Attempt to extract description and parameter docs from an MCP tool object.

    Returns a dict with keys:
    - description: Optional[str]
    - param_docs: Dict[str, str]
    """
    meta: Dict[str, Any] = {"description": None, "param_docs": {}, "schema": None}

    # Direct description fields
    for attr in ("description", "doc", "docs"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, str) and val.strip():
            meta["description"] = val.strip()
            break

    # JSON schema-like fields
    schema = None
    for attr in ("schema", "input_schema", "parameters", "spec"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, dict) and val:
            schema = val
            break

    if schema:
        meta["schema"] = schema
        # Top-level description
        if not meta["description"] and isinstance(schema.get("description"), str):
            meta["description"] = schema.get("description")
        # Parameters (OpenAI/MCP-style JSON schema)
        params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
        props = params_obj.get("properties") if isinstance(params_obj, dict) else None
        if isinstance(props, dict):
            for pname, pdef in props.items():
                desc = pdef.get("description") if isinstance(pdef, dict) else None
                if isinstance(desc, str) and desc.strip():
                    meta["param_docs"][pname] = desc.strip()

    return meta

def discover_tools():
    """Discover MCP tools from the server.

    Priority:
    1) Use server.get_tool_registry() if available
    2) Use server.mcp registry if available
    2) Fallback to scanning public callables in server module (excluding helpers)
    """
    tools: Dict[str, ToolInfo] = {}

    mcp = getattr(server, 'mcp', None)
    registry = None
    try:
        get_reg = getattr(server, "get_tool_registry", None)
        if callable(get_reg):
            reg = get_reg()
            if reg and hasattr(reg, "items"):
                registry = reg
    except Exception as e:
        _debug(f"get_tool_registry failed: {e}")
    if mcp is not None:
        # Try common registry attribute names on FastMCP
        for attr in ("tools", "_tools", "registry", "tool_registry", "_tool_registry"):
            reg = getattr(mcp, attr, None)
            if reg and hasattr(reg, 'items'):
                registry = reg
                break

    if registry:
        pkg_prefix = server.__name__.rsplit('.', 1)[0] + '.'
        for name, obj in registry.items():
            func = _extract_function_from_tool_obj(obj)
            mod = getattr(func, '__module__', None) if func else None
            if func and isinstance(mod, str) and (mod == server.__name__ or mod.startswith(pkg_prefix)):
                meta = _extract_metadata_from_tool_obj(obj)
                tools[name] = {"func": func, "meta": meta}

    if not tools:
        # Fallback: scan server module for likely tool functions
        for name in dir(server):
            if name.startswith('_'):
                continue
            if name in {"main", "MT5Connection"}:  # skip non-tool exports
                continue
            obj = getattr(server, name)
            if callable(obj) and getattr(obj, '__module__', None) == server.__name__:
                # Heuristic: prefer functions with a docstring and at least 0-5 params
                try:
                    sig = inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                if isinstance(obj, type):
                    continue  # skip classes
                if name.endswith(('_wrapper',)):
                    continue
                # Avoid internal helpers
                if name in {"_group_symbols", "_auto_connect_wrapper"}:
                    continue
                tools[name] = {"func": obj, "meta": {"description": None, "param_docs": {}}}

    return tools

def _resolve_param_kwargs(
    param: Dict[str, Any],
    param_docs: Optional[Dict[str, str]],
    cmd_name: Optional[str] = None,
    param_names: Optional[set] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Resolve CLI argument kwargs and determine if parameter is a mapping type."""
    def _escape_argparse_help(text: Optional[str]) -> Optional[str]:
        # argparse expands help strings using old-style % formatting; escape literal percents.
        return text.replace('%', '%%') if isinstance(text, str) else text

    def _looks_like_forecast_method_literal(ptype: Any) -> bool:
        try:
            origin = get_origin(ptype)
            if origin is not Literal:
                return False
            args = set(str(v) for v in get_args(ptype) if v is not None)
            # Heuristic: Forecast methods always include at least one of these canonical names.
            return bool(args.intersection({'theta', 'naive', 'arima', 'chronos2', 'statsforecast'}))
        except Exception:
            return False

    desc = None
    if param_docs and param['name'] in param_docs:
        desc = param_docs[param['name']]
    hint = desc or _PARAM_HINTS.get(param['name'])
    kwargs = {'help': _escape_argparse_help(hint) or f"{param['name']} parameter", 'dest': param['name']}
    is_mapping_type = False

    # Dynamically populate choices for 'method' parameter
    if param['name'] == 'method' and (
        (cmd_name in {'forecast_generate', 'forecast_conformal_intervals', 'forecast_tune_genetic'})
        or _looks_like_forecast_method_literal(param.get('type'))
    ):
        # If the tool exposes the newer (library, model) selection, don't explode
        # help output by enumerating every possible method name.
        if param_names and ('library' in param_names or 'model' in param_names):
            # No choices -> free string. Users can use --library/--model for guided selection.
            pass
        else:
            try:
                from mtdata.forecast.registry import ForecastRegistry

                # Best-effort import: optional method modules may fail to import if their
                # third-party deps are missing; still surface whatever registers successfully.
                for mod_name in (
                    "mtdata.forecast.methods.classical",
                    "mtdata.forecast.methods.ets_arima",
                    "mtdata.forecast.methods.statsforecast",
                    "mtdata.forecast.methods.mlforecast",
                    "mtdata.forecast.methods.pretrained",
                    "mtdata.forecast.methods.neural",
                    "mtdata.forecast.methods.sktime",
                    "mtdata.forecast.methods.analog",
                    "mtdata.forecast.methods.monte_carlo",
                ):
                    try:
                        __import__(mod_name)
                    except Exception as import_ex:
                        _debug(f"Skipping method module import '{mod_name}': {import_ex}")

                kwargs['choices'] = ForecastRegistry.get_all_method_names()
            except Exception as e:
                _debug(f"Failed to dynamically load forecast methods for CLI: {e}")
                # Fallback to static type choices if dynamic loading fails
                ptype = param.get('type')
                origin = get_origin(ptype)
                if origin is Literal:
                    kwargs['choices'] = [str(v) for v in get_args(ptype) if v is not None]
    else:
        # Handle other types
        try:
            ptype = param.get('type')
            origin = get_origin(ptype)
            
            # Optional[T] -> T Unwrap first
            base_type = ptype
            if origin is not None and str(origin).endswith('Union'):
                args = [a for a in get_args(ptype) if a is not type(None)]
                if len(args) == 1:
                    base_type = args[0]
                    origin = get_origin(base_type)

            # Check for mapping type on the unwrapped base type
            is_typed_dict = hasattr(base_type, '__annotations__') and isinstance(getattr(base_type, '__annotations__', {}), dict)
            is_mapping_type = (base_type in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict

            # Default to string parsing unless we can provide a better type.
            kwargs['type'] = str

            if base_type in (int, float, str):
                kwargs['type'] = base_type
            elif base_type is bool:
                kwargs['type'] = str
                kwargs['choices'] = ['true', 'false']
                kwargs['metavar'] = 'bool'

            
            if origin in (list, tuple):
                inner = get_args(ptype)[0] if get_args(ptype) else None
                inner_origin = get_origin(inner)
                if inner_origin and str(inner_origin).endswith('Literal'):
                    choices = [str(v) for v in get_args(inner)]
                    if choices:
                        kwargs['choices'] = choices
                    kwargs['type'] = str
                    kwargs['nargs'] = '+'
                else:
                    kwargs['type'] = str
            elif origin and str(origin).endswith('Literal'):
                choices = [str(v) for v in get_args(base_type)]
                if choices:
                    kwargs['choices'] = choices
                kwargs['type'] = str
        except Exception as e:
            _debug(f"Type resolution failed for param '{param['name']}': {e}")
            kwargs['type'] = str
        
    # Handle defaults (do not force a default for tri-state bools)
    if not param['required'] and not (param['type'] == bool and param['default'] is None):
        kwargs['default'] = param['default']

    return kwargs, is_mapping_type

def add_dynamic_arguments(parser, param_info, param_docs: Optional[Dict[str, str]] = None, cmd_name: Optional[str] = None):
    """Add arguments to parser based on parameter info.

    Adds both hyphen and underscore long-option aliases and sets dest to the
    original param name (snake_case) so downstream mapping works.
    Also casts Optional[int|float|bool] to their base types for argparse.
    """
    for param in param_info['params']:
        hyph = f"--{param['name'].replace('_', '-')}"
        uscr = f"--{param['name']}"
        
        param_names = {p.get('name') for p in (param_info.get('params') or []) if isinstance(p, dict)}
        kwargs, is_mapping_type = _resolve_param_kwargs(param, param_docs, cmd_name, param_names=param_names)
        
        # Add positional argument for first required parameter
        if param['required'] and param == param_info['params'][0]:
            # Special-case forecast_generate UX: allow omitting the symbol to show
            # contextual help (e.g. listing models for a library).
            if cmd_name == "forecast_generate":
                parser.add_argument(
                    param['name'],
                    nargs='?',
                    help=f"{param['name']} (omit to list models when --library is set)",
                )
            else:
                parser.add_argument(param['name'], help=f"{param['name']} (required)")
        else:
            # For mapping-like params (e.g., --simplify), allow bare flag: '--simplify' triggers defaults
            if is_mapping_type:
                local_kwargs = dict(kwargs)
                local_kwargs['nargs'] = '?'
                local_kwargs['const'] = '__PRESENT__'
                parser.add_argument(hyph, uscr, **local_kwargs)
            else:
                parser.add_argument(hyph, uscr, **kwargs)

        # If this parameter is mapping-like, add a companion --<name>-params to pass extra kwargs
        if is_mapping_type:
            parser.add_argument(
                f"--{param['name'].replace('_','-')}-params",
                f"--{param['name']}_params",
                dest=f"{param['name']}_params",
                type=str,
                default=None,
                help=f"Extra params for {param['name']} (key=value[,key=value])"
            )

def _parse_kv_string(s: str) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' (commas or spaces) into a dict. Delegates to utils implementation."""
    try:
        from ..utils.utils import parse_kv_or_json
        result = parse_kv_or_json(s)
        return result if result else None
    except Exception as e:
        _debug(f"Failed to parse kv string '{s}': {e}")
        return None


def _coerce_cli_scalar(v: str) -> Any:
    s = v.strip()
    if not s:
        return s
    sl = s.lower()
    if sl == "true":
        return True
    if sl == "false":
        return False
    if sl == "null" or sl == "none":
        return None
    # JSON value (string/number/list/object/bool/null)
    if s[0] in ('{', '[', '"') or sl in ("true", "false", "null") or s.replace(".", "", 1).isdigit():
        try:
            import json
            return json.loads(s)
        except Exception:
            pass
    # Number
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _parse_set_overrides(items: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """Parse repeated --set entries like 'model.sp=24' into nested dicts."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in items or []:
        if not isinstance(item, str) or not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        left, right = item.split("=", 1)
        left = left.strip()
        if "." not in left:
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        section, key = left.split(".", 1)
        section = section.strip().lower()
        key = key.strip()
        if not section or not key:
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        out.setdefault(section, {})[key] = _coerce_cli_scalar(right)
    return out


def _merge_dict(dst: Optional[Dict[str, Any]], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = dict(dst or {})
    for k, v in (src or {}).items():
        d[k] = v
    return d


def _add_forecast_generate_args(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.description = "Generate forecasts with a structured pipeline (model + optional preprocessing)."

    cmd_parser.add_argument(
        "symbol",
        nargs="?",
        help="symbol (omit to list models when --library is set)",
    )

    group_model = cmd_parser.add_argument_group("Model Selection")
    group_model.add_argument(
        "--method",
        dest="method",
        type=str,
        default=None,
        help="Native mtdata method name (e.g. theta, arima, analog). If --library is set and --model is omitted, this is treated as --model for backward compatibility.",
    )
    group_model.add_argument(
        "--library",
        dest="library",
        type=str,
        choices=["native", "statsforecast", "sktime", "mlforecast", "pretrained"],
        default=None,
        help="External library/group to use with --model.",
    )
    group_model.add_argument(
        "--model",
        dest="model",
        type=str,
        default=None,
        help="Model name within --library (e.g. AutoARIMA; ThetaForecaster; or dotted class path).",
    )
    group_model.add_argument(
        "--model-params",
        dest="model_params",
        type=str,
        default=None,
        help="Model constructor/adapter params as JSON or k=v pairs (e.g. 'sp=24').",
    )
    group_model.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="SECTION.KEY=VALUE",
        help="Override nested params (sections: model, denoise, features, dimred, target). Repeatable.",
    )

    # Backward-compat aliases (hidden): keep old names working
    group_model.add_argument("--params", dest="model_params", type=str, default=None, help=argparse.SUPPRESS)
    group_model.add_argument("--params-params", dest="model_params_kv", type=str, default=None, help=argparse.SUPPRESS)

    group_window = cmd_parser.add_argument_group("Window & Target")
    group_window.add_argument("--horizon", type=int, default=12, help="Forecast horizon in bars.")
    group_window.add_argument("--lookback", type=int, default=None, help="Historical bars to use (default: auto).")
    group_window.add_argument(
        "--quantity",
        choices=["price", "return", "volatility"],
        default="price",
        help="Quantity to model.",
    )
    group_window.add_argument(
        "--target",
        choices=["price", "return"],
        default="price",
        help=argparse.SUPPRESS,
    )

    group_data = cmd_parser.add_argument_group("Data Selection")
    group_data.add_argument("--timeframe", type=str, default="H1", help="MT5 timeframe (e.g. H1/M30/D1).")
    group_data.add_argument("--as-of", dest="as_of", type=str, default=None, help="Override reference time for 'now'.")

    group_uncertainty = cmd_parser.add_argument_group("Uncertainty")
    group_uncertainty.add_argument("--ci-alpha", dest="ci_alpha", type=float, default=0.05, help="CI alpha (0.05 => 95%%).")

    group_pipe = cmd_parser.add_argument_group("Pipeline (Optional)")
    group_pipe.add_argument("--denoise", type=str, default=None, help="Denoise preset name or JSON spec (e.g. 'wavelet').")
    group_pipe.add_argument("--denoise-params", dest="denoise_params", type=str, default=None, help="Extra denoise params k=v pairs.")
    group_pipe.add_argument("--features", type=str, default=None, help="Feature spec JSON or shorthand.")
    group_pipe.add_argument("--features-params", dest="features_params", type=str, default=None, help="Extra features params k=v pairs.")
    group_pipe.add_argument("--dimred-method", dest="dimred_method", type=str, default=None, help="Dimensionality reduction method (e.g. pca).")
    group_pipe.add_argument("--dimred-params", dest="dimred_params", type=str, default=None, help="Dimred params as JSON or k=v pairs.")
    group_pipe.add_argument("--dimred-params-params", dest="dimred_params_kv", type=str, default=None, help=argparse.SUPPRESS)
    group_pipe.add_argument("--target-spec", dest="target_spec", type=str, default=None, help="Target spec JSON or k=v pairs.")
    group_pipe.add_argument("--target-spec-params", dest="target_spec_params", type=str, default=None, help="Extra target spec params k=v pairs.")
    group_pipe.add_argument("--future-covariates", dest="future_covariates", type=str, default=None, help="Comma-separated date features (e.g. hour,dow,is_holiday).")
    group_pipe.add_argument("--country", type=str, default=None, help="Country code for holiday calendar (e.g. US).")

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
    def command_func(args):
        # CLI-only convenience: if forecast_generate is invoked without symbol,
        # show contextual help and optionally list library models.
        if cmd_name == "forecast_generate" and not getattr(args, "symbol", None):
            lib = getattr(args, "library", None)
            if lib:
                try:
                    out = server.forecast_list_library_models(library=str(lib))  # type: ignore[attr-defined]
                    text = _format_result_minimal(out, verbose=getattr(args, "verbose", False))
                    if text:
                        print(text)
                except Exception:
                    lib = None

            # If the user explicitly asked for a library, treat this as a successful
            # list-models invocation (no need to dump full argparse help).
            if lib:
                return

            # Otherwise, show help and exit non-zero to signal missing args.
            if cmd_parser is not None:
                cmd_parser.print_help()
            raise SystemExit(2)

        # Build kwargs from args
        kwargs = {}
        for param in func_info['params']:
            param_name = param['name']
            arg_value = getattr(args, param_name, param['default'])
            
            # Normalize boolean values coming as strings
            if param.get('type') == bool and isinstance(arg_value, str):
                if arg_value.lower() == 'true':
                    arg_value = True
                elif arg_value.lower() == 'false':
                    arg_value = False
            # Handle mapping-like params for CLI convenience
            try:
                ptype = param.get('type')
                origin = get_origin(ptype)
                
                # Unwrap Optional
                base_type = ptype
                if origin is not None and str(origin).endswith('Union'):
                    args_t = [a for a in get_args(ptype) if a is not type(None)]
                    if len(args_t) == 1:
                        base_type = args_t[0]
                        origin = get_origin(base_type)

                is_typed_dict = hasattr(base_type, '__annotations__') and isinstance(getattr(base_type, '__annotations__', {}), dict)
                is_mapping = (base_type in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict
            except Exception:
                is_mapping = False
            # Bare flag sentinel: treat as empty mapping to trigger defaults
            if is_mapping and arg_value == '__PRESENT__':
                arg_value = {}
            # For mapping-like params, support shorthand and companion '<name>_params'
            if is_mapping:
                # Try to parse JSON/KV string if it looks like one
                if isinstance(arg_value, str) and arg_value.strip():
                    if arg_value.strip().startswith('{'):
                         parsed = _parse_kv_string(arg_value)
                         if parsed is not None:
                             arg_value = parsed
                    # Shorthand: --simplify lttb  -> {"method":"lttb"}
                    elif not arg_value.strip().startswith('{'):
                        arg_value = {"method": arg_value.strip()}
                # Companion params: --simplify-params 'points=100,ratio=0.5'
                extra_param_name = f"{param_name}_params"
                extra_val = getattr(args, extra_param_name, None)
                if isinstance(extra_val, str) and extra_val.strip():
                    extra = _parse_kv_string(extra_val)
                    if extra:
                        if arg_value is None or arg_value == {}:
                            arg_value = extra
                        elif isinstance(arg_value, dict):
                            # merge without clobbering keys explicitly present in arg_value
                            for k, v in extra.items():
                                if k not in arg_value:
                                    arg_value[k] = v
                        else:
                            # Unexpected type; replace
                            arg_value = extra
            
            # Only include non-None values
            if arg_value is not None:
                kwargs[param_name] = arg_value
        
        # Call the function (tools now return minimal plain text for API and CLI)
        # Request raw output so we can control formatting in CLI (e.g. verbose flag)
        kwargs['__cli_raw'] = True
        result = func_info['func'](**kwargs)

        # If the tool already returned text, print it exactly (no stripping)
        if isinstance(result, str):
            print(result)
            return

        # Otherwise, use the same shared minimal formatter as the server
        # Pass verbose flag if available (default to False for cleaner output)
        verbose = getattr(args, 'verbose', False)
        minimal_output = _format_result_minimal(result, verbose=verbose)
        if minimal_output:
            print(minimal_output)
        return

    return command_func

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
        for param in func_info['params']:
            tname = _type_name(param['type']) if param['type'] else 'str'
            if param['required']:
                arg_strs.append(f"{param['name']}<{tname}>")
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
    lines.append("  # Basic forecast with Theta method (fast, univariate)")
    lines.append("  python cli.py forecast_generate EURUSD --timeframe H1 --method theta --horizon 24")
    lines.append("")
    lines.append("  # Foundation model (Chronos-2) with covariates and quantiles")
    lines.append("  python cli.py forecast_generate BTCUSD --timeframe H1 --method chronos2 --horizon 12 \\")
    lines.append("    --features \"include=open,high future_covariates=hour,dow,is_holiday\" \\")
    lines.append("    --country US --verbose")
    lines.append("")
    lines.append("  # Rolling backtest for accuracy check")
    lines.append("  python cli.py forecast_backtest_run EURUSD --timeframe H1 --methods theta,seasonal_naive \\")
    lines.append("    --steps 5 --horizon 12")
    return "\n".join(lines)



_EXTENDED_HELP_EXAMPLE_HINTS: Dict[str, Any] = {
    'symbol': 'EURUSD',
    'timeframe': 'H1',
    'method': 'nhits',
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
    base = "python cli.py " + " ".join(base_parts)
    advanced = None
    if optional_tokens:
        adv_parts = base_parts + optional_tokens[:2]
        advanced = "python cli.py " + " ".join(adv_parts)
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
    matches = _match_commands(functions, query)
    if not matches:
        print(f"No commands match '{query}'.")
        print("Available commands:")
        for name in sorted(functions.keys()):
            print(f"  {name}")
        print("\nTip: run `python cli.py --help` to view the full list.")
        return
    print(f"Extended help for query: {query}")
    print("")
    for name, tool, func_info in matches:
        meta = tool.get('meta') or {}
        summary = meta.get('description') or _first_line(func_info.get('doc'))
        required = [p['name'] for p in func_info['params'] if p['required']]
        optional = [p['name'] for p in func_info['params'] if not p['required']]
        base_example, advanced_example = _build_usage_examples(name, func_info)
        print(name)
        if summary:
            print(f"  Summary: {summary}")
        if required:
            print(f"  Required: {', '.join(required)}")
        if optional:
            print(f"  Optional: {', '.join(optional[:6])}")
        print(f"  Example: {base_example}")
        if advanced_example and advanced_example != base_example:
            print(f"  Example+: {advanced_example}")
        print(f"  More: python cli.py {name} --help")
        print("")
def main():
    """Main CLI entry point with dynamic parameter discovery"""
    # Discover functions to expose dynamically
    functions = discover_tools()
    if not functions:
        print("No tools discovered from server module.", file=sys.stderr)
        return 1
    help_query = _extract_help_query(sys.argv[1:])
    if help_query:
        _print_extended_help(functions, help_query)
        return 0

    
    parser = argparse.ArgumentParser(
        description="Dynamic CLI for MetaTrader5 MCP tools (CSV-first output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(functions),
    )
    # Add unified global parameters
    add_global_args_to_parser(parser)

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
        cmd_parser = subparsers.add_parser(
            cmd_name, 
            help=((meta.get('description') or func_info['doc'].split('\n')[0] if func_info['doc'] else f"Execute {cmd_name}").replace('%', '%%')),
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Add global parameters to each subparser, excluding any that conflict with function params
        existing_param_names = [p['name'] for p in func_info['params']]
        exclude_globals = list(existing_param_names)
        if cmd_name == 'report_generate':
            exclude_globals.append('timeframe')
        add_global_args_to_parser(cmd_parser, exclude_params=exclude_globals)
        
        # Add dynamic arguments
        add_dynamic_arguments(cmd_parser, func_info, meta.get('param_docs'), cmd_name=cmd_name)
        
        # Set the command function
        cmd_parser.set_defaults(func=create_command_function(func_info, cmd_name, cmd_parser=cmd_parser))

    # Custom forecast_generate parser (grouped UX + backward-compat aliases)
    if forecast_tool is not None:
        cmd_name = "forecast_generate"
        func = forecast_tool["func"]
        func_info = forecast_tool_info or get_function_info(func)
        meta = forecast_tool.get("meta") or {}
        cmd_parser = subparsers.add_parser(
            cmd_name,
            help=((meta.get('description') or func_info['doc'].split('\n')[0] if func_info['doc'] else f"Execute {cmd_name}").replace('%', '%%')),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        # Add global parameters to each subparser, excluding any that conflict
        exclude_globals = ["symbol", "timeframe", "verbose"]  # handled manually
        add_global_args_to_parser(cmd_parser, exclude_params=exclude_globals)
        _add_forecast_generate_args(cmd_parser)

        def _forecast_generate_cmd(args):
            # List models if symbol omitted and library is specified.
            if not getattr(args, "symbol", None):
                lib = getattr(args, "library", None)
                if lib:
                    out = server.forecast_list_library_models(library=str(lib))  # type: ignore[attr-defined]
                    text = _format_result_minimal(out, verbose=getattr(args, "verbose", False))
                    if text:
                        print(text)
                    return
                cmd_parser.print_help()
                raise SystemExit(2)

            # Validate selection: method OR library/model.
            if args.library and not args.model and args.method:
                # Backward compatible: `--library X --method Y` means `--library X --model Y`.
                args.model = args.method
                args.method = None
            if args.library and not args.model:
                raise ValueError("--library requires --model (or use --method as a backward-compatible alias)")
            if args.model and not args.library:
                raise ValueError("--model requires --library")

            # Model params
            model_params = _parse_kv_string(args.model_params) if isinstance(args.model_params, str) else None
            if getattr(args, "model_params_kv", None):
                model_params = _merge_dict(model_params, _parse_kv_string(args.model_params_kv))

            # Pipeline mapping-like params
            denoise = None
            if args.denoise:
                denoise = {"method": args.denoise.strip()}
                if args.denoise.strip().startswith("{"):
                    parsed = _parse_kv_string(args.denoise)
                    denoise = parsed if parsed is not None else denoise
            if args.denoise_params:
                denoise = _merge_dict(denoise, _parse_kv_string(args.denoise_params))

            features = _parse_kv_string(args.features) if args.features else None
            if args.features and not args.features.strip().startswith("{"):
                # Accept shorthand like "include=close,volume" (already handled by parse_kv_or_json)
                pass
            if args.features_params:
                features = _merge_dict(features, _parse_kv_string(args.features_params))

            dimred_params = _parse_kv_string(args.dimred_params) if args.dimred_params else None
            if getattr(args, "dimred_params_kv", None):
                dimred_params = _merge_dict(dimred_params, _parse_kv_string(args.dimred_params_kv))
            target_spec = _parse_kv_string(args.target_spec) if args.target_spec else None
            if args.target_spec_params:
                target_spec = _merge_dict(target_spec, _parse_kv_string(args.target_spec_params))

            # --set overrides (sections: model/denoise/features/dimred/target)
            overrides = _parse_set_overrides(args.set_overrides)
            model_params = _merge_dict(model_params, overrides.get("model"))
            denoise = _merge_dict(denoise, overrides.get("denoise"))
            features = _merge_dict(features, overrides.get("features"))
            dimred_params = _merge_dict(dimred_params, overrides.get("dimred"))
            target_spec = _merge_dict(target_spec, overrides.get("target"))

            future_covariates = None
            if args.future_covariates:
                future_covariates = [s.strip() for s in str(args.future_covariates).split(",") if s.strip()]

            kwargs = {
                "symbol": args.symbol,
                "timeframe": args.timeframe,
                "method": args.method,
                "library": args.library,
                "model": args.model,
                "horizon": int(args.horizon),
                "lookback": args.lookback,
                "as_of": args.as_of,
                "params": model_params,
                "ci_alpha": args.ci_alpha,
                "quantity": args.quantity,
                "target": args.target,
                "denoise": denoise or None,
                "features": features or None,
                "dimred_method": args.dimred_method,
                "dimred_params": dimred_params or None,
                "target_spec": target_spec or None,
                "future_covariates": future_covariates,
                "country": args.country,
            }

            if getattr(args, "print_config", False):
                print(_format_result_minimal({"forecast_generate": kwargs}, verbose=True))
                return

            kwargs["__cli_raw"] = True
            out = func(**kwargs)
            text = _format_result_minimal(out, verbose=getattr(args, "verbose", False))
            if text:
                print(text)

        cmd_parser.set_defaults(func=_forecast_generate_cmd)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
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

