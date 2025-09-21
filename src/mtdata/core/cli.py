#!/usr/bin/env python3
"""
Dynamic CLI wrapper for testing MetaTrader5 MCP server functions
Automatically discovers function parameters and creates CLI arguments
"""

import argparse
import io
import csv
import sys
import inspect
import os
from typing import get_type_hints, get_origin, get_args, Optional, Dict, Any, List
import json
import math
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





def _is_scalar_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return all(_is_empty_value(v) for v in value)
    if isinstance(value, dict):
        return all(_is_empty_value(v) for v in value.values())
    return False


def _minify_number(num: float) -> str:
    try:
        f = float(num)
    except Exception:
        return str(num)
    if not math.isfinite(f):
        return str(num)
    text = f"{f:.8f}".rstrip('0').rstrip('.')
    return text if text else '0'


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _minify_number(value)
    return str(value)


def _stringify_cell(value: Any) -> str:
    if _is_scalar_value(value):
        return _stringify_scalar(value)
    if isinstance(value, list):
        values = [v for v in value if not _is_empty_value(v)]
        if not values:
            return ""
        if all(_is_scalar_value(v) for v in values):
            return "|".join(_stringify_scalar(v) for v in values)
        return "; ".join(_stringify_cell(v) for v in values if not _is_empty_value(v))
    if isinstance(value, dict):
        parts = []
        for key, subval in value.items():
            if _is_empty_value(subval):
                continue
            parts.append(f"{key}={_stringify_cell(subval)}")
        return "; ".join(parts)
    return str(value)


def _indent_text(text: str, indent: str = "  ") -> str:
    return "\n".join(f"{indent}{line}" if line else indent.rstrip() for line in text.splitlines())


def _list_of_dicts_to_csv(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    headers: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            if key not in headers:
                headers.append(key)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(headers)
    for item in items:
        row = [_stringify_cell(item.get(header)) for header in headers]
        writer.writerow(row)
    return buffer.getvalue().rstrip("\n")


def _format_complex_value(value: Any) -> str:
    if _is_scalar_value(value):
        return _stringify_scalar(value)
    if isinstance(value, list):
        values = [v for v in value if not _is_empty_value(v)]
        if not values:
            return ""
        if all(isinstance(v, dict) for v in values):
            return _list_of_dicts_to_csv(values)
        if all(_is_scalar_value(v) for v in values):
            return ", ".join(_stringify_scalar(v) for v in values)
        parts = []
        for entry in values:
            formatted = _format_complex_value(entry)
            if formatted:
                parts.append(formatted)
        return "\n".join(parts)
    if isinstance(value, dict):
        lines = []
        for key, subvalue in value.items():
            if _is_empty_value(subvalue):
                continue
            formatted = _format_complex_value(subvalue)
            if not formatted:
                continue
            if "\n" in formatted:
                lines.append(f"{key}:\n{_indent_text(formatted)}")
            else:
                lines.append(f"{key}: {formatted}")
        return "\n".join(lines)
    return _stringify_scalar(value)


def _format_meta_block(meta: Dict[str, Any]) -> str:
    lines = []
    for key, value in meta.items():
        if _is_empty_value(value):
            continue
        formatted = _format_complex_value(value)
        if not formatted:
            continue
        if "\n" in formatted:
            lines.append(f"{key}:\n{_indent_text(formatted)}")
        else:
            lines.append(f"{key}: {formatted}")
    return "\n".join(lines)


def _format_result_minimal(result: Any) -> str:
    # Delegate to shared formatter used by the server so CLI output matches API output exactly
    try:
        return _shared_minimal(result)
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
    1) Use server.mcp registry if available
    2) Fallback to scanning public callables in server module (excluding helpers)
    """
    tools: Dict[str, ToolInfo] = {}

    mcp = getattr(server, 'mcp', None)
    registry = None
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

def add_dynamic_arguments(parser, param_info, param_docs: Optional[Dict[str, str]] = None):
    """Add arguments to parser based on parameter info.

    Adds both hyphen and underscore long-option aliases and sets dest to the
    original param name (snake_case) so downstream mapping works.
    Also casts Optional[int|float|bool] to their base types for argparse.
    """
    for param in param_info['params']:
        hyph = f"--{param['name'].replace('_', '-')}"
        uscr = f"--{param['name']}"
        desc = None
        if param_docs and param['name'] in param_docs:
            desc = param_docs[param['name']]
        hint = desc or _PARAM_HINTS.get(param['name'])
        kwargs = {'help': hint or f"{param['name']} parameter", 'dest': param['name']}
        
        # Handle different types
        if param['type'] == int:
            kwargs['type'] = int
        elif param['type'] == float:
            kwargs['type'] = float
        elif param['type'] == bool:
            # Support tri-state booleans via explicit true/false value
            kwargs['type'] = str
            kwargs['choices'] = ['true', 'false']
            kwargs['metavar'] = 'bool'
        else:
            # Detect Literal and List[Literal] choices
            try:
                ptype = param.get('type')
                origin = get_origin(ptype)
                # Optional[T] -> T for argparse casting
                if origin is not None and str(origin).endswith('Union'):
                    args = [a for a in get_args(ptype) if a is not type(None)]  # noqa: E721
                    if len(args) == 1:
                        base = args[0]
                        if base in (int, float, str):
                            kwargs['type'] = base
                        elif base is bool:
                            kwargs['type'] = str
                            kwargs['choices'] = ['true', 'false']
                            kwargs['metavar'] = 'bool'
                        origin = None
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
                    choices = [str(v) for v in get_args(ptype)]
                    if choices:
                        kwargs['choices'] = choices
                    kwargs['type'] = str
                else:
                    kwargs['type'] = str
            except Exception as e:
                _debug(f"Type resolution failed for param '{param['name']}': {e}")
                kwargs['type'] = str
            
        # Handle defaults (do not force a default for tri-state bools)
        if not param['required'] and not (param['type'] == bool and param['default'] is None):
            kwargs['default'] = param['default']
        
        # Add positional argument for first required parameter
        if param['required'] and param == param_info['params'][0]:
            parser.add_argument(param['name'], help=f"{param['name']} (required)")
        else:
            # For mapping-like params (e.g., --simplify), allow bare flag: '--simplify' triggers defaults
            try:
                ptype = param.get('type')
                origin = get_origin(ptype)
                is_typed_dict = hasattr(ptype, '__annotations__') and isinstance(getattr(ptype, '__annotations__', {}), dict)
                is_mapping_type = (ptype in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict
            except Exception as e:
                _debug(f"Mapping type check failed for param '{param['name']}': {e}")
                is_mapping_type = False
            if is_mapping_type:
                local_kwargs = dict(kwargs)
                local_kwargs['nargs'] = '?'
                local_kwargs['const'] = '__PRESENT__'
                parser.add_argument(hyph, uscr, **local_kwargs)
            else:
                parser.add_argument(hyph, uscr, **kwargs)

        # If this parameter is mapping-like, add a companion --<name>-params to pass extra kwargs
        try:
            ptype = param.get('type')
            origin = get_origin(ptype)
            is_typed_dict = hasattr(ptype, '__annotations__') and isinstance(getattr(ptype, '__annotations__', {}), dict)
            is_mapping = (ptype in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict
        except Exception as e:
            _debug(f"Mapping type check failed for companion params of '{param['name']}': {e}")
            is_mapping = False
        if is_mapping:
            parser.add_argument(
                f"--{param['name'].replace('_','-')}-params",
                f"--{param['name']}_params",
                dest=f"{param['name']}_params",
                type=str,
                default=None,
                help=f"Extra params for {param['name']} (key=value[,key=value])"
            )

def _parse_kv_string(s: str) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' (commas or spaces) into a dict. Returns None if not parseable.

    Note: JSON strings are no longer parsed by the CLI.
    """
    try:
        if not s:
            return None
        parts = []
        # Allow comma and whitespace as separators
        for token in s.replace(',', ' ').split():
            parts.append(token)
        out: Dict[str, Any] = {}
        for part in parts:
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            k = k.strip()
            v = v.strip()
            # Try to parse JSON scalars
            if v.lower() in ('true','false'):
                out[k] = (v.lower() == 'true')
                continue
            try:
                if v.isdigit():
                    out[k] = int(v)
                    continue
                fv = float(v)
                out[k] = fv
                continue
            except Exception:
                pass
            # Strip surrounding quotes
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            out[k] = v
        return out if out else None
    except Exception as e:
        _debug(f"Failed to parse kv string '{s}': {e}")
        return None


def create_command_function(func_info, cmd_name: str = ""):
    """Create a command function that calls the MCP function dynamically"""
    def command_func(args):
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
                is_typed_dict = hasattr(ptype, '__annotations__') and isinstance(getattr(ptype, '__annotations__', {}), dict)
                is_mapping = (ptype in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict
            except Exception:
                is_mapping = False
            # Bare flag sentinel: treat as empty mapping to trigger defaults
            if is_mapping and arg_value == '__PRESENT__':
                arg_value = {}
            # For mapping-like params, support shorthand and companion '<name>_params'
            if is_mapping:
                # Shorthand: --simplify lttb  -> {"method":"lttb"}
                if isinstance(arg_value, str) and arg_value.strip() and not arg_value.strip().startswith('{'):
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
        result = func_info['func'](**kwargs)

        # If the tool already returned text, print it exactly (no stripping)
        if isinstance(result, str):
            print(result)
            return

        # Otherwise, use the same shared minimal formatter as the server
        minimal_output = _format_result_minimal(result)
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
        meta = tool.get('meta') or {}
        info = get_function_info(func)
        # Overlay from schema for defaults/required
        schema = meta.get('schema') or {}
        # Enrich with shared $defs and timeframe refs; or build minimal if missing
        schema = enrich_schema_with_shared_defs(schema, info)
        # Persist enriched schema back into meta for downstream help building
        tool['meta']['schema'] = schema
        params_obj = schema.get('parameters') if isinstance(schema.get('parameters'), dict) else schema
        schema_props = params_obj.get('properties') if isinstance(params_obj, dict) else {}
        schema_required = set(params_obj.get('required', [])) if isinstance(params_obj, dict) else set()
        for p in info['params']:
            prop = schema_props.get(p['name']) if isinstance(schema_props, dict) else None
            if isinstance(prop, dict) and 'default' in prop and p['default'] is None:
                p['default'] = prop['default']
            if p['name'] in schema_required:
                p['required'] = True
        arg_strs = []
        for p in info['params']:
            tname = _type_name(p['type']) if p['type'] else 'str'
            if p['required']:
                arg_strs.append(f"{p['name']}<{tname}>")
            else:
                default = p['default']
                # Always display default, even when None (unset)
                arg_strs.append(
                    f"--{p['name'].replace('_','-')}<{tname}>=[{default}]"
                )
        desc = meta.get('description') or _first_line(info['doc'])
        lines.append(f"  {cmd_name}: {' '.join(arg_strs) if arg_strs else '(no args)'}")
        if desc:
            lines.append(f"    - {desc}")
    lines.append("")
    lines.append("Type Conventions:")
    lines.append("  - int: integer")
    lines.append("  - str: string")
    lines.append("  - bool: pass true|false (e.g., --flag true)")
    return "\n".join(lines)

def main():
    """Main CLI entry point with dynamic parameter discovery"""
    # Discover functions to expose dynamically
    functions = discover_tools()
    if not functions:
        print("No tools discovered from server module.", file=sys.stderr)
        return 1
    
    parser = argparse.ArgumentParser(
        description="Dynamic CLI for MetaTrader5 MCP tools (CSV-first output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_build_epilog(functions),
    )
    # Add unified global parameters
    add_global_args_to_parser(parser)

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dynamically create subparsers for each function
    for cmd_name, tool in sorted(functions.items()):
        func = tool['func']
        meta = tool.get('meta') or {}
        func_info = get_function_info(func)
        # Overlay defaults and required flags from schema if available
        schema = meta.get('schema') or {}
        schema = enrich_schema_with_shared_defs(schema, func_info)
        params_obj = schema.get('parameters') if isinstance(schema.get('parameters'), dict) else schema
        schema_props = params_obj.get('properties') if isinstance(params_obj, dict) else {}
        schema_required = set(params_obj.get('required', [])) if isinstance(params_obj, dict) else set()
        for p in func_info['params']:
            prop = schema_props.get(p['name']) if isinstance(schema_props, dict) else None
            if isinstance(prop, dict) and 'default' in prop and p['default'] is None:
                p['default'] = prop['default']
            if p['name'] in schema_required:
                p['required'] = True
        
        # Create subparser
        cmd_parser = subparsers.add_parser(
            cmd_name, 
            help=(meta.get('description') or func_info['doc'].split('\n')[0] if func_info['doc'] else f"Execute {cmd_name}")
        )
        
        # Add global parameters to each subparser, excluding any that conflict with function params
        existing_param_names = [p['name'] for p in func_info['params']]
        exclude_globals = list(existing_param_names)
        if cmd_name == 'report_generate':
            exclude_globals.append('timeframe')
        add_global_args_to_parser(cmd_parser, exclude_params=exclude_globals)
        
        # Add dynamic arguments
        add_dynamic_arguments(cmd_parser, func_info, meta.get('param_docs'))
        
        # Set the command function
        cmd_parser.set_defaults(func=create_command_function(func_info, cmd_name))
    
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

