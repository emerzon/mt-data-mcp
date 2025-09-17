#!/usr/bin/env python3
"""
Dynamic CLI wrapper for testing MetaTrader5 MCP server functions
Automatically discovers function parameters and creates CLI arguments
"""

import argparse
import io
import csv
from collections import OrderedDict
import sys
import json
import inspect
import os
from typing import get_type_hints, get_origin, get_args, Optional, Dict, Any

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
from .schema import enrich_schema_with_shared_defs, get_function_info as _schema_get_function_info

# Types for discovered metadata
ToolInfo = Dict[str, Any]

def print_csv_result(result):
    """Print CSV result if available"""
    if 'csv_data' in result:
        if 'csv_header' in result:
            print(result['csv_header'])
        if result['csv_data']:
            print(result['csv_data'])
        return True
    return False

def convert_csv_to_json(result):
    """Convert CSV data in result to proper structured JSON"""
    if 'csv_data' not in result or 'csv_header' not in result:
        return result
    
    # Parse CSV header and data
    header = result['csv_header']
    data = result['csv_data']
    
    if not header or not data:
        return result
    
    # Split header into columns
    columns = [col.strip() for col in header.split(',')]
    
    # Split data into rows
    rows = []
    for line in data.split('\n'):
        if line.strip():
            values = [val.strip() for val in line.split(',')]
            # Create a dictionary for each row
            row_dict = {}
            for i, col in enumerate(columns):
                value = values[i] if i < len(values) else ''
                # Try to convert numeric values
                try:
                    if '.' in value:
                        row_dict[col] = float(value)
                    elif value.isdigit():
                        row_dict[col] = int(value)
                    else:
                        row_dict[col] = value
                except (ValueError, TypeError):
                    row_dict[col] = value
            rows.append(row_dict)
    
    # Create new result with structured data
    structured_result = {k: v for k, v in result.items() if k not in ['csv_data', 'csv_header']}
    structured_result['data'] = rows
    structured_result['count'] = len(rows)
    
    return structured_result

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
        kwargs = {'help': desc or f"{param['name']} parameter", 'dest': param['name']}
        
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
                help=f"Extra params for {param['name']} (JSON or key=value[,key=value])"
            )

def _parse_kv_string(s: str) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' (commas or spaces) into a dict. Returns None if not parseable."""
    try:
        if not s:
            return None
        # Try JSON first
        s_strip = s.strip()
        if (s_strip.startswith('{') and s_strip.endswith('}')):
            return json.loads(s_strip)
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
            # Decode JSON for mapping/sequence typed params when passed as strings
            try:
                ptype = param.get('type')
                origin = get_origin(ptype)
                is_typed_dict = hasattr(ptype, '__annotations__') and isinstance(getattr(ptype, '__annotations__', {}), dict)
                is_mapping = (ptype in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict
                is_sequence = (ptype in (list, tuple)) or (origin in (list, tuple))
            except Exception:
                is_mapping = False
                is_sequence = False
            # Bare flag sentinel: treat as empty mapping to trigger defaults
            if is_mapping and arg_value == '__PRESENT__':
                arg_value = {}
            if isinstance(arg_value, str) and (is_mapping or is_sequence):
                s = arg_value.strip()
                if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                    try:
                        arg_value = json.loads(s)
                    except Exception as e:
                        _debug(f"JSON decode failed for param '{param_name}': {e}")

            # For mapping-like params, support shorthand and companion '<name>_params'
            if is_mapping:
                # Shorthand: --simplify lttb  -> {"method":"lttb"}
                if isinstance(arg_value, str) and arg_value.strip() and not (arg_value.strip().startswith('{')):
                    arg_value = {"method": arg_value.strip()}
                # Companion params: --simplify-params 'points=100,ratio=0.5'
                extra_param_name = f"{param_name}_params"
                extra_val = getattr(args, extra_param_name, None)
                if isinstance(extra_val, str) and extra_val.strip():
                    extra = _parse_kv_string(extra_val)
                    if extra is None:
                        try:
                            extra = json.loads(extra_val)
                        except Exception as e:
                            _debug(f"JSON decode failed for extra params '{extra_param_name}': {e}")
                            extra = None
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
        
        # Call the function
        result = func_info['func'](**kwargs)

        # Prefer markdown output when provided
        if isinstance(result, dict) and 'markdown' in result:
            try:
                output = result.get('markdown')
                if output is not None:
                    print(str(output))
                    return
            except Exception:
                pass

        # Handle output
        # Respect global format preference
        try:
            preferred = getattr(args, 'format', 'csv')
        except Exception:
            preferred = 'csv'
            
        if preferred == 'csv':
            # Special CSV shaping for certain commands
            if cmd_name == 'indicators_list':
                try:
                    structured = convert_csv_to_json(result)
                    rows = structured.get('data') or []
                    # We will output as category,name regardless of server header order
                    header = 'category,name'
                    # Preserve original order while grouping by category
                    groups: OrderedDict = OrderedDict()
                    for r in rows:
                        if not isinstance(r, dict):
                            continue
                        cat = r.get('category') or 'Uncategorized'
                        name = r.get('name')
                        if not name:
                            continue
                        groups.setdefault(cat, []).append(name)
                    # Build grouped CSV: show category once, then blank for subsequent entries
                    buf = io.StringIO()
                    writer = csv.writer(buf, lineterminator='\n')
                    # Only write data rows; print header separately for consistency with other commands
                    for cat, names in groups.items():
                        for i, name in enumerate(names):
                            writer.writerow([cat if i == 0 else '', name])
                    print(header)
                    print(buf.getvalue().rstrip('\n'))
                    return
                except Exception as e:
                    # Fallback to default CSV printing
                    _debug(f"CSV shaping failed for indicators_list: {e}")
            if print_csv_result(result):
                return
            # No CSV payload available; fall back to JSON pretty print
            try:
                print(json.dumps(result, indent=2, sort_keys=True, default=str, ensure_ascii=False))
            except Exception as e:
                _debug(f"Failed to pretty-print JSON; falling back to str: {e}")
                print(str(result))
            return
        elif preferred == 'json':
            # Convert CSV data to proper structured JSON if present
            structured_result = convert_csv_to_json(result)
            # Special JSON shaping for certain commands
            try:
                if cmd_name == 'indicators_list' and isinstance(structured_result, dict):
                    rows = structured_result.get('data') or []
                    groups = {}
                    for r in rows:
                        try:
                            cat = (r.get('category') if isinstance(r, dict) else None) or 'Uncategorized'
                            name = r.get('name') if isinstance(r, dict) else None
                            if name:
                                groups.setdefault(cat, []).append(name)
                        except Exception as e:
                            _debug(f"Grouping indicator row failed: {e}")
                            continue
                structured_result = {
                        'success': bool(structured_result.get('success', True)),
                        'categories': groups,
                        'count': sum(len(v) for v in groups.values())
                    }
            except Exception as e:
                _debug(f"Special JSON shaping failed: {e}")
            try:
                print(json.dumps(structured_result, indent=2, sort_keys=True, default=str, ensure_ascii=False))
            except Exception as e:
                # Fallback in unlikely case of serialization issue
                _debug(f"JSON serialization failed; falling back to str: {e}")
                print(str(structured_result))
        else:
            # Fallback: print as JSON
            try:
                print(json.dumps(result, indent=2, sort_keys=True, default=str, ensure_ascii=False))
            except Exception as e:
                _debug(f"JSON serialization failed (fallback json branch); falling back to str: {e}")
                print(str(result))
    
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
            exclude_globals.append('format')
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

