import json
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, get_args

from pydantic import ValidationError

from ...data.requests import (
    _normalize_indicator_specs as _shared_normalize_indicator_specs,
)
from ...output_contract import resolve_output_contract


def parse_kv_string(s: str, *, debug: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' or JSON into a dict."""
    try:
        from ....utils.utils import parse_kv_or_json

        result = parse_kv_or_json(s)
        return result if result else None
    except Exception as exc:
        debug(f"Failed to parse kv string '{s}': {exc}")
        return None


def normalize_cli_list_value(value: Any) -> Any:  # noqa: C901
    """Normalize CLI list values from comma, whitespace, or JSON input."""
    if value is None:
        return None

    out: List[Any] = []

    def _split_compact_tokens(text: str) -> List[str]:
        s = str(text or "").strip()
        if not s:
            return []
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        in_quote: Optional[str] = None
        for ch in s:
            if in_quote:
                current.append(ch)
                if ch == in_quote:
                    in_quote = None
                continue
            if ch in ('"', "'"):
                in_quote = ch
                current.append(ch)
                continue
            if ch in "([{":
                depth += 1
                current.append(ch)
                continue
            if ch in ")]}":
                depth = max(0, depth - 1)
                current.append(ch)
                continue
            if ch == "," and depth == 0:
                token = "".join(current).strip()
                if token:
                    parts.append(token)
                current = []
                continue
            current.append(ch)
        token = "".join(current).strip()
        if token:
            parts.append(token)
        if len(parts) > 1:
            return parts
        return [token for token in s.split() if token]

    def _add_text_tokens(text: str) -> None:
        s = str(text or "").strip()
        if not s:
            return
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            token = item.strip()
                            if token:
                                out.append(token)
                        elif item is not None:
                            out.append(item)
                    return
            except Exception:
                pass
        for token in _split_compact_tokens(s):
            value_token = token.strip()
            if value_token:
                out.append(value_token)

    if isinstance(value, str):
        _add_text_tokens(value)
        return out
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str):
                _add_text_tokens(item)
            elif item is not None:
                out.append(item)
        return out
    return value


def coerce_cli_scalar(v: str) -> Any:
    s = v.strip()
    if not s:
        return s
    sl = s.lower()
    if sl == "true":
        return True
    if sl == "false":
        return False
    if sl in ("null", "none"):
        return None
    if s[0] in ("{", "[", '"') or sl in ("true", "false", "null") or s.replace(".", "", 1).isdigit():
        try:
            return json.loads(s)
        except Exception:
            pass
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def parse_set_overrides(
    items: Optional[List[str]],
    *,
    coerce_cli_scalar: Callable[[str], Any],
) -> Dict[str, Dict[str, Any]]:
    """Parse repeated --set entries like 'method.sp=24' into nested dicts."""
    out: Dict[str, Dict[str, Any]] = {}

    def _assign_path(root: Dict[str, Any], parts: List[str], value: Any) -> None:
        node = root
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                existing = {}
                node[part] = existing
            node = existing
        node[parts[-1]] = value

    for item in items or []:
        if not isinstance(item, str) or not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        left, right = item.split("=", 1)
        left = left.strip()
        if "." not in left:
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        parts = [part.strip() for part in left.split(".")]
        if len(parts) < 2 or not parts[0] or not all(parts[1:]):
            raise ValueError(f"Invalid --set '{item}': expected section.key=value")
        section = parts[0].lower()
        _assign_path(out.setdefault(section, {}), parts[1:], coerce_cli_scalar(right))
    return out


def merge_dict(dst: Optional[Dict[str, Any]], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(dst or {})
    for key, value in (src or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


_SIMPLIFY_METHOD_DESCRIPTIONS = {
    "lttb": "fast bucket-based selection",
    "rdp": "Douglas-Peucker line simplification",
    "pla": "piecewise linear approximation",
    "apca": "adaptive piecewise constant approximation",
}


def create_command_function(  # noqa: C901
    func_info: Dict[str, Any],
    *,
    cmd_name: str,
    render_cli_result: Callable[..., None],
    result_has_tool_error: Callable[[Any], bool],
    normalize_cli_list_value: Callable[[Any], Any],
    parse_kv_string: Callable[[str], Optional[Dict[str, Any]]],
    unwrap_optional_type: Callable[[Any], Tuple[Any, Any]],
    is_typed_dict_type: Callable[[Any], bool],
    invoke_tool_function: Optional[Callable[..., Any]] = None,
) -> Callable[[Any], int]:
    """Build a CLI command callable for a tool function."""

    def _build_cli_error(message: str) -> Dict[str, Any]:
        return {"error": str(message).strip() or "Invalid command input."}

    def _literal_choices_for_param(param: Optional[Dict[str, Any]]) -> Optional[List[str]]:
        if not isinstance(param, dict):
            return None
        try:
            ptype, origin = unwrap_optional_type(param.get("type"))
        except Exception:
            return None
        if origin is Literal or str(origin) in {"typing.Literal", "<class 'typing.Literal'>"}:
            choices = [str(value) for value in get_args(ptype) if value is not None]
            return choices or None
        return None

    def _normalize_indicator_specs(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = normalize_cli_list_value(value)
        return _shared_normalize_indicator_specs(value)

    def _friendly_validation_error(exc: ValidationError) -> str:
        try:
            errors = exc.errors()
        except Exception:
            return str(exc)
        messages: List[str] = []
        for item in errors:
            loc = ".".join(str(part) for part in item.get("loc", ()))
            msg = str(item.get("msg") or "Invalid value.")
            if "indicators" in loc and "params" in loc and any(
                marker in msg.lower() for marker in ("list", "dict", "dictionary", "mapping", "valid")
            ):
                return (
                    "'params' must be a list of numeric values like [14] "
                    'or a named numeric map like {"length": 14}.'
                )
            if loc.endswith("simplify.method") and ("input should be" in msg.lower() or "literal" in msg.lower()):
                choices = ", ".join(
                    f"{name} ({description})"
                    for name, description in _SIMPLIFY_METHOD_DESCRIPTIONS.items()
                )
                return f"simplify.method must be one of: {choices}."
            if loc:
                messages.append(f"{loc}: {msg}")
            else:
                messages.append(msg)
        return "; ".join(messages) or str(exc)

    def command_func(args: Any) -> int:  # noqa: C901
        kwargs: Dict[str, Any] = {}
        missing_required: List[str] = []
        mapping_param_names: set[str] = set()
        for param in func_info["params"]:
            try:
                base_type, origin = unwrap_optional_type(param.get("type"))
                if (
                    base_type in (dict, Dict)
                    or origin in (dict, Dict)
                    or is_typed_dict_type(base_type)
                ):
                    mapping_param_names.add(param["name"])
            except Exception:
                continue
        try:
            set_overrides = parse_set_overrides(
                getattr(args, "set_overrides", None),
                coerce_cli_scalar=coerce_cli_scalar,
            )
        except ValueError as exc:
            render_cli_result(_build_cli_error(str(exc)), args=args, cmd_name=cmd_name)
            return 1
        unknown_sections = sorted(set(set_overrides) - mapping_param_names)
        if unknown_sections:
            allowed = ", ".join(sorted(mapping_param_names)) or "none"
            render_cli_result(
                _build_cli_error(
                    f"Unknown --set section(s): {', '.join(unknown_sections)}. "
                    f"Use one of: {allowed}."
                ),
                args=args,
                cmd_name=cmd_name,
            )
            return 1
        for param in func_info["params"]:
            param_name = param["name"]
            arg_value = getattr(args, param_name, param["default"])

            if param.get("type") is bool and isinstance(arg_value, str):
                if arg_value.lower() == "true":
                    arg_value = True
                elif arg_value.lower() == "false":
                    arg_value = False

            try:
                ptype = param.get("type")
                base_type, origin = unwrap_optional_type(ptype)

                is_typed_dict = is_typed_dict_type(base_type)
                is_mapping = (base_type in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict
                is_list_like = origin in (list, tuple)
            except Exception:
                is_mapping = False
                is_list_like = False

            if is_mapping and arg_value == "__PRESENT__":
                arg_value = {}
            if is_list_like:
                arg_value = normalize_cli_list_value(arg_value)
                if param_name == "indicators":
                    try:
                        arg_value = _normalize_indicator_specs(arg_value)
                    except ValueError as exc:
                        render_cli_result(_build_cli_error(str(exc)), args=args, cmd_name=cmd_name)
                        return 1
            if is_mapping:
                if isinstance(arg_value, str) and arg_value.strip():
                    if arg_value.strip().startswith("{"):
                        parsed = parse_kv_string(arg_value)
                        if parsed is not None:
                            arg_value = parsed
                    else:
                        parsed = parse_kv_string(arg_value)
                        if parsed is not None:
                            arg_value = parsed
                        else:
                            arg_value = {"method": arg_value.strip()}

                extra_param_name = f"{param_name}_params"
                extra_val = getattr(args, extra_param_name, None)
                if isinstance(extra_val, str) and extra_val.strip():
                    extra = parse_kv_string(extra_val)
                    if extra:
                        if arg_value is None or arg_value == {}:
                            arg_value = extra
                        elif isinstance(arg_value, dict):
                            for key, value in extra.items():
                                if key not in arg_value:
                                    arg_value[key] = value
                        else:
                            arg_value = extra
                if param_name in set_overrides:
                    if arg_value is None or arg_value in ("", "__PRESENT__"):
                        arg_value = {}
                    if isinstance(arg_value, dict):
                        arg_value = merge_dict(arg_value, set_overrides.get(param_name))
                    else:
                        arg_value = set_overrides.get(param_name)

            if param["required"] and arg_value in (None, ""):
                missing_required.append(param_name)
                continue
            if arg_value is not None:
                kwargs[param_name] = arg_value

        if missing_required:
            missing_text = ", ".join(missing_required)
            message = f"Missing required argument(s): {missing_text}."
            if len(missing_required) == 1:
                missing_name = missing_required[0]
                param_def = next((param for param in func_info["params"] if param.get("name") == missing_name), None)
                choices = _literal_choices_for_param(param_def)
                if choices:
                    message = f"Missing required argument '{missing_name}'. Valid values: {', '.join(choices)}."
            render_cli_result(
                _build_cli_error(message),
                args=args,
                cmd_name=cmd_name,
            )
            return 1

        request_model = func_info.get("request_model")
        request_param_name = func_info.get("request_param_name")
        if request_model is not None and request_param_name:
            try:
                kwargs = {request_param_name: request_model(**kwargs)}
            except ValidationError as exc:
                render_cli_result(
                    _build_cli_error(_friendly_validation_error(exc)),
                    args=args,
                    cmd_name=cmd_name,
                )
                return 1

        kwargs["__cli_raw"] = True
        if invoke_tool_function is not None:
            result = invoke_tool_function(
                func_info["func"],
                args=args,
                cmd_name=cmd_name,
                kwargs=kwargs,
            )
        else:
            result = func_info["func"](**kwargs)
        render_cli_result(result, args=args, cmd_name=cmd_name)
        return 1 if result_has_tool_error(result) else 0

    return command_func
