import json
from typing import Any, Callable, Dict, List, Optional, Tuple


def parse_kv_string(s: str, *, debug: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    """Parse 'k=v,k2=v2' or JSON into a dict."""
    try:
        from ..utils.utils import parse_kv_or_json

        result = parse_kv_or_json(s)
        return result if result else None
    except Exception as exc:
        debug(f"Failed to parse kv string '{s}': {exc}")
        return None


def normalize_cli_list_value(value: Any) -> Any:
    """Normalize CLI list values from comma, whitespace, or JSON input."""
    if value is None:
        return None

    out: List[Any] = []

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
        for token in s.replace(",", " ").split():
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
        out.setdefault(section, {})[key] = coerce_cli_scalar(right)
    return out


def merge_dict(dst: Optional[Dict[str, Any]], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(dst or {})
    for key, value in (src or {}).items():
        merged[key] = value
    return merged


def create_command_function(
    func_info: Dict[str, Any],
    *,
    cmd_name: str,
    render_cli_result: Callable[..., None],
    result_has_tool_error: Callable[[Any], bool],
    normalize_cli_list_value: Callable[[Any], Any],
    parse_kv_string: Callable[[str], Optional[Dict[str, Any]]],
    unwrap_optional_type: Callable[[Any], Tuple[Any, Any]],
    is_typed_dict_type: Callable[[Any], bool],
) -> Callable[[Any], int]:
    """Build a CLI command callable for a tool function."""

    def command_func(args: Any) -> int:
        kwargs: Dict[str, Any] = {}
        for param in func_info["params"]:
            param_name = param["name"]
            arg_value = getattr(args, param_name, param["default"])

            if param.get("type") == bool and isinstance(arg_value, str):
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
            if is_mapping:
                if isinstance(arg_value, str) and arg_value.strip():
                    if arg_value.strip().startswith("{"):
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

            if arg_value is not None:
                kwargs[param_name] = arg_value

        request_model = func_info.get("request_model")
        request_param_name = func_info.get("request_param_name")
        if request_model is not None and request_param_name:
            kwargs = {request_param_name: request_model(**kwargs)}

        kwargs["__cli_raw"] = True
        result = func_info["func"](**kwargs)
        render_cli_result(result, args=args, cmd_name=cmd_name)
        return 1 if result_has_tool_error(result) else 0

    return command_func
