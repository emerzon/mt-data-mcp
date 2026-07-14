import ast
import json
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, get_args

from pydantic import ValidationError

from ...data.requests import (
    _normalize_indicator_specs as _shared_normalize_indicator_specs,
)
from ...error_envelope import build_error_payload
from ...output_contract import normalize_output_extras

LIVE_TRADE_MUTATION_TOOLS = frozenset({"trade_place", "trade_modify", "trade_close"})
LIVE_TRADE_MUTATION_WARNING = (
    "LIVE ORDER WARNING: this command can send real MT5 trade requests when "
    "--dry-run false. Preview mode is the default."
)


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
            parsed: Any = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(s)
                    break
                except Exception:
                    parsed = None
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        token = item.strip()
                        if token:
                            out.append(token)
                    elif item is not None:
                        out.append(item)
                return
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
        if s[0] in ("{", "["):
            try:
                return ast.literal_eval(s)
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

    def _is_model_type(value: Any) -> bool:
        return isinstance(value, type) and (
            callable(getattr(value, "model_validate", None))
            or callable(getattr(value, "parse_obj", None))
        )

    def _build_cli_error(
        message: str,
        *,
        code: str = "cli_invalid_arguments",
        remediation: Optional[str] = None,
    ) -> Dict[str, Any]:
        return build_error_payload(
            str(message).strip() or "Invalid command input.",
            code=code,
            operation=cmd_name,
            remediation=(
                remediation
                or f"Run 'mtdata-cli {cmd_name} --help' for accepted arguments."
            ),
        )

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
        return _shared_normalize_indicator_specs(value)

    def _parse_wait_event_spec_text(text: str) -> Any:
        s = str(text or "").strip()
        if not s:
            return []
        if s[0] in "[{":
            parsed: Any = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(s)
                    break
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        if "=" in s:
            parsed_map = parse_kv_string(s)
            if parsed_map is not None:
                return [parsed_map]
        parsed_tokens = normalize_cli_list_value(s)
        if isinstance(parsed_tokens, list):
            return [
                {"type": item.strip()} if isinstance(item, str) and item.strip() else item
                for item in parsed_tokens
                if item not in (None, "")
            ]
        return parsed_tokens

    def _normalize_wait_event_specs(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            return _parse_wait_event_spec_text(value)
        if isinstance(value, (list, tuple)):
            out: List[Any] = []
            for item in value:
                if isinstance(item, str):
                    parsed = _parse_wait_event_spec_text(item)
                    if isinstance(parsed, list):
                        out.extend(parsed)
                    elif parsed is not None:
                        out.append(parsed)
                elif item is not None:
                    out.append(item)
            return out
        return value

    def _friendly_validation_error(exc: ValidationError) -> str:
        try:
            errors = exc.errors()
        except Exception:
            return str(exc)
        messages: List[str] = []
        for item in errors:
            loc = ".".join(str(part) for part in item.get("loc", ()))
            msg = str(item.get("msg") or "Invalid value.")
            if cmd_name == "wait_event" and loc.split(".", 1)[0] in {"watch_for", "end_on"}:
                return (
                    "wait_event watch_for/end_on must be arrays of event objects. "
                    "Example: --watch-for '[{\"type\":\"price_change\","
                    "\"threshold_value\":0.1,\"threshold_mode\":\"fixed_pct\"}]' "
                    "--end-on '[{\"type\":\"candle_close\",\"timeframe\":\"M1\"}]'."
                )
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
                    or _is_model_type(base_type)
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
                is_mapping = (
                    (base_type in (dict, Dict))
                    or (origin in (dict, Dict))
                    or is_typed_dict
                    or _is_model_type(base_type)
                )
                is_list_like = origin in (list, tuple)
            except Exception:
                is_mapping = False
                is_list_like = False

            if is_mapping and arg_value == "__PRESENT__":
                arg_value = {}
            if is_list_like:
                if param_name == "indicators":
                    try:
                        arg_value = _normalize_indicator_specs(arg_value)
                    except ValueError as exc:
                        render_cli_result(_build_cli_error(str(exc)), args=args, cmd_name=cmd_name)
                        return 1
                elif cmd_name == "wait_event" and param_name in {"watch_for", "end_on"}:
                    arg_value = _normalize_wait_event_specs(arg_value)
                else:
                    arg_value = normalize_cli_list_value(arg_value)
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
                        elif (
                            param_name == "simplify"
                            and arg_value.strip().lower()
                            in {"on", "auto", "off", "none", "null", "true", "false"}
                        ):
                            # Preserve request-model shortcuts for its BeforeValidator.
                            arg_value = arg_value.strip()
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
                elif missing_name in {"symbol", "symbols"}:
                    message += " Use symbols_list to browse available broker symbols."
            if cmd_name in LIVE_TRADE_MUTATION_TOOLS:
                message += f" {LIVE_TRADE_MUTATION_WARNING}"
            render_cli_result(
                _build_cli_error(
                    message,
                    code="cli_missing_required",
                    remediation=(
                        f"Provide: {missing_text}. Run "
                        f"'mtdata-cli {cmd_name} --help' for examples."
                    ),
                ),
                args=args,
                cmd_name=cmd_name,
            )
            return 1

        if (
            cmd_name in {"trade_place", "trade_modify"}
            and kwargs.get("idempotency_key") not in (None, "")
        ):
            render_cli_result(
                _build_cli_error(
                    "idempotency_key is not supported by one-shot CLI commands "
                    "because the in-memory replay cache ends with the process.",
                    code="cli_process_local_idempotency_unsupported",
                    remediation=(
                        "Use the MCP HTTP/SSE server or Web API when retries need "
                        "a shared idempotency cache, or omit idempotency_key."
                    ),
                ),
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

        normalized_extras = normalize_output_extras(getattr(args, "extras", None))
        if normalized_extras:
            kwargs["extras"] = normalized_extras
        fields = getattr(args, "fields", None)
        if fields:
            kwargs["fields"] = fields
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
