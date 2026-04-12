import argparse
import inspect
from typing import Any, Callable, Dict, Optional, Tuple

ToolInfo = Dict[str, Any]


_BAR_LIMIT_ALIAS_COMMANDS: set[str] = {
    "causal_discover_signals",
    "data_fetch_candles",
    "labels_triple_barrier",
    "patterns_detect",
    "regime_detect",
}

_OPTIONAL_FIRST_POSITIONAL_PARAMS: set[tuple[str, str]] = {
    ("finviz_news", "symbol"),
    ("correlation_matrix", "symbols"),
    ("cointegration_test", "symbols"),
    ("market_scan", "symbols"),
}

_COMMAND_PARAM_CHOICE_OVERRIDES: Dict[tuple[str, str], list[str]] = {
    (
        "forecast_barrier_optimize",
        "method",
    ): [
        "mc_gbm",
        "mc_gbm_bb",
        "hmm_mc",
        "garch",
        "bootstrap",
        "heston",
        "jump_diffusion",
        "auto",
    ],
}


_COMMAND_PARAM_HELP_OVERRIDES: Dict[tuple[str, str], str] = {
    ("data_fetch_candles", "indicators"): "Technical indicators. Use names like rsi, bb, or compact specs like sma(20) and macd(12,26,9). Use parentheses for params, not sma,20.",
    ("forecast_barrier_optimize", "method"): "Barrier simulation method: mc_gbm, mc_gbm_bb, hmm_mc, garch, bootstrap, heston, jump_diffusion, or auto.",
    ("forecast_quantlib_barrier_price", "option_type"): "Option side: call or put.",
    ("forecast_tune_optuna", "search_space"): "Optuna search space (JSON or k=v).",
    ("indicators_list", "detail"): "Output detail: compact table or full rows with aliases and descriptions.",
    ("labels_triple_barrier", "output"): "Output mode: full, summary, compact, or summary_only (alias for summary).",
    ("market_depth_fetch", "compact"): "Fail if DOM is unavailable instead of falling back to a ticker snapshot. Alias: --require-dom.",
    ("report_generate", "output"): "Output format: formatted text or markdown.",
    ("trade_modify", "expiration"): "Pending order expiration time (dateparser string, UTC epoch seconds, or GTC token).",
    ("trade_place", "expiration"): "Pending order expiration time (dateparser string, UTC epoch seconds, or GTC token).",
}

_VOLATILITY_METHOD_LITERAL_MARKERS = {
    "ewma",
    "parkinson",
    "gk",
    "rs",
    "yang_zhang",
    "rolling_std",
    "realized_kernel",
    "har_rv",
    "garch_t",
    "egarch_t",
    "gjr_garch_t",
    "figarch",
}

_FORECAST_METHOD_LITERAL_MARKERS = {
    "theta",
    "naive",
    "arima",
    "chronos2",
    "statsforecast",
}


def _normalize_bool_choice(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_forecast_method_literal(
    ptype: Any,
    *,
    is_literal_origin: Callable[[Any], bool],
    get_origin_func: Callable[[Any], Any],
    get_args_func: Callable[[Any], Tuple[Any, ...]],
) -> bool:
    try:
        origin = get_origin_func(ptype)
        if not is_literal_origin(origin):
            return False
        args = {str(v) for v in get_args_func(ptype) if v is not None}
        if args.intersection(_VOLATILITY_METHOD_LITERAL_MARKERS):
            return False
        return bool(args.intersection(_FORECAST_METHOD_LITERAL_MARKERS))
    except Exception:
        return False


def _dedupe_flags(*flags: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flag for flag in flags if flag))


def _canonicalize_long_option(flag: str) -> str:
    text = str(flag or "").strip()
    if not text.startswith("--"):
        return text
    if "=" in text:
        option, value = text.split("=", 1)
        return f"{option.replace('_', '-')}={value}"
    return text.replace("_", "-")


def _split_visible_and_hidden_flags(*flags: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    visible: list[str] = []
    hidden: list[str] = []
    for flag in _dedupe_flags(*flags):
        canonical = _canonicalize_long_option(flag)
        if canonical and canonical not in visible:
            visible.append(canonical)
        if flag != canonical and flag not in hidden:
            hidden.append(flag)
    return tuple(visible), tuple(hidden)


def should_expose_cli_param(*, cmd_name: Optional[str], param_name: str) -> bool:
    """Return whether a function parameter should surface as a user CLI argument."""
    if str(cmd_name or "") == "labels_triple_barrier" and str(param_name or "") == "summary_only":
        return False
    return True


def get_function_info(
    func: Any,
    *,
    schema_get_function_info: Callable[[Any], Dict[str, Any]],
    flatten_request_model_param: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Attach the underlying callable to schema introspection data."""
    info = schema_get_function_info(func)
    info["func"] = func
    info = flatten_request_model_param(info)
    if not info.get("doc"):
        info["doc"] = f"Execute {info.get('name') or getattr(func, '__name__', 'function')}"
    for param in info.get("params", []):
        if param.get("type") is None:
            param["type"] = str
        if "required" not in param:
            param["required"] = param.get("default") is None
    return info


def apply_schema_overrides(
    tool: ToolInfo,
    func_info: Dict[str, Any],
    *,
    enrich_schema_with_shared_defs: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply JSON schema defaults and required flags to CLI parameter metadata."""
    meta = tool.setdefault("meta", {})
    schema = meta.get("schema") or {}
    schema = enrich_schema_with_shared_defs(schema, func_info)
    meta["schema"] = schema
    params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
    schema_props = params_obj.get("properties") if isinstance(params_obj, dict) else {}
    schema_required = set(params_obj.get("required", [])) if isinstance(params_obj, dict) else set()
    for param in func_info.get("params", []):
        prop = schema_props.get(param["name"]) if isinstance(schema_props, dict) else None
        if isinstance(prop, dict) and "default" in prop and param.get("default") is None:
            param["default"] = prop["default"]
        if param["name"] in schema_required:
            param["required"] = True
    return schema


def extract_function_from_tool_obj(tool_obj: Any) -> Any:
    """Best-effort extraction of the underlying function from an MCP tool object."""
    for attr in ("func", "function", "callable", "handler", "wrapped", "_func"):
        if hasattr(tool_obj, attr) and callable(getattr(tool_obj, attr)):
            return getattr(tool_obj, attr)
    if callable(tool_obj):
        return tool_obj
    return None


def extract_metadata_from_tool_obj(tool_obj: Any) -> Dict[str, Any]:
    """Extract tool descriptions and per-parameter docs from registry objects."""
    meta: Dict[str, Any] = {"description": None, "param_docs": {}, "schema": None}

    for attr in ("description", "doc", "docs"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, str) and val.strip():
            meta["description"] = val.strip()
            break

    schema = None
    for attr in ("schema", "input_schema", "parameters", "spec"):
        val = getattr(tool_obj, attr, None)
        if isinstance(val, dict) and val:
            schema = val
            break

    if schema:
        meta["schema"] = schema
        if not meta["description"] and isinstance(schema.get("description"), str):
            meta["description"] = schema.get("description")
        params_obj = schema.get("parameters") if isinstance(schema.get("parameters"), dict) else schema
        props = params_obj.get("properties") if isinstance(params_obj, dict) else None
        if isinstance(props, dict):
            for pname, pdef in props.items():
                desc = pdef.get("description") if isinstance(pdef, dict) else None
                if isinstance(desc, str) and desc.strip():
                    meta["param_docs"][pname] = desc.strip()

    return meta


def discover_tools(
    *,
    bootstrap_tools: Callable[[], Tuple[Any, ...]],
    get_registered_tools: Callable[[], Any],
    mcp: Any,
    get_mcp_registry: Callable[[Any], Any],
    debug: Callable[[str], None],
    extract_function_from_tool_obj: Callable[[Any], Any],
    extract_metadata_from_tool_obj: Callable[[Any], Dict[str, Any]],
) -> Dict[str, ToolInfo]:
    """Discover CLI-visible tools from the bootstrap and MCP registries."""
    tools: Dict[str, ToolInfo] = {}

    def _module_is_visible(module_name: Any, allowed_modules: set[str], allowed_prefixes: tuple[str, ...]) -> bool:
        if not isinstance(module_name, str):
            return False
        if module_name in allowed_modules:
            return True
        return any(module_name.startswith(prefix) for prefix in allowed_prefixes)

    registry = None
    bootstrapped_modules: Tuple[Any, ...] = ()
    try:
        bootstrapped_modules = tuple(bootstrap_tools())
    except Exception as exc:
        debug(f"bootstrap_tools failed: {exc}")
    try:
        reg = get_registered_tools()
        if reg and hasattr(reg, "items"):
            registry = reg
    except Exception as exc:
        debug(f"get_registered_tools failed: {exc}")
    if mcp is not None:
        registry = get_mcp_registry(mcp) or registry

    module_names = {
        str(getattr(module, "__name__", "")).strip()
        for module in bootstrapped_modules
        if getattr(module, "__name__", None)
    }
    module_prefixes = tuple(
        f"{module_name.rsplit('.', 1)[0]}."
        for module_name in module_names
        if "." in module_name
    )
    if registry and hasattr(registry, "items"):
        for name, obj in registry.items():
            func = extract_function_from_tool_obj(obj)
            mod = getattr(func, "__module__", None) if func else None
            if func and (not module_names or _module_is_visible(mod, module_names, module_prefixes)):
                meta = extract_metadata_from_tool_obj(obj)
                tools[name] = {"func": func, "meta": meta}

    if tools:
        return tools

    for module in bootstrapped_modules:
        module_name = getattr(module, "__name__", None)
        if not isinstance(module_name, str):
            continue
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name)
            if callable(obj) and getattr(obj, "__module__", None) == module_name:
                try:
                    inspect.signature(obj)
                except (TypeError, ValueError):
                    continue
                if isinstance(obj, type):
                    continue
                if name.endswith(("_wrapper",)):
                    continue
                tools[name] = {"func": obj, "meta": {"description": None, "param_docs": {}}}

    return tools


def _load_forecast_method_choices(debug: Callable[[str], None]) -> Optional[list[str]]:
    try:
        from mtdata.forecast.registry import ForecastRegistry

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
            except Exception as import_exc:
                debug(f"Skipping method module import '{mod_name}': {import_exc}")

        return ForecastRegistry.get_all_method_names()
    except Exception as exc:
        debug(f"Failed to dynamically load forecast methods for CLI: {exc}")
        return None


def resolve_param_kwargs(
    param: Dict[str, Any],
    param_docs: Optional[Dict[str, str]],
    *,
    cmd_name: Optional[str],
    param_names: Optional[set],
    param_hints: Dict[str, str],
    debug: Callable[[str], None],
    is_literal_origin: Callable[[Any], bool],
    unwrap_optional_type: Callable[[Any], Tuple[Any, Any]],
    is_typed_dict_type: Callable[[Any], bool],
    get_origin: Callable[[Any], Any],
    get_args: Callable[[Any], Tuple[Any, ...]],
) -> Tuple[Dict[str, Any], bool]:
    """Resolve argparse kwargs for a single CLI parameter."""

    def _escape_argparse_help(text: Optional[str]) -> Optional[str]:
        return text.replace("%", "%%") if isinstance(text, str) else text

    desc = None
    if param_docs and param["name"] in param_docs:
        desc = param_docs[param["name"]]
    hint = desc or param_hints.get(param["name"])
    override_help = _COMMAND_PARAM_HELP_OVERRIDES.get((str(cmd_name or ""), str(param["name"])))
    if override_help:
        hint = override_help
    kwargs = {"help": _escape_argparse_help(hint) or f"{param['name']} parameter", "dest": param["name"]}
    is_mapping_type = False

    if param["name"] == "method" and (
        (cmd_name in {"forecast_generate", "forecast_conformal_intervals", "forecast_tune_genetic", "forecast_tune_optuna"})
        or _is_forecast_method_literal(
            param.get("type"),
            is_literal_origin=is_literal_origin,
            get_origin_func=get_origin,
            get_args_func=get_args,
        )
    ):
        if not (param_names and "library" in param_names):
            help_suffix = " Use forecast_list_methods to browse available methods."
            if "forecast_list_methods" not in kwargs["help"]:
                kwargs["help"] = f"{kwargs['help']}{help_suffix}"
            kwargs["metavar"] = "METHOD"
    else:
        try:
            ptype = param.get("type")
            base_type, origin = unwrap_optional_type(ptype)

            is_typed_dict = is_typed_dict_type(base_type)
            is_mapping_type = (base_type in (dict, Dict)) or (origin in (dict, Dict)) or is_typed_dict

            kwargs["type"] = str

            if base_type in (int, float, str):
                kwargs["type"] = base_type
            elif base_type is bool:
                kwargs["type"] = _normalize_bool_choice
                kwargs["choices"] = ["true", "false"]
                kwargs["metavar"] = "bool"

            if origin in (list, tuple):
                inner = get_args(ptype)[0] if get_args(ptype) else None
                inner_origin = get_origin(inner)
                if is_literal_origin(inner_origin):
                    choices = [str(v) for v in get_args(inner)]
                    if choices:
                        kwargs["choices"] = choices
                    kwargs["type"] = str
                    kwargs["nargs"] = "+"
                else:
                    kwargs["type"] = str
                    kwargs["nargs"] = "+"
            elif is_literal_origin(origin):
                choices = [str(v) for v in get_args(base_type)]
                if choices:
                    kwargs["choices"] = choices
                kwargs["type"] = str
        except Exception as exc:
            debug(f"Type resolution failed for param '{param['name']}': {exc}")
            kwargs["type"] = str

    if not param["required"] and not (param["type"] is bool and param["default"] is None):
        kwargs["default"] = param["default"]

    choice_override = _COMMAND_PARAM_CHOICE_OVERRIDES.get((str(cmd_name or ""), str(param["name"])))
    if choice_override:
        kwargs["choices"] = list(choice_override)
        kwargs["type"] = lambda value: str(value or "").strip().lower()

    if (str(cmd_name or ""), str(param["name"])) == ("indicators_list", "category"):
        kwargs["type"] = lambda value: str(value or "").strip().lower()

    return kwargs, is_mapping_type


def add_dynamic_arguments(
    parser: Any,
    param_info: Dict[str, Any],
    *,
    resolve_param_kwargs: Callable[..., Tuple[Dict[str, Any], bool]],
    param_docs: Optional[Dict[str, str]] = None,
    cmd_name: Optional[str] = None,
) -> None:
    """Add CLI arguments for an introspected function schema."""
    def _extra_option_flags(param_name: str, cmd_name_value: Optional[str]) -> tuple[str, ...]:
        extras: list[str] = []
        if param_name == "limit" and cmd_name_value in _BAR_LIMIT_ALIAS_COMMANDS:
            extras.append("--bars")
        if cmd_name_value == "trade_history" and param_name == "position_ticket":
            extras.append("--ticket")
        if cmd_name_value == "market_depth_fetch" and param_name == "compact":
            extras.extend(["--require-dom", "--require_dom"])
        return tuple(extras)

    for param in param_info["params"]:
        if not should_expose_cli_param(cmd_name=cmd_name, param_name=str(param.get("name") or "")):
            continue
        hyph = f"--{param['name'].replace('_', '-')}"
        uscr = f"--{param['name']}"
        option_flags, hidden_option_flags = _split_visible_and_hidden_flags(
            hyph,
            uscr,
            *_extra_option_flags(param["name"], cmd_name),
        )

        param_names = {p.get("name") for p in (param_info.get("params") or []) if isinstance(p, dict)}
        kwargs, is_mapping_type = resolve_param_kwargs(
            param,
            param_docs,
            cmd_name=cmd_name,
            param_names=param_names,
        )

        is_optional_bool = param.get("type") is bool and not param.get("required", False)
        allow_optional_first_positional = (
            param == param_info["params"][0]
            and (str(cmd_name or ""), str(param["name"])) in _OPTIONAL_FIRST_POSITIONAL_PARAMS
        )

        if param["required"] and param == param_info["params"][0]:
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            positional_kwargs["nargs"] = "?"
            positional_kwargs["default"] = argparse.SUPPRESS
            parser.add_argument(param["name"], **positional_kwargs)
            hidden_alias_kwargs = dict(kwargs)
            hidden_alias_kwargs["help"] = argparse.SUPPRESS
            if option_flags:
                parser.add_argument(*option_flags, **hidden_alias_kwargs)
            if hidden_option_flags:
                parser.add_argument(*hidden_option_flags, **hidden_alias_kwargs)
        elif allow_optional_first_positional:
            positional_kwargs = {k: v for k, v in kwargs.items() if k in ("help", "type", "choices", "metavar")}
            positional_kwargs["nargs"] = "?"
            positional_kwargs["default"] = argparse.SUPPRESS
            parser.add_argument(param["name"], **positional_kwargs)
            hidden_alias_kwargs = dict(kwargs)
            hidden_alias_kwargs["help"] = argparse.SUPPRESS
            if option_flags:
                parser.add_argument(*option_flags, **hidden_alias_kwargs)
            if hidden_option_flags:
                parser.add_argument(*hidden_option_flags, **hidden_alias_kwargs)
        else:
            if is_optional_bool:
                local_kwargs = dict(kwargs)
                local_kwargs["nargs"] = "?"
                local_kwargs["const"] = "true"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
                no_flags, no_hidden_flags = _split_visible_and_hidden_flags(
                    f"--no-{param['name'].replace('_', '-')}",
                    f"--no_{param['name']}",
                )
                if no_flags:
                    parser.add_argument(
                        *no_flags,
                        dest=param["name"],
                        action="store_const",
                        const="false",
                        help=argparse.SUPPRESS,
                    )
                if no_hidden_flags:
                    hidden_no_kwargs = {
                        "dest": param["name"],
                        "action": "store_const",
                        "const": "false",
                        "help": argparse.SUPPRESS,
                    }
                    parser.add_argument(*no_hidden_flags, **hidden_no_kwargs)
            elif is_mapping_type:
                local_kwargs = dict(kwargs)
                local_kwargs["nargs"] = "?"
                local_kwargs["const"] = "__PRESENT__"
                if option_flags:
                    parser.add_argument(*option_flags, **local_kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(local_kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
            else:
                if option_flags:
                    parser.add_argument(*option_flags, **kwargs)
                if hidden_option_flags:
                    hidden_kwargs = dict(kwargs)
                    hidden_kwargs["help"] = argparse.SUPPRESS
                    parser.add_argument(*hidden_option_flags, **hidden_kwargs)
        if cmd_name == "trade_history" and param["name"] == "minutes_back":
            parser.add_argument(
                "--days",
                dest="_trade_history_days",
                type=float,
                default=argparse.SUPPRESS,
                metavar="DAYS",
                help="Alias for --minutes-back expressed in days.",
            )

        if is_mapping_type:
            params_flags = _dedupe_flags(
                f"--{param['name'].replace('_', '-')}-params",
                f"--{param['name']}_params",
            )
            parser.add_argument(
                *params_flags,
                dest=f"{param['name']}_params",
                type=str,
                default=None,
                help=f"Extra params for {param['name']} (key=value[,key=value])",
            )
