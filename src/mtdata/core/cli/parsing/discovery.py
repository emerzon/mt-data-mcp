import argparse
import inspect
from typing import Any, Callable, Dict, Optional, Tuple

ToolInfo = Dict[str, Any]


_OPTIONAL_FIRST_POSITIONAL_PARAMS: set[tuple[str, str]] = {
    ("finviz_forex", "symbol"),
    ("finviz_news", "symbol"),
    ("news", "symbol"),
    ("correlation_matrix", "symbols"),
    ("cointegration_test", "symbols"),
    ("market_scan", "symbols"),
    ("causal_discover_signals", "symbols"),
    ("market_status", "symbol"),
    ("trade_get_open", "symbol"),
    ("trade_get_pending", "symbol"),
    ("trade_place", "symbol"),
    ("trade_risk_analyze", "symbol"),
    ("trade_var_cvar_calculate", "symbol"),
    ("forecast_list_library_models", "library"),
    ("wait_event", "symbol"),
}

_HIDDEN_OPTIONAL_FIRST_POSITIONAL_FLAGS: set[tuple[str, str]] = {
    ("correlation_matrix", "symbols"),
    ("cointegration_test", "symbols"),
    ("causal_discover_signals", "symbols"),
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
    (
        "patterns_detect",
        "mode",
    ): [
        "all",
        "candlestick",
        "classic",
        "harmonic",
        "fractal",
        "elliott",
    ],
}


_COMMAND_PARAM_HELP_OVERRIDES: Dict[tuple[str, str], str] = {
    ("data_fetch_candles", "indicators"): "Technical indicators. Prefer compact CLI specs like 'rsi(14),macd(12,26,9)' or JSON arrays like '[{\"name\":\"rsi\",\"params\":[14]}]'. Bare names, rsi_14, sma=20, and named params like rsi(length=14) also work. Use params syntax, not sma,20.",
    ("indicators_list", "trading_style"): "Filter indicators by common trading workflow: intraday, swing, or position.",
    ("trade_place", "magic"): "MT5 magic number: integer strategy/order identifier used to group EA or strategy trades. Defaults to configured order_magic when omitted.",
    ("trade_get_open", "magic"): "MT5 magic number filter for positions from one strategy or EA. Omit for all magic numbers.",
    ("trade_get_pending", "magic"): "MT5 magic number filter for pending orders from one strategy or EA. Omit for all magic numbers.",
    ("trade_close", "magic"): "MT5 magic number filter when closing matching positions. Omit for all magic numbers.",
    ("wait_event", "magic"): "MT5 magic number filter for account events from one strategy or EA. Omit for all magic numbers.",
    ("finviz_screen", "filters"): "Filter key=value pairs, operator aliases like beta_under=1, Finviz shorthand, or JSON object. Examples: 'country=USA,marketcap=mega', 'pe_under=15,beta_under=1', 'cap_largeover,exch_nyse', '{\"Exchange\":\"NASDAQ\",\"Sector\":\"Technology\"}'. Common keys include Exchange, Index, Sector, Industry, Country, Market Cap., P/E, Dividend Yield, RSI (14), Average Volume, and Price.",
    ("finviz_screen", "limit"): "Max screener results to return on this page.",
    ("finviz_screen", "order"): "Finviz sort key. Example: -marketcap for descending or price for ascending.",
    ("finviz_news", "limit"): "Max news items to return on this page.",
    ("finviz_insider", "limit"): "Max insider trades to return on this page.",
    ("finviz_calendar", "start"): "Start date (YYYY-MM-DD).",
    ("finviz_calendar", "end"): "End date (YYYY-MM-DD).",
    ("forecast_barrier_optimize", "method"): "Barrier simulation method: mc_gbm, mc_gbm_bb, hmm_mc, garch, bootstrap, heston, jump_diffusion, or auto.",
    ("causal_discover_signals", "symbols"): (
        "Comma-separated MT5 symbols (e.g. EURUSD,GBPUSD); one symbol auto-expands "
        "to its MT5 group. Optional with --group."
    ),
    ("options_barrier_price", "option_type"): "Option side: call or put.",
    ("options_chain", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX."
    ),
    ("options_expirations", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX."
    ),
    ("options_heston_calibrate", "symbol"): (
        "Underlying symbol for listed options, e.g. AAPL or SPX."
    ),
    ("forecast_tune_optuna", "search_space"): "Optuna search space (JSON or k=v).",
    ("indicators_list", "detail"): "Output detail: compact table or full rows with aliases and descriptions.",
    ("market_snapshot", "sections"): (
        "Analysis modules to include: quote, levels, patterns, regime, forecast, "
        "or all. Defaults to quote,levels,patterns."
    ),
    ("market_snapshot", "detail"): (
        "Field verbosity inside selected sections; full does not add sections. "
        "Use --sections all for every snapshot module."
    ),
    ("causal_discover_signals", "limit"): "Max causal link rows to return.",
    ("causal_discover_signals", "window_bars"): (
        "Historical bars per symbol used for causal tests."
    ),
    ("cointegration_test", "symbols"): (
        "Comma-separated MT5 symbols (e.g. EURUSD,GBPUSD); one symbol auto-expands "
        "to its MT5 group. Optional with --group."
    ),
    ("cointegration_test", "limit"): "Max cointegration pair rows to return.",
    ("cointegration_test", "window_bars"): (
        "Historical bars per symbol used for the cointegration test window."
    ),
    ("correlation_matrix", "limit"): "Max correlation pair rows to return.",
    ("correlation_matrix", "window_bars"): (
        "Historical bars per symbol used for the correlation window."
    ),
    ("correlation_matrix", "symbols"): (
        "Comma-separated MT5 symbols (e.g. EURUSD,GBPUSD); one symbol auto-expands "
        "to its MT5 group. Optional with --group."
    ),
    ("market_scan", "symbols"): (
        "Comma-separated MT5 symbols to scan. Optional with --group."
    ),
    ("market_scan", "preset"): (
        "Built-in scan preset: oversold, overbought, high-volume, tight-spread, "
        "gap-up, or gap-down. Explicit filter flags override preset defaults."
    ),
    ("market_scan", "rank_order"): (
        "Sort direction for ranked rows: auto, asc/ascending, or desc/descending. "
        "Auto keeps tight spreads and oversold RSI ascending; most other ranks descending."
    ),
    ("outliers_detect", "score_fields"): (
        "Comma-separated candle features to score: return, volume, and/or range."
    ),
    ("outliers_detect", "threshold"): (
        "Positive robust-deviation cutoff; 3.5 is a common MAD threshold."
    ),
    ("labels_triple_barrier", "detail"): (
        "Detail level: compact (small outcome sample), standard (recent lookback rows), "
        "summary, or full."
    ),
    ("labels_triple_barrier", "limit"): (
        "Historical bars fetched for labeling; not an output row limit. "
        "Too-small values are raised to cover lookback plus horizon."
    ),
    ("labels_triple_barrier", "lookback"): (
        "Recent labeled entries used for compact/summary stats and samples; "
        "limit controls fetched history."
    ),
    ("market_scan", "limit"): "Max matching symbols to return.",
    ("market_depth_fetch", "require_dom"): "Fail if DOM is unavailable instead of falling back to a quote snapshot.",
    ("patterns_detect", "limit"): (
        "Historical bars fetched for pattern analysis; use top_k for compact "
        "top-pattern count."
    ),
    ("patterns_detect", "mode"): "Pattern mode: all, candlestick, classic, harmonic, fractal, or elliott.",
    ("patterns_detect", "engine"): (
        "Classic-mode engine: native or stock_pattern. Omitted classic calls "
        "use native; invalid for other modes."
    ),
    ("report_generate", "template"): (
        "Report template: minimal fast context+forecast, basic balanced default, "
        "advanced regimes/HAR/conformal, scalping M5, intraday H1, swing H4/D1, "
        "or position D1/W1. Runtime cost: minimal is the quick path; "
        "basic/advanced and style templates may invoke multiple MT5 fetches plus "
        "pivots, patterns, backtests, barriers, and regime checks."
    ),
    ("temporal_analyze", "lookback"): (
        "Historical bars used when start/end are omitted. Defaults to a "
        "timeframe-aware seasonal window: 210 days for day-of-week, 60 days "
        "for hour/session, 730 days for month, and 365 days for overall "
        "analysis, bounded to 200-20000 bars (H1 session: 1440 bars)."
    ),
    ("regime_detect", "limit"): (
        "Historical bars fetched for regime detection. Defaults to the effective "
        "lookback plus warmup bars; use max_regimes for compact output count."
    ),
    ("symbols_list", "limit"): "Max symbols or groups to return.",
    ("symbols_top_markets", "rank_by"): (
        "Leaderboard to compute: abs_price_change_pct (default), all, "
        "spread/spread_pct, tick_volume, price_change/price_change_pct, "
        "or abs_price_change/abs_price_change_pct."
    ),
    ("symbols_top_markets", "limit"): (
        "Max symbols for the selected ranking; per leaderboard when rank_by=all."
    ),
    ("trade_close", "close_all"): (
        "Close all matching open positions instead of a single ticket."
    ),
    ("trade_close", "dry_run"): (
        "Preview the close request without sending it to the broker."
    ),
    ("trade_close", "profit_only"): "Only close positions currently in profit.",
    ("trade_close", "loss_only"): "Only close positions currently at a loss.",
    ("trade_close", "close_priority"): (
        "When multiple positions match, close loss_first, profit_first, or largest_first."
    ),
    ("trade_modify", "dry_run"): (
        "Preview the modification without sending it to the broker."
    ),
    ("trade_modify", "idempotency_key"): (
        "Optional in-process dedupe key; replays the prior matching modify result."
    ),
    ("trade_place", "dry_run"): (
        "Preview the order without sending it to the broker."
    ),
    ("trade_place", "detail"): (
        "Dry-run preview detail: compact for key checks, full for execution diagnostics."
    ),
    ("trade_place", "require_sl_tp"): (
        "Require both stop_loss and take_profit for market orders."
    ),
    ("trade_place", "auto_close_on_sl_tp_fail"): (
        "If TP/SL attachment fails after a market fill, try to close the unprotected position."
    ),
    ("trade_history", "minutes_back"): (
        "History lookback in minutes. Defaults to 10080 minutes (7 days) when "
        "start/end and minutes_back are omitted."
    ),
    ("trade_journal_analyze", "minutes_back"): (
        "Journal history lookback in minutes. Defaults to 10080 minutes (7 days) "
        "when start/end and minutes_back are omitted."
    ),
    ("trade_modify", "expiration"): "Pending order expiration time (dateparser string, UTC epoch seconds, or GTC token).",
    ("trade_place", "expiration"): "Pending order expiration time (dateparser string, UTC epoch seconds, or GTC token).",
    ("wait_event", "symbol"): "Trading symbol (e.g. EURUSD).",
    ("wait_event", "timeframe"): (
        "Candle/event timeframe. Defaults to M1 for faster event polling; "
        "set H1 for hourly boundaries."
    ),
    ("wait_event", "watch_for"): (
        "Event names or event objects. Examples: order_filled, "
        "'{\"type\":\"order_filled\",\"symbol\":\"EURUSD\"}'. "
        "Use candle_close for a boundary wait."
    ),
    ("wait_event", "end_on"): (
        "Boundary event names or objects. Example: candle_close or "
        "'{\"type\":\"candle_close\",\"timeframe\":\"H1\"}'."
    ),
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


def _normalize_cli_choice_value(value: Any) -> str:
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
    if str(cmd_name or "") == "finviz_calendar" and str(param_name or "") in {"date_from", "date_to"}:
        return False
    if str(cmd_name or "") == "wait_event" and str(param_name or "") == "instrument":
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
    errors: Optional[list[str]] = None,
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
        message = f"bootstrap_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    try:
        reg = get_registered_tools()
        if reg and hasattr(reg, "items"):
            registry = reg
    except Exception as exc:
        message = f"get_registered_tools failed: {exc}"
        debug(message)
        if errors is not None:
            errors.append(message)
    if mcp is not None:
        try:
            registry = get_mcp_registry(mcp) or registry
        except Exception as exc:
            message = f"get_mcp_registry failed: {exc}"
            debug(message)
            if errors is not None:
                errors.append(message)

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

    def _is_model_type(value: Any) -> bool:
        return isinstance(value, type) and (
            callable(getattr(value, "model_validate", None))
            or callable(getattr(value, "parse_obj", None))
        )

    def _escape_argparse_help(text: Optional[str]) -> Optional[str]:
        return text.replace("%", "%%") if isinstance(text, str) else text

    desc = None
    if param_docs and param["name"] in param_docs:
        desc = param_docs[param["name"]]
    hint = desc or param_hints.get(param["name"])
    override_help = _COMMAND_PARAM_HELP_OVERRIDES.get((str(cmd_name or ""), str(param["name"])))
    if override_help:
        hint = override_help
    fallback_help = f"Value for {str(param['name']).replace('_', ' ')}."
    kwargs = {"help": _escape_argparse_help(hint) or fallback_help, "dest": param["name"]}
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
            is_mapping_type = (
                (base_type in (dict, Dict))
                or (origin in (dict, Dict))
                or is_typed_dict
                or _is_model_type(base_type)
            )

            kwargs["type"] = str

            if base_type in (int, float, str):
                kwargs["type"] = base_type
            elif base_type is bool:
                kwargs["type"] = _normalize_cli_choice_value
                kwargs["choices"] = ["true", "false"]

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

    choice_override_key = (str(cmd_name or ""), str(param["name"]))
    choice_override = _COMMAND_PARAM_CHOICE_OVERRIDES.get(choice_override_key)
    if choice_override:
        kwargs["choices"] = list(choice_override)
        kwargs["type"] = _normalize_cli_choice_value

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
    has_mapping_param = False

    def _extra_option_flags(param_name: str, cmd_name_value: Optional[str]) -> tuple[str, ...]:
        extras: list[str] = []
        if cmd_name_value == "trade_history" and param_name == "position_ticket":
            extras.append("--ticket")
        if cmd_name_value == "forecast_backtest_run" and param_name == "methods":
            extras.append("--method")
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
            option_kwargs = dict(kwargs)
            if (
                str(param["name"]) != "symbols"
                or (str(cmd_name or ""), str(param["name"])) in _HIDDEN_OPTIONAL_FIRST_POSITIONAL_FLAGS
            ):
                option_kwargs["help"] = argparse.SUPPRESS
            if option_flags:
                parser.add_argument(*option_flags, **option_kwargs)
            if hidden_option_flags:
                hidden_option_kwargs = dict(kwargs)
                hidden_option_kwargs["help"] = argparse.SUPPRESS
                parser.add_argument(*hidden_option_flags, **hidden_option_kwargs)
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
        if str(param["name"]) == "minutes_back" and str(cmd_name or "").startswith("trade_"):
            parser.add_argument(
                "--days",
                dest="_trade_days",
                type=float,
                default=argparse.SUPPRESS,
                metavar="DAYS",
                help="Alias for --minutes-back expressed in days.",
        )

        if is_mapping_type:
            has_mapping_param = True
            if param["name"] == "params":
                continue
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
    if has_mapping_param:
        parser.add_argument(
            "--set",
            dest="set_overrides",
            action="append",
            default=None,
            metavar="PARAM.KEY=VALUE",
            help="Override nested mapping params, e.g. --set params.window=64.",
        )
