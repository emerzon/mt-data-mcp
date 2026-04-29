"""
Shared JSON schema helpers for CLI/server tool inputs.

Provides reusable $defs such as TimeframeSpec and helpers to apply them
to per-tool parameter schemas.
"""
import inspect
import logging
import types
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from typing_extensions import TypedDict

try:
    import annotationlib
except Exception:  # pragma: no cover - Python 3.14+ should provide this
    annotationlib = None

from .constants import TIMEFRAME_MAP
from .parameter_contracts import PARAMETER_HELP

_logger = logging.getLogger(__name__)

PARAM_HINTS = {
    **PARAMETER_HELP,
    "direction": "Trade direction (long/short).",
    "limit": "Max rows/bars to return.",
    "start": "Start time (dateparser).",
    "end": "End time (dateparser).",
    "group_by": "Temporal grouping: dow/day_of_week, hour, month, or all.",
    "day_of_week": "Weekday filter (0-6 or Mon..Sun).",
    "month": "Month filter (1-12 or Jan..Dec).",
    "time_range": "Time-of-day filter 'HH:MM-HH:MM' (start inclusive, end exclusive; wraps midnight).",
    "rank_by": (
        "Ranking to compute for market scans: abs_price_change_pct, "
        "price_change_pct, tick_volume, rsi, or spread_pct."
    ),
    "return_mode": "Return calculation mode: pct or log.",
    "ohlcv": "OHLCV column selector (e.g. 'close', 'high,low').",
    "indicators": "Indicators as compact specs like 'rsi_14', 'rsi(length=14)', 'macd(12,26,9)', or 'macd(fast=12,slow=26,signal=9)', or JSON like '[{\"name\":\"rsi\",\"params\":{\"length\":14}}]'. Bare names such as 'rsi' are also accepted.",
    "denoise": "Denoise preset name or JSON spec. Examples: --denoise kalman or --denoise '{\"method\":\"kalman\",\"params\":{\"lookback\":100}}'.",
    "simplify": "Simplify preset name or JSON spec. Examples: --simplify select, --simplify '{\"mode\":\"select\",\"method\":\"lttb\",\"ratio\":0.2}', or --simplify select --simplify-params \"ratio=0.2\".",
    "include_incomplete": "Include the latest forming candle; defaults to false, so responses expose has_forming_candle/incomplete_candles_skipped while returned rows stay on closed bars.",
    "method": "Method/algorithm for this tool.",
    "mode": "Mode for this tool.",
    "engine": "Detection engine or comma-separated engines (for ensemble mode).",
    "transform": "Preprocessing transform applied before analysis, such as log_return, pct, diff, level, or log_level depending on the tool.",
    "min_overlap": "Minimum overlapping transformed samples required for each pair before a pairwise statistic is calculated.",
    "min_regime_bars": "Minimum bars a detected regime must span; shorter runs are merged to reduce noisy state flicker.",
    "ensemble": "Enable multi-engine consensus merge when true.",
    "ensemble_weights": "Optional JSON weight map used when ensemble aggregation is weighted.",
    "library": "Forecast library/group (e.g. native, statsforecast, sktime).",
    "model": "Model identifier for this tool.",
    "horizon": "Forecast horizon in bars.",
    "steps": "Number of backtest anchors or steps to run.",
    "spacing": "Spacing between backtest anchors (in bars).",
    "methods": "One or more method names (comma-separated or space-separated).",
    "alpha": "Alpha parameter for the selected method.",
    "params": "Method parameters as JSON or k=v pairs. Examples: --params alpha=0.3,beta=0.1 or --params '{\"alpha\":0.3,\"beta\":0.1}'.",
    "params_per_method": "Per-method params map (e.g. {method: {k: v}}).",
    "as_of": "Reference time override (dateparser).",
    "ci_alpha": "Confidence interval alpha.",
    "features": "Feature spec as JSON or k=v pairs. Examples: --features lag=3,rolling=5 or --features '{\"lag\":3,\"rolling\":5}'.",
    "dimred_method": "Dimred method (e.g. pca, tsne).",
    "dimred_params": "Dimred params (JSON or k=v).",
    "target_spec": "Target spec (JSON or k=v).",
    "quantity": "Quantity to model (price/return/volatility).",
    "target": "Target series (price/return).",
    "points": "Target point count.",
    "ratio": "Target compression ratio.",
    "epsilon": "Tolerance value (e.g. RDP).",
    "max_error": "Max approximation error.",
    "segments": "Segment count.",
    "bucket_seconds": "Resample bucket size in seconds.",
    "buffer_seconds": "Extra seconds to wait after the candle close before returning.",
    "max_wait_seconds": "Maximum seconds this tool may block before returning timing metadata instead.",
    "schema": "Encoding schema (e.g. delta).",
    "bits": "Bits per symbol for encoding schemas.",
    "paa": "PAA segments for symbolic representation.",
    "znorm": "Apply z-normalization before processing.",
    "threshold_pct": "Segmentation change threshold (percent).",
    "threshold": "Change-point probability threshold (0-1).",
    "value_col": "Column name to use for value-based operations.",
    "lookback": "Historical bars to use.",
    "spacing_pct": "Spacing as percent of duration.",
    "history_kind": "History type (deals or orders).",
    "list_mode": "List mode (symbols or groups).",
    "volume": "Order volume (lots).",
    "comment": "Order comment tag.",
    "deviation": "Max slippage (points).",
    "order_type": "Required order type: BUY/SELL for market orders, or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP for pending orders.",
    "price": "Entry price (required for pending).",
    "stop_loss": "Stop-loss price.",
    "take_profit": "Take-profit price.",
    "require_sl_tp": "For market orders, require both stop_loss and take_profit and fail if protection cannot be attached. Defaults to true.",
    "auto_close_on_sl_tp_fail": "If a filled market order cannot attach TP/SL, immediately try to close the unprotected position. Defaults to true.",
    "ticket": "Ticket/order ID.",
    "expiration": "Expiration time/date. For trade orders accepts a dateparser string, UTC epoch seconds, or GTC token; for option tools accepts YYYY-MM-DD.",
    "profit_only": "Close only profitable positions when true.",
    "loss_only": "Close only losing positions when true.",
    "position_ticket": "Filter history rows by the linked position ticket.",
    "deal_ticket": "Filter deal history by deal ticket.",
    "order_ticket": "Filter history by order ticket.",
    "minutes_back": "Look back this many minutes from end/now instead of using start.",
    "min_strength": "Minimum candlestick pattern strength (0-1).",
    "min_gap": "Minimum bars between detected patterns.",
    "robust_only": "Only return high-confidence patterns when true.",
    "whitelist": "Comma-separated pattern names to include.",
    "universe": "Symbol scan universe: visible (fast default) or all (includes hidden tradable symbols and may be slower).",
    "series_time": "Series timestamp format (string or epoch).",
    "include_completed": "Include completed patterns (otherwise forming only).",
    "include_series": "Include raw series in output.",
    "config": "Pattern-specific config overrides (JSON or k=v).",
    "top_k": "Return top-K results.",
    "tp_abs": "Take-profit absolute price level.",
    "sl_abs": "Stop-loss absolute price level.",
    "tp_pct": "Take-profit percent move (e.g. 2.0 for 2%).",
    "sl_pct": "Stop-loss percent move (e.g. 1.0 for 1%).",
    "tp_ticks": "Take-profit barrier distance in ticks.",
    "sl_ticks": "Stop-loss barrier distance in ticks.",
    "label_on": "Barrier evaluation basis: close or high_low.",
    "barrier": "Barrier level for closed-form probability.",
    "mu": "Drift override for closed-form barrier method.",
    "sigma": "Volatility override for closed-form barrier method.",
    "search_space": "Genetic search space (JSON or k=v).",
    "metric": "Optimization metric.",
    "population": "Population size.",
    "generations": "Generation count.",
    "n_trials": "Optuna trial count.",
    "timeout": "Optimization timeout in seconds.",
    "n_jobs": "Parallel worker count for optimization.",
    "sampler": "Optuna sampler strategy.",
    "pruner": "Optuna pruner strategy.",
    "study_name": "Optuna study name (for persistence/resume).",
    "storage": "Optuna storage URL (for persistence/resume).",
    "crossover_rate": "Genetic crossover probability (0-1).",
    "mutation_rate": "Genetic mutation probability (0-1).",
    "seed": "Random seed for reproducibility.",
    "trade_threshold": "Trade threshold for backtests.",
    "slippage_bps": "Backtest slippage in basis points.",
    "objective": "Optimization objective.",
    "return_grid": "Include full grid results in output.",
    "viable_only": "Only return viable barrier candidates when true.",
    "concise": "Return a shorter barrier-optimization payload when true.",
    "grid_style": "TP/SL grid style.",
    "preset": "TP/SL grid preset. Common examples: volatility, conservative, aggressive.",
    "tp_min": "Minimum TP level for grid (pct or tick-size units depending on mode).",
    "tp_max": "Maximum TP level for grid (pct or tick-size units depending on mode).",
    "tp_steps": "Number of TP grid steps.",
    "sl_min": "Minimum SL level for grid (pct or tick-size units depending on mode).",
    "sl_max": "Maximum SL level for grid (pct or tick-size units depending on mode).",
    "sl_steps": "Number of SL grid steps.",
    "vol_window": "Lookback window for volatility-based grid.",
    "vol_min_mult": "Minimum volatility multiple for grid.",
    "vol_max_mult": "Maximum volatility multiple for grid.",
    "vol_steps": "Number of volatility grid steps.",
    "vol_floor_pct": "Minimum TP/SL percent floor when using volatility grid.",
    "vol_floor_ticks": "Minimum volatility-derived barrier distance in ticks.",
    "ratio_min": "Minimum TP/SL ratio for ratio grid.",
    "ratio_max": "Maximum TP/SL ratio for ratio grid.",
    "ratio_steps": "Number of ratio grid steps.",
    "refine": "Run a second-stage refinement around best grid point.",
    "refine_radius": "Refinement radius around best grid point.",
    "refine_steps": "Number of refinement steps per axis.",
    "optimizer": "Barrier optimizer backend: grid or optuna.",
    "fast_defaults": "Use a faster low-cost optimization profile (fewer sims/steps/trials). Example: '--fast-defaults true'.",
    "search_profile": "Search intensity profile: fast, medium, or long.",
    "profile": "Alias for search_profile in params payloads.",
    "statistical_robustness": "Enable statistical robustness diagnostics for the selected barrier pair.",
    "target_ci_width": "Requested probability CI width used for minimum-simulation guidance.",
    "n_seeds_stability": "Number of alternate seed re-runs for cross-seed stability checks.",
    "enable_bootstrap": "Enable bootstrap uncertainty estimation for selected metrics.",
    "n_bootstrap": "Number of bootstrap resamples when bootstrap uncertainty is enabled.",
    "enable_convergence_check": "Enable convergence diagnostics for the selected objective metric.",
    "convergence_window": "Rolling window size used by the convergence diagnostic.",
    "convergence_threshold": "Tolerance threshold used by the convergence diagnostic.",
    "enable_power_analysis": "Enable statistical power analysis for the selected candidate.",
    "power_effect_size": "Minimum detectable effect size assumed by power analysis.",
    "enable_sensitivity_analysis": "Enable local TP/SL sensitivity analysis around the selected candidate.",
    "sensitivity_params": "List of barrier parameters to vary during sensitivity analysis.",
    "ensemble_methods": "Comma-list or array of member simulators for method=ensemble.",
    "ensemble_agg": "Ensemble aggregation: median or weighted_mean (equal weights by default).",
    "optuna_pareto": "Enable Optuna multi-objective Pareto optimization.",
    "optuna_pareto_objectives": "JSON metric->direction map for Pareto optimization.",
    "pareto_limit": "Maximum number of Pareto-front rows returned.",
    "option_type": "Option side filter: call, put, or both.",
    "min_open_interest": "Minimum option open interest filter.",
    "min_volume": "Minimum option volume filter.",
    "risk_free_rate": "Annual risk-free rate for option pricing/calibration.",
    "dividend_yield": "Annual dividend yield for option pricing/calibration.",
    "maturity_days": "Option time-to-maturity in days.",
    "strike": "Option strike price.",
    "barrier_type": "Barrier style: up_in, up_out, down_in, down_out.",
    "rebate": "Barrier option rebate paid upon knock-in/out.",
    "volatility": "Annualized volatility input for pricing.",
    # Finviz parameters
    "news_type": "News type: 'news' or 'blogs'.",
    "filters": "JSON filter dict for stock screener.",
    "view": "Screener view: overview, valuation, financial, ownership, performance, technical.",
    "option": "Insider activity type: latest, top week, top owner trade, insider buy, insider sale.",
    "period": "Finviz earnings period: This Week, Next Week, Previous Week, This Month.",
    "impact": "Economic calendar impact filter: low, medium, high.",
    "page": "Page number for pagination (default 1).",
}


_TIMEFRAME_CHOICES = tuple(sorted(TIMEFRAME_MAP.keys()))
TimeframeLiteral = Literal[_TIMEFRAME_CHOICES]  # type: ignore
AutoTimeframeLiteral = Union[TimeframeLiteral, Literal["auto"]]
CANONICAL_OUTPUT_SHAPE_DETAILS = ("compact", "standard", "summary", "full")
CANONICAL_OUTPUT_DETAIL_ALIASES = types.MappingProxyType(
    {"summary_only": "summary", "standard": "compact", "summary": "compact"}
)
DetailLiteral = Literal["compact", "standard", "summary", "full"]
CompactFullDetailLiteral = DetailLiteral
CompactStandardFullDetailLiteral = DetailLiteral
SummaryCompactFullDetailLiteral = Literal["full", "summary", "compact"]

# ---- Technical Indicators (dynamic discovery and application) ----
def _load_indicator_doc_choices(
    list_ta_indicators_docs: Optional[Any] = None,
) -> Tuple[List[str], List[str]]:
    """Load small schema enums for indicator docs.

    Indicator names intentionally stay runtime-validated strings to avoid
    inflating MCP schemas with the full indicator registry.
    """
    if list_ta_indicators_docs is None:
        try:
            from ..utils.indicators import list_ta_indicators as list_ta_indicators_docs
        except Exception as ex:
            _logger.warning(
                "Indicator schema choices are unavailable because indicator metadata could not be imported: %s",
                ex,
            )
            return [], []
    try:
        docs = list_ta_indicators_docs(detailed=False)
    except Exception as ex:
        _logger.warning(
            "Indicator schema choices are unavailable because indicator metadata loading failed: %s",
            ex,
        )
        return [], []
    if not isinstance(docs, list):
        _logger.warning(
            "Indicator schema choices are unavailable because indicator metadata returned %s instead of a list.",
            type(docs).__name__,
        )
        return [], []

    categories = sorted(
        {it.get("category") for it in docs if isinstance(it, dict) and it.get("category")}
    )
    return categories, []


_CATEGORY_CHOICES, _INDICATOR_NAME_CHOICES = _load_indicator_doc_choices()

if _CATEGORY_CHOICES:
    # Create a Literal type alias dynamically
    CategoryLiteral = Literal[tuple(_CATEGORY_CHOICES)]  # type: ignore
else:
    CategoryLiteral = str  # fallback

IndicatorNameLiteral = str

class IndicatorSpec(TypedDict, total=False):
    """Structured TI spec: name with optional numeric params.

    Note: 'name' accepts any string to allow compact forms like "rsi(20)".
    The optional 'params' field accepts either positional numeric values or
    a named numeric parameter map.
    """
    name: str
    params: Union[List[float], Dict[str, float]]

# ---- Denoising (spec + application) ----
# Allowed denoising methods for first phase (no extra dependencies)
_DENOISE_METHODS = (
    "none",        # no-op
    "ema",         # exponential moving average
    "sma",         # simple moving average
    "median",      # rolling median
    "lowpass_fft", # zero-phase FFT low-pass
    "butterworth", # Butterworth IIR filter
    "hp",          # Hodrick-Prescott trend filter
    "savgol",      # Savitzky-Golay smoothing
    "tv",          # total variation denoising
    "kalman",      # 1D Kalman filter smoothing
    "hampel",      # Hampel outlier filter
    "bilateral",   # bilateral smoothing
    "wavelet_packet", # wavelet packet denoise
    "ssa",         # singular spectrum analysis
    "l1_trend",    # L1 trend filtering
    "lms",         # adaptive LMS filter
    "rls",         # adaptive RLS filter
    "beta",        # beta-IRLS smoothing
    "vmd",         # variational mode decomposition
    "loess",       # LOESS/LOWESS smoothing
    "stl",         # seasonal-trend decomposition
    "whittaker",   # Whittaker smoothing
    "gaussian",    # Gaussian kernel smoothing
    "wavelet",     # wavelet shrinkage (PyWavelets optional)
    "emd",         # empirical mode decomposition (PyEMD optional)
    "eemd",        # ensemble EMD (PyEMD optional)
    "ceemdan",     # complementary EEMD with adaptive noise (PyEMD optional)    
)

try:
    DenoiseMethodLiteral = Literal[_DENOISE_METHODS]  # type: ignore
except Exception:
    DenoiseMethodLiteral = str  # fallback for typing

class DenoiseSpec(TypedDict, total=False):
    method: DenoiseMethodLiteral  # type: ignore
    params: Dict[str, Any]
    columns: List[str]
    when: Literal['pre_ti', 'post_ti']  # type: ignore
    causality: Literal['causal', 'zero_phase']  # type: ignore
    keep_original: bool
    suffix: str

# ---- Simplify (schema for MCP) ----
_SIMPLIFY_MODES = (
    'select',        # pick representative existing rows
    'approximate',   # aggregate between selected rows
    'resample',      # time-bucket aggregation
    'encode',        # compact encodings (envelope, delta)
    'segment',       # swing points (e.g., ZigZag)
    'symbolic',      # SAX symbolic representation
)
_SIMPLIFY_METHODS = (
    'lttb', 'rdp', 'pla', 'apca'
)
try:
    SimplifyModeLiteral = Literal[_SIMPLIFY_MODES]  # type: ignore
except Exception:
    SimplifyModeLiteral = str
try:
    SimplifyMethodLiteral = Literal[_SIMPLIFY_METHODS]  # type: ignore
except Exception:
    SimplifyMethodLiteral = str
try:
    EncodeSchemaLiteral = Literal['envelope','delta']  # type: ignore
    SymbolicSchemaLiteral = Literal['sax']  # type: ignore
except Exception:
    EncodeSchemaLiteral = str
    SymbolicSchemaLiteral = str

class SimplifySpec(TypedDict, total=False):
    # Common
    mode: SimplifyModeLiteral  # type: ignore
    method: SimplifyMethodLiteral  # type: ignore
    points: int
    ratio: float
    # RDP/PLA/APCA specifics
    epsilon: float
    max_error: float
    segments: int
    # Resample
    bucket_seconds: int
    # Encode specifics
    schema: EncodeSchemaLiteral  # 'envelope' | 'delta' (or 'sax' when mode='symbolic')
    bits: int
    as_chars: bool
    alphabet: str
    scale: float
    zero_char: str
    # Segment specifics
    algo: Literal['zigzag','zz']  # type: ignore
    threshold_pct: float
    value_col: str
    # Symbolic specifics
    paa: int
    znorm: bool

# ---- Pivot Point methods (enums) ----
_PIVOT_METHODS = (
    "classic",
    "fibonacci",
    "camarilla",
    "woodie",
    "demark",
)

try:
    PivotMethodLiteral = Literal[_PIVOT_METHODS]  # type: ignore
except Exception:
    PivotMethodLiteral = str  # fallback for typing

# ---- Fast Forecast methods (enums) ----
# Dynamically fetch available methods + ensemble
# We need to ensure methods are registered, but avoiding heavy imports if possible.
# However, to get the full list, we essentially need to import the method modules.
# For schema purposes, we might want a superset or a hardcoded list if imports are too heavy.
# Given this is a CLI tool, maybe hardcoding is safer for startup time, but it drifts.
# Let's stick to the hardcoded list for now but ensure it's up to date with our knowledge.

# ---- Fast Forecast methods (enums) ----
#
# Use a conservative static list here. Importing the forecast registry during
# shared schema module import pulls in optional forecast method stacks and their
# heavy dependencies (for example torch), which makes unrelated tools noisy and
# slow at startup.
_FALLBACK_FORECAST_METHODS: Tuple[str, ...] = (
    "naive",
    "seasonal_naive",
    "drift",
    "theta",
    "fourier_ols",
    "ses",
    "holt",
    "holt_winters_add",
    "holt_winters_mul",
    "ets",
    "arima",
    "sarima",
    "mc_gbm",
    "hmm_mc",
    "mlforecast",
    "mlf_rf",
    "mlf_lightgbm",
    "statsforecast",
    "sktime",
    "chronos2",
    "chronos_bolt",
    "timesfm",
    "lag_llama",
    "ensemble",
    "analog",
)

_FORECAST_METHODS: Tuple[str, ...] = _FALLBACK_FORECAST_METHODS

ForecastLibraryLiteral = Literal[
    "native",
    "statsforecast",
    "sktime",
    "mlforecast",
    "pretrained",
]

try:
    ForecastMethodLiteral = Literal[_FORECAST_METHODS]  # type: ignore
except Exception:
    ForecastMethodLiteral = str  # fallback for typing



def shared_defs() -> Dict[str, Any]:
    """Return shared $defs for input schemas (e.g., TimeframeSpec).

    Note: Additional shared enums (SimplifyMode, etc.) are injected by the server.
    """
    return {
        "TimeframeSpec": {
            "type": "string",
            "enum": sorted(TIMEFRAME_MAP.keys()),
            "description": "MT5 timeframe code (e.g. H1/M30/D1)",
        }
    }


def complex_defs() -> Dict[str, Any]:
    """Return complex reusable definitions for nested params.

    These use $ref to shared enums that the server injects (e.g., SimplifyMode).
    """
    return {
        "IndicatorSpec": {
            "type": "object",
            "properties": {
                "name": {"$ref": "#/$defs/IndicatorName"},
                "params": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["name"],
            "additionalProperties": False,
            "description": "Indicator name plus optional numeric params.",
        },
        "DenoiseSpec": {
            "type": "object",
            "properties": {
                "method": {"$ref": "#/$defs/DenoiseMethod"},
                "params": {"type": "object", "description": "Method-specific overrides", "additionalProperties": True},
                "columns": {"type": "array", "items": {"type": "string"}},
                "when": {"$ref": "#/$defs/WhenSpec"},
                "causality": {"$ref": "#/$defs/CausalitySpec"},
                "keep_original": {"type": "boolean"},
                "suffix": {"type": "string"},
            },
            "required": ["method"],
            "additionalProperties": False,
            "description": "Denoise spec: method plus optional columns/params.",
        },
        "SimplifySpec": {
            "type": "object",
            "properties": {
                "mode": {"$ref": "#/$defs/SimplifyMode"},
                "method": {"$ref": "#/$defs/SimplifyMethod"},
                "points": {"type": "integer"},
                "ratio": {"type": "number"},
                "epsilon": {"type": "number"},
                "max_error": {"type": "number"},
                "segments": {"type": "integer"},
                "bucket_seconds": {"type": "integer"},
                "schema": {"oneOf": [
                    {"$ref": "#/$defs/EncodeSchema"},
                    {"$ref": "#/$defs/SymbolicSchema"}
                ]},
                "bits": {"type": "integer"},
                "as_chars": {"type": "boolean"},
                "alphabet": {"type": "string"},
                "scale": {"type": "number"},
                "zero_char": {"type": "string"},
                "algo": {"type": "string", "enum": ["zigzag","zz"]},
                "threshold_pct": {"type": "number"},
                "value_col": {"type": "string"},
                "paa": {"type": "integer"},
                "znorm": {"type": "boolean"},
            },
            "additionalProperties": False,
            "description": "Simplify/segment/encode spec for outputs.",
        },
    }


def _ensure_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
    if "$defs" not in schema or not isinstance(schema.get("$defs"), dict):
        schema["$defs"] = {}
    # Merge shared defs without overwriting existing keys
    defs = schema["$defs"]
    for k, v in shared_defs().items():
        defs.setdefault(k, v)
    return schema





def apply_param_hints(schema: Dict[str, Any]) -> Dict[str, Any]:
    params_obj = _parameters_obj(schema)
    props = params_obj.get("properties", {}) if isinstance(params_obj, dict) else {}
    for name, prop in list(props.items()):
        if not isinstance(prop, dict):
            continue
        hint = PARAM_HINTS.get(name)
        if hint and not prop.get("description"):
            prop["description"] = hint
    return schema

def _parameters_obj(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Get or create the OpenAI/MCP-style parameters object inside a schema."""
    if not isinstance(schema.get("parameters"), dict):
        schema["parameters"] = {"type": "object", "properties": {}}
    params = schema["parameters"]
    if not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    return params


def apply_timeframe_ref(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Replace simple timeframe property shapes with a $ref to TimeframeSpec.

    Looks for common parameter names and applies a $ref if present.
    """
    _ensure_defs(schema)
    params = _parameters_obj(schema)
    props = params["properties"]
    for key in ("timeframe", "target_timeframe", "source_timeframe"):
        if key in props and isinstance(props.get(key), dict):
            props[key] = {"$ref": "#/$defs/TimeframeSpec"}
    return schema



def _allow_null(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of schema that also accepts null."""
    updated = dict(schema)
    schema_type = updated.get("type")
    if schema_type is None:
        # Avoid explicit {"type": "null"} in anyOf/oneOf constructs as it triggers
        # "Cannot apply filter 'string' to type: NullValue" in some MCP clients (Jinja).
        # For optional parameters, relying on "required": False is sufficient.
        return updated

    if isinstance(schema_type, list):
        if "null" not in schema_type:
            updated["type"] = schema_type + ["null"]
    else:
        if schema_type != "null":
            updated["type"] = [schema_type, "null"]
    return updated


_TYPED_DICT_REFS = {
    "IndicatorSpec": "#/$defs/IndicatorSpec",
    "DenoiseSpec": "#/$defs/DenoiseSpec",
    "SimplifySpec": "#/$defs/SimplifySpec",
}

_ANNOTATION_VALUE_FORMAT = getattr(getattr(annotationlib, "Format", None), "VALUE", None)


def _get_runtime_signature(obj: Any) -> inspect.Signature:
    """Resolve a signature with evaluated annotations when available."""
    if _ANNOTATION_VALUE_FORMAT is not None:
        try:
            return inspect.signature(obj, eval_str=True, annotation_format=_ANNOTATION_VALUE_FORMAT)
        except Exception:
            pass
    return inspect.signature(obj)


def _get_runtime_annotations(obj: Any) -> Dict[str, Any]:
    """Resolve runtime annotations using the 3.14 annotation API when available."""
    if annotationlib is not None and _ANNOTATION_VALUE_FORMAT is not None:
        try:
            resolved = annotationlib.get_annotations(obj, eval_str=True, format=_ANNOTATION_VALUE_FORMAT)
            if isinstance(resolved, dict):
                return resolved
        except Exception:
            pass
    try:
        resolved = get_type_hints(obj)
        if isinstance(resolved, dict):
            return resolved
    except Exception:
        pass
    raw = getattr(obj, "__annotations__", None)
    return raw if isinstance(raw, dict) else {}


def _is_typed_dict_type(type_hint: Any) -> bool:
    try:
        if is_typeddict(type_hint):
            return True
    except Exception:
        pass
    annotations = getattr(type_hint, "__annotations__", None)
    return isinstance(annotations, dict) and (
        getattr(type_hint, "__required_keys__", None) is not None
        or getattr(type_hint, "__optional_keys__", None) is not None
    )


def _type_hint_to_schema(type_hint: Any) -> Dict[str, Any]:  # noqa: C901
    """Convert a Python type hint to a minimal JSON Schema fragment."""
    if type_hint is None:
        return {"type": "string"}
    if type_hint is Any:  # allow arbitrary content
        return {}
    origin = get_origin(type_hint)
    if origin is Literal:
        literals = [lit for lit in get_args(type_hint) if lit is not None]
        if not literals:
            return {"type": "string"}
        literal_types = {type(lit) for lit in literals}
        if literal_types == {bool}:
            return {"type": "boolean"}
        if literal_types == {int}:
            return {"type": "integer", "enum": literals}
        if literal_types == {float}:
            return {"type": "number", "enum": literals}
        return {"type": "string", "enum": [str(lit) for lit in literals]}
    if origin in (Union, types.UnionType):
        args = list(get_args(type_hint))
        allow_null = False
        non_null_args = []
        for arg in args:
            if arg is type(None):
                allow_null = True
            else:
                non_null_args.append(arg)
        if not non_null_args:
            return {"type": "null"}
        if len(non_null_args) == 1:
            schema = _type_hint_to_schema(non_null_args[0])
        else:
            schema = {"oneOf": [_type_hint_to_schema(arg) for arg in non_null_args]}
        if allow_null:
            schema = _allow_null(schema)
        return schema
    if origin in (list, List, tuple, Tuple, set, frozenset):
        args = get_args(type_hint)
        item_type = args[0] if args else Any
        item_schema = _type_hint_to_schema(item_type)
        # Ensure items schema defaults to accepting any value if empty
        if not item_schema:
            item_schema = {}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Dict):
        args = get_args(type_hint)
        value_type = args[1] if len(args) > 1 else Any
        value_schema = _type_hint_to_schema(value_type)
        if not value_schema:
            value_schema = {}
        return {"type": "object", "additionalProperties": value_schema or True}
    # Handle direct builtins and aliases
    if type_hint in (str, bytes):
        return {"type": "string"}
    if type_hint is int:
        return {"type": "integer"}
    if type_hint is float:
        return {"type": "number"}
    if type_hint is bool:
        return {"type": "boolean"}
    if type_hint is dict:
        return {"type": "object", "additionalProperties": True}
    if type_hint is list or type_hint is tuple:
        return {"type": "array"}
    ref_name = getattr(type_hint, "__name__", "")
    if ref_name in _TYPED_DICT_REFS:
        return {"$ref": _TYPED_DICT_REFS[ref_name]}
    if _is_typed_dict_type(type_hint) and ref_name:
        ref = _TYPED_DICT_REFS.get(ref_name)
        if ref:
            return {"$ref": ref}
    return {"type": "string"}

def build_minimal_schema(func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal parameters schema from a discovered function description.

    - Only includes parameter names and required flags.
    - Applies TimeframeSpec $ref to known timeframe param names.
    """
    schema: Dict[str, Any] = {"parameters": {"type": "object", "properties": {}, "required": []}}
    props = schema["parameters"]["properties"]
    req = schema["parameters"]["required"]
    for p in func_info.get("params", []):
        name = p.get("name")
        if not name:
            continue
        prop_schema = _type_hint_to_schema(p.get("type"))
        if not prop_schema:
            prop_schema = {"type": "string"}
        props[name] = prop_schema
        default_val = p.get("default")
        if default_val is not None:
            if isinstance(default_val, (str, int, float, bool, list, dict)):
                try:
                    props[name]["default"] = default_val
                except Exception:
                    pass
        if p.get("required"):
            req.append(name)
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    apply_param_hints(schema)
    return schema


def enrich_schema_with_shared_defs(schema: Dict[str, Any], func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure schema has $defs and timeframe refs. If empty, build minimal one."""
    if not isinstance(schema, dict) or not schema:
        schema = build_minimal_schema(func_info)
        return schema
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    apply_param_hints(schema)
    return schema




def get_shared_enum_lists() -> Dict[str, List[str]]:
    """Return enum lists used to enrich schemas when attaching to tools."""
    enums: Dict[str, List[str]] = {
        "DENOISE_METHODS": list(_DENOISE_METHODS),
        "SIMPLIFY_MODES": list(_SIMPLIFY_MODES),
        "SIMPLIFY_METHODS": list(_SIMPLIFY_METHODS),
        "PIVOT_METHODS": list(_PIVOT_METHODS),
        "FORECAST_METHODS": list(_FORECAST_METHODS),
    }
    if _CATEGORY_CHOICES:
        enums["CATEGORY_CHOICES"] = list(_CATEGORY_CHOICES)
    if _INDICATOR_NAME_CHOICES:
        enums["INDICATOR_NAME_CHOICES"] = list(_INDICATOR_NAME_CHOICES)
    return enums


def get_function_info(func: Any) -> Dict[str, Any]:
    """Extract minimal parameter info from a function for schema building."""
    # Introspect original function if wrapped
    try:
        target = inspect.unwrap(func)
    except Exception:
        target = func
    sig = _get_runtime_signature(target)
    type_hints = _get_runtime_annotations(target)

    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = type_hints.get(name, param.annotation)
        if annotation is inspect._empty:
            annotation = None
        params.append({
            "name": name,
            "required": param.default == inspect._empty,  # type: ignore[attr-defined]
            "default": None if param.default == inspect._empty else param.default,  # type: ignore[attr-defined]
            "type": annotation,
        })

    return {
        "name": getattr(target, "__name__", ""),
        "doc": inspect.getdoc(target) or "",
        "params": params,
    }
