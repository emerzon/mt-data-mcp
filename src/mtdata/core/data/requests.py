from __future__ import annotations

import json
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)

from ...shared.schema import (
    CompactFullDetailLiteral,
    CompactStandardFullDetailLiteral,
    DenoiseSpec,
    IndicatorSpec,
    SimplifySpec,
    TimeframeLiteral,
)
from ...utils.coercion import coerce_finite_float
from ..output_contract import normalize_output_detail

_INDICATOR_FORMAT_HELP = (
    "Use bare names like 'rsi', underscore forms like 'rsi_14', "
    "compact specs like 'sma(20)' and 'macd(12,26,9)', or named specs like "
    "'rsi(length=14)' and 'macd(fast=12,slow=26,signal=9)'."
)
def _reject_removed_field(values: Any, *, field_name: str, replacement: str) -> Any:
    if isinstance(values, dict) and field_name in values:
        raise ValueError(f"{field_name} was removed; use {replacement}")
    return values


def _split_indicator_tokens(spec: str) -> List[str]:
    text = str(spec or "").strip()
    if not text:
        return []
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    in_quote: Optional[str] = None
    for ch in text:
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
    return parts


def _indicator_numeric_value_error(raw_text: str, source_spec: str) -> ValueError:
    return ValueError(
        f"Indicator params must be numeric. Invalid value {raw_text!r} in {source_spec!r}."
    )


def _parse_indicator_numeric_value(value: Any, *, raw_text: str, source_spec: str) -> float:
    parsed = value
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            raise _indicator_numeric_value_error(raw_text, source_spec)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed_float = coerce_finite_float(text)
            if parsed_float is None:
                try:
                    raise ValueError(text)
                except ValueError as exc:
                    raise _indicator_numeric_value_error(raw_text, source_spec) from exc
            parsed = parsed_float
    parsed_float = coerce_finite_float(parsed)
    if isinstance(parsed, bool) or parsed_float is None:
        raise _indicator_numeric_value_error(raw_text, source_spec)
    return parsed_float


def _normalize_indicator_param_mapping(params: Dict[Any, Any], *, source_spec: str) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for raw_key, raw_value in params.items():
        key = str(raw_key or "").strip()
        if not key:
            raise ValueError("Indicator param names must be non-empty strings.")
        normalized[key] = _parse_indicator_numeric_value(
            raw_value,
            raw_text=f"{key}={raw_value}",
            source_spec=source_spec,
        )
    return normalized


def _normalize_indicator_entry(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        normalized = dict(value)
        if "params" not in normalized and "kwargs" in normalized:
            normalized["params"] = normalized.pop("kwargs")
        params = normalized.get("params")
        source_spec = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if isinstance(params, dict):
            normalized["params"] = _normalize_indicator_param_mapping(params, source_spec=source_spec)
        elif isinstance(params, (list, tuple)):
            normalized["params"] = [
                _parse_indicator_numeric_value(
                    item,
                    raw_text=str(item),
                    source_spec=source_spec,
                )
                for item in params
            ]
        elif params is not None:
            raise ValueError(
                "'params' must be a list of numeric values like [14] or a named numeric map like {\"length\": 14}."
            )
        return normalized
    if value is None:
        raise ValueError("Indicator entries cannot be null.")
    if not isinstance(value, str):
        raise ValueError("Indicators must be strings or objects.")

    stripped = value.strip()
    if not stripped:
        raise ValueError("Indicator entries cannot be empty.")

    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
        except Exception as exc:
            raise ValueError(f"Invalid indicator JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Indicator JSON entries must be objects with 'name' and optional 'params'.")
        return _normalize_indicator_entry(parsed)

    match = re.fullmatch(r"([A-Za-z0-9_]+)(?:\((.*)\))?", stripped)
    if not match:
        raise ValueError(f"Invalid indicator format. {_INDICATOR_FORMAT_HELP}")

    name = match.group(1)
    params_blob = match.group(2)
    if params_blob is None or not params_blob.strip():
        return {"name": name}

    positional: List[float] = []
    named: Dict[str, float] = {}
    for raw_part in _split_indicator_tokens(params_blob):
        part = raw_part.strip()
        if not part:
            continue
        if "=" in part:
            key, raw_value = part.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid named indicator param in {stripped!r}.")
            named[key] = _parse_indicator_numeric_value(
                raw_value,
                raw_text=part,
                source_spec=stripped,
            )
            continue
        positional.append(
            _parse_indicator_numeric_value(
                part,
                raw_text=part,
                source_spec=stripped,
            )
        )

    if named and positional:
        raise ValueError(
            f"Indicator params cannot mix positional and named values in {stripped!r}. "
            "Use either macd(12,26,9) or macd(fast=12,slow=26,signal=9)."
        )

    return {"name": name, "params": named or positional}


def _normalize_indicator_specs(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [_normalize_indicator_entry(item) for item in parsed]
        return [_normalize_indicator_entry(token) for token in _split_indicator_tokens(stripped)]
    if isinstance(value, list):
        return [_normalize_indicator_entry(item) for item in value]
    return value


def _validate_positive_limit(value: int) -> int:
    if int(value) <= 0:
        raise ValueError("limit must be greater than 0.")
    return int(value)


def _validate_non_negative(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    value_f = coerce_finite_float(value)
    if value_f is None:
        raise ValueError(f"{name} must be finite.")
    if value_f < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return value_f


def _validate_positive_float(value: float, name: str) -> float:
    value_f = coerce_finite_float(value)
    if value_f is None:
        raise ValueError(f"{name} must be finite.")
    if value_f <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value_f


def _validate_optional_ticket(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    value_i = int(value)
    if value_i <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value_i


def _validate_indicator_entries(value: Any) -> Any:
    if value is None or not isinstance(value, list):
        return value

    validated: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            validated.append(item)
            continue
        name = str(item.get("name") or "").strip()
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", name):
            raise ValueError("Indicator params must use parentheses, e.g. sma(20), not sma,20.")
        normalized = dict(item)
        if name:
            normalized["name"] = name
        validated.append(normalized)
    return validated


IndicatorSpecsInput = Annotated[
    Optional[List[IndicatorSpec]],
    BeforeValidator(
        _normalize_indicator_specs,
        json_schema_input_type=Optional[Union[str, List[IndicatorSpec]]],
    ),
    AfterValidator(_validate_indicator_entries),
]


def _normalize_simplify_input(value: Any) -> Any:
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, bool):
        return {} if value else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null", "off"}:
            return None
        if normalized in {"on", "auto"}:
            return {}
        raise ValueError(
            "simplify must be a dict such as {'method': 'lttb', 'points': 100}, "
            "a boolean, or use on/auto to enable defaults and off to disable."
        )
    return value


SimplifySpecInput = Annotated[
    Optional[SimplifySpec],
    BeforeValidator(
        _normalize_simplify_input,
        json_schema_input_type=Optional[Union[bool, str, SimplifySpec]],
    ),
]


class DataFetchCandlesRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    detail: CompactFullDetailLiteral = "compact"
    limit: int = 50
    start: Optional[str] = None
    end: Optional[str] = None
    ohlcv: Optional[str] = Field(
        None,
        description=(
            "Candle fields to include. Use all, ohlcv, ohlc, close/price, compact "
            "letters from o/h/l/c/v, or comma-separated names such as "
            "open,high,low,close,volume."
        ),
        examples=["ohlcv", "close", "open,high,low,close,volume"],
    )
    indicators: IndicatorSpecsInput = None
    denoise: Optional[DenoiseSpec] = None
    simplify: SimplifySpecInput = None
    include_spread: bool = Field(
        False,
        description=(
            "Append MT5 historical candle spread values. Defaults false because many "
            "symbols/timeframes return missing or zero historical spread and the extra "
            "column increases every row."
        ),
    )
    include_incomplete: bool = False
    allow_stale: bool = False

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Symbol is required and cannot be empty")
        return str(value).strip().upper()

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        return _validate_positive_limit(value)


class DataFetchTicksRequest(BaseModel):
    symbol: str
    limit: int = 100
    start: Optional[str] = None
    end: Optional[str] = None
    simplify: SimplifySpecInput = None
    detail: CompactStandardFullDetailLiteral = "compact"

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("Symbol is required and cannot be empty")
        return str(value).strip().upper()

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_output(cls, values: Any) -> Any:
        values = _reject_removed_field(values, field_name="output", replacement="json")
        return _reject_removed_field(values, field_name="output_mode", replacement="extras")

    @field_validator("detail", mode="before")
    @classmethod
    def _normalize_detail(cls, value: Any) -> Any:
        return normalize_output_detail(
            value,
            default="compact",
        )

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        return _validate_positive_limit(value)


class WaitCandleRequest(BaseModel):
    timeframe: TimeframeLiteral = "H1"
    buffer_seconds: float = 1.0
    max_wait_seconds: Optional[float] = 3600.0

    @field_validator("buffer_seconds")
    @classmethod
    def _validate_buffer_seconds(cls, value: float) -> float:
        validated = _validate_non_negative(value, "buffer_seconds")
        if validated is None:
            raise ValueError("buffer_seconds must be greater than or equal to 0.")
        return validated

    @field_validator("max_wait_seconds")
    @classmethod
    def _validate_max_wait_seconds(cls, value: Optional[float]) -> Optional[float]:
        return _validate_non_negative(value, "max_wait_seconds")


class WaitEventWindow(BaseModel):
    kind: Literal["minutes", "ticks"] = "minutes"
    value: float = 5.0

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: float) -> float:
        return _validate_positive_float(value, "window.value")


class _WaitAccountEventBase(BaseModel):
    symbol: Optional[str] = None
    order_ticket: Optional[int] = None
    position_ticket: Optional[int] = None
    magic: Optional[int] = None
    side: Optional[Literal["buy", "sell"]] = None

    @field_validator("order_ticket")
    @classmethod
    def _validate_order_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "order_ticket")

    @field_validator("position_ticket")
    @classmethod
    def _validate_position_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "position_ticket")


class CandleCloseEventSpec(BaseModel):
    type: Literal["candle_close"] = "candle_close"
    timeframe: Optional[TimeframeLiteral] = None
    buffer_seconds: Optional[float] = None

    @field_validator("buffer_seconds")
    @classmethod
    def _validate_buffer_seconds(cls, value: Optional[float]) -> Optional[float]:
        return _validate_non_negative(value, "buffer_seconds")


class OrderCreatedEventSpec(_WaitAccountEventBase):
    type: Literal["order_created"] = "order_created"


class OrderFilledEventSpec(_WaitAccountEventBase):
    type: Literal["order_filled"] = "order_filled"


class OrderCancelledEventSpec(_WaitAccountEventBase):
    type: Literal["order_cancelled"] = "order_cancelled"


class PositionOpenedEventSpec(_WaitAccountEventBase):
    type: Literal["position_opened"] = "position_opened"


class PositionClosedEventSpec(_WaitAccountEventBase):
    type: Literal["position_closed"] = "position_closed"


class TpHitEventSpec(_WaitAccountEventBase):
    type: Literal["tp_hit"] = "tp_hit"


class SlHitEventSpec(_WaitAccountEventBase):
    type: Literal["sl_hit"] = "sl_hit"


class PriceChangeEventSpec(BaseModel):
    type: Literal["price_change"] = "price_change"
    symbol: Optional[str] = None
    window: WaitEventWindow = Field(default_factory=WaitEventWindow)
    baseline_window: WaitEventWindow = Field(
        default_factory=lambda: WaitEventWindow(kind="minutes", value=60.0)
    )
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"
    threshold_mode: Literal["fixed_pct", "ratio_to_baseline", "zscore"] = "ratio_to_baseline"
    threshold_value: float = 2.0

    @field_validator("threshold_value")
    @classmethod
    def _validate_threshold_value(cls, value: float) -> float:
        return _validate_positive_float(value, "threshold_value")


class VolumeSpikeEventSpec(BaseModel):
    type: Literal["volume_spike"] = "volume_spike"
    symbol: Optional[str] = None
    window: WaitEventWindow = Field(default_factory=WaitEventWindow)
    baseline_window: WaitEventWindow = Field(
        default_factory=lambda: WaitEventWindow(kind="minutes", value=60.0)
    )
    source: Literal["auto", "tick_count", "volume", "volume_real"] = "auto"
    threshold_mode: Literal["ratio_to_baseline", "zscore"] = "ratio_to_baseline"
    threshold_value: float = 2.0

    @field_validator("threshold_value")
    @classmethod
    def _validate_threshold_value(cls, value: float) -> float:
        return _validate_positive_float(value, "threshold_value")


class TickCountSpikeEventSpec(BaseModel):
    type: Literal["tick_count_spike"] = "tick_count_spike"
    symbol: Optional[str] = None
    window: WaitEventWindow = Field(default_factory=WaitEventWindow)
    baseline_window: WaitEventWindow = Field(
        default_factory=lambda: WaitEventWindow(kind="minutes", value=60.0)
    )
    threshold_mode: Literal["ratio_to_baseline", "zscore"] = "ratio_to_baseline"
    threshold_value: float = 2.0

    @field_validator("threshold_value")
    @classmethod
    def _validate_threshold_value(cls, value: float) -> float:
        return _validate_positive_float(value, "threshold_value")


class SpreadSpikeEventSpec(BaseModel):
    type: Literal["spread_spike"] = "spread_spike"
    symbol: Optional[str] = None
    window: WaitEventWindow = Field(default_factory=WaitEventWindow)
    baseline_window: WaitEventWindow = Field(
        default_factory=lambda: WaitEventWindow(kind="minutes", value=60.0)
    )
    threshold_mode: Literal["ratio_to_baseline", "zscore"] = "ratio_to_baseline"
    threshold_value: float = 2.0

    @field_validator("threshold_value")
    @classmethod
    def _validate_threshold_value(cls, value: float) -> float:
        return _validate_positive_float(value, "threshold_value")


class TickCountDroughtEventSpec(BaseModel):
    type: Literal["tick_count_drought"] = "tick_count_drought"
    symbol: Optional[str] = None
    window: WaitEventWindow = Field(default_factory=WaitEventWindow)
    baseline_window: WaitEventWindow = Field(
        default_factory=lambda: WaitEventWindow(kind="minutes", value=60.0)
    )
    threshold_mode: Literal["ratio_to_baseline", "zscore"] = "ratio_to_baseline"
    threshold_value: float = 0.5

    @field_validator("threshold_value")
    @classmethod
    def _validate_threshold_value(cls, value: float) -> float:
        return _validate_positive_float(value, "threshold_value")


class RangeExpansionEventSpec(BaseModel):
    type: Literal["range_expansion"] = "range_expansion"
    symbol: Optional[str] = None
    window: WaitEventWindow = Field(default_factory=WaitEventWindow)
    baseline_window: WaitEventWindow = Field(
        default_factory=lambda: WaitEventWindow(kind="minutes", value=60.0)
    )
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    threshold_mode: Literal["ratio_to_baseline", "zscore"] = "ratio_to_baseline"
    threshold_value: float = 2.0

    @field_validator("threshold_value")
    @classmethod
    def _validate_threshold_value(cls, value: float) -> float:
        return _validate_positive_float(value, "threshold_value")


class PriceTouchLevelEventSpec(BaseModel):
    type: Literal["price_touch_level"] = "price_touch_level"
    symbol: Optional[str] = None
    level: float
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"
    tolerance: float = 0.0

    @field_validator("tolerance")
    @classmethod
    def _validate_tolerance(cls, value: float) -> float:
        validated = _validate_non_negative(value, "tolerance")
        return 0.0 if validated is None else float(validated)


class PriceBreakLevelEventSpec(BaseModel):
    type: Literal["price_break_level"] = "price_break_level"
    symbol: Optional[str] = None
    level: float
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"
    tolerance: float = 0.0
    confirm_ticks: int = 1

    @field_validator("tolerance")
    @classmethod
    def _validate_tolerance(cls, value: float) -> float:
        validated = _validate_non_negative(value, "tolerance")
        return 0.0 if validated is None else float(validated)

    @field_validator("confirm_ticks")
    @classmethod
    def _validate_confirm_ticks(cls, value: int) -> int:
        if int(value) < 1:
            raise ValueError("confirm_ticks must be greater than or equal to 1.")
        return int(value)


class PriceEnterZoneEventSpec(BaseModel):
    type: Literal["price_enter_zone"] = "price_enter_zone"
    symbol: Optional[str] = None
    lower: float
    upper: float
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"
    direction: Literal["up", "down", "either"] = "either"

    @model_validator(mode="after")
    def _validate_bounds(self) -> "PriceEnterZoneEventSpec":
        if float(self.upper) <= float(self.lower):
            raise ValueError("upper must be greater than lower.")
        return self


class PendingNearFillEventSpec(_WaitAccountEventBase):
    type: Literal["pending_near_fill"] = "pending_near_fill"
    distance: float = 0.0
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"

    @field_validator("distance")
    @classmethod
    def _validate_distance(cls, value: float) -> float:
        validated = _validate_non_negative(value, "distance")
        return 0.0 if validated is None else float(validated)


class StopThreatEventSpec(_WaitAccountEventBase):
    type: Literal["stop_threat"] = "stop_threat"
    distance: float = 0.0
    price_source: Literal["auto", "bid", "ask", "mid", "last"] = "auto"

    @field_validator("distance")
    @classmethod
    def _validate_distance(cls, value: float) -> float:
        validated = _validate_non_negative(value, "distance")
        return 0.0 if validated is None else float(validated)


WaitWatchEventSpec = Annotated[
    OrderCreatedEventSpec
    | OrderFilledEventSpec
    | OrderCancelledEventSpec
    | PositionOpenedEventSpec
    | PositionClosedEventSpec
    | TpHitEventSpec
    | SlHitEventSpec
    | PriceChangeEventSpec
    | VolumeSpikeEventSpec
    | TickCountSpikeEventSpec
    | SpreadSpikeEventSpec
    | TickCountDroughtEventSpec
    | RangeExpansionEventSpec
    | PriceTouchLevelEventSpec
    | PriceBreakLevelEventSpec
    | PriceEnterZoneEventSpec
    | PendingNearFillEventSpec
    | StopThreatEventSpec,
    Field(discriminator="type"),
]

WaitBoundaryEventSpec = CandleCloseEventSpec


class WaitEventRequest(BaseModel):
    model_config = {"extra": "forbid"}

    watch_for: Optional[List[WaitWatchEventSpec]] = None
    end_on: List[WaitBoundaryEventSpec] = Field(default_factory=list)
    symbol: Optional[str] = None
    timeframe: Optional[TimeframeLiteral] = None
    order_ticket: Optional[int] = None
    position_ticket: Optional[int] = None
    magic: Optional[int] = None
    side: Optional[Literal["buy", "sell"]] = None
    buffer_seconds: float = 1.0
    poll_interval_seconds: float = 0.5
    max_wait_seconds: Optional[float] = 86400.0
    accept_preexisting: bool = False

    @field_validator("order_ticket")
    @classmethod
    def _validate_order_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "order_ticket")

    @field_validator("position_ticket")
    @classmethod
    def _validate_position_ticket(cls, value: Optional[int]) -> Optional[int]:
        return _validate_optional_ticket(value, "position_ticket")

    @field_validator("buffer_seconds")
    @classmethod
    def _validate_buffer_seconds(cls, value: float) -> float:
        validated = _validate_non_negative(value, "buffer_seconds")
        if validated is None:
            raise ValueError("buffer_seconds must be greater than or equal to 0.")
        return validated

    @field_validator("poll_interval_seconds")
    @classmethod
    def _validate_poll_interval_seconds(cls, value: float) -> float:
        return _validate_positive_float(value, "poll_interval_seconds")

    @field_validator("max_wait_seconds")
    @classmethod
    def _validate_max_wait_seconds(cls, value: Optional[float]) -> Optional[float]:
        return _validate_non_negative(value, "max_wait_seconds")

    @model_validator(mode="after")
    def _validate_explicit_empty_watchers(self) -> "WaitEventRequest":
        if self.watch_for == [] and not self.end_on and self.timeframe is None:
            raise ValueError(
                "watch_for cannot be an explicit empty list unless end_on or timeframe is provided."
            )
        return self
