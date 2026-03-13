from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator

from .schema import DenoiseSpec, IndicatorSpec, SimplifySpec, TimeframeLiteral


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


def _normalize_indicator_entry(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        params = value.get("params")
        if isinstance(params, dict):
            raise ValueError("'params' must be a list of numbers, e.g., [14], not a dict.")
        return dict(value)
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
        raise ValueError(
            "Invalid indicator format. Use bare names like 'rsi' or compact specs like 'sma(20)' and 'macd(12,26,9)'."
        )

    name = match.group(1)
    params_blob = match.group(2)
    if params_blob is None or not params_blob.strip():
        return {"name": name}

    params_out: List[float] = []
    for raw_part in params_blob.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            parsed = json.loads(part)
        except Exception:
            try:
                parsed = float(part)
            except Exception as exc:
                raise ValueError(
                    f"Indicator params must be numeric. Invalid value {part!r} in {stripped!r}."
                ) from exc
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            raise ValueError(
                f"Indicator params must be numeric. Invalid value {part!r} in {stripped!r}."
            )
        params_out.append(float(parsed))

    return {"name": name, "params": params_out}


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


class DataFetchCandlesRequest(BaseModel):
    symbol: str
    timeframe: TimeframeLiteral = "H1"
    limit: int = 200
    start: Optional[str] = None
    end: Optional[str] = None
    ohlcv: Optional[str] = None
    indicators: Optional[List[IndicatorSpec]] = None
    denoise: Optional[DenoiseSpec] = None
    simplify: Optional[SimplifySpec] = None

    @field_validator("indicators", mode="before")
    @classmethod
    def _coerce_indicators(cls, value: Any) -> Any:
        return _normalize_indicator_specs(value)

    @field_validator("indicators")
    @classmethod
    def _validate_indicators(cls, value: Any) -> Any:
        return _validate_indicator_entries(value)

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        return _validate_positive_limit(value)


class DataFetchTicksRequest(BaseModel):
    symbol: str
    limit: int = 200
    start: Optional[str] = None
    end: Optional[str] = None
    simplify: Optional[SimplifySpec] = None
    output: Literal["summary", "stats", "rows"] = "summary"

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
        value_f = float(value)
        if value_f < 0:
            raise ValueError("buffer_seconds must be greater than or equal to 0.")
        return value_f

    @field_validator("max_wait_seconds")
    @classmethod
    def _validate_max_wait_seconds(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        value_f = float(value)
        if value_f < 0:
            raise ValueError("max_wait_seconds must be greater than or equal to 0.")
        return value_f
