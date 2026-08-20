"""Stable metadata for inspecting and replaying trained-model identity."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from .requests import MAX_FORECAST_HORIZON

_FINGERPRINT_VERSION = 1
_WINDOW_FIELDS = ("lookback", "as_of", "start", "end")


def _json_safe(value: Any) -> Any:
    """Return the typed JSON representation persisted by the model store."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def build_model_reuse_metadata(
    fingerprint: Mapping[str, Any],
    data_scope: str,
    training_window: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build inspectable identity metadata and a replayable generate request."""
    canonical = _json_safe(dict(fingerprint))
    params = dict(canonical.get("params") or {})
    training_context = params.pop("_training_context", None)
    quantity = str(params.pop("quantity", "price"))
    params["seasonality"] = int(canonical.get("seasonality") or 1)

    timeframe = str(canonical.get("timeframe") or "")
    scope = str(data_scope)
    suffix = f"_{timeframe}" if timeframe else ""
    symbol = scope[: -len(suffix)] if suffix and scope.endswith(suffix) else scope
    request: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "method": str(canonical.get("method") or ""),
        "horizon": int(canonical.get("horizon") or 0),
        "quantity": quantity,
        "params": params,
    }
    if isinstance(training_context, Mapping):
        for request_field, context_field in (
            ("denoise", "denoise"),
            ("features", "features"),
            ("target_spec", "target_spec"),
            ("dimred", "dimred"),
        ):
            value = training_context.get(context_field)
            if value not in (None, {}, []):
                request[request_field] = _json_safe(value)
    for field in _WINDOW_FIELDS:
        value = (training_window or {}).get(field)
        if value is not None:
            request[field] = _json_safe(value)

    return {
        "compatibility_fingerprint_version": _FINGERPRINT_VERSION,
        "compatibility_fingerprint": canonical,
        "reuse_request": request,
    }


def fingerprint_mismatches(
    stored: Optional[Mapping[str, Any]],
    requested: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Describe differing model-identity dimensions for machine-readable errors."""
    stored_values = _json_safe(dict(stored or {}))
    requested_values = _json_safe(dict(requested))
    mismatches: Dict[str, Dict[str, Any]] = {}
    for field in sorted(set(stored_values) | set(requested_values)):
        if stored_values.get(field) != requested_values.get(field):
            mismatches[field] = {
                "stored": stored_values.get(field),
                "requested": requested_values.get(field),
            }
    return mismatches


def describe_request_compatibility(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify whether listing metadata can construct a valid reuse request."""
    fingerprint = metadata.get("compatibility_fingerprint")
    request = metadata.get("reuse_request")
    if not isinstance(fingerprint, Mapping) or not isinstance(request, Mapping):
        return {
            "status": "unknown",
            "reason": "compatibility_metadata_missing",
        }
    try:
        horizon = int(request.get("horizon"))
    except (TypeError, ValueError):
        return {"status": "unusable", "reason": "invalid_horizon"}
    if not 1 <= horizon <= MAX_FORECAST_HORIZON:
        return {
            "status": "unusable",
            "reason": "horizon_out_of_supported_range",
            "supported_horizon": {"minimum": 1, "maximum": MAX_FORECAST_HORIZON},
        }
    return {"status": "ready"}
