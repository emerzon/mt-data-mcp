"""Options chain, expiration, and pricing tools."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import Field

from ..shared.schema import DetailLiteral
from ._mcp_instance import mcp
from .execution_logging import run_logged_operation
from .output_contract import normalize_output_verbosity_detail

logger = logging.getLogger(__name__)

_OPTIONS_CHAIN_COMPACT_FIELDS = (
    "side",
    "contract",
    "strike",
    "last",
    "bid",
    "ask",
    "volume",
    "open_interest",
)
_OPTIONS_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.^=_/-]{0,63}$")
_OPTIONS_EXPIRATION_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_options_symbol(
    symbol: Any,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    normalized = str(symbol or "").strip().upper()
    if not normalized:
        return None, {
            "success": False,
            "error": "symbol is required",
            "error_code": "invalid_symbol",
        }
    if _OPTIONS_SYMBOL_PATTERN.fullmatch(normalized) is None:
        return None, {
            "success": False,
            "error": (
                f"Invalid symbol: {symbol}. Use 1-64 letters, digits, or common "
                "market-symbol characters: . ^ = _ / -."
            ),
            "error_code": "invalid_symbol",
        }
    return normalized, None


def _normalize_option_expiration(
    expiration: Any,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if expiration in (None, ""):
        return None, None
    normalized = str(expiration).strip()
    try:
        if _OPTIONS_EXPIRATION_PATTERN.fullmatch(normalized) is None:
            raise ValueError
        date.fromisoformat(normalized)
    except ValueError:
        return None, {
            "success": False,
            "error": (
                f"Invalid expiration: {expiration!r}. Expected a calendar date "
                "in YYYY-MM-DD format, for example 2026-07-17."
            ),
            "error_code": "invalid_expiration",
            "parameter": "expiration",
            "value": expiration,
            "expected_format": "YYYY-MM-DD",
        }
    return normalized, None


def _validate_options_integer(
    parameter: str,
    value: Any,
    *,
    minimum: int,
) -> Optional[Dict[str, Any]]:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = minimum - 1
    if not isinstance(value, bool) and numeric >= minimum:
        return None
    return {
        "success": False,
        "error": f"{parameter} must be greater than or equal to {minimum}.",
        "error_code": "invalid_input",
        "parameter": parameter,
        "value": value,
        "minimum": minimum,
    }


def _validate_options_valuation_date(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        if _OPTIONS_EXPIRATION_PATTERN.fullmatch(text) is None:
            raise ValueError
        date.fromisoformat(text)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid valuation_date: {value}. Use YYYY-MM-DD.",
            "error_code": "invalid_valuation_date",
            "parameter": "valuation_date",
            "value": value,
            "expected_format": "YYYY-MM-DD",
        }
    return None


def _run_options_operation(
    operation: str,
    *,
    func,
    **fields: Any,
) -> Dict[str, Any]:
    return run_logged_operation(
        logger,
        operation=operation,
        func=func,
        **fields,
    )


def _options_detail_mode(detail: str) -> str:
    return normalize_output_verbosity_detail(detail, default="compact")


def _options_provider_readiness() -> Dict[str, Any]:
    from ..bootstrap.settings import options_data_config

    provider = str(getattr(options_data_config, "provider", "yahoo") or "yahoo").strip().lower()
    api_key_configured = bool(getattr(options_data_config, "api_key", None))
    if provider == "tradier" and not api_key_configured:
        effective_provider = "yahoo"
        configured_provider_ready = False
        configured_provider_status = "misconfigured_using_fallback"
        recommendation = (
            "Configured Tradier mode is missing MTDATA_OPTIONS_API_KEY. mtdata will "
            "retry anonymous Yahoo cookie/crumb access as a best-effort fallback, but reliable "
            "options chains still require Tradier credentials."
        )
    else:
        effective_provider = (
            "tradier" if provider == "auto" and api_key_configured
            else "yahoo" if provider == "auto"
            else provider
        )
        configured_provider_ready = effective_provider == "yahoo" or (
            effective_provider == "tradier" and api_key_configured
        )
        configured_provider_status = (
            "ready" if configured_provider_ready else "unsupported"
        )
        recommendation = (
            "Yahoo options data uses anonymous cookie/crumb access and may still return "
            "401/429. For reliable chains, set MTDATA_OPTIONS_PROVIDER=tradier and "
            "MTDATA_OPTIONS_API_KEY."
        ) if effective_provider == "yahoo" else None
    chain_request_supported = effective_provider in {"yahoo", "tradier"}
    usable_now = chain_request_supported
    chain_provider_ready = usable_now
    chain_data_ready = usable_now
    provider_mode = (
        "anonymous_fallback" if effective_provider == "yahoo" else "credentialed"
    )
    action_required = None if usable_now else "configure_options_provider"
    remediation = (
        None
        if usable_now
        else "Set MTDATA_OPTIONS_PROVIDER to yahoo or configure Tradier credentials."
    )
    warnings = []
    if provider_mode == "anonymous_fallback":
        warnings.append(
            "Options chain access is using anonymous Yahoo cookie/crumb fallback; "
            "it is best-effort and may return 401/429."
        )
    out = {
        "configured_provider": provider,
        "effective_provider": effective_provider,
        "api_key_configured": api_key_configured,
        "configured_provider_ready": configured_provider_ready,
        "configured_provider_status": configured_provider_status,
        "local_tools_ready": True,
        "chain_provider_ready": chain_provider_ready,
        "chain_data_ready": chain_data_ready,
        "chain_request_supported": chain_request_supported,
        "usable_now": usable_now,
        "live_chain_requests_expected_to_work": chain_request_supported,
        "live_chain_expectation_basis": (
            "authenticated_provider"
            if effective_provider == "tradier" and api_key_configured
            else "best_effort_anonymous_provider"
            if chain_request_supported
            else "unsupported_provider"
        ),
        "degraded": bool(provider_mode == "anonymous_fallback"),
        "provider_mode": provider_mode,
        "supported_providers": ["tradier", "yahoo"],
        "chain_dependent_tools": [
            "options_expirations",
            "options_chain",
            "options_heston_calibrate",
        ],
        "local_tools": ["options_barrier_price"],
        "action_required": action_required,
        "recommended_action": (
            "configure_tradier_credentials"
            if provider_mode == "anonymous_fallback"
            else None
        ),
        "recommendation": recommendation,
        "remediation": remediation,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def _options_chain_provider_gate(tool_name: str) -> Optional[Dict[str, Any]]:
    readiness = _options_provider_readiness()
    if readiness.get("chain_request_supported") is True:
        return None
    provider = readiness.get("effective_provider")
    error_code = (
        "options_provider_auth"
        if provider == "tradier"
        else "options_provider_unavailable"
    )
    return {
        "success": False,
        "error": (
            f"{tool_name} requires a configured options-chain provider. "
            "Run options_provider_status for setup details."
        ),
        "error_code": error_code,
        "provider": provider,
        "configured_provider": readiness.get("configured_provider"),
        "chain_data_ready": False,
        "action_required": readiness.get("action_required"),
        "next_tool": "options_provider_status",
        "env_vars": ["MTDATA_OPTIONS_PROVIDER", "MTDATA_OPTIONS_API_KEY"],
        "remediation": readiness.get("remediation"),
    }


def _compact_option_contract(row: Any) -> Any:
    if not isinstance(row, dict):
        return row
    return {
        key: row[key]
        for key in _OPTIONS_CHAIN_COMPACT_FIELDS
        if key in row and row[key] is not None
    }


def _barrier_pricing_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = payload.get("params_used")
    source = params if isinstance(params, dict) else payload
    inputs = {
        key: source[key]
        for key in (
            "risk_free_rate",
            "dividend_yield",
            "volatility",
            "rebate",
        )
        if source.get(key) is not None
    }
    if inputs:
        inputs["rate_unit"] = "decimal_fraction"
        if "volatility" in inputs:
            inputs["volatility_unit"] = "decimal_fraction"
    return inputs


def _apply_options_detail(
    payload: Dict[str, Any],
    *,
    detail: str,
    kind: str,
) -> Dict[str, Any]:
    if not isinstance(payload, dict) or not payload.get("success"):
        return payload
    detail_mode = _options_detail_mode(detail)
    out = dict(payload)
    out["detail"] = detail_mode
    if kind == "barrier_price":
        out.setdefault(
            "units",
            {
                "price": "premium_per_underlying_unit",
                "delta": "premium_change_per_underlying_price_unit",
                "gamma": (
                    "premium_change_per_squared_underlying_price_unit"
                ),
                "vega": "premium_change_per_1.0_decimal_volatility",
            },
        )
        pricing_inputs = _barrier_pricing_inputs(out)
        if pricing_inputs:
            out["pricing_inputs"] = pricing_inputs
    if detail_mode == "full":
        return out

    if kind == "expirations":
        return {
            key: out[key]
            for key in (
                "success",
                "provider",
                "configured_provider",
                "provider_effective",
                "cached",
                "data_age_seconds",
                "as_of",
                "freshness",
                "freshness_reason",
                "underlying_price_source",
                "underlying_price_session",
                "symbol",
                "expirations",
                "expiration_count",
                "warnings",
                "detail",
            )
            if key in out
        }
    if kind == "chain":
        compact = {
            key: out[key]
            for key in (
                "success",
                "provider",
                "configured_provider",
                "provider_effective",
                "cached",
                "data_age_seconds",
                "as_of",
                "freshness",
                "freshness_reason",
                "underlying_price_source",
                "underlying_price_session",
                "symbol",
                "expiration",
                "underlying_price",
                "currency",
                "option_type",
                "count",
                "calls_count",
                "puts_count",
                "limit",
                "limit_source",
                "warnings",
                "detail",
            )
            if key in out
        }
        options = out.get("options")
        if isinstance(options, list):
            compact["options"] = [_compact_option_contract(row) for row in options]
        return compact
    if kind == "barrier_price":
        return {
            key: out[key]
            for key in (
                "success",
                "option_type",
                "barrier_type",
                "spot",
                "spot_as_of",
                "spot_data_age_seconds",
                "spot_freshness",
                "spot_source",
                "spot_session",
                "strike",
                "barrier",
                "maturity_days",
                "valuation_date",
                "valuation_timezone",
                "valuation_date_source",
                "maturity_date",
                "time_to_maturity_years",
                "price",
                "delta",
                "greeks_status",
                "greeks_method",
                "greeks_warnings",
                "units",
                "pricing_assumptions",
                "pricing_inputs",
                "pricing_note",
                "detail",
            )
            if key in out
        }
    if kind == "heston_calibrate":
        compact = {
            key: out[key]
            for key in (
                "success",
                "symbol",
                "expiration",
                "valuation_date",
                "valuation_timezone",
                "valuation_date_source",
                "days_to_expiry",
                "contracts_used",
                "spot",
                "calibration_error_rmse",
                "params",
                "pricing_assumptions",
                "detail",
            )
            if key in out
        }
        return compact
    return out


@mcp.tool()
def options_provider_status(
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Report configured options-chain provider readiness without querying market data."""
    payload: Dict[str, Any] = {
        "success": True,
        **_options_provider_readiness(),
    }
    if _options_detail_mode(detail) == "full":
        from ..bootstrap.settings import options_data_config

        payload["tradier_docs"] = "https://documentation.tradier.com/"
        payload["base_url"] = getattr(options_data_config, "base_url", None)
    elif payload.get("action_required") and payload.get("remediation"):
        payload["remediation_hint"] = (
            "Reliable options-chain access requires Tradier credentials."
        )
        payload["next_steps"] = [
            "Set MTDATA_OPTIONS_PROVIDER=tradier.",
            "Set MTDATA_OPTIONS_API_KEY to a Tradier API token, then restart mtdata.",
            "Yahoo cookie/crumb fallback is best-effort and may still return 401/429.",
        ]
        payload.pop("remediation", None)
    elif payload.get("recommendation"):
        payload["recommendation_hint"] = (
            "Anonymous Yahoo is usable now but remains best-effort."
        )
    return _run_options_operation(
        "options_provider_status",
        detail=detail,
        func=lambda: payload,
    )


@mcp.tool()
def options_expirations(
    symbol: str,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Fetch option expirations using the configured options-chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance uses anonymous
    cookie/crumb negotiation and may still return 401 responses. When provider mode is `tradier` or
    `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured. For
    reliable options-chain data, configure Tradier with
    MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY. Tradier API
    tokens: https://documentation.tradier.com/.
    """
    from ..services.options_service import get_options_expirations as _impl

    symbol_value, symbol_error = _normalize_options_symbol(symbol)
    if symbol_error is not None or symbol_value is None:
        return _run_options_operation(
            "options_expirations",
            symbol=symbol,
            detail=detail,
            func=lambda: symbol_error or {"error": "symbol is required"},
        )
    gate = _options_chain_provider_gate("options_expirations")
    if gate is not None:
        return _run_options_operation(
            "options_expirations",
            symbol=symbol_value,
            detail=detail,
            func=lambda: gate,
        )

    return _run_options_operation(
        "options_expirations",
        symbol=symbol_value,
        detail=detail,
        func=lambda: _apply_options_detail(
            _impl(symbol=symbol_value),
            detail=detail,
            kind="expirations",
        ),
    )


@mcp.tool()
def options_chain(
    symbol: str,
    expiration: Optional[str] = None,
    option_type: Literal["call", "put", "both"] = "both",  # type: ignore
    min_open_interest: Annotated[int, Field(ge=0)] = 0,
    min_volume: Annotated[int, Field(ge=0)] = 0,
    limit: Annotated[Optional[int], Field(ge=1)] = None,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Fetch option-chain snapshots using the configured chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance uses anonymous
    cookie/crumb negotiation and may still return 401 responses. When provider mode is `tradier` or
    `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured. For
    reliable options-chain data, configure Tradier with
    MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY. Tradier API
    tokens: https://documentation.tradier.com/.

    Compact output defaults to the 20 contracts nearest the underlying price,
    balanced across calls and puts. Full detail defaults to 200 contracts.
    Pass ``limit`` explicitly to override either default.
    """
    from ..services.options_service import get_options_chain as _impl

    symbol_value, symbol_error = _normalize_options_symbol(symbol)
    if symbol_error is not None or symbol_value is None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol,
            detail=detail,
            func=lambda: symbol_error or {"error": "symbol is required"},
        )
    expiration_value, expiration_error = _normalize_option_expiration(expiration)
    if expiration_error is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration,
            detail=detail,
            func=lambda: expiration_error,
        )
    detail_mode = _options_detail_mode(detail)
    effective_limit = (
        int(limit)
        if limit is not None
        else 200
        if detail_mode == "full"
        else 20
    )
    input_error = next(
        (
            error
            for error in (
                _validate_options_integer(
                    "min_open_interest", min_open_interest, minimum=0
                ),
                _validate_options_integer("min_volume", min_volume, minimum=0),
                _validate_options_integer("limit", effective_limit, minimum=1),
            )
            if error is not None
        ),
        None,
    )
    if input_error is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration_value,
            detail=detail,
            func=lambda: input_error,
        )
    gate = _options_chain_provider_gate("options_chain")
    if gate is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration_value,
            option_type=option_type,
            limit=effective_limit,
            detail=detail,
            func=lambda: gate,
        )

    return _run_options_operation(
        "options_chain",
        symbol=symbol_value,
        expiration=expiration_value,
        option_type=option_type,
        limit=effective_limit,
        detail=detail,
        func=lambda: _apply_options_detail(
            {
                **_impl(
                    symbol=symbol_value,
                    expiration=expiration_value,
                    option_type=option_type,
                    min_open_interest=int(min_open_interest),
                    min_volume=int(min_volume),
                    limit=effective_limit,
                ),
                "limit": effective_limit,
                "limit_source": (
                    "request" if limit is not None else f"{detail_mode}_default"
                ),
            },
            detail=detail,
            kind="chain",
        ),
    )


@mcp.tool()
def options_barrier_price(
    spot: float,
    strike: float,
    barrier: float,
    maturity_days: int,
    option_type: Literal["call", "put"] = "call",  # type: ignore
    barrier_type: Literal["up_in", "up_out", "down_in", "down_out"] = "up_out",  # type: ignore
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    volatility: float = 0.2,
    rebate: float = 0.0,
    valuation_date: Optional[str] = None,
    calendar: str = "UnitedStates.NYSE",
    maturity_basis: Literal["calendar_days", "business_days"] = "calendar_days",  # type: ignore
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Price a barrier option using QuantLib with optional calendar overrides."""
    from ..forecast.quantlib_tools import price_barrier_option_quantlib as _impl

    def _run() -> Dict[str, Any]:
        payload = _impl(
            spot=float(spot),
            strike=float(strike),
            barrier=float(barrier),
            maturity_days=int(maturity_days),
            option_type=option_type,
            barrier_type=barrier_type,
            risk_free_rate=float(risk_free_rate),
            dividend_yield=float(dividend_yield),
            volatility=float(volatility),
            rebate=float(rebate),
            valuation_date=valuation_date,
            calendar=calendar,
            maturity_basis=maturity_basis,
        )
        if isinstance(payload, dict) and payload.get("success"):
            payload.update(
                {
                    "option_type": option_type,
                    "barrier_type": barrier_type,
                    "spot": float(spot),
                    "strike": float(strike),
                    "barrier": float(barrier),
                    "maturity_days": int(maturity_days),
                    "price_basis": (
                        "premium per underlying unit, in the same currency/units as "
                        "the supplied spot, strike and barrier (no symbol context)."
                    ),
                    "pricing_note": (
                        f"{barrier_type} {option_type}: spot={float(spot)}, "
                        f"strike={float(strike)}, barrier={float(barrier)}."
                    ),
                }
            )
        return _apply_options_detail(
            payload,
            detail=detail,
            kind="barrier_price",
        )

    return _run_options_operation(
        "options_barrier_price",
        option_type=option_type,
        barrier_type=barrier_type,
        maturity_days=maturity_days,
        valuation_date=valuation_date,
        calendar=calendar,
        maturity_basis=maturity_basis,
        detail=detail,
        func=_run,
    )


@mcp.tool()
def options_heston_calibrate(
    symbol: str,
    expiration: Optional[str] = None,
    valuation_date: Optional[str] = None,
    option_type: Literal["call", "put", "both"] = "call",  # type: ignore
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    min_open_interest: Annotated[int, Field(ge=0)] = 0,
    min_volume: Annotated[int, Field(ge=0)] = 0,
    max_contracts: Annotated[int, Field(ge=5)] = 25,
    calendar: str = "UnitedStates.NYSE",
    maturity_basis: Literal["calendar_days", "business_days"] = "calendar_days",  # type: ignore
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Calibrate Heston from the configured options-chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance uses anonymous
    cookie/crumb negotiation and may still return 401 responses. When provider mode is `tradier` or
    `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured. For
    reliable options-chain data, configure Tradier with
    MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY. Tradier API
    tokens: https://documentation.tradier.com/. Use `calendar` and
    `maturity_basis` to override the default `UnitedStates.NYSE` /
    `calendar_days` maturity assumptions.
    """
    from ..forecast.quantlib_tools import (
        calibrate_heston_quantlib_from_options as _impl,
    )

    symbol_value, symbol_error = _normalize_options_symbol(symbol)
    if symbol_error is not None or symbol_value is None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol,
            detail=detail,
            func=lambda: symbol_error or {"error": "symbol is required"},
        )
    expiration_value, expiration_error = _normalize_option_expiration(expiration)
    if expiration_error is not None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol_value,
            expiration=expiration,
            detail=detail,
            func=lambda: expiration_error,
        )
    input_error = next(
        (
            error
            for error in (
                _validate_options_integer(
                    "min_open_interest", min_open_interest, minimum=0
                ),
                _validate_options_integer("min_volume", min_volume, minimum=0),
                _validate_options_integer(
                    "max_contracts", max_contracts, minimum=5
                ),
                _validate_options_valuation_date(valuation_date),
            )
            if error is not None
        ),
        None,
    )
    if input_error is not None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol_value,
            expiration=expiration_value,
            valuation_date=valuation_date,
            detail=detail,
            func=lambda: input_error,
        )
    gate = _options_chain_provider_gate("options_heston_calibrate")
    if gate is not None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol_value,
            expiration=expiration_value,
            option_type=option_type,
            max_contracts=max_contracts,
            detail=detail,
            func=lambda: gate,
        )

    return _run_options_operation(
        "options_heston_calibrate",
        symbol=symbol_value,
        expiration=expiration_value,
        valuation_date=valuation_date,
        option_type=option_type,
        max_contracts=max_contracts,
        detail=detail,
        func=lambda: _apply_options_detail(
            _impl(
                symbol=symbol_value,
                expiration=expiration_value,
                valuation_date=valuation_date,
                option_type=option_type,
                risk_free_rate=float(risk_free_rate),
                dividend_yield=float(dividend_yield),
                min_open_interest=int(min_open_interest),
                min_volume=int(min_volume),
                max_contracts=int(max_contracts),
                calendar=calendar,
                maturity_basis=maturity_basis,
            ),
            detail=detail,
            kind="heston_calibrate",
        ),
    )
