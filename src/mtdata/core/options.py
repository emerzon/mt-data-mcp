"""Options chain, expiration, and pricing tools."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Literal, Optional

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
        action_required = "configure_options_provider"
        remediation = (
            "Configured Tradier mode is missing MTDATA_OPTIONS_API_KEY. mtdata will "
            "retry unauthenticated Yahoo as a best-effort fallback, but reliable "
            "options chains still require Tradier credentials."
        )
    else:
        effective_provider = (
            "tradier" if provider == "auto" and api_key_configured
            else "yahoo" if provider == "auto"
            else provider
        )
        configured_provider_ready = True
        action_required = None
        remediation = (
            "Yahoo options data is unauthenticated best-effort data and may return "
            "401/429. For reliable chains, set MTDATA_OPTIONS_PROVIDER=tradier and "
            "MTDATA_OPTIONS_API_KEY."
        ) if effective_provider == "yahoo" else None
    chain_data_access_available = effective_provider in {"yahoo", "tradier"}
    chain_provider_ready = effective_provider == "tradier" and api_key_configured
    chain_data_ready = chain_provider_ready
    provider_mode = "best_effort" if effective_provider == "yahoo" else "authenticated"
    warnings = []
    if provider_mode == "best_effort":
        warnings.append(
            "Options chain access is using unauthenticated Yahoo fallback; "
            "it is best-effort and may return 401/429."
        )
    out = {
        "configured_provider": provider,
        "effective_provider": effective_provider,
        "api_key_configured": api_key_configured,
        "configured_provider_ready": configured_provider_ready,
        "local_tools_ready": True,
        "chain_provider_ready": chain_provider_ready,
        "chain_data_ready": chain_data_ready,
        "chain_data_access_available": chain_data_access_available,
        "degraded": bool(provider_mode == "best_effort"),
        "provider_mode": provider_mode,
        "supported_providers": ["tradier", "yahoo"],
        "chain_dependent_tools": [
            "options_expirations",
            "options_chain",
            "options_heston_calibrate",
        ],
        "local_tools": ["options_barrier_price"],
        "action_required": action_required,
        "remediation": remediation,
    }
    if warnings:
        out["warnings"] = warnings
    return out


def _options_chain_provider_gate(tool_name: str) -> Optional[Dict[str, Any]]:
    readiness = _options_provider_readiness()
    if readiness.get("chain_data_access_available") is True:
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
                "symbol",
                "expiration",
                "underlying_price",
                "currency",
                "option_type",
                "count",
                "calls_count",
                "puts_count",
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
                "strike",
                "barrier",
                "maturity_days",
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
    elif payload.get("remediation"):
        payload["remediation_hint"] = (
            "Reliable options-chain access requires Tradier credentials."
        )
        payload["next_steps"] = [
            "Set MTDATA_OPTIONS_PROVIDER=tradier.",
            "Set MTDATA_OPTIONS_API_KEY to a Tradier API token, then restart mtdata.",
            "Yahoo fallback is best-effort only and may still return 401/429.",
        ]
        payload.pop("remediation", None)
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

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance is an unauthenticated
    fallback and may return 401 responses. When provider mode is `tradier` or
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
    min_open_interest: int = 0,
    min_volume: int = 0,
    limit: int = 200,
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Fetch option-chain snapshots using the configured chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance is an unauthenticated
    fallback and may return 401 responses. When provider mode is `tradier` or
    `auto`, mtdata retries Yahoo if Tradier is unavailable or misconfigured. For
    reliable options-chain data, configure Tradier with
    MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY. Tradier API
    tokens: https://documentation.tradier.com/.
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
    gate = _options_chain_provider_gate("options_chain")
    if gate is not None:
        return _run_options_operation(
            "options_chain",
            symbol=symbol_value,
            expiration=expiration,
            option_type=option_type,
            limit=limit,
            detail=detail,
            func=lambda: gate,
        )

    return _run_options_operation(
        "options_chain",
        symbol=symbol_value,
        expiration=expiration,
        option_type=option_type,
        limit=limit,
        detail=detail,
        func=lambda: _apply_options_detail(
            _impl(
                symbol=symbol_value,
                expiration=expiration,
                option_type=option_type,
                min_open_interest=int(min_open_interest),
                min_volume=int(min_volume),
                limit=int(limit),
            ),
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
    min_open_interest: int = 0,
    min_volume: int = 0,
    max_contracts: int = 25,
    calendar: str = "UnitedStates.NYSE",
    maturity_basis: Literal["calendar_days", "business_days"] = "calendar_days",  # type: ignore
    detail: DetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Calibrate Heston from the configured options-chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance is an unauthenticated
    fallback and may return 401 responses. When provider mode is `tradier` or
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
    gate = _options_chain_provider_gate("options_heston_calibrate")
    if gate is not None:
        return _run_options_operation(
            "options_heston_calibrate",
            symbol=symbol_value,
            expiration=expiration,
            option_type=option_type,
            max_contracts=max_contracts,
            detail=detail,
            func=lambda: gate,
        )

    return _run_options_operation(
        "options_heston_calibrate",
        symbol=symbol_value,
        expiration=expiration,
        valuation_date=valuation_date,
        option_type=option_type,
        max_contracts=max_contracts,
        detail=detail,
        func=lambda: _apply_options_detail(
            _impl(
                symbol=symbol_value,
                expiration=expiration,
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
