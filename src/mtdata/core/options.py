"""Options chain, expiration, and pricing tools."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Literal, Optional

from ..shared.schema import CompactFullDetailLiteral
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
    effective_provider = (
        "tradier" if provider == "auto" and api_key_configured
        else "yahoo" if provider == "auto"
        else provider
    )
    chain_data_ready = effective_provider == "yahoo" or (
        effective_provider == "tradier" and api_key_configured
    )
    return {
        "configured_provider": provider,
        "effective_provider": effective_provider,
        "api_key_configured": api_key_configured,
        "chain_data_ready": chain_data_ready,
        "provider_mode": "best_effort" if effective_provider == "yahoo" else "authenticated",
        "supported_providers": ["tradier", "yahoo"],
        "chain_dependent_tools": [
            "options_expirations",
            "options_chain",
            "options_heston_calibrate",
        ],
        "local_tools": ["options_barrier_price"],
        "action_required": None if chain_data_ready else "configure_options_provider",
        "remediation": (
            "Yahoo options data is unauthenticated best-effort data and may return 401/429. "
            "For reliable chains, set MTDATA_OPTIONS_PROVIDER=tradier and MTDATA_OPTIONS_API_KEY."
        )
        if effective_provider == "yahoo"
        else None if chain_data_ready else (
            "For reliable options chains, set MTDATA_OPTIONS_PROVIDER=tradier "
            "and MTDATA_OPTIONS_API_KEY. Yahoo is an unauthenticated fallback "
            "and may return 401/429."
        ),
    }


def _options_chain_provider_gate(tool_name: str) -> Optional[Dict[str, Any]]:
    readiness = _options_provider_readiness()
    if readiness.get("chain_data_ready") is True:
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
                "cached",
                "data_age_seconds",
                "symbol",
                "expirations",
                "expiration_count",
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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
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
            "Use Yahoo only as an unauthenticated fallback that may return 401/429.",
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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Fetch option expirations using the configured options-chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance is an unauthenticated
    fallback and may return 401 responses. For reliable options-chain data,
    configure Tradier with MTDATA_OPTIONS_PROVIDER=tradier and
    MTDATA_OPTIONS_API_KEY. Tradier API tokens: https://documentation.tradier.com/.
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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Fetch option-chain snapshots using the configured chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance is an unauthenticated
    fallback and may return 401 responses. For reliable options-chain data,
    configure Tradier with MTDATA_OPTIONS_PROVIDER=tradier and
    MTDATA_OPTIONS_API_KEY. Tradier API tokens: https://documentation.tradier.com/.
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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Price a barrier option using QuantLib."""
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
    detail: CompactFullDetailLiteral = "compact",  # type: ignore
) -> Dict[str, Any]:
    """Calibrate Heston from the configured options-chain provider.

    Tradier requires MTDATA_OPTIONS_API_KEY. Yahoo Finance is an unauthenticated
    fallback and may return 401 responses. For reliable options-chain data,
    configure Tradier with MTDATA_OPTIONS_PROVIDER=tradier and
    MTDATA_OPTIONS_API_KEY. Tradier API tokens: https://documentation.tradier.com/.
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
            ),
            detail=detail,
            kind="heston_calibrate",
        ),
    )
