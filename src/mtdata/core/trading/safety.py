"""Pre-trade guardrails and safety rails."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .sizing import _resolve_risk_tick_value


class TradeSafetyPolicy(BaseModel):
    """Configurable pre-trade safety rails."""

    max_volume: Optional[float] = None
    require_stop_loss: bool = False
    max_deviation: Optional[int] = None
    reduce_only: bool = False


class AccountRiskLimits(BaseModel):
    """Configurable account-level risk thresholds."""

    min_margin_level_pct: Optional[float] = None
    max_floating_loss: Optional[float] = None
    max_total_exposure_lots: Optional[float] = None


class WalletRiskLimits(BaseModel):
    """Portfolio risk caps measured against account wallet metrics."""

    max_risk_pct_of_equity: Optional[float] = None
    max_risk_pct_of_balance: Optional[float] = None
    max_risk_pct_of_free_margin: Optional[float] = None


class TradeGuardrailsConfig(BaseModel):
    """Top-level trade guardrail configuration."""

    enabled: bool = False
    trading_enabled: bool = True
    allowed_symbols: List[str] = Field(default_factory=list)
    blocked_symbols: List[str] = Field(default_factory=list)
    max_volume: Optional[float] = None
    max_volume_by_symbol: Dict[str, float] = Field(default_factory=dict)
    safety_policy: TradeSafetyPolicy = Field(default_factory=TradeSafetyPolicy)
    account_risk_limits: AccountRiskLimits = Field(default_factory=AccountRiskLimits)
    wallet_risk_limits: WalletRiskLimits = Field(default_factory=WalletRiskLimits)

    def is_enabled(self) -> bool:
        return _guardrails_active(self)


def _safe_float_attr(obj: Any, name: str) -> Optional[float]:
    """Extract a float attribute safely, returning None on failure."""
    try:
        val = getattr(obj, name, None)
        if val is None:
            return None
        fv = float(val)
        return fv if math.isfinite(fv) else None
    except TypeError, ValueError:
        return None


def _normalize_stop_loss_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or math.isclose(numeric, 0.0, abs_tol=1e-12):
        return None
    return numeric


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_side(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if not text:
        return None
    if text.startswith("BUY"):
        return "BUY"
    if text.startswith("SELL"):
        return "SELL"
    return text


def _model_has_values(model: Any) -> bool:
    if hasattr(model, "model_dump"):
        values = getattr(model, "model_dump")().values()
    elif isinstance(model, dict):
        values = model.values()
    else:
        values = vars(model).values()
    for value in values:
        if isinstance(value, bool):
            if value:
                return True
            continue
        if isinstance(value, dict):
            if value:
                return True
            continue
        if isinstance(value, list):
            if value:
                return True
            continue
        if value is not None:
            return True
    return False


def _guardrails_active(config: Optional[Any]) -> bool:
    if config is None:
        return False
    if config.enabled:
        return True
    if not config.trading_enabled:
        return True
    return any(
        (
            bool(config.allowed_symbols),
            bool(config.blocked_symbols),
            config.max_volume is not None,
            bool(config.max_volume_by_symbol),
            _model_has_values(config.safety_policy),
            _model_has_values(config.account_risk_limits),
            _model_has_values(config.wallet_risk_limits),
        )
    )


def _wallet_limits_active(limits: Optional[WalletRiskLimits]) -> bool:
    return limits is not None and _model_has_values(limits)


def _build_guardrail_block(
    violations: List[str],
    *,
    rule: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": "Trade blocked by guardrails.",
        "guardrail_blocked": True,
        "guardrail_rule": rule,
        "violations": violations,
    }
    if context:
        payload["guardrail_context"] = context
    return payload


def _evaluate_safety_policy(
    policy: Optional[TradeSafetyPolicy],
    *,
    volume: Optional[float] = None,
    stop_loss: Optional[float] = None,
    deviation: Optional[int] = None,
    side: Optional[str] = None,
    existing_side: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate *policy* against the given order parameters."""
    if policy is None:
        return None

    violations: List[str] = []

    if policy.max_volume is not None and volume is not None:
        try:
            if math.isfinite(volume) and volume > policy.max_volume:
                violations.append(
                    f"Volume {volume} exceeds safety limit of {policy.max_volume}."
                )
        except TypeError, ValueError:
            pass

    if policy.require_stop_loss:
        if stop_loss is None or not math.isfinite(stop_loss):
            violations.append("Safety policy requires a stop-loss on every order.")

    if policy.max_deviation is not None and deviation is not None:
        try:
            if int(deviation) > policy.max_deviation:
                violations.append(
                    f"Deviation {deviation} exceeds safety limit of {policy.max_deviation}."
                )
        except TypeError, ValueError:
            pass

    if policy.reduce_only and side is not None:
        opposite = {"BUY": "SELL", "SELL": "BUY"}
        normalized_existing_side = _normalize_side(existing_side)
        normalized_side = _normalize_side(side)
        if normalized_existing_side is None:
            violations.append("Reduce-only policy: no existing position to reduce.")
        elif normalized_side != opposite.get(normalized_existing_side, ""):
            violations.append(
                f"Reduce-only policy: order side {side} does not reduce "
                f"existing {existing_side} position."
            )

    if not violations:
        return None

    return {
        "error": "Order blocked by safety policy.",
        "violations": violations,
    }


def _evaluate_account_risk_gate(
    limits: Optional[AccountRiskLimits],
    *,
    account_info: Any = None,
    new_volume: float = 0.0,
    existing_volume: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Evaluate *limits* against current account state."""
    if limits is None:
        return None

    violations: List[str] = []

    if limits.min_margin_level_pct is not None and account_info is not None:
        margin_level = _safe_float_attr(account_info, "margin_level")
        if margin_level is not None and margin_level < limits.min_margin_level_pct:
            violations.append(
                f"Margin level {margin_level:.1f}% is below the "
                f"minimum threshold of {limits.min_margin_level_pct:.1f}%."
            )

    if limits.max_floating_loss is not None and account_info is not None:
        profit = _safe_float_attr(account_info, "profit")
        if profit is not None and profit < 0 and abs(profit) > limits.max_floating_loss:
            violations.append(
                f"Floating loss ${abs(profit):.2f} exceeds the "
                f"limit of ${limits.max_floating_loss:.2f}."
            )

    if limits.max_total_exposure_lots is not None:
        total_after = existing_volume + new_volume
        if total_after > limits.max_total_exposure_lots:
            violations.append(
                f"Total exposure {total_after:.2f} lots would exceed the "
                f"limit of {limits.max_total_exposure_lots:.2f} lots."
            )

    if not violations:
        return None

    return {
        "error": "Order blocked by account risk gate.",
        "violations": violations,
    }


def _evaluate_symbol_guardrails(
    config: TradeGuardrailsConfig,
    *,
    symbol: str,
    volume: Optional[float],
) -> Optional[Dict[str, Any]]:
    violations: List[str] = []
    normalized_symbol = _normalize_symbol(symbol)

    if not config.trading_enabled:
        violations.append("Trading is disabled by guardrail configuration.")

    if normalized_symbol:
        blocked = {_normalize_symbol(item) for item in config.blocked_symbols}
        allowed = {_normalize_symbol(item) for item in config.allowed_symbols}
        if normalized_symbol in blocked:
            violations.append(
                f"Symbol {normalized_symbol} is blocked by guardrail policy."
            )
        elif allowed and normalized_symbol not in allowed:
            violations.append(
                f"Symbol {normalized_symbol} is not in the configured allowlist."
            )

    if volume is not None and math.isfinite(float(volume)):
        limits = []
        if config.max_volume is not None:
            limits.append(("global", float(config.max_volume)))

        # If max_volume_by_symbol is configured, it acts as a whitelist
        if config.max_volume_by_symbol:
            symbol_limit = config.max_volume_by_symbol.get(normalized_symbol)
            if symbol_limit is None:
                violations.append(
                    f"Symbol {normalized_symbol} is not in the configured max_volume_by_symbol allowlist."
                )
            else:
                limits.append((normalized_symbol, float(symbol_limit)))

        for scope, cap in limits:
            if float(volume) > cap:
                scope_label = (
                    "global max volume"
                    if scope == "global"
                    else f"max volume for {normalized_symbol}"
                )
                violations.append(
                    f"Volume {float(volume)} exceeds the configured {scope_label} of {cap}."
                )

    if not violations:
        return None
    return _build_guardrail_block(
        violations,
        rule="symbol_policy",
        context={"symbol": normalized_symbol or symbol, "volume": volume},
    )


def _estimate_order_risk_currency(
    *,
    symbol_info: Any,
    volume: float,
    entry_price: float,
    stop_loss: Optional[float],
    side: str,
) -> tuple[Optional[float], Optional[str]]:
    normalized_stop_loss = _normalize_stop_loss_value(stop_loss)
    if normalized_stop_loss is None:
        return None, "stop_loss_missing"

    tick_size = _safe_float_attr(symbol_info, "trade_tick_size")
    tick_value = _safe_float_attr(symbol_info, "trade_tick_value")
    tick_value_loss = _safe_float_attr(symbol_info, "trade_tick_value_loss")
    risk_tick_value = _resolve_risk_tick_value(
        tick_value=tick_value or float("nan"),
        tick_value_loss=tick_value_loss,
    )
    if tick_size is None or tick_size <= 0:
        return None, "tick_size_invalid"
    if not math.isfinite(risk_tick_value) or risk_tick_value <= 0:
        return None, "tick_value_invalid"

    normalized_side = _normalize_side(side)
    if normalized_side == "BUY":
        risk_ticks = (float(entry_price) - normalized_stop_loss) / tick_size
    else:
        risk_ticks = (normalized_stop_loss - float(entry_price)) / tick_size
    if not math.isfinite(risk_ticks) or risk_ticks <= 0:
        return None, "stop_loss_wrong_side"

    risk_currency = abs(float(volume) * risk_ticks * risk_tick_value)
    if not math.isfinite(risk_currency) or risk_currency < 0:
        return None, "risk_invalid"
    return risk_currency, None


def _position_side(position: Any) -> Optional[str]:
    text = _normalize_side(getattr(position, "type", None))
    if text in {"BUY", "SELL"}:
        return text
    raw_type = getattr(position, "type", None)
    try:
        numeric_type = int(raw_type)
    except Exception:
        numeric_type = None
    if numeric_type == 0:
        return "BUY"
    if numeric_type == 1:
        return "SELL"
    return None


def _sum_existing_exposure_lots(existing_positions: Optional[List[Any]]) -> float:
    total = 0.0
    for position in list(existing_positions or []):
        volume = _safe_float_attr(position, "volume")
        if volume is not None:
            total += abs(volume)
    return total


def _resolve_existing_symbol_side(
    *,
    symbol: str,
    existing_positions: Optional[List[Any]],
) -> Optional[str]:
    normalized_symbol = _normalize_symbol(symbol)
    net_volume = 0.0
    for position in list(existing_positions or []):
        if _normalize_symbol(getattr(position, "symbol", None)) != normalized_symbol:
            continue
        volume = _safe_float_attr(position, "volume")
        side = _position_side(position)
        if volume is None or side is None:
            continue
        net_volume += float(volume) if side == "BUY" else -float(volume)
    if net_volume > 0:
        return "BUY"
    if net_volume < 0:
        return "SELL"
    return None


def _total_portfolio_risk_currency(
    *,
    existing_positions: Optional[List[Any]],
    symbol_info_resolver: Optional[Callable[[str], Any]],
) -> tuple[Optional[float], List[str]]:
    total = 0.0
    issues: List[str] = []
    if not existing_positions:
        return 0.0, issues
    if symbol_info_resolver is None:
        return None, ["symbol metadata resolver unavailable"]

    for position in list(existing_positions):
        symbol = _normalize_symbol(getattr(position, "symbol", None))
        if not symbol:
            issues.append("position symbol missing")
            continue
        side = _position_side(position)
        if side is None:
            issues.append(f"{symbol}: unable to determine position side")
            continue
        volume = _safe_float_attr(position, "volume")
        entry_price = _safe_float_attr(position, "price_open")
        stop_loss = _safe_float_attr(position, "sl")
        if volume is None or entry_price is None:
            issues.append(f"{symbol}: position metadata is incomplete")
            continue
        if stop_loss is None or math.isclose(stop_loss, 0.0, abs_tol=1e-12):
            issues.append(
                f"{symbol}: existing position risk cannot be quantified without a stop-loss."
            )
            continue
        symbol_info = symbol_info_resolver(symbol)
        if symbol_info is None:
            issues.append(f"{symbol}: symbol metadata unavailable")
            continue
        risk_currency, risk_error = _estimate_order_risk_currency(
            symbol_info=symbol_info,
            volume=abs(volume),
            entry_price=entry_price,
            stop_loss=stop_loss,
            side=side,
        )
        if risk_currency is None:
            issues.append(f"{symbol}: unable to quantify risk ({risk_error}).")
            continue
        total += risk_currency

    return total, issues


def _evaluate_wallet_risk_limits(
    limits: Optional[WalletRiskLimits],
    *,
    account_info: Any,
    existing_positions: Optional[List[Any]],
    symbol_info: Any,
    symbol_info_resolver: Optional[Callable[[str], Any]],
    symbol: str,
    volume: Optional[float],
    entry_price: Optional[float],
    stop_loss: Optional[float],
    side: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not _wallet_limits_active(limits):
        return None

    violations: List[str] = []
    normalized_side = _normalize_side(side)
    if account_info is None:
        violations.append(
            "Account information is required to enforce wallet risk guardrails."
        )
    if symbol_info is None:
        violations.append(
            f"Symbol metadata is required to enforce wallet risk for {symbol}."
        )
    if volume is None or not math.isfinite(float(volume)) or float(volume) <= 0:
        violations.append(
            "Volume must be finite and positive to enforce wallet risk guardrails."
        )
    if entry_price is None or not math.isfinite(float(entry_price)):
        violations.append("Entry price is required to enforce wallet risk guardrails.")
    if stop_loss is None or not math.isfinite(float(stop_loss)):
        violations.append("Wallet risk guardrails require a stop-loss on the order.")
    if normalized_side is None:
        violations.append("Order side is required to enforce wallet risk guardrails.")
    if violations:
        return _build_guardrail_block(
            violations,
            rule="wallet_risk",
            context={"symbol": _normalize_symbol(symbol)},
        )

    existing_total_risk, existing_issues = _total_portfolio_risk_currency(
        existing_positions=existing_positions,
        symbol_info_resolver=symbol_info_resolver,
    )
    if existing_total_risk is None:
        existing_total_risk = 0.0
    if existing_issues:
        return _build_guardrail_block(
            existing_issues,
            rule="wallet_risk",
            context={"symbol": _normalize_symbol(symbol)},
        )

    candidate_risk, candidate_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=float(volume),
        entry_price=float(entry_price),
        stop_loss=float(stop_loss),
        side=normalized_side,
    )
    if candidate_risk is None:
        return _build_guardrail_block(
            [f"Unable to quantify candidate order risk ({candidate_error})."],
            rule="wallet_risk",
            context={"symbol": _normalize_symbol(symbol)},
        )

    total_after = float(existing_total_risk) + float(candidate_risk)
    thresholds = (
        (
            "equity",
            _safe_float_attr(account_info, "equity"),
            limits.max_risk_pct_of_equity,
        ),
        (
            "balance",
            _safe_float_attr(account_info, "balance"),
            limits.max_risk_pct_of_balance,
        ),
        (
            "free_margin",
            _safe_float_attr(account_info, "margin_free"),
            limits.max_risk_pct_of_free_margin,
        ),
    )
    for basis_name, basis_value, limit_pct in thresholds:
        if limit_pct is None:
            continue
        if basis_value is None or basis_value <= 0:
            violations.append(
                f"Cannot enforce max risk pct of {basis_name}: account {basis_name} is unavailable."
            )
            continue
        risk_pct = (total_after / basis_value) * 100.0
        if risk_pct > float(limit_pct):
            violations.append(
                f"Total risk after this trade would be {risk_pct:.2f}% of {basis_name}, "
                f"exceeding the configured limit of {float(limit_pct):.2f}%."
            )

    if not violations:
        return None
    return _build_guardrail_block(
        violations,
        rule="wallet_risk",
        context={
            "symbol": _normalize_symbol(symbol),
            "candidate_risk": round(float(candidate_risk), 2),
            "portfolio_risk_after": round(float(total_after), 2),
        },
    )


def evaluate_trade_guardrails(
    config: Optional[TradeGuardrailsConfig],
    *,
    symbol: str,
    volume: Optional[float],
    stop_loss: Optional[float] = None,
    deviation: Optional[int] = None,
    side: Optional[str] = None,
    entry_price: Optional[float] = None,
    account_info: Any = None,
    existing_positions: Optional[List[Any]] = None,
    symbol_info: Any = None,
    symbol_info_resolver: Optional[Callable[[str], Any]] = None,
    enforce_symbol_rules: bool = True,
    enforce_safety_policy: bool = True,
    enforce_account_risk: bool = True,
    enforce_wallet_risk: bool = True,
) -> Optional[Dict[str, Any]]:
    """Evaluate configured guardrails against an order request."""
    if not _guardrails_active(config):
        return None

    normalized_symbol = _normalize_symbol(symbol)
    normalized_side = _normalize_side(side)

    if enforce_symbol_rules:
        symbol_result = _evaluate_symbol_guardrails(
            config,
            symbol=normalized_symbol,
            volume=volume,
        )
        if symbol_result is not None:
            return symbol_result

    if enforce_safety_policy:
        existing_side = _resolve_existing_symbol_side(
            symbol=normalized_symbol,
            existing_positions=existing_positions,
        )
        safety_result = _evaluate_safety_policy(
            config.safety_policy,
            volume=volume,
            stop_loss=stop_loss,
            deviation=deviation,
            side=normalized_side,
            existing_side=existing_side,
        )
        if safety_result is not None:
            return _build_guardrail_block(
                list(safety_result.get("violations") or []),
                rule="safety_policy",
                context={"symbol": normalized_symbol, "side": normalized_side},
            )

    existing_volume = _sum_existing_exposure_lots(existing_positions)
    if enforce_account_risk:
        account_result = _evaluate_account_risk_gate(
            config.account_risk_limits,
            account_info=account_info,
            new_volume=abs(float(volume or 0.0)),
            existing_volume=existing_volume,
        )
        if account_result is not None:
            return _build_guardrail_block(
                list(account_result.get("violations") or []),
                rule="account_risk",
                context={"symbol": normalized_symbol, "volume": volume},
            )

    if enforce_wallet_risk:
        wallet_result = _evaluate_wallet_risk_limits(
            config.wallet_risk_limits,
            account_info=account_info,
            existing_positions=existing_positions,
            symbol_info=symbol_info,
            symbol_info_resolver=symbol_info_resolver,
            symbol=normalized_symbol,
            volume=volume,
            entry_price=entry_price,
            stop_loss=stop_loss,
            side=normalized_side,
        )
        if wallet_result is not None:
            return wallet_result

    return None


def preview_trade_guardrails(
    config: Optional[TradeGuardrailsConfig],
    *,
    symbol: str,
    volume: Optional[float],
    stop_loss: Optional[float] = None,
    deviation: Optional[int] = None,
    side: Optional[str] = None,
) -> Dict[str, Any]:
    """Produce a dry-run friendly preview of guardrail checks."""
    enabled = _guardrails_active(config)
    if not enabled:
        return {
            "enabled": False,
            "blocked": False,
            "checks_not_performed": [],
            "message": "No trade guardrails are configured.",
        }

    static_result = evaluate_trade_guardrails(
        config,
        symbol=symbol,
        volume=volume,
        stop_loss=stop_loss,
        deviation=deviation,
        side=side,
        enforce_account_risk=False,
        enforce_wallet_risk=False,
    )
    checks_not_performed: List[str] = []
    if _model_has_values(config.account_risk_limits):
        checks_not_performed.append("account_risk")
    if _wallet_limits_active(config.wallet_risk_limits):
        checks_not_performed.append("wallet_risk")
    return {
        "enabled": True,
        "blocked": static_result is not None,
        "violations": list(static_result.get("violations") or [])
        if static_result
        else [],
        "rule": static_result.get("guardrail_rule") if static_result else None,
        "checks_not_performed": checks_not_performed,
    }


def pending_order_risk_increased(
    *,
    symbol_info: Any,
    side: str,
    volume: float,
    existing_entry_price: Optional[float],
    existing_stop_loss: Optional[float],
    candidate_entry_price: Optional[float],
    candidate_stop_loss: Optional[float],
) -> bool:
    """Return True when a pending-order modification increases downside risk."""
    current_sl = _normalize_stop_loss_value(_safe_float_attr(
        type("Obj", (), {"value": existing_stop_loss})(), "value"
    ))
    next_sl = _normalize_stop_loss_value(_safe_float_attr(
        type("Obj", (), {"value": candidate_stop_loss})(), "value"
    ))
    current_entry = _safe_float_attr(
        type("Obj", (), {"value": existing_entry_price})(), "value"
    )
    next_entry = _safe_float_attr(
        type("Obj", (), {"value": candidate_entry_price})(), "value"
    )

    if current_sl is None and next_sl is None:
        return False
    if current_sl is not None and next_sl is None:
        return True
    if next_sl is None or next_entry is None:
        return False

    next_risk, next_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=volume,
        entry_price=next_entry,
        stop_loss=next_sl,
        side=side,
    )
    if next_risk is None:
        return next_error in {
            "stop_loss_wrong_side",
            "tick_size_invalid",
            "tick_value_invalid",
        }

    if current_sl is None or current_entry is None:
        return True
    current_risk, current_error = _estimate_order_risk_currency(
        symbol_info=symbol_info,
        volume=volume,
        entry_price=current_entry,
        stop_loss=current_sl,
        side=side,
    )
    if current_risk is None:
        return True if current_error else False
    return next_risk > (current_risk + 1e-9)
