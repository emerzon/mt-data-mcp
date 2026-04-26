import math
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from ..shared.constants import TIMEFRAME_MAP
from ..shared.schema import CompactFullDetailLiteral, DenoiseSpec, TimeframeLiteral
from ..shared.validators import invalid_timeframe_error
from ..utils.denoise import normalize_denoise_spec as _normalize_denoise_spec
from ..utils.utils import _format_time_minimal
from .common import (
    bars_per_year as _bars_per_year,
)
from .common import (
    fetch_history as _fetch_history,
)
from .common import (
    log_returns_from_prices as _log_returns_from_prices,
)
from .common import (
    quantity_to_target as _quantity_to_target,
)
from .contracts import (
    AnchorMetadata,
    BacktestEvaluationContract,
    CuratedPreparedInputs,
    DataPreparationContract,
    DeclarativeStrategyContract,
    ForecastArtifact,
    ForecastEvaluationContext,
    ForecastExecutionContract,
    ForecastModelContract,
    RealizedPathArtifact,
    StrategyEvaluationResult,
    StrategyTradeIntent,
)
from .exceptions import ForecastError, raise_if_error_result
from .forecast import forecast
from .volatility import forecast_volatility


def _attach_request_metadata(
    result: Dict[str, Any],
    *,
    request: Dict[str, Any],
    resolved_request: Optional[Dict[str, Any]] = None,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    out = dict(result)
    # Only include request metadata in full detail mode
    if detail == "full":
        out["request"] = deepcopy(request)
        # Only include resolved_request if it differs from request
        if resolved_request is not None and resolved_request != request:
            out["resolved_request"] = deepcopy(resolved_request)
    return out


def _compact_metrics_payload(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}

    out = dict(metrics)
    sample_warning = out.pop("sample_warning", None)
    sample_notice = out.get("sample_notice")
    if sample_warning and not isinstance(sample_notice, dict):
        out["sample_notice"] = {
            "code": "annualization_suppressed_low_sample",
            "trades_observed": out.get("trades_observed"),
            "minimum_trades": out.get("min_trades_for_annualization"),
        }
    return out


def _unavailable_performance_metrics(reason: str, slippage_bps: float) -> Dict[str, Any]:
    return {
        "avg_return": None,
        "avg_return_per_trade": None,
        "win_rate": None,
        "max_drawdown": None,
        "trades_per_year": None,
        "trades_observed": 0,
        "slippage_bps": float(slippage_bps),
        "metrics_available": False,
        "metrics_reason": str(reason),
    }


def _attach_metrics_status(
    payload: Dict[str, Any],
    *,
    metrics: Dict[str, Any],
    slippage_bps: float,
    unavailable_reason: str,
) -> None:
    if metrics:
        payload["metrics"] = metrics
        payload["metrics_available"] = True
        payload["slippage_bps"] = float(slippage_bps)
        return

    payload["metrics"] = _unavailable_performance_metrics(
        unavailable_reason,
        slippage_bps,
    )
    payload["metrics_available"] = False
    payload["metrics_reason"] = str(unavailable_reason)
    if unavailable_reason == "no_non_flat_trades":
        payload["trade_status"] = "flat"
    payload["slippage_bps"] = float(slippage_bps)


def _get_forecast_methods_data_safe() -> Dict[str, Any]:
    """Safely fetch forecast methods metadata.

    Falls back to a minimal set of classical methods if discovery fails.
    Only 'method' and 'available' keys are required by this module.
    """
    try:
        from .forecast_registry import get_forecast_methods_data as _get
        data = _get()
        if isinstance(data, dict) and 'methods' in data:
            return data
    except Exception:
        pass
    return {
        'methods': [
            {'method': 'naive', 'available': True},
            {'method': 'drift', 'available': True},
            {'method': 'seasonal_naive', 'available': True},
            {'method': 'theta', 'available': True},
            {'method': 'fourier_ols', 'available': True},
        ]
    }
_MIN_ANNUALIZATION_TRADES = 30
_MIN_ANNUALIZATION_YEARS = 0.25


def _compute_performance_metrics(
    returns: List[float],
    timeframe: str,
    horizon: int,
    slippage_bps: float,
    trade_spacing_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute portfolio-level performance statistics from per-trade returns."""

    metrics: Dict[str, Any] = {}
    if not returns:
        return metrics

    arr = np.asarray([float(r) for r in returns if r is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return metrics

    bars_per_year = _bars_per_year(timeframe)
    cadence = max(1, int(trade_spacing_bars)) if trade_spacing_bars is not None else max(1, int(horizon))
    trades_per_year = float(bars_per_year / cadence) if math.isfinite(bars_per_year) else float('nan')

    avg_return = float(np.mean(arr))
    win_rate = float(np.mean(arr > 0.0)) if arr.size > 0 else float('nan')
    std_ret = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    enough_trades = int(arr.size) >= int(_MIN_ANNUALIZATION_TRADES)
    sharpe = float('nan')
    if enough_trades and std_ret > 1e-12 and math.isfinite(trades_per_year) and trades_per_year > 0:
        sharpe = float((avg_return / std_ret) * math.sqrt(trades_per_year))

    equity = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.where(peak == 0.0, 1.0, peak) - 1.0
    max_drawdown = float(abs(np.min(drawdowns))) if drawdowns.size > 0 else float('nan')

    years = float(arr.size / trades_per_year) if math.isfinite(trades_per_year) and trades_per_year > 0 else float('nan')
    annual_return = float('nan')
    if (
        enough_trades
        and math.isfinite(years)
        and years >= _MIN_ANNUALIZATION_YEARS
        and equity.size > 0
        and equity[-1] > 0
    ):
        try:
            annual_return = float(equity[-1] ** (1.0 / years) - 1.0)
        except Exception:
            annual_return = float('nan')
    calmar = float('nan')
    if max_drawdown > 0 and math.isfinite(max_drawdown) and math.isfinite(annual_return):
        calmar = float(annual_return / max_drawdown)

    def _finite_or_none(value: float) -> Optional[float]:
        try:
            value_f = float(value)
        except Exception:
            return None
        if not math.isfinite(value_f):
            return None
        return value_f

    metrics.update({
        "avg_return_per_trade": avg_return,
        "win_rate": win_rate,
        "win_rate_display": f"{win_rate:.1%}",
        "sharpe_ratio": _finite_or_none(sharpe),
        "max_drawdown": max_drawdown,
        "calmar_ratio": _finite_or_none(calmar),
        "annual_return": _finite_or_none(annual_return),
        "trades_per_year": trades_per_year,
        "trades_observed": int(arr.size),
        "slippage_bps": float(slippage_bps),
    })
    if not enough_trades:
        metrics["sample_notice"] = {
            "code": "annualization_suppressed_low_sample",
            "trades_observed": int(arr.size),
            "minimum_trades": int(_MIN_ANNUALIZATION_TRADES),
        }
        metrics["sample_warning"] = (
            f"Only {int(arr.size)} trades. Annualized risk metrics "
            f"(Sharpe/Calmar/annual_return) are suppressed below {_MIN_ANNUALIZATION_TRADES} trades."
        )
        metrics["min_trades_for_annualization"] = float(_MIN_ANNUALIZATION_TRADES)
    return metrics


def _normalize_detail_mode(value: Any) -> Literal["compact", "full"]:
    detail_mode = str(value or "compact").strip().lower()
    if detail_mode not in {"compact", "full"}:
        return "compact"
    return detail_mode  # type: ignore[return-value]


def _contract_payload(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    return dict(model.model_dump(exclude_none=True))


def _feature_names_from_spec(features: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(features, dict):
        return []

    names: List[str] = []
    for key in ("ti", "indicators", "exog", "future_covariates"):
        raw_value = features.get(key)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            tokens = [token.strip() for token in raw_value.replace(",", " ").split() if token.strip()]
        elif isinstance(raw_value, (list, tuple, set)):
            tokens = [str(token).strip() for token in raw_value if str(token).strip()]
        else:
            tokens = [str(raw_value).strip()] if str(raw_value).strip() else []
        names.extend(tokens)
    return list(dict.fromkeys(names))


def _build_curated_prepared_inputs(
    *,
    features: Optional[Dict[str, Any]],
    anchor_history: Optional[Any],
    entry_price: float,
    expected_return: Optional[float],
) -> CuratedPreparedInputs:
    scalars: Dict[str, float] = {}
    if math.isfinite(entry_price):
        scalars["entry_price"] = float(entry_price)
    if expected_return is not None and math.isfinite(float(expected_return)):
        scalars["expected_return"] = float(expected_return)
    if anchor_history is not None and getattr(anchor_history, "empty", True) is False:
        for col in ("close", "close_dn"):
            if col in anchor_history.columns:
                try:
                    value = float(anchor_history[col].iloc[-1])
                except Exception:
                    continue
                if math.isfinite(value):
                    scalars[col] = value
    return CuratedPreparedInputs(
        scalars=scalars,
        feature_names=_feature_names_from_spec(features),
    )


def _build_forecast_threshold_strategy_contract(
    trade_threshold: float,
) -> DeclarativeStrategyContract:
    threshold_value = float(trade_threshold or 0.0)
    return DeclarativeStrategyContract(
        name="forecast-threshold",
        description="Built-in bridge strategy for forecast_backtest.",
        entry={
            "type": "forecast_threshold",
            "signal": "expected_return",
            "long_above": threshold_value,
            "short_below": -threshold_value,
        },
        exits=[{"type": "forecast_target"}],
    )


def _resolve_strategy_signal_value(
    signal: str,
    *,
    context: ForecastEvaluationContext,
) -> Optional[float]:
    if signal == "expected_return":
        return context.forecast.expected_return
    if signal == "forecast_sum":
        if not context.forecast.values:
            return None
        return float(np.nansum(np.asarray(context.forecast.values, dtype=float)))
    if signal == "forecast_last":
        if not context.forecast.values:
            return None
        return float(context.forecast.values[-1])
    return None


def _resolve_size_fraction(
    *,
    strategy_contract: DeclarativeStrategyContract,
    context: ForecastEvaluationContext,
) -> float:
    sizing = strategy_contract.position_sizing
    if sizing.type == "fixed_fraction":
        return float(sizing.fraction)
    confidence = context.forecast.confidence
    if confidence is None or not math.isfinite(float(confidence)):
        return float(sizing.min_fraction)
    confidence_value = min(1.0, max(0.0, float(confidence)))
    span = float(sizing.max_fraction) - float(sizing.min_fraction)
    return float(sizing.min_fraction) + span * confidence_value


def _evaluate_forecast_strategy(
    strategy_contract: DeclarativeStrategyContract,
    *,
    context: ForecastEvaluationContext,
) -> StrategyEvaluationResult:
    triggered_filters: List[str] = []
    for filter_rule in strategy_contract.filters:
        if filter_rule.type == "min_confidence":
            confidence = context.forecast.confidence
            if confidence is None or not math.isfinite(float(confidence)) or float(confidence) < float(filter_rule.min_confidence):
                triggered_filters.append(filter_rule.type)
        elif filter_rule.type == "prepared_input_threshold":
            scalar_value = context.prepared_inputs.scalars.get(filter_rule.key)
            numeric_value: Optional[float]
            if isinstance(scalar_value, (int, float)):
                numeric_value = float(scalar_value)
            else:
                numeric_value = None
            if numeric_value is None:
                triggered_filters.append(filter_rule.type)
            elif filter_rule.min_value is not None and numeric_value < float(filter_rule.min_value):
                triggered_filters.append(filter_rule.type)
            elif filter_rule.max_value is not None and numeric_value > float(filter_rule.max_value):
                triggered_filters.append(filter_rule.type)
        if triggered_filters:
            return StrategyEvaluationResult(
                intent=StrategyTradeIntent(
                    direction="flat",
                    size_fraction=0.0,
                    reason="blocked_by_filters",
                ),
                skipped=True,
                triggered_filters=triggered_filters,
                metadata={
                    "planned_exit_types": [exit_rule.type for exit_rule in strategy_contract.exits],
                },
            )

    signal_value = _resolve_strategy_signal_value(strategy_contract.entry.signal, context=context)
    direction = "flat"
    reason = "signal_not_actionable"
    if signal_value is not None and math.isfinite(float(signal_value)):
        if strategy_contract.entry.type == "forecast_threshold":
            if (
                strategy_contract.entry.long_above is not None
                and float(signal_value) > float(strategy_contract.entry.long_above)
            ):
                direction = "long"
                reason = "threshold_long"
            elif (
                strategy_contract.entry.short_below is not None
                and float(signal_value) < float(strategy_contract.entry.short_below)
            ):
                direction = "short"
                reason = "threshold_short"
            else:
                reason = "threshold_not_met"
        elif float(signal_value) > 0.0:
            direction = "long"
            reason = "positive_signal"
        elif float(signal_value) < 0.0:
            direction = "short"
            reason = "negative_signal"
        else:
            reason = "zero_signal"

    size_fraction = 0.0 if direction == "flat" else _resolve_size_fraction(
        strategy_contract=strategy_contract,
        context=context,
    )
    return StrategyEvaluationResult(
        intent=StrategyTradeIntent(
            direction=direction,  # type: ignore[arg-type]
            size_fraction=size_fraction,
            reason=reason,
            target_return=context.forecast.expected_return if direction != "flat" else None,
            metadata={"signal_value": signal_value},
        ),
        skipped=direction == "flat",
        metadata={
            "planned_exit_types": [exit_rule.type for exit_rule in strategy_contract.exits],
        },
    )


def _build_forecast_evaluation_context(
    *,
    execution_contract: ForecastExecutionContract,
    anchor_time: str,
    anchor_index: int,
    entry_price: float,
    forecast_values: List[float],
    realized_values: List[float],
    realized_timestamps: List[float],
    expected_return: Optional[float],
    target_value: Optional[float],
    anchor_history: Optional[Any],
) -> ForecastEvaluationContext:
    quantity = str(execution_contract.model.quantity).lower().strip()
    kind = "price_path"
    if quantity == "return":
        kind = "return_path"
    elif quantity == "volatility":
        kind = "volatility_path"
    return ForecastEvaluationContext(
        anchor=AnchorMetadata(
            anchor_time=anchor_time,
            horizon=int(execution_contract.evaluation.horizon),
            anchor_index=int(anchor_index),
            entry_price=float(entry_price) if math.isfinite(entry_price) else None,
        ),
        forecast=ForecastArtifact(
            kind=kind,  # type: ignore[arg-type]
            values=[float(value) for value in forecast_values],
            expected_return=float(expected_return) if expected_return is not None and math.isfinite(float(expected_return)) else None,
            target_value=float(target_value) if target_value is not None and math.isfinite(float(target_value)) else None,
            metadata={"method": execution_contract.model.method},
        ),
        realized=RealizedPathArtifact(
            values=[float(value) for value in realized_values],
            timestamps=[_format_time_minimal(float(ts)) for ts in realized_timestamps],
        ),
        prepared_inputs=_build_curated_prepared_inputs(
            features=execution_contract.data_preparation.features,
            anchor_history=anchor_history,
            entry_price=entry_price,
            expected_return=expected_return,
        ),
        model=execution_contract.model,
        evaluation=execution_contract.evaluation,
    )


def _build_execution_contract_metadata(
    execution_contract: ForecastExecutionContract,
) -> Dict[str, Any]:
    return {
        "data_preparation": _contract_payload(execution_contract.data_preparation),
        "model": _contract_payload(execution_contract.model),
        "evaluation": _contract_payload(execution_contract.evaluation),
        "input_mode": execution_contract.inferred_input_mode(),
    }


def _strategy_signal_label(value: float) -> str:
    if value > 0:
        return "long"
    if value < 0:
        return "short"
    return "flat"


def _build_strategy_signal_series(
    df: Any,
    *,
    strategy: str,
    position_mode: str,
    fast_period: int,
    slow_period: int,
    rsi_length: int,
    oversold: float,
    overbought: float,
) -> tuple[Any, Dict[str, Any], int]:
    close = df["close"].astype(float)
    diagnostics: Dict[str, Any] = {"fast_ma": None, "slow_ma": None, "rsi": None}

    if strategy == "sma_cross":
        fast_ma = close.rolling(window=int(fast_period), min_periods=int(fast_period)).mean()
        slow_ma = close.rolling(window=int(slow_period), min_periods=int(slow_period)).mean()
        signal = fast_ma * 0.0
        signal[:] = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))
        signal[(~np.isfinite(fast_ma)) | (~np.isfinite(slow_ma))] = np.nan
        diagnostics["fast_ma"] = fast_ma
        diagnostics["slow_ma"] = slow_ma
        warmup = int(slow_period)
    elif strategy == "ema_cross":
        fast_ma = close.ewm(span=int(fast_period), adjust=False, min_periods=int(fast_period)).mean()
        slow_ma = close.ewm(span=int(slow_period), adjust=False, min_periods=int(slow_period)).mean()
        signal = fast_ma * 0.0
        signal[:] = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))
        signal[(~np.isfinite(fast_ma)) | (~np.isfinite(slow_ma))] = np.nan
        diagnostics["fast_ma"] = fast_ma
        diagnostics["slow_ma"] = slow_ma
        warmup = int(slow_period)
    elif strategy == "rsi_reversion":
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / float(rsi_length), adjust=False, min_periods=int(rsi_length)).mean()
        avg_loss = loss.ewm(alpha=1.0 / float(rsi_length), adjust=False, min_periods=int(rsi_length)).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(avg_loss != 0.0, 100.0)
        rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
        signal = rsi * 0.0
        signal[:] = np.where(rsi < float(oversold), 1.0, np.where(rsi > float(overbought), -1.0, 0.0))
        signal[~np.isfinite(rsi)] = np.nan
        diagnostics["rsi"] = rsi
        warmup = int(rsi_length) + 1
    else:
        raise ForecastError(f"Unsupported strategy '{strategy}'")

    if position_mode == "long_only":
        signal = signal.where(~np.isfinite(signal), np.where(signal > 0.0, 1.0, 0.0))
    return signal, diagnostics, warmup


def _build_strategy_trade(
    *,
    direction: int,
    entry_idx: int,
    exit_idx: int,
    entry_time: float,
    exit_time: float,
    entry_price: float,
    exit_price: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    gross_return = float(direction) * ((float(exit_price) - float(entry_price)) / float(entry_price))
    if gross_return <= -0.999:
        gross_return = -0.999
    slip = float(abs(slippage_bps) or 0.0) / 10000.0
    net_return = gross_return - (2.0 * slip)
    if net_return <= -0.999:
        net_return = -0.999
    return {
        "direction": _strategy_signal_label(float(direction)),
        "entry_time": _format_time_minimal(float(entry_time)),
        "exit_time": _format_time_minimal(float(exit_time)),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "bars_held": int(max(1, exit_idx - entry_idx)),
        "return_gross": gross_return,
        "return_net": net_return,
    }


def strategy_backtest(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    strategy: Literal["sma_cross", "ema_cross", "rsi_reversion"] = "sma_cross",  # type: ignore
    lookback: int = 500,
    detail: CompactFullDetailLiteral = "compact",
    position_mode: Literal["long_only", "long_short"] = "long_short",  # type: ignore
    fast_period: int = 10,
    slow_period: int = 30,
    rsi_length: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
    max_hold_bars: Optional[int] = None,
    slippage_bps: float = 0.0,
) -> Dict[str, Any]:
    try:
        request_payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy,
            "lookback": lookback,
            "detail": detail,
            "position_mode": position_mode,
            "fast_period": fast_period,
            "slow_period": slow_period,
            "rsi_length": rsi_length,
            "oversold": oversold,
            "overbought": overbought,
            "max_hold_bars": max_hold_bars,
            "slippage_bps": slippage_bps,
        }
        strategy_value = str(strategy or "sma_cross").strip().lower()
        if strategy_value not in {"sma_cross", "ema_cross", "rsi_reversion"}:
            return {"error": "strategy must be one of: sma_cross, ema_cross, rsi_reversion"}
        position_mode_value = str(position_mode or "long_short").strip().lower()
        if position_mode_value not in {"long_only", "long_short"}:
            return {"error": "position_mode must be 'long_only' or 'long_short'"}
        detail_mode = _normalize_detail_mode(detail)
        if str(detail or "compact").strip().lower() not in {"compact", "full"}:
            return {"error": "detail must be 'compact' or 'full'"}
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        if int(lookback) < 5:
            return {"error": "lookback must be at least 5"}
        if int(fast_period) >= int(slow_period):
            return {"error": "fast_period must be less than slow_period"}
        if float(oversold) >= float(overbought):
            return {"error": "oversold must be less than overbought"}

        if strategy_value in {"sma_cross", "ema_cross"}:
            warmup_bars = max(int(slow_period), 5)
        else:
            warmup_bars = max(int(rsi_length) + 1, 5)
        need = int(lookback) + int(warmup_bars) + 5
        try:
            df = _fetch_history(symbol, timeframe, int(need), as_of=None)
        except Exception as ex:
            return {"error": str(ex)}
        if len(df) < max(int(lookback), warmup_bars + 5):
            return {"error": "Not enough closed bars for strategy backtest"}

        signal_series, diagnostics, signal_warmup = _build_strategy_signal_series(
            df,
            strategy=strategy_value,
            position_mode=position_mode_value,
            fast_period=int(fast_period),
            slow_period=int(slow_period),
            rsi_length=int(rsi_length),
            oversold=float(oversold),
            overbought=float(overbought),
        )

        start_signal_idx = max(int(signal_warmup), len(df) - int(lookback))
        times = df["time"].astype(float).to_numpy()
        opens = df["open"].astype(float).to_numpy()
        closes = df["close"].astype(float).to_numpy()
        signals = signal_series.to_numpy(dtype=float)

        trades: List[Dict[str, Any]] = []
        current_direction = 0
        entry_idx = None
        entry_time = None
        entry_price = None

        def _execution_price(bar_idx: int) -> Optional[float]:
            if bar_idx < 0 or bar_idx >= len(opens):
                return None
            open_price = float(opens[bar_idx])
            if math.isfinite(open_price) and open_price > 0.0:
                return open_price
            close_price = float(closes[bar_idx])
            if math.isfinite(close_price) and close_price > 0.0:
                return close_price
            return None

        for signal_idx in range(int(start_signal_idx), len(df) - 1):
            raw_signal = float(signals[signal_idx]) if math.isfinite(float(signals[signal_idx])) else 0.0
            desired_direction = int(np.sign(raw_signal))
            action_idx = int(signal_idx + 1)
            action_price = _execution_price(action_idx)
            if action_price is None:
                continue

            if current_direction == 0:
                if desired_direction != 0:
                    current_direction = desired_direction
                    entry_idx = action_idx
                    entry_time = float(times[action_idx])
                    entry_price = float(action_price)
                continue

            assert entry_idx is not None and entry_time is not None and entry_price is not None
            bars_held = int(action_idx - entry_idx)
            hit_max_hold = max_hold_bars is not None and bars_held >= int(max_hold_bars)
            if desired_direction == current_direction and not hit_max_hold:
                continue

            trades.append(
                _build_strategy_trade(
                    direction=current_direction,
                    entry_idx=int(entry_idx),
                    exit_idx=action_idx,
                    entry_time=float(entry_time),
                    exit_time=float(times[action_idx]),
                    entry_price=float(entry_price),
                    exit_price=float(action_price),
                    slippage_bps=float(slippage_bps),
                )
            )
            current_direction = 0
            entry_idx = None
            entry_time = None
            entry_price = None

            if desired_direction != 0:
                current_direction = desired_direction
                entry_idx = action_idx
                entry_time = float(times[action_idx])
                entry_price = float(action_price)

        if current_direction != 0 and entry_idx is not None and entry_time is not None and entry_price is not None:
            final_exit_idx = len(df) - 1
            final_exit_price = float(closes[final_exit_idx])
            if not math.isfinite(final_exit_price) or final_exit_price <= 0.0:
                final_exit_price = _execution_price(final_exit_idx) or float(entry_price)
            trades.append(
                _build_strategy_trade(
                    direction=current_direction,
                    entry_idx=int(entry_idx),
                    exit_idx=int(final_exit_idx),
                    entry_time=float(entry_time),
                    exit_time=float(times[final_exit_idx]),
                    entry_price=float(entry_price),
                    exit_price=float(final_exit_price),
                    slippage_bps=float(slippage_bps),
                )
            )

        trade_returns = [float(trade["return_net"]) for trade in trades if trade.get("return_net") is not None]
        entry_indices = []
        for trade in trades:
            entry_time_text = str(trade.get("entry_time") or "")
            try:
                entry_idx = next(i for i, ts in enumerate(times) if _format_time_minimal(float(ts)) == entry_time_text)
            except Exception:
                entry_idx = None
            if entry_idx is not None:
                entry_indices.append(entry_idx)
        trade_spacing = None
        if len(entry_indices) > 1:
            trade_spacing = int(np.median(np.diff(entry_indices)))

        metrics = _compute_performance_metrics(
            trade_returns,
            timeframe,
            1,
            float(slippage_bps),
            trade_spacing_bars=trade_spacing,
        ) if trade_returns else {}
        if detail_mode == "compact" and metrics:
            metrics = _compact_metrics_payload(metrics)

        gross_equity = np.cumprod([1.0 + float(trade["return_gross"]) for trade in trades]) if trades else np.array([1.0])
        net_equity = np.cumprod([1.0 + float(trade["return_net"]) for trade in trades]) if trades else np.array([1.0])
        long_trades = int(sum(1 for trade in trades if trade.get("direction") == "long"))
        short_trades = int(sum(1 for trade in trades if trade.get("direction") == "short"))
        last_idx = len(df) - 1
        last_signal_value = float(signals[last_idx]) if math.isfinite(float(signals[last_idx])) else 0.0

        data_contract = DataPreparationContract(
            symbol=symbol,
            timeframe=timeframe,
            lookback=int(need),
        )
        evaluation_contract = BacktestEvaluationContract(
            horizon=1,
            steps=1,
            spacing=1,
            slippage_bps=float(slippage_bps),
            detail=detail_mode,
        )

        _strategy_params: Dict[str, Any] = {
            "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
        }
        if strategy_value in {"sma_cross", "ema_cross"}:
            _strategy_params["fast_period"] = int(fast_period)
            _strategy_params["slow_period"] = int(slow_period)
        if strategy_value == "rsi_reversion":
            _strategy_params["rsi_length"] = int(rsi_length)
            _strategy_params["oversold"] = float(oversold)
            _strategy_params["overbought"] = float(overbought)
        _params: Dict[str, Any] = {
            "lookback": int(lookback),
            "slippage_bps": float(slippage_bps),
            **_strategy_params,
        }

        result: Dict[str, Any] = {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": strategy_value,
            "detail": detail_mode,
            "position_mode": position_mode_value,
            "parameters": _params,
            "summary": {
                "bars_used": int(lookback),
                "warmup_bars": int(signal_warmup),
                "num_trades": int(len(trades)),
                "long_trades": long_trades,
                "short_trades": short_trades,
                "gross_return": float(gross_equity[-1] - 1.0),
                "net_return": float(net_equity[-1] - 1.0),
            },
            "metrics": metrics,
            "last_signal": {
                "signal": _strategy_signal_label(last_signal_value),
                "close": float(closes[last_idx]),
                "fast_ma": float(diagnostics["fast_ma"].iloc[last_idx]) if diagnostics.get("fast_ma") is not None and np.isfinite(float(diagnostics["fast_ma"].iloc[last_idx])) else None,
                "slow_ma": float(diagnostics["slow_ma"].iloc[last_idx]) if diagnostics.get("slow_ma") is not None and np.isfinite(float(diagnostics["slow_ma"].iloc[last_idx])) else None,
                "rsi": float(diagnostics["rsi"].iloc[last_idx]) if diagnostics.get("rsi") is not None and np.isfinite(float(diagnostics["rsi"].iloc[last_idx])) else None,
                "time": _format_time_minimal(float(times[last_idx])),
            },
        }
        if detail_mode == "full":
            result["contracts"] = {
                "data_preparation": _contract_payload(data_contract),
                "evaluation": _contract_payload(evaluation_contract),
                "strategy": {
                    "kind": "legacy_indicator_strategy",
                    "name": strategy_value,
                    "position_mode": position_mode_value,
                    "parameters": dict(_strategy_params),
                },
            }
        if trades:
            if detail_mode == "full":
                result["trades"] = trades
                # Add enriched detail for full mode: equity curve, drawdowns, monthly breakdown, trade distribution
                
                # Build equity curve with timestamps
                equity_curve = []
                cumulative_net = 1.0
                trade_exit_times = {}
                for i, trade in enumerate(trades):
                    exit_time_str = str(trade.get("exit_time") or "")
                    if exit_time_str:
                        try:
                            exit_idx = next(j for j, ts in enumerate(times) if _format_time_minimal(float(ts)) == exit_time_str)
                            trade_exit_times[exit_idx] = i
                        except Exception:
                            pass
                
                for idx in sorted(trade_exit_times.keys()):
                    trade_idx = trade_exit_times[idx]
                    trade_net_return = float(trades[trade_idx].get("return_net") or 0.0)
                    cumulative_net *= (1.0 + trade_net_return)
                    equity_curve.append({
                        "time": _format_time_minimal(float(times[idx])),
                        "equity": cumulative_net,
                    })
                
                if equity_curve:
                    result["equity_curve"] = equity_curve
                
                # Calculate drawdown periods
                drawdown_periods = []
                if equity_curve and len(equity_curve) > 1:
                    peak_equity = 1.0
                    for point in equity_curve:
                        if point["equity"] > peak_equity:
                            peak_equity = point["equity"]
                    
                    current_peak = 1.0
                    current_peak_time = equity_curve[0]["time"] if equity_curve else None
                    for point in equity_curve:
                        if point["equity"] > current_peak:
                            current_peak = point["equity"]
                            current_peak_time = point["time"]
                        else:
                            dd_depth = (point["equity"] - current_peak) / max(current_peak, 1e-10)
                            if dd_depth < -0.0001:  # Only report material drawdowns
                                drawdown_periods.append({
                                    "start": current_peak_time,
                                    "end": point["time"],
                                    "depth": dd_depth,
                                })
                
                if drawdown_periods:
                    result["drawdown_periods"] = drawdown_periods
                
                # Monthly breakdown
                monthly_stats = {}
                for trade in trades:
                    exit_time_str = str(trade.get("exit_time") or "")
                    if exit_time_str and len(exit_time_str) >= 7:
                        month_key = exit_time_str[:7]  # "2026-03" format
                        if month_key not in monthly_stats:
                            monthly_stats[month_key] = {
                                "trades": 0,
                                "winning": 0,
                                "losing": 0,
                                "returns": [],
                            }
                        monthly_stats[month_key]["trades"] += 1
                        ret = float(trade.get("return_net") or 0.0)
                        monthly_stats[month_key]["returns"].append(ret)
                        if ret > 0:
                            monthly_stats[month_key]["winning"] += 1
                        elif ret < 0:
                            monthly_stats[month_key]["losing"] += 1
                
                monthly_breakdown = []
                for month_key in sorted(monthly_stats.keys()):
                    stats = monthly_stats[month_key]
                    month_return = float(np.prod([1.0 + r for r in stats["returns"]]) - 1.0) if stats["returns"] else 0.0
                    monthly_breakdown.append({
                        "month": month_key,
                        "return": month_return,
                        "trades": stats["trades"],
                        "winning": stats["winning"],
                        "losing": stats["losing"],
                    })
                
                if monthly_breakdown:
                    result["monthly_breakdown"] = monthly_breakdown
                
                # Trade distribution statistics
                if trades:
                    winning_trades = [t for t in trades if float(t.get("return_net") or 0.0) > 0.0]
                    losing_trades = [t for t in trades if float(t.get("return_net") or 0.0) < 0.0]
                    breakeven_trades = [t for t in trades if float(t.get("return_net") or 0.0) == 0.0]
                    
                    trade_distribution = {}
                    
                    if winning_trades:
                        winning_returns = [float(t.get("return_net") or 0.0) for t in winning_trades]
                        trade_distribution["winning"] = {
                            "count": len(winning_trades),
                            "avg_return": float(np.mean(winning_returns)),
                            "max": float(np.max(winning_returns)),
                            "min": float(np.min(winning_returns)),
                        }
                    
                    if losing_trades:
                        losing_returns = [float(t.get("return_net") or 0.0) for t in losing_trades]
                        trade_distribution["losing"] = {
                            "count": len(losing_trades),
                            "avg_return": float(np.mean(losing_returns)),
                            "max": float(np.max(losing_returns)),
                            "min": float(np.min(losing_returns)),
                        }
                    
                    if breakeven_trades:
                        trade_distribution["breakeven"] = {
                            "count": len(breakeven_trades),
                        }
                    
                    if trade_distribution:
                        result["trade_distribution"] = trade_distribution
        else:
            result["no_action"] = True
            result["message"] = "The strategy generated no trades on the requested history."
        return _attach_request_metadata(
            result,
            request=request_payload,
            resolved_request={
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": strategy_value,
                "lookback": int(lookback),
                "detail": detail_mode,
                "position_mode": position_mode_value,
                "fast_period": int(fast_period),
                "slow_period": int(slow_period),
                "rsi_length": int(rsi_length),
                "oversold": float(oversold),
                "overbought": float(overbought),
                "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
                "slippage_bps": float(slippage_bps),
            },
            detail=detail_mode,
        )
    except Exception as e:
        return {"error": f"Error in strategy_backtest: {str(e)}"}


execute_strategy_backtest = strategy_backtest


def forecast_backtest(  # noqa: C901
    symbol: str,
    timeframe: TimeframeLiteral = "H1",
    horizon: int = 12,
    steps: int = 5,
    spacing: int = 20,
    methods: Optional[List[str]] = None,
    params_per_method: Optional[Dict[str, Any]] = None,
    quantity: Literal['price','return','volatility'] = 'price',  # type: ignore
    denoise: Optional[DenoiseSpec] = None,
    anchors: Optional[List[str]] = None,
    # Unified per-run tuning applied to all methods (unless overridden in params_per_method)
    params: Optional[Dict[str, Any]] = None,
    # Feature engineering for exogenous/multivariate models
    features: Optional[Dict[str, Any]] = None,
    dimred_method: Optional[str] = None,
    dimred_params: Optional[Dict[str, Any]] = None,
    slippage_bps: float = 0.0,
    trade_threshold: float = 0.0,
    detail: CompactFullDetailLiteral = "compact",
) -> Dict[str, Any]:
    """Rolling-origin backtest over historical anchors using the forecast tool.

    Parameters: symbol, timeframe, horizon, steps, spacing, methods?, params_per_method?, quantity, denoise?
    - Picks `steps` anchor points spaced `spacing` bars apart, each with `horizon` future bars for validation.
    - For each method, runs our `forecast` as-of that anchor and reports MAE/RMSE/directional accuracy.
    """
    try:
        request_payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": horizon,
            "steps": steps,
            "spacing": spacing,
            "methods": methods,
            "params_per_method": params_per_method,
            "quantity": quantity,
            "denoise": denoise,
            "anchors": anchors,
            "params": params,
            "features": features,
            "dimred_method": dimred_method,
            "dimred_params": dimred_params,
            "slippage_bps": slippage_bps,
            "trade_threshold": trade_threshold,
            "detail": detail,
        }
        __stage = 'start'
        detail_mode = _normalize_detail_mode(detail)
        include_paths = detail_mode == "full"
        if timeframe not in TIMEFRAME_MAP:
            return {"error": invalid_timeframe_error(timeframe, TIMEFRAME_MAP)}
        if (
            not anchors
            and int(steps) > 1
            and int(spacing) < int(horizon)
        ):
            return {
                "error": "spacing must be greater than or equal to horizon when steps > 1"
            }

        # Fetch sufficient history via shared helper; ensure enough bars for anchors
        if anchors and isinstance(anchors, (list, tuple)) and len(anchors) > 0:
            need = int(len(anchors)) * int(horizon) + 600
        else:
            need = int(steps) * int(spacing) + int(horizon) + 400
        try:
            df = _fetch_history(symbol, timeframe, int(need), as_of=None)
        except Exception as ex:
            return {"error": str(ex)}
        if len(df) < (int(horizon) + 50):
            return {"error": "Not enough closed bars for backtest"}

        # Determine anchor indices (explicit anchors or rolling from end)
        total = len(df)
        anchor_indices: List[int] = []
        if anchors and isinstance(anchors, (list, tuple)) and len(anchors) > 0:
            tvals = df['time'].astype(float).to_numpy()
            tstr = [_format_time_minimal(ts) for ts in tvals]
            idx_by_time = {s: i for i, s in enumerate(tstr)}
            for s in anchors:
                i = idx_by_time.get(str(s).strip())
                if i is not None and (i + int(horizon)) < total:
                    anchor_indices.append(int(i))
        else:
            pos = total - int(horizon) - 1
            for _ in range(int(steps)):
                if pos <= 1:
                    break
                anchor_indices.append(int(pos))
                pos -= int(spacing)
            anchor_indices = list(reversed(anchor_indices))
        if not anchor_indices:
            return {"error": "Failed to determine backtest anchors"}
        if len(anchor_indices) > 1:
            sorted_anchor_indices = sorted(anchor_indices)
            for prev_idx, curr_idx in zip(sorted_anchor_indices, sorted_anchor_indices[1:]):
                if (curr_idx - prev_idx) < int(horizon):
                    prev_anchor = _format_time_minimal(float(df['time'].iloc[prev_idx]))
                    curr_anchor = _format_time_minimal(float(df['time'].iloc[curr_idx]))
                    return {
                        "error": (
                            "Explicit backtest anchors must be at least horizon bars apart to prevent "
                            f"data leakage: {prev_anchor} -> {curr_anchor}"
                        )
                    }

        # Normalize methods input (allow comma or whitespace separated string)
        if isinstance(methods, str):
            txt = methods.strip()
            if "," in txt:
                methods = [s.strip() for s in txt.split(",") if s.strip()]
            else:
                methods = [s for s in txt.split() if s]

        # Default methods based on quantity
        if not methods:
            if quantity == 'volatility':
                methods = ['ewma', 'parkinson']
            else:
                methods_info = _get_forecast_methods_data_safe()
                avail = [m['method'] for m in methods_info.get('methods', []) if m.get('available')]
                preferred = ['naive', 'drift', 'seasonal_naive', 'theta', 'fourier_ols', 'sf_autoarima', 'sf_theta']
                methods = [m for m in preferred if m in avail]
                if not methods:
                    methods = [m for m in ('naive', 'drift', 'theta') if m in avail]
        params_map = dict(params_per_method or {})
        target_mode = _quantity_to_target(quantity)

        # Build ground-truth windows for each anchor
        closes = df['close'].astype(float).to_numpy()
        times = df['time'].astype(float).to_numpy()
        actual_windows: Dict[int, Tuple[List[float], List[float]]] = {}
        for idx in anchor_indices:
            if idx + int(horizon) >= len(closes):
                continue
            if target_mode == 'return' and quantity != 'volatility':
                prev = np.maximum(closes[idx: idx + int(horizon)], 1e-12)
                nxt = np.maximum(closes[idx + 1: idx + 1 + int(horizon)], 1e-12)
                with np.errstate(divide='ignore', invalid='ignore'):
                    actual = np.log(nxt / prev).tolist()
            else:
                actual = closes[idx + 1: idx + 1 + int(horizon)].tolist()
            ts = times[idx + 1: idx + 1 + int(horizon)].tolist()
            if len(actual) != int(horizon) or len(ts) != int(horizon):
                continue
            actual_windows[idx] = (actual, ts)
        if not actual_windows:
            return {"error": "No valid validation windows found"}

        # Normalize denoise spec once for the whole run (uniform across methods)
        try:
            _dn_used = _normalize_denoise_spec(denoise, default_when='pre_ti') if denoise is not None else None
        except Exception:
            _dn_used = None

        data_contract = DataPreparationContract(
            symbol=symbol,
            timeframe=timeframe,
            lookback=int(need),
            denoise=_dn_used,
            features=dict(features) if isinstance(features, dict) else features,
            dimred_method=dimred_method,
            dimred_params=dict(dimred_params) if isinstance(dimred_params, dict) else dimred_params,
        )
        evaluation_contract = BacktestEvaluationContract(
            horizon=int(horizon),
            steps=int(steps),
            spacing=int(spacing),
            anchors=list(anchors) if isinstance(anchors, (list, tuple)) else None,
            slippage_bps=float(slippage_bps),
            detail=detail_mode,
        )
        strategy_contract = (
            _build_forecast_threshold_strategy_contract(float(trade_threshold or 0.0))
            if quantity != "volatility"
            else None
        )

        # Run forecasts per method and compute metrics
        results: Dict[str, Any] = {}
        contract_methods: Dict[str, Any] = {}
        for method in methods:
            per_anchor = []
            execution_contract: Optional[ForecastExecutionContract] = None
            for idx in anchor_indices:
                if idx not in actual_windows:
                    continue
                anchor_time = _format_time_minimal(times[idx])
                truth, ts = actual_windows[idx]
                try:
                    if quantity == 'volatility':
                        # Volatility forecast: allow proxy in params map (params_map[method].get('proxy'))
                        pm_raw = params_map.get(method) or {}
                        pm = dict(pm_raw) if isinstance(pm_raw, dict) else {}
                        proxy = pm.pop('proxy', None) if isinstance(pm, dict) else None
                        if execution_contract is None:
                            execution_contract = ForecastExecutionContract(
                                data_preparation=data_contract,
                                model=ForecastModelContract(
                                    method=method,
                                    params=pm if isinstance(pm, dict) else None,
                                    quantity=quantity,
                                ),
                                evaluation=evaluation_contract,
                            )
                        r = raise_if_error_result(forecast_volatility(  # type: ignore
                            symbol=symbol,
                            timeframe=timeframe,
                            method=method,  # type: ignore
                            horizon=int(horizon),
                            as_of=anchor_time,
                            params=pm if isinstance(pm, dict) else None,
                            proxy=proxy,  # type: ignore
                            denoise=_dn_used,
                        ))
                    else:
                        # Choose per-method params falling back to global params
                        pm = params_map.get(method)
                        if pm is None:
                            pm = params
                        anchor_history = df.iloc[: idx + 1].copy()
                        if execution_contract is None:
                            execution_contract = ForecastExecutionContract(
                                data_preparation=data_contract,
                                model=ForecastModelContract(
                                    method=method,
                                    params=pm if isinstance(pm, dict) else None,
                                    quantity=quantity,
                                ),
                                evaluation=evaluation_contract,
                            )
                        r = raise_if_error_result(forecast(
                            symbol=symbol,
                            timeframe=timeframe,
                            method=method,  # type: ignore[arg-type]
                            horizon=int(horizon),
                            as_of=anchor_time,
                            params=pm,
                            quantity=quantity,  # type: ignore[arg-type]
                            denoise=_dn_used,
                            features=features,
                            dimred_method=dimred_method,
                            dimred_params=dimred_params,
                            prefetched_df=anchor_history,
                        ))
                except Exception as ex:
                    per_anchor.append({"anchor": anchor_time, "success": False, "error": str(ex)})
                    continue
                if quantity == 'volatility':
                    # Compute realized horizon sigma from the anchor close through the future path.
                    act = np.array(truth, dtype=float)
                    path = np.concatenate(([float(closes[idx])], act)) if act.size > 0 else act
                    r_act = _log_returns_from_prices(path) if path.size >= 2 else np.array([], dtype=float)
                    realized_sigma = (
                        float(np.sqrt(np.sum(np.square(np.clip(r_act, -1e6, 1e6)))))
                        if r_act.size > 0
                        else float('nan')
                    )
                    pred_sigma = float(r.get('horizon_sigma_return', float('nan')))
                    mae = float(abs(pred_sigma - realized_sigma)) if np.isfinite(pred_sigma) and np.isfinite(realized_sigma) else float('nan')
                    rmse = mae
                    per_anchor.append({
                        "anchor": anchor_time,
                        "success": np.isfinite(pred_sigma) and np.isfinite(realized_sigma),
                        "mae": mae,
                        "rmse": rmse,
                        "forecast_sigma": pred_sigma,
                        "realized_sigma": realized_sigma,
                    })
                else:
                    if target_mode == 'return':
                        fc = r.get('forecast_return')
                        if not fc:
                            per_anchor.append({"anchor": anchor_time, "success": False, "error": "Missing forecast_return for return mode"})
                            continue
                    else:
                        fc = r.get('forecast_price')
                    if not fc:
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "Empty forecast"})
                        continue
                    fcv = np.array(fc, dtype=float)
                    act = np.array(truth, dtype=float)
                    m = min(len(fcv), len(act))
                    if m <= 0:
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "No overlap"})
                        continue
                    if not np.all(np.isfinite(fcv[:m])):
                        per_anchor.append({"anchor": anchor_time, "success": False, "error": "Non-finite forecast values"})
                        continue
                    mae = float(np.mean(np.abs(fcv[:m] - act[:m])))
                    rmse = float(np.sqrt(np.mean((fcv[:m] - act[:m])**2)))
                    if target_mode == 'return':
                        # Return mode: DA = did forecast correctly predict the sign of each bar's return?
                        da = float(np.mean(np.sign(fcv[:m]) == np.sign(act[:m]))) if m > 0 else float('nan')
                    elif m > 1:
                        da = float(np.mean(np.sign(np.diff(fcv[:m])) == np.sign(np.diff(act[:m]))))
                    else:
                        da = float('nan')
                    entry_price = float(closes[idx]) if idx < len(closes) else float('nan')
                    if target_mode == 'return':
                        expected_move = float(np.nansum(fcv[:m]))
                    else:
                        expected_move = float((float(fcv[m - 1]) - entry_price)) if math.isfinite(entry_price) else float('nan')
                    expected_return = float('nan')
                    if target_mode == 'return':
                        try:
                            expected_return = float(math.exp(expected_move) - 1.0)
                        except Exception:
                            expected_return = float('nan')
                    elif math.isfinite(entry_price) and entry_price != 0.0:
                        expected_return = expected_move / entry_price
                    evaluation_context = _build_forecast_evaluation_context(
                        execution_contract=execution_contract if execution_contract is not None else ForecastExecutionContract(
                            data_preparation=data_contract,
                            model=ForecastModelContract(method=method, quantity=quantity),
                            evaluation=evaluation_contract,
                        ),
                        anchor_time=anchor_time,
                        anchor_index=int(idx),
                        entry_price=entry_price,
                        forecast_values=[float(v) for v in fcv[:m].tolist()],
                        realized_values=[float(v) for v in act[:m].tolist()],
                        realized_timestamps=ts[:m],
                        expected_return=expected_return if math.isfinite(expected_return) else None,
                        target_value=float(np.nansum(fcv[:m])) if target_mode == 'return' else float(fcv[m - 1]),
                        anchor_history=anchor_history,
                    )
                    strategy_eval = _evaluate_forecast_strategy(
                        strategy_contract if strategy_contract is not None else _build_forecast_threshold_strategy_contract(0.0),
                        context=evaluation_context,
                    )
                    intent_direction = str(strategy_eval.intent.direction)
                    direction = 0
                    if intent_direction == "long":
                        direction = 1
                    elif intent_direction == "short":
                        direction = -1
                    position = intent_direction
                    gross_return = float('nan')
                    net_return = float('nan')
                    exit_price = float('nan')
                    exit_step = m - 1
                    if direction != 0:
                        if target_mode == 'return':
                            try:
                                realized_path = np.array(act[:m], dtype=float)
                                if not np.all(np.isfinite(realized_path)):
                                    realized_path = np.nan_to_num(realized_path, nan=0.0, posinf=0.0, neginf=0.0)
                                cum_log = np.cumsum(realized_path)
                                forecast_target_log = float(np.nansum(fcv[:m]))
                                if math.isfinite(forecast_target_log) and abs(forecast_target_log) > 0:
                                    if direction > 0:
                                        hit_idx = np.where(cum_log >= forecast_target_log)[0]
                                    else:
                                        hit_idx = np.where(cum_log <= forecast_target_log)[0]
                                    if hit_idx.size > 0:
                                        exit_step = int(hit_idx[0])
                                realized_log = float(cum_log[exit_step]) if cum_log.size else 0.0
                                gross_return = direction * float(math.exp(realized_log) - 1.0)
                                exit_idx = idx + exit_step + 1
                                exit_price = float(closes[exit_idx]) if exit_idx < len(closes) else float('nan')
                            except Exception:
                                gross_return = float('nan')
                            slip = float(abs(slippage_bps) or 0.0) / 10000.0
                            net_return = gross_return - 2.0 * slip
                            if net_return <= -0.999:
                                net_return = -0.999
                        elif math.isfinite(entry_price) and entry_price != 0.0:
                            try:
                                forecast_target_price = float(fcv[m - 1])
                                realized_prices = np.array(act[:m], dtype=float)
                                if math.isfinite(forecast_target_price):
                                    if direction > 0:
                                        hit_idx = np.where(realized_prices >= forecast_target_price)[0]
                                    else:
                                        hit_idx = np.where(realized_prices <= forecast_target_price)[0]
                                    if hit_idx.size > 0:
                                        exit_step = int(hit_idx[0])
                                exit_price = float(realized_prices[exit_step]) if realized_prices.size else float('nan')
                            except Exception:
                                exit_price = float('nan')
                            if math.isfinite(exit_price):
                                gross_return = direction * ((exit_price - entry_price) / entry_price)
                                slip = float(abs(slippage_bps) or 0.0) / 10000.0
                                net_return = gross_return - 2.0 * slip
                                if net_return <= -0.999:
                                    net_return = -0.999
                    elif direction == 0:
                        gross_return = 0.0
                        net_return = 0.0
                    detail_row = {
                        "anchor": anchor_time,
                        "success": True,
                        "mae": mae,
                        "rmse": rmse,
                        "directional_accuracy": da,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "exit_step": int(exit_step) + 1 if m > 0 else 0,
                        "expected_return": expected_return,
                        "position": position,
                        "trade_return_gross": gross_return,
                        "trade_return": net_return,
                    }
                    if include_paths:
                        detail_row["strategy_intent"] = strategy_eval.intent.model_dump(exclude_none=True)
                        detail_row["strategy_context"] = {
                            "top_level_keys": list(evaluation_context.top_level_context_keys()),
                            "visible_inputs": evaluation_context.visible_prepared_input_names(),
                        }
                    if include_paths:
                        detail_row["forecast"] = [float(v) for v in fcv[:m].tolist()]
                        detail_row["actual"] = [float(v) for v in act[:m].tolist()]
                    else:
                        detail_row["horizon_used"] = int(m)
                        detail_row["forecast_end"] = float(fcv[m - 1]) if m > 0 else None
                        detail_row["actual_end"] = float(act[m - 1]) if m > 0 else None
                    per_anchor.append(detail_row)
            if execution_contract is not None and detail_mode == "full":
                contract_methods[method] = _build_execution_contract_metadata(execution_contract)
            # Aggregate
            ok = [x for x in per_anchor if x.get('success')]
            if ok:
                agg = {
                    "success": True,
                    "avg_mae": float(np.mean([x['mae'] for x in ok])),
                    "avg_rmse": float(np.mean([x['rmse'] for x in ok])),
                    "successful_tests": len(ok),
                    "num_tests": len(per_anchor),
                    "details": per_anchor,
                }
                if quantity != 'volatility':
                    da_vals = [x.get('directional_accuracy') for x in ok]
                    da_vals = [v for v in da_vals if v is not None and np.isfinite(v)]
                    if da_vals:
                        agg["avg_directional_accuracy"] = float(np.mean(da_vals))
                    # Exclude flat positions from trade metrics
                    trade_returns = [
                        float(x['trade_return']) for x in ok
                        if x.get('trade_return') is not None
                        and x.get('position') != 'flat'
                        and np.isfinite(x['trade_return'])
                    ]
                    # Compute actual trade spacing from anchor indices
                    _spacing: Optional[int] = None
                    if len(anchor_indices) > 1:
                        _diffs = [anchor_indices[i + 1] - anchor_indices[i] for i in range(len(anchor_indices) - 1)]
                        _spacing = int(np.median(_diffs))
                    metrics = _compute_performance_metrics(
                        trade_returns, timeframe, int(horizon), float(slippage_bps),
                        trade_spacing_bars=_spacing,
                    ) if trade_returns else {}
                    if metrics:
                        if detail_mode == "compact":
                            metrics = _compact_metrics_payload(metrics)
                    _attach_metrics_status(
                        agg,
                        metrics=metrics,
                        slippage_bps=float(slippage_bps),
                        unavailable_reason="no_non_flat_trades",
                    )
                else:
                    _attach_metrics_status(
                        agg,
                        metrics={},
                        slippage_bps=float(slippage_bps),
                        unavailable_reason="not_applicable_for_volatility",
                    )
                if _dn_used:
                    agg["denoise_used"] = _dn_used
                results[method] = agg
            else:
                results[method] = {
                    "success": False,
                    "successful_tests": 0,
                    "num_tests": len(per_anchor),
                    "details": per_anchor,
                    "slippage_bps": float(slippage_bps),
                    "metrics": _unavailable_performance_metrics(
                        "no_successful_tests",
                        float(slippage_bps),
                    ),
                    "metrics_available": False,
                    "metrics_reason": "no_successful_tests",
                }

        result_payload = {
            "success": True,
            "slippage_bps": float(slippage_bps),
            "trade_threshold": float(trade_threshold or 0.0),
            "detail": detail_mode,
            "results": results,
        }
        if detail_mode == "full":
            result_payload["contracts"] = {
                "data_preparation": _contract_payload(data_contract),
                "evaluation": _contract_payload(evaluation_contract),
                "strategy": (
                    _contract_payload(strategy_contract)
                    if strategy_contract is not None
                    else {"kind": "volatility_evaluation", "trade_generation": False}
                ),
                "methods": contract_methods,
            }
        return _attach_request_metadata(
            result_payload,
            request=request_payload,
            resolved_request={
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon": int(horizon),
                "steps": int(steps),
                "spacing": int(spacing),
                "methods": list(methods or []),
                "params_per_method": dict(params_map),
                "quantity": quantity,
                "denoise": _dn_used,
                "anchors": list(anchors) if isinstance(anchors, (list, tuple)) else anchors,
                "params": dict(params) if isinstance(params, dict) else params,
                "features": dict(features) if isinstance(features, dict) else features,
                "dimred_method": dimred_method,
                "dimred_params": dict(dimred_params) if isinstance(dimred_params, dict) else dimred_params,
                "slippage_bps": float(slippage_bps),
                "trade_threshold": float(trade_threshold or 0.0),
                "detail": detail_mode,
            },
            detail=detail_mode,
        )
    except Exception as e:
        return {"error": f"Error in forecast_backtest: {str(e)}"}


def execute_forecast_backtest(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Internal backtest entrypoint that raises typed errors for result payload failures."""
    try:
        return raise_if_error_result(forecast_backtest(*args, **kwargs))
    except ForecastError:
        raise
    except Exception as exc:
        raise ForecastError(str(exc)) from exc
