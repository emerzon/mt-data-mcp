from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ...shared.schema import DenoiseSpec
from ..report.utils import (
    adapt_forecast_payload_for_report,
    now_utc_iso,
    parse_table_tail,
    report_section_enabled,
    resolve_report_context_indicators,
)
from .basic import _TREND_COMPACT_LEGEND, _compute_compact_trend, _get_raw_result

_MINIMAL_SKIPPED_SECTIONS = (
    "pivot",
    "contexts_multi",
    "pivot_multi",
    "volatility",
    "backtest",
    "barriers",
    "patterns",
)


def _iter_requested_methods(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        for token in value.split(","):
            text = str(token or "").strip()
            if text:
                yield text
        return
    if isinstance(value, (list, tuple)):
        for token in value:
            text = str(token or "").strip()
            if text:
                yield text


def _resolve_minimal_forecast_method(params: Dict[str, Any]) -> str:
    direct_method = str(params.get("method") or "").strip()
    if direct_method:
        return direct_method
    for method_name in _iter_requested_methods(params.get("methods")):
        return method_name
    return "theta"


def template_minimal(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    p = dict(params or {})
    tf = str(p.get("timeframe", "H1"))
    start = p.get("start")
    end = p.get("end")
    forecast_method = _resolve_minimal_forecast_method(p)
    forecast_library = str(p.get("library") or "native").strip() or "native"

    report: Dict[str, Any] = {
        "meta": {
            "symbol": symbol,
            "timeframe": tf,
            "horizon": int(horizon),
            "template": "minimal",
            "generated_at": now_utc_iso(),
            "fast_path": True,
            "skipped_sections": list(_MINIMAL_SKIPPED_SECTIONS),
        },
        "sections": {},
    }

    indicators = resolve_report_context_indicators(
        p,
        default="ema(20),ema(50),rsi(14),macd(12,26,9)",
    )
    from ..data import data_fetch_candles

    ctx = (
        _get_raw_result(
            data_fetch_candles,
            symbol=symbol,
            timeframe=tf,
            limit=int(p.get("context_limit", 200)),
            start=start,
            end=end,
            indicators=indicators,  # type: ignore[arg-type]
            denoise=denoise,
            simplify={"mode": "select", "method": "lttb", "ratio": 0.2},  # type: ignore[arg-type]
        )
        if report_section_enabled(p, "context")
        else {"error": "context section not requested"}
    )

    if "error" in ctx:
        report["sections"]["context"] = {"error": ctx["error"]}
    else:
        tail_n = int(p.get("context_tail", 40))
        tail_rows = parse_table_tail(ctx, tail=tail_n)
        if not tail_rows:
            if isinstance(ctx, dict) and isinstance(ctx.get("bars"), list):
                tail_rows = ctx.get("bars")[-tail_n:]  # type: ignore[index]
            elif isinstance(ctx, dict) and isinstance(ctx.get("data"), list):
                tail_rows = ctx.get("data")[-tail_n:]  # type: ignore[index]
            elif isinstance(ctx, list):
                tail_rows = ctx
            else:
                tail_rows = []

        if not tail_rows:
            report["sections"]["context"] = {"error": "No candle data available for context section."}
        else:
            last = tail_rows[-1] if tail_rows else {}
            compact = _compute_compact_trend(tail_rows)
            ctx_obj: Dict[str, Any] = {
                "symbol": symbol,
                "timeframe": tf,
                "last_snapshot": last,
                "notes": "Minimal template keeps only candle context plus a direct forecast.",
            }
            timezone_label = ctx.get("timezone") if isinstance(ctx, dict) else None
            if timezone_label not in (None, "", [], {}):
                ctx_obj["timezone"] = timezone_label
            if compact:
                ctx_obj["trend_compact"] = compact
                ctx_obj["trend_compact_legend"] = dict(_TREND_COMPACT_LEGEND)
            report["sections"]["context"] = ctx_obj

    from ..forecast import forecast_generate

    forecast_kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": tf,
        "library": forecast_library,
        "method": forecast_method,
        "horizon": int(horizon),
        "start": start,
        "end": end,
        "denoise": denoise,
    }
    forecast_params = p.get("forecast_params")
    if isinstance(forecast_params, dict) and forecast_params:
        forecast_kwargs["params"] = forecast_params

    fc = (
        _get_raw_result(forecast_generate, **forecast_kwargs)
        if report_section_enabled(p, "forecast")
        else {"error": "forecast section not requested"}
    )
    if "error" in fc:
        report["sections"]["forecast"] = {
            "error": fc["error"],
            "method": forecast_method,
            "library": forecast_library,
            "selection_mode": "direct",
            "selection_note": "Minimal template skips backtest ranking and barrier optimization.",
        }
        if not report_section_enabled(p, "forecast"):
            report["sections"].pop("forecast", None)
        if not report_section_enabled(p, "context"):
            report["sections"].pop("context", None)
        return report

    forecast_section = {
        "method": forecast_method,
        "library": forecast_library,
        "selection_mode": "direct",
        "selection_note": "Minimal template skips backtest ranking and barrier optimization.",
    }
    forecast_section.update(adapt_forecast_payload_for_report(fc))
    report["sections"]["forecast"] = forecast_section
    if not report_section_enabled(p, "context"):
        report["sections"].pop("context", None)
    return report
