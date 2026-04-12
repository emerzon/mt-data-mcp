from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .patterns_requests import PatternsDetectRequest
from .patterns_support import (
    _compact_all_mode_payload,
    _elliott_completed_preview,
    _elliott_hidden_completed_note,
)

_ALL_MODE_TIMEFRAMES = ("H1", "H4", "D1")

_CLASSIC_CONFIG_EXTRA_KEYS = {
    "ensemble_weights",
    "native_multiscale",
    "native_multiscale_overlap",
    "native_scale_factors",
    "native_scales",
    "stock_bars_left",
    "stock_bars_right",
}


def _unknown_config_keys_for_mode(mode: str, unknown_keys: List[str]) -> List[str]:
    mode_key = str(mode).strip().lower()
    if mode_key == "classic":
        return [key for key in unknown_keys if key not in _CLASSIC_CONFIG_EXTRA_KEYS]
    return list(unknown_keys)


@dataclass(frozen=True)
class PatternsDetectDeps:
    compact_patterns_payload: Any
    fetch_pattern_data: Any
    classic_cfg_cls: Any
    elliott_cfg_cls: Any
    apply_config_to_obj: Any
    select_classic_engines: Any
    available_classic_engines: Any
    run_classic_engine: Any
    resolve_engine_weights: Any
    merge_classic_ensemble: Any
    enrich_classic_patterns: Any
    summarize_engine_findings: Any
    summarize_pattern_bias: Any
    build_pattern_response: Any
    format_elliott_patterns: Any
    detect_candlestick_patterns: Any
    elliott_timeframe_suggestion: Any
    resolve_elliott_scan_timeframes: Any
    format_time_minimal: Any
    to_float_np: Any


def run_patterns_detect(  # noqa: C901
    request: PatternsDetectRequest,
    deps: PatternsDetectDeps,
) -> Dict[str, Any]:
    tf_norm: Optional[str] = (
        str(request.timeframe).strip().upper() if request.timeframe is not None else None
    )
    if not tf_norm:
        tf_norm = None

    mode_value = str(request.mode).strip().lower()
    if mode_value == "chart":
        mode_value = "classic"
    detail_value = str(request.detail).strip().lower()
    if detail_value not in ("compact", "full"):
        return {"error": "Invalid detail. Use 'compact' or 'full'."}

    if mode_value == "candlestick":
        tf_single = tf_norm or "H1"
        last_n_bars_val: Optional[int] = None
        if request.last_n_bars is not None:
            try:
                last_n_bars_val = int(request.last_n_bars)
            except Exception:
                return {"error": "last_n_bars must be a positive integer."}
            if last_n_bars_val <= 0:
                return {"error": "last_n_bars must be >= 1."}
        out = deps.detect_candlestick_patterns(
            symbol=request.symbol,
            timeframe=tf_single,
            limit=request.limit,
            min_strength=request.min_strength,
            min_gap=request.min_gap,
            robust_only=request.robust_only,
            whitelist=request.whitelist,
            top_k=request.top_k,
            last_n_bars=last_n_bars_val,
            config=request.config if isinstance(request.config, dict) else None,
        )
        if detail_value == "compact":
            return deps.compact_patterns_payload(out if isinstance(out, dict) else {"data": out})
        return out

    if mode_value == "classic":
        tf_single = tf_norm or "H1"
        cfg = deps.classic_cfg_cls()
        unknown_cfg = _unknown_config_keys_for_mode(
            mode_value,
            deps.apply_config_to_obj(cfg, request.config),
        )
        if unknown_cfg:
            return {"error": f"Invalid config key(s): {sorted(unknown_cfg)}"}
        df, err = deps.fetch_pattern_data(request.symbol, tf_single, request.limit, request.denoise)
        if err:
            return err
        engines, invalid_engines = deps.select_classic_engines(request.engine, request.ensemble)
        if invalid_engines:
            return {
                "error": (
                    f"Invalid classic engine(s): {invalid_engines}. "
                    f"Valid options: {list(deps.available_classic_engines())}"
                )
            }

        per_engine: Dict[str, List[Dict[str, Any]]] = {}
        engine_errors: Dict[str, str] = {}
        for eng in engines:
            patt_rows, eng_err = deps.run_classic_engine(
                eng, request.symbol, df, cfg, request.config
            )
            if eng_err:
                engine_errors[eng] = eng_err
            per_engine[eng] = patt_rows

        non_empty = {engine_name: rows for engine_name, rows in per_engine.items() if rows}
        if not non_empty:
            if engine_errors and len(engine_errors) == len(engines):
                return {
                    "error": "No classic engines produced results",
                    "engines_run": engines,
                    "engine_errors": engine_errors,
                }
            resp = deps.build_pattern_response(
                request.symbol,
                tf_single,
                request.limit,
                mode_value,
                [],
                request.include_completed,
                request.include_series,
                request.series_time,
                df,
                detail=detail_value,
            )
            resp["engine"] = "ensemble" if (bool(request.ensemble) or len(engines) > 1) else engines[0]
            resp["engines_run"] = engines
            resp["engine_findings"] = deps.summarize_engine_findings(
                per_engine, engines, request.include_completed
            )
            if engine_errors:
                resp["engine_errors"] = engine_errors
            return resp

        run_ensemble = bool(request.ensemble) or len(non_empty) > 1
        if run_ensemble:
            weight_map = deps.resolve_engine_weights(
                engines,
                request.ensemble_weights
                if isinstance(request.ensemble_weights, dict)
                else (
                    (request.config or {}).get("ensemble_weights")
                    if isinstance(request.config, dict)
                    else None
                ),
            )
            out_list = deps.merge_classic_ensemble(non_empty, weight_map)
        else:
            only_engine = next(iter(non_empty.keys()))
            out_list = list(non_empty.get(only_engine, []))

        out_list = deps.enrich_classic_patterns(out_list, df, cfg)
        resp = deps.build_pattern_response(
            request.symbol,
            tf_single,
            request.limit,
            mode_value,
            out_list,
            request.include_completed,
            request.include_series,
            request.series_time,
            df,
            detail=detail_value,
        )
        resp["engine"] = "ensemble" if run_ensemble else next(iter(non_empty.keys()))
        resp["engines_run"] = engines
        resp["engine_findings"] = deps.summarize_engine_findings(
            per_engine, engines, request.include_completed
        )
        signal_summary = None
        if detail_value == "compact":
            summary = resp.get("summary")
            if isinstance(summary, dict):
                signal_summary = summary.get("signal_bias")
        else:
            rows = resp.get("patterns")
            if isinstance(rows, list):
                signal_summary = deps.summarize_pattern_bias(rows)
        if signal_summary:
            resp["signal_summary"] = signal_summary
        if engine_errors:
            resp["engine_errors"] = engine_errors
        return resp

    if mode_value == "elliott":
        cfg = deps.elliott_cfg_cls()
        unknown_cfg = _unknown_config_keys_for_mode(
            mode_value,
            deps.apply_config_to_obj(cfg, request.config),
        )
        if unknown_cfg:
            return {"error": f"Invalid config key(s): {sorted(unknown_cfg)}"}

        if tf_norm:
            df, err = deps.fetch_pattern_data(request.symbol, tf_norm, request.limit, request.denoise)
            if err:
                return err

            out_list = deps.format_elliott_patterns(df, cfg)
            return deps.build_pattern_response(
                request.symbol,
                tf_norm,
                request.limit,
                mode_value,
                out_list,
                request.include_completed,
                request.include_series,
                request.series_time,
                df,
                detail=detail_value,
            )

        scanned_timeframes = deps.resolve_elliott_scan_timeframes(cfg)
        findings: List[Dict[str, Any]] = []
        combined_patterns: List[Dict[str, Any]] = []
        failed_timeframes: Dict[str, str] = {}
        series_by_timeframe: Dict[str, Dict[str, Any]] = {}
        warnings_out: List[str] = []
        completed_hidden_total = 0
        hidden_completed_rows_total: List[Dict[str, Any]] = []

        for tf in scanned_timeframes:
            df, err = deps.fetch_pattern_data(request.symbol, tf, request.limit, request.denoise)
            if err:
                failed_timeframes[tf] = str(err.get("error", "Unknown error"))
                continue

            tf_patterns = deps.format_elliott_patterns(df, cfg)
            filtered = (
                tf_patterns
                if request.include_completed
                else [
                    d for d in tf_patterns if str(d.get("status", "")).lower() == "forming"
                ]
            )
            completed_hidden = 0 if request.include_completed else int(
                sum(1 for d in tf_patterns if str(d.get("status", "")).lower() == "completed")
            )
            completed_hidden_total += int(completed_hidden)
            hidden_completed_rows = [
                {**dict(d), "timeframe": tf}
                for d in tf_patterns
                if str(d.get("status", "")).lower() == "completed"
            ]
            completed_preview = _elliott_completed_preview(hidden_completed_rows)
            hidden_completed_rows_total.extend(hidden_completed_rows)
            finding_row: Dict[str, Any] = {
                "timeframe": tf,
                "n_patterns": int(len(filtered)),
                "patterns": filtered,
            }
            if completed_hidden > 0:
                finding_row["completed_patterns_hidden"] = int(completed_hidden)
                if completed_preview:
                    finding_row["completed_patterns_preview"] = completed_preview
                finding_row["note"] = _elliott_hidden_completed_note(completed_hidden, completed_preview)
            tf_warnings = df.attrs.get("warnings")
            if isinstance(tf_warnings, list) and tf_warnings:
                finding_row["warnings"] = [str(w) for w in tf_warnings if str(w)]
                warnings_out.extend(f"{tf}: {str(w)}" for w in tf_warnings if str(w))
            if int(len(filtered)) == 0:
                if completed_hidden > 0:
                    finding_row["diagnostic"] = (
                        f"No forming Elliott Wave structures detected in {int(request.limit)} {tf} bars. "
                        f"{int(completed_hidden)} completed structure(s) were detected but hidden by default. "
                        f"{deps.elliott_timeframe_suggestion(tf)}"
                    )
                else:
                    finding_row["diagnostic"] = (
                        f"No valid Elliott Wave structures detected in {int(request.limit)} {tf} bars. "
                        f"{deps.elliott_timeframe_suggestion(tf)}"
                    )
            findings.append(finding_row)

            for row in filtered:
                merged = dict(row)
                merged["timeframe"] = tf
                combined_patterns.append(merged)

            if request.include_series:
                series_payload: Dict[str, Any] = {
                    "series_close": [float(v) for v in deps.to_float_np(df.get("close")).tolist()]
                }
                if "time" in df.columns:
                    if str(request.series_time).lower() == "epoch":
                        series_payload["series_epoch"] = [
                            float(v) for v in deps.to_float_np(df.get("time")).tolist()
                        ]
                    else:
                        series_payload["series_time"] = [
                            deps.format_time_minimal(float(v))
                            for v in deps.to_float_np(df.get("time")).tolist()
                        ]
                series_by_timeframe[tf] = series_payload

        if not findings:
            return {
                "error": (
                    f"Failed to fetch sufficient bars for {request.symbol} across all timeframes"
                ),
                "failed_timeframes": failed_timeframes,
            }

        resp: Dict[str, Any] = {
            "success": True,
            "symbol": request.symbol,
            "timeframe": "ALL",
            "lookback": int(request.limit),
            "mode": mode_value,
            "scanned_timeframes": scanned_timeframes,
            "findings": findings,
            "patterns": combined_patterns,
            "n_patterns": int(len(combined_patterns)),
        }
        if int(len(combined_patterns)) == 0:
            if completed_hidden_total > 0:
                resp["diagnostic"] = (
                    "No forming Elliott Wave structures were detected across scanned timeframes. "
                    f"{int(completed_hidden_total)} completed structure(s) were detected but hidden by default. "
                    "Try increasing lookback or focusing on higher-structure windows like H4/D1."
                )
            else:
                resp["diagnostic"] = (
                    "No valid Elliott Wave structures were detected across scanned timeframes. "
                    "Try increasing lookback or focusing on higher-structure windows like H4/D1."
                )
        if completed_hidden_total > 0:
            resp["completed_patterns_hidden"] = int(completed_hidden_total)
            completed_preview_total = _elliott_completed_preview(hidden_completed_rows_total)
            if completed_preview_total:
                resp["completed_patterns_preview"] = completed_preview_total
            resp["note"] = _elliott_hidden_completed_note(completed_hidden_total, completed_preview_total)
        if failed_timeframes:
            resp["failed_timeframes"] = failed_timeframes
        if warnings_out:
            resp["warnings"] = warnings_out
        if request.include_series:
            resp["series_by_timeframe"] = series_by_timeframe
        if detail_value == "compact":
            return deps.compact_patterns_payload(resp)
        return resp

    if mode_value == "all":
        timeframes = [tf_norm] if tf_norm else list(_ALL_MODE_TIMEFRAMES)

        candlestick_patterns: List[Dict[str, Any]] = []
        classic_patterns: List[Dict[str, Any]] = []
        elliott_patterns: List[Dict[str, Any]] = []
        section_errors: Dict[str, Dict[str, str]] = {}

        classic_cfg = deps.classic_cfg_cls()
        elliott_cfg = deps.elliott_cfg_cls()
        if isinstance(request.config, dict):
            deps.apply_config_to_obj(classic_cfg, request.config)
            deps.apply_config_to_obj(elliott_cfg, request.config)

        effective_top_k = max(request.top_k, 3)

        for tf in timeframes:
            # ── Candlestick ──
            try:
                candle_result = deps.detect_candlestick_patterns(
                    symbol=request.symbol,
                    timeframe=tf,
                    limit=request.limit,
                    min_strength=request.min_strength,
                    min_gap=request.min_gap,
                    robust_only=request.robust_only,
                    whitelist=request.whitelist,
                    top_k=effective_top_k,
                    last_n_bars=request.last_n_bars,
                    config=request.config if isinstance(request.config, dict) else None,
                )
                if isinstance(candle_result, dict) and not candle_result.get("error"):
                    rows = candle_result.get("data", [])
                    if isinstance(rows, list):
                        for row in rows:
                            if isinstance(row, dict):
                                row["timeframe"] = tf
                                candlestick_patterns.append(row)
                elif isinstance(candle_result, dict) and candle_result.get("error"):
                    section_errors.setdefault("candlestick", {})[tf] = str(
                        candle_result["error"]
                    )
            except Exception as exc:
                section_errors.setdefault("candlestick", {})[tf] = str(exc)

            # ── Shared data fetch for classic + elliott ──
            df, fetch_err = deps.fetch_pattern_data(
                request.symbol, tf, request.limit, request.denoise
            )
            if fetch_err:
                err_msg = str(fetch_err.get("error", "data fetch failed"))
                section_errors.setdefault("classic", {})[tf] = err_msg
                section_errors.setdefault("elliott", {})[tf] = err_msg
                continue

            # ── Classic (native engine) ──
            try:
                patt_rows, eng_err = deps.run_classic_engine(
                    "native", request.symbol, df, classic_cfg, request.config
                )
                if eng_err:
                    section_errors.setdefault("classic", {})[tf] = eng_err
                if patt_rows:
                    enriched = deps.enrich_classic_patterns(patt_rows, df, classic_cfg)
                    for row in enriched:
                        if isinstance(row, dict):
                            row["timeframe"] = tf
                            classic_patterns.append(row)
            except Exception as exc:
                section_errors.setdefault("classic", {})[tf] = str(exc)

            # ── Elliott ──
            try:
                elliott_rows = deps.format_elliott_patterns(df, elliott_cfg)
                for row in elliott_rows:
                    if isinstance(row, dict):
                        row["timeframe"] = tf
                        elliott_patterns.append(row)
            except Exception as exc:
                section_errors.setdefault("elliott", {})[tf] = str(exc)

        # Filter completed patterns for classic/elliott
        if not request.include_completed:
            classic_patterns = [
                r for r in classic_patterns
                if str(r.get("status", "")).lower() != "completed"
            ]
            elliott_patterns = [
                r for r in elliott_patterns
                if str(r.get("status", "")).lower() != "completed"
            ]

        # Sort each section by confidence descending
        def _conf_key(r: Dict[str, Any]) -> float:
            try:
                return float(r.get("confidence", 0))
            except (TypeError, ValueError):
                return 0.0

        candlestick_patterns.sort(key=_conf_key, reverse=True)
        classic_patterns.sort(key=_conf_key, reverse=True)
        elliott_patterns.sort(key=_conf_key, reverse=True)

        total = len(candlestick_patterns) + len(classic_patterns) + len(elliott_patterns)

        if total == 0 and section_errors:
            flat: Dict[str, str] = {}
            for section, tf_errs in section_errors.items():
                for tf_name, msg in tf_errs.items():
                    flat[f"{section}/{tf_name}"] = msg
            return {
                "error": "No patterns detected across any mode or timeframe.",
                "details": flat,
            }

        resp: Dict[str, Any] = {
            "success": True,
            "symbol": request.symbol,
            "mode": "all",
            "timeframes": timeframes,
            "candlestick": {
                "n_patterns": len(candlestick_patterns),
                "patterns": candlestick_patterns,
            },
            "classic": {
                "n_patterns": len(classic_patterns),
                "patterns": classic_patterns,
            },
            "elliott": {
                "n_patterns": len(elliott_patterns),
                "patterns": elliott_patterns,
            },
            "total_patterns": total,
        }

        if classic_patterns:
            bias = deps.summarize_pattern_bias(classic_patterns)
            if bias:
                resp["classic"]["signal_bias"] = bias

        if section_errors:
            resp["errors"] = section_errors

        if detail_value == "compact":
            return _compact_all_mode_payload(resp)
        return resp

    return {"error": f"Unknown mode: {request.mode}. Use all, candlestick, classic/chart, or elliott."}
