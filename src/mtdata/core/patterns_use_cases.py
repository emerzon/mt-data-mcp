from __future__ import annotations

from typing import Any, Dict, List, Optional

from .patterns_requests import PatternsDetectRequest


def run_patterns_detect(
    request: PatternsDetectRequest,
    *,
    timeframe_map: Dict[str, Any],
    compact_patterns_payload: Any,
    fetch_pattern_data: Any,
    classic_cfg_cls: Any,
    elliott_cfg_cls: Any,
    apply_config_to_obj: Any,
    select_classic_engines: Any,
    available_classic_engines: Any,
    run_classic_engine: Any,
    resolve_engine_weights: Any,
    merge_classic_ensemble: Any,
    enrich_classic_patterns: Any,
    summarize_engine_findings: Any,
    summarize_pattern_bias: Any,
    build_pattern_response: Any,
    format_elliott_patterns: Any,
    detect_candlestick_patterns: Any,
    elliott_timeframe_suggestion: Any,
    format_time_minimal: Any,
    to_float_np: Any,
) -> Dict[str, Any]:
    try:
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
            out = detect_candlestick_patterns(
                symbol=request.symbol,
                timeframe=tf_single,
                limit=request.limit,
                min_strength=request.min_strength,
                min_gap=request.min_gap,
                robust_only=request.robust_only,
                whitelist=request.whitelist,
                top_k=request.top_k,
                last_n_bars=last_n_bars_val,
            )
            if detail_value == "compact":
                return compact_patterns_payload(out if isinstance(out, dict) else {"data": out})
            return out

        if mode_value == "classic":
            tf_single = tf_norm or "H1"
            df, err = fetch_pattern_data(request.symbol, tf_single, request.limit, request.denoise)
            if err:
                return err

            cfg = classic_cfg_cls()
            apply_config_to_obj(cfg, request.config)
            engines, invalid_engines = select_classic_engines(request.engine, request.ensemble)
            if invalid_engines:
                return {
                    "error": (
                        f"Invalid classic engine(s): {invalid_engines}. "
                        f"Valid options: {list(available_classic_engines())}"
                    )
                }

            per_engine: Dict[str, List[Dict[str, Any]]] = {}
            engine_errors: Dict[str, str] = {}
            for eng in engines:
                patt_rows, eng_err = run_classic_engine(
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
                resp = build_pattern_response(
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
                resp["engine_findings"] = summarize_engine_findings(
                    per_engine, engines, request.include_completed
                )
                if engine_errors:
                    resp["engine_errors"] = engine_errors
                return resp

            run_ensemble = bool(request.ensemble) or len(non_empty) > 1
            if run_ensemble:
                weight_map = resolve_engine_weights(
                    engines,
                    request.ensemble_weights
                    if isinstance(request.ensemble_weights, dict)
                    else (
                        (request.config or {}).get("ensemble_weights")
                        if isinstance(request.config, dict)
                        else None
                    ),
                )
                out_list = merge_classic_ensemble(non_empty, weight_map)
            else:
                only_engine = next(iter(non_empty.keys()))
                out_list = list(non_empty.get(only_engine, []))

            out_list = enrich_classic_patterns(out_list, df)
            visible_rows = (
                out_list
                if request.include_completed
                else [row for row in out_list if str(row.get("status", "")).lower() == "forming"]
            )

            resp = build_pattern_response(
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
            resp["engine_findings"] = summarize_engine_findings(
                per_engine, engines, request.include_completed
            )
            signal_summary = summarize_pattern_bias(visible_rows)
            if signal_summary:
                resp["signal_summary"] = signal_summary
            if engine_errors:
                resp["engine_errors"] = engine_errors
            return resp

        if mode_value == "elliott":
            cfg = elliott_cfg_cls()
            apply_config_to_obj(cfg, request.config)

            if tf_norm:
                df, err = fetch_pattern_data(request.symbol, tf_norm, request.limit, request.denoise)
                if err:
                    return err

                out_list = format_elliott_patterns(df, cfg)
                return build_pattern_response(
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

            scanned_timeframes = list(timeframe_map.keys())
            findings: List[Dict[str, Any]] = []
            combined_patterns: List[Dict[str, Any]] = []
            failed_timeframes: Dict[str, str] = {}
            series_by_timeframe: Dict[str, Dict[str, Any]] = {}

            for tf in scanned_timeframes:
                df, err = fetch_pattern_data(request.symbol, tf, request.limit, request.denoise)
                if err:
                    failed_timeframes[tf] = str(err.get("error", "Unknown error"))
                    continue

                tf_patterns = format_elliott_patterns(df, cfg)
                filtered = (
                    tf_patterns
                    if request.include_completed
                    else [
                        d for d in tf_patterns if str(d.get("status", "")).lower() == "forming"
                    ]
                )
                finding_row: Dict[str, Any] = {
                    "timeframe": tf,
                    "n_patterns": int(len(filtered)),
                    "patterns": filtered,
                }
                if int(len(filtered)) == 0:
                    finding_row["diagnostic"] = (
                        f"No valid Elliott Wave structures detected in {int(request.limit)} {tf} bars. "
                        f"{elliott_timeframe_suggestion(tf)}"
                    )
                findings.append(finding_row)

                for row in filtered:
                    merged = dict(row)
                    merged["timeframe"] = tf
                    combined_patterns.append(merged)

                if request.include_series:
                    series_payload: Dict[str, Any] = {
                        "series_close": [float(v) for v in to_float_np(df.get("close")).tolist()]
                    }
                    if "time" in df.columns:
                        if str(request.series_time).lower() == "epoch":
                            series_payload["series_epoch"] = [
                                float(v) for v in to_float_np(df.get("time")).tolist()
                            ]
                        else:
                            series_payload["series_time"] = [
                                format_time_minimal(float(v))
                                for v in to_float_np(df.get("time")).tolist()
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
                resp["diagnostic"] = (
                    "No valid Elliott Wave structures were detected across scanned timeframes. "
                    "Try increasing lookback or focusing on higher-structure windows like H4/D1."
                )
            if failed_timeframes:
                resp["failed_timeframes"] = failed_timeframes
            if request.include_series:
                resp["series_by_timeframe"] = series_by_timeframe
            if detail_value == "compact":
                return compact_patterns_payload(resp)
            return resp

        return {"error": f"Unknown mode: {request.mode}. Use candlestick, classic/chart, or elliott."}

    except Exception as exc:
        return {"error": f"Error detecting patterns: {str(exc)}"}
