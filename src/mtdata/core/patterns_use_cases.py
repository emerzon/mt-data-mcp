from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .patterns_requests import PatternsDetectRequest
from .patterns_support import (
    _build_highlights,
    _compact_all_mode_payload,
    _elliott_completed_preview,
    _elliott_hidden_completed_note,
    score_all_mode_patterns,
)

_ALL_MODE_TIMEFRAMES = ("M30", "H1", "H4", "D1", "W1")

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


def _all_mode_invalid_config_keys(
    config: Optional[Dict[str, Any]],
    *,
    classic_cfg: Any,
    elliott_cfg: Any,
    fractal_cfg: Any,
    classic_invalid: List[str],
    elliott_invalid: List[str],
    fractal_invalid: List[str],
) -> List[str]:
    if not isinstance(config, dict):
        return []

    known_keys = set(_CLASSIC_CONFIG_EXTRA_KEYS)
    for cfg in (classic_cfg, elliott_cfg, fractal_cfg):
        try:
            known_keys.update(vars(cfg).keys())
        except Exception:
            continue

    classic_invalid_set = {str(key) for key in classic_invalid}
    elliott_invalid_set = {str(key) for key in elliott_invalid}
    fractal_invalid_set = {str(key) for key in fractal_invalid}

    out: List[str] = []
    for key in config.keys():
        key_str = str(key)
        if key_str in _CLASSIC_CONFIG_EXTRA_KEYS:
            continue
        if key_str not in known_keys:
            out.append(key_str)
            continue

        applicable_invalid_sets: List[set[str]] = []
        if hasattr(classic_cfg, key_str):
            applicable_invalid_sets.append(classic_invalid_set)
        if hasattr(elliott_cfg, key_str):
            applicable_invalid_sets.append(elliott_invalid_set)
        if hasattr(fractal_cfg, key_str):
            applicable_invalid_sets.append(fractal_invalid_set)

        if applicable_invalid_sets and all(
            key_str in invalid_set for invalid_set in applicable_invalid_sets
        ):
            out.append(key_str)

    return list(dict.fromkeys(out))


@dataclass(frozen=True)
class PatternsDetectDeps:
    compact_patterns_payload: Any
    fetch_pattern_data: Any
    classic_cfg_cls: Any
    elliott_cfg_cls: Any
    fractal_cfg_cls: Any
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
    format_fractal_patterns: Any
    detect_candlestick_patterns: Any
    elliott_timeframe_suggestion: Any
    resolve_elliott_scan_timeframes: Any
    validate_fractal_config: Any
    summarize_fractal_context: Any
    format_time_minimal: Any
    to_float_np: Any


def run_patterns_detect(  # noqa: C901
    request: PatternsDetectRequest,
    deps: PatternsDetectDeps,
) -> Dict[str, Any]:
    tf_norm: Optional[str] = (
        str(request.timeframe).strip().upper()
        if request.timeframe is not None
        else None
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
            return deps.compact_patterns_payload(
                out if isinstance(out, dict) else {"data": out}
            )
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
        df, err = deps.fetch_pattern_data(
            request.symbol, tf_single, request.limit, request.denoise
        )
        if err:
            return err
        engines, invalid_engines = deps.select_classic_engines(
            request.engine, request.ensemble
        )
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

        non_empty = {
            engine_name: rows for engine_name, rows in per_engine.items() if rows
        }
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
            resp["engine"] = (
                "ensemble"
                if (bool(request.ensemble) or len(engines) > 1)
                else engines[0]
            )
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

    if mode_value == "fractal":
        tf_single = tf_norm or "H1"
        cfg = deps.fractal_cfg_cls()
        unknown_cfg = _unknown_config_keys_for_mode(
            mode_value,
            deps.apply_config_to_obj(cfg, request.config),
        )
        if unknown_cfg:
            return {"error": f"Invalid config key(s): {sorted(unknown_cfg)}"}
        config_errors = deps.validate_fractal_config(cfg)
        if config_errors:
            return {"error": f"Invalid fractal config: {config_errors[0]}"}

        df, err = deps.fetch_pattern_data(
            request.symbol, tf_single, request.limit, request.denoise
        )
        if err:
            return err

        out_list = deps.format_fractal_patterns(df, cfg)
        visible_rows = (
            list(out_list)
            if request.include_completed
            else [
                row
                for row in out_list
                if str(row.get("status", "")).strip().lower() == "forming"
            ]
        )
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
        fractal_context = deps.summarize_fractal_context(visible_rows)
        if fractal_context:
            resp.update(fractal_context)
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
            df, err = deps.fetch_pattern_data(
                request.symbol, tf_norm, request.limit, request.denoise
            )
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
            df, err = deps.fetch_pattern_data(
                request.symbol, tf, request.limit, request.denoise
            )
            if err:
                failed_timeframes[tf] = str(err.get("error", "Unknown error"))
                continue

            tf_patterns = deps.format_elliott_patterns(df, cfg)
            filtered = (
                tf_patterns
                if request.include_completed
                else [
                    d
                    for d in tf_patterns
                    if str(d.get("status", "")).lower() == "forming"
                ]
            )
            completed_hidden = (
                0
                if request.include_completed
                else int(
                    sum(
                        1
                        for d in tf_patterns
                        if str(d.get("status", "")).lower() == "completed"
                    )
                )
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
                finding_row["note"] = _elliott_hidden_completed_note(
                    completed_hidden, completed_preview
                )
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
                    "series_close": [
                        float(v) for v in deps.to_float_np(df.get("close")).tolist()
                    ]
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
            completed_preview_total = _elliott_completed_preview(
                hidden_completed_rows_total
            )
            if completed_preview_total:
                resp["completed_patterns_preview"] = completed_preview_total
            resp["note"] = _elliott_hidden_completed_note(
                completed_hidden_total, completed_preview_total
            )
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
        fractal_patterns: List[Dict[str, Any]] = []
        section_errors: Dict[str, Dict[str, str]] = {}

        classic_cfg = deps.classic_cfg_cls()
        elliott_cfg = deps.elliott_cfg_cls()
        fractal_cfg = deps.fractal_cfg_cls()

        # Enable auto-complete for stale forming patterns by default in "all" mode
        # This prevents ancient patterns from showing as "forming" indefinitely
        classic_cfg.auto_complete_stale_forming = True
        classic_cfg.stale_completion_recent_bars = (
            10  # More aggressive - patterns ending >10 bars ago = completed
        )
        # Use 1/3 of the lookback limit as max pattern age for relevant results
        # This ensures patterns are recent regardless of timeframe
        classic_cfg.max_pattern_age_bars = max(100, request.limit // 3)
        # Limit pattern span to prevent ancient long-running patterns (e.g., 3-year trend lines)
        # Cap at 150 bars max to prevent multi-year patterns on weekly charts while
        # still allowing reasonable patterns (150 bars on W1 = ~3 years, on D1 = ~6 months)
        span_limit = min(150, int(request.limit * 0.5))
        classic_cfg.max_pattern_span_bars = max(60, span_limit)
        fractal_cfg.max_age_bars = max(100, request.limit // 3)

        classic_invalid: List[str] = []
        elliott_invalid: List[str] = []
        fractal_invalid: List[str] = []
        if isinstance(request.config, dict):
            classic_invalid = deps.apply_config_to_obj(classic_cfg, request.config)
            elliott_invalid = deps.apply_config_to_obj(elliott_cfg, request.config)
            fractal_invalid = deps.apply_config_to_obj(fractal_cfg, request.config)

        fractal_unapplied_keys = _all_mode_invalid_config_keys(
            request.config,
            classic_cfg=classic_cfg,
            elliott_cfg=elliott_cfg,
            fractal_cfg=fractal_cfg,
            classic_invalid=classic_invalid,
            elliott_invalid=elliott_invalid,
            fractal_invalid=fractal_invalid,
        )
        if fractal_unapplied_keys:
            msg = f"Invalid config key(s): {fractal_unapplied_keys}"
            section_errors["fractal"] = {tf: msg for tf in timeframes}

        fractal_config_errors = deps.validate_fractal_config(fractal_cfg)
        if fractal_config_errors:
            msg = f"Invalid fractal config: {fractal_config_errors[0]}"
            section_errors.setdefault("fractal", {})
            for tf in timeframes:
                section_errors["fractal"][tf] = msg

        # Elliott waves are multi-bar structures: a wave ending 20+ bars from
        # the tip is still actively developing.  The default recent_bars=3 is
        # too tight for all-mode and causes every pattern to be marked
        # "completed" then filtered.  Use 10 % of the fetch window (floor 20).
        if not (isinstance(request.config, dict) and "recent_bars" in request.config):
            elliott_cfg.recent_bars = max(20, request.limit // 10)

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
                section_errors.setdefault("fractal", {})[tf] = err_msg
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

            # ── Fractal ──
            if tf not in section_errors.get("fractal", {}):
                try:
                    fractal_rows = deps.format_fractal_patterns(df, fractal_cfg)
                    for row in fractal_rows:
                        if isinstance(row, dict):
                            row["timeframe"] = tf
                            fractal_patterns.append(row)
                except Exception as exc:
                    section_errors.setdefault("fractal", {})[tf] = str(exc)

        # Filter completed patterns for classic/elliott/fractal
        if not request.include_completed:
            classic_patterns = [
                r
                for r in classic_patterns
                if str(r.get("status", "")).lower() != "completed"
            ]
            elliott_patterns = [
                r
                for r in elliott_patterns
                if str(r.get("status", "")).lower() != "completed"
            ]
            fractal_patterns = [
                r
                for r in fractal_patterns
                if str(r.get("status", "")).lower() != "completed"
            ]

        # Score and sort each section by relevance (confidence + recency)
        score_all_mode_patterns(candlestick_patterns, request.limit)
        score_all_mode_patterns(classic_patterns, request.limit)
        score_all_mode_patterns(elliott_patterns, request.limit)
        score_all_mode_patterns(fractal_patterns, request.limit)

        total = (
            len(candlestick_patterns)
            + len(classic_patterns)
            + len(elliott_patterns)
            + len(fractal_patterns)
        )

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
            "fractal": {
                "n_patterns": len(fractal_patterns),
                "patterns": fractal_patterns,
            },
            "total_patterns": total,
        }

        if classic_patterns:
            bias = deps.summarize_pattern_bias(classic_patterns)
            if bias:
                resp["classic"]["signal_bias"] = bias
        if fractal_patterns:
            bias = deps.summarize_pattern_bias(fractal_patterns)
            if bias:
                resp["fractal"]["signal_bias"] = bias
            fractal_context = deps.summarize_fractal_context(fractal_patterns)
            if fractal_context:
                resp["fractal"].update(fractal_context)

        if section_errors:
            resp["errors"] = section_errors

        # Merged cross-section highlights for quick trader read
        resp["highlights"] = _build_highlights(resp, limit=5)

        if detail_value == "compact":
            return _compact_all_mode_payload(resp)
        return resp

    return {
        "error": (
            f"Unknown mode: {request.mode}. "
            "Use all, candlestick, classic/chart, fractal, or elliott."
        )
    }
