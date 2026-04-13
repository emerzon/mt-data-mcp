"""Payload consolidation utilities for regime detection output formatting."""

from typing import Any, Dict, List, Optional

import numpy as np

from .smoothing import _canonicalize_regime_labels


def _interpret_regime_label(
    regime_id: int,
    mu: Optional[float],
    sigma: Optional[float],
    method: str,
    total_states: int,
    target: Optional[str] = "return",
) -> str:
    """Generate human-readable regime label from statistical parameters.

    For HMM/MS-AR with mean/volatility: creates descriptive labels like
    'low_return_low_vol' or 'high_return_high_vol'.
    For other methods: returns basic regime identifier.
    """
    target_name = str(target or "return").strip().lower()

    if target_name == "price":
        if total_states == 2:
            return "low_price_regime" if regime_id == 0 else "high_price_regime"
        if total_states == 3:
            labels = ("low_price_regime", "mid_price_regime", "high_price_regime")
            if 0 <= regime_id < len(labels):
                return labels[regime_id]
        if total_states == 4:
            labels = (
                "lowest_price_regime",
                "low_price_regime",
                "high_price_regime",
                "highest_price_regime",
            )
            if 0 <= regime_id < len(labels):
                return labels[regime_id]
        return f"price_regime_{regime_id}"

    if method in ("hmm", "ms_ar") and mu is not None and sigma is not None:
        # Determine return level
        if abs(mu) < 0.0001:  # ~0.01% threshold
            return_level = "neutral"
        elif mu > 0:
            return_level = "positive"
        else:
            return_level = "negative"

        # Determine volatility level (relative terms)
        if sigma < 0.0005:  # Very low vol
            vol_level = "very_low_vol"
        elif sigma < 0.001:  # Low vol (~0.1%)
            vol_level = "low_vol"
        elif sigma < 0.003:  # Moderate vol (~0.3%)
            vol_level = "mod_vol"
        else:  # High vol
            vol_level = "high_vol"

        return f"{return_level}_{vol_level}"

    # Fallback for methods without mu/sigma
    if total_states <= 2:
        return "regime_bearish" if regime_id == 0 else "regime_bullish"
    return f"regime_{regime_id}"


def _normalize_label_mapping(label_mapping: Any) -> Dict[int, int]:
    """Normalize serialized old->new state mappings to integer keys/values."""
    if not isinstance(label_mapping, dict):
        return {}

    normalized: Dict[int, int] = {}
    for old_state, new_state in label_mapping.items():
        try:
            normalized[int(old_state)] = int(new_state)
        except (TypeError, ValueError):
            continue
    return normalized


def _build_regime_descriptions(
    regime_params: Optional[Dict[str, Any]],
    method: str,
    *,
    target: Optional[str] = "return",
    label_mapping: Any = None,
) -> Dict[int, Dict[str, Any]]:
    """Build a mapping of regime_id -> descriptive info from regime_params.

    regime_params contains: weights, mu (mean returns/levels), sigma (dispersion)
    Returns dict with descriptive labels and statistics for each regime.
    """
    if not regime_params or method not in ("hmm", "ms_ar"):
        return {}

    mu_list = regime_params.get("mu")
    sigma_list = regime_params.get("sigma")
    weights_list = regime_params.get("weights")

    if not isinstance(mu_list, (list, tuple)) or not isinstance(
        sigma_list, (list, tuple)
    ):
        return {}

    descriptions: Dict[int, Dict[str, Any]] = {}
    n_states = len(mu_list)
    normalized_mapping = _normalize_label_mapping(label_mapping)
    indexed_states = (
        sorted(normalized_mapping.items(), key=lambda item: item[1])
        if normalized_mapping
        else [(i, i) for i in range(n_states)]
    )
    described_states = len(indexed_states) if indexed_states else n_states
    target_name = str(target or "return").strip().lower()

    for source_state, regime_id in indexed_states:
        if source_state < 0 or source_state >= n_states:
            continue
        mu = float(mu_list[source_state]) if source_state < len(mu_list) else None
        sigma = (
            float(sigma_list[source_state]) if source_state < len(sigma_list) else None
        )
        weight = (
            float(weights_list[source_state])
            if weights_list and source_state < len(weights_list)
            else None
        )

        label = _interpret_regime_label(
            regime_id,
            mu,
            sigma,
            method,
            described_states,
            target=target_name,
        )

        desc: Dict[str, Any] = {"label": label}
        if mu is not None:
            if target_name == "price":
                desc["mean_price"] = round(mu, 6)
            else:
                desc["mean_return"] = round(mu, 6)
                desc["mean_return_pct"] = round(mu * 100, 4)
        if sigma is not None:
            if target_name == "price":
                desc["std_dev"] = round(sigma, 6)
            else:
                desc["volatility"] = round(sigma, 6)
                desc["volatility_pct"] = round(sigma * 100, 4)
        if weight is not None:
            desc["weight"] = round(weight, 4)

        descriptions[regime_id] = desc

    return descriptions


def _consolidate_payload(  # noqa: C901
    payload: Dict[str, Any],
    method: str,
    output_mode: str,
    include_series: bool = False,
) -> Dict[str, Any]:
    """Consolidate time series into regime segments and restructure payload."""
    try:
        times = payload.get("times")
        if not times or not isinstance(times, list):
            return payload

        # Prepare consolidation
        segments: List[Dict[str, Any]] = []

        # Extract states/regimes
        states: List[int] = []
        probs: List[Any] = []
        cps_idx: set = set()  # Initialize for all methods to avoid unbound reference

        if method == "bocpd":
            # For BOCPD, we define regimes by change points
            # We can create a 'regime_id' that increments at each CP
            # We also look at 'change_points' list in payload
            if "change_points" in payload and isinstance(
                payload["change_points"], list
            ):
                for cp in payload["change_points"]:
                    if isinstance(cp, dict) and "idx" in cp:
                        cps_idx.add(cp["idx"])

            curr_regime = 0
            # Reconstruct per-step state
            for i in range(len(times)):
                if i in cps_idx:
                    curr_regime += 1
                states.append(curr_regime)

            # Probs
            raw_probs = payload.get("cp_prob")
            if isinstance(raw_probs, list):
                probs = raw_probs
            else:
                probs = [0.0] * len(times)

            raw_series = payload.get("_series_values")
            if isinstance(raw_series, list) and states:
                canon_state, _, canon_meta = _canonicalize_regime_labels(
                    np.asarray(states, dtype=int),
                    None,
                    np.asarray(raw_series, dtype=float),
                )
                states = [int(value) for value in canon_state.tolist()]
                params_used = payload.get("params_used")
                if isinstance(params_used, dict) and canon_meta.get("relabeled", False):
                    params_used["relabeled"] = True
                    params_used["label_mapping"] = canon_meta.get("mapping", {})

        elif method in ("ms_ar", "hmm", "clustering"):
            raw_state = payload.get("state")
            if isinstance(raw_state, list):
                states = raw_state

            # Probs
            # structure is usually list of lists [ [p0, p1...], ... ]
            raw_probs = payload.get("state_probabilities")
            # We might just store the max prob or the prob of the current state?
            if isinstance(raw_probs, list) and raw_probs:
                if isinstance(raw_probs[0], list):
                    # Pick prob of selected state
                    for s, p_vec in zip(states, raw_probs):
                        if isinstance(p_vec, list) and 0 <= s < len(p_vec):
                            probs.append(p_vec[s])
                        else:
                            probs.append(None)
                else:
                    probs = raw_probs  # Should not happen based on current logic but safe fallback

        if not states or len(states) != len(times):
            # Fallback if creation failed
            return payload

        filtered_entries = []
        for idx, state in enumerate(states):
            try:
                state_value = int(state)
            except Exception:
                state_value = None
            if state_value is None or state_value < 0:
                continue
            prob_value = probs[idx] if idx < len(probs) else None
            filtered_entries.append((idx, times[idx], state_value, prob_value))
        if not filtered_entries:
            times = []
            states = []
            probs = []
            original_indices = []
        elif len(filtered_entries) != len(times):
            times = [entry[1] for entry in filtered_entries]
            states = [entry[2] for entry in filtered_entries]
            probs = [entry[3] for entry in filtered_entries]
            original_indices = [entry[0] for entry in filtered_entries]
        else:
            original_indices = list(range(len(times)))

        # Consolidate
        # Loop through
        if times:
            curr_start = times[0]
            curr_state = states[0]
            curr_start_index = original_indices[0]
            curr_prob_sum = 0.0
            curr_count = 0
            curr_transition_conf = None
            if method == "bocpd" and curr_start_index in cps_idx and probs:
                try:
                    curr_transition_conf = float(probs[0])
                except Exception:
                    curr_transition_conf = None

            i = 0
            while i < len(times):
                t = times[i]
                s = states[i]
                p = probs[i] if i < len(probs) and probs[i] is not None else 0.0

                # Check change (state change)
                # For BOCPD, 's' changes exactly at CP.
                if s != curr_state and curr_count > 0:
                    # close segment
                    avg_prob = curr_prob_sum / max(1, curr_count)
                    segments.append(
                        {
                            "start": curr_start,
                            "end": times[i - 1] if i > 0 else curr_start,
                            "duration": curr_count,
                            "regime": curr_state,  # state ID or regime ID
                            "confidence": avg_prob,  # average prob of being in this state/regime
                            "transition_conf": curr_transition_conf,
                        }
                    )
                    # New segment
                    curr_start = t
                    curr_state = s
                    curr_start_index = original_indices[i]
                    curr_prob_sum = 0.0
                    curr_count = 0
                    curr_transition_conf = None
                    if method == "bocpd":
                        try:
                            curr_transition_conf = float(p)
                        except Exception:
                            curr_transition_conf = None

                curr_prob_sum += p
                curr_count += 1
                i += 1

            # Final segment
            if curr_count > 0:
                avg_prob = curr_prob_sum / max(1, curr_count)
                segments.append(
                    {
                        "start": curr_start,
                        "end": times[-1],
                        "duration": curr_count,
                        "regime": curr_state,
                        "confidence": avg_prob,
                        "transition_conf": curr_transition_conf,
                    }
                )

        # Post-process segments for readability
        # For BOCPD, 'confidence' is avg cp_prob which is usually low except at edges.
        # Maybe we want the PEAK prob? or just drop it.
        # For HMM, 'confidence' is avg prob of that state.

        params_used = payload.get("params_used")
        label_mapping = (
            params_used.get("label_mapping") if isinstance(params_used, dict) else None
        )

        # Build regime descriptions for labeling segments
        regime_params = payload.get("regime_params")
        regime_descriptions = _build_regime_descriptions(
            regime_params,
            method,
            target=payload.get("target"),
            label_mapping=label_mapping,
        )

        final_segments = []
        for _i, seg in enumerate(segments):
            regime_id = seg["regime"]
            row: Dict[str, Any] = {
                "start": seg["start"],
                "end": seg["end"],
                "bars": seg["duration"],
                "regime": regime_id,
            }
            # Add descriptive label if available
            if regime_id in regime_descriptions:
                row["label"] = regime_descriptions[regime_id].get(
                    "label", f"regime_{regime_id}"
                )
            if method != "bocpd":
                row["avg_conf"] = round(seg["confidence"], 4)
            elif seg.get("transition_conf") is not None:
                row["transition_conf"] = round(float(seg["transition_conf"]), 4)
            final_segments.append(row)

        # Restructure Payload
        # We want 'regimes' to be the MAIN table.
        # We want to hide raw series under 'series' if output='full'.

        new_payload = {
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "method": payload.get("method"),
            "success": True,
        }

        if "threshold" in payload:
            new_payload["threshold"] = payload["threshold"]
        if "reliability" in payload:
            new_payload["reliability"] = payload["reliability"]
        if "tuning_hint" in payload:
            new_payload["tuning_hint"] = payload["tuning_hint"]
        if "warnings" in payload:
            new_payload["warnings"] = payload["warnings"]

        # Add consolidated table
        new_payload["regimes"] = final_segments

        if regime_descriptions:
            new_payload["regime_info"] = regime_descriptions

        # Handle raw series
        if output_mode == "full" and include_series:
            series_data = {}
            for k in [
                "times",
                "cp_prob",
                "state",
                "state_probabilities",
                "change_points",
            ]:
                if k in payload:
                    series_data[k] = payload[k]
            new_payload["series"] = series_data
        elif output_mode == "compact" and include_series:
            # Maybe keep tail of series in 'series'?
            series_data = {}
            for k in ["times", "cp_prob", "state"]:
                if k in payload:
                    series_data[k] = payload[k]  # Already truncated by caller?
            new_payload["series"] = series_data

        # Add params
        if "params_used" in payload:
            new_payload["params_used"] = payload["params_used"]

        return new_payload

    except Exception as e:
        # Fallback to original payload on error
        payload["consolidation_error"] = str(e)
        payload["success"] = False
        return payload


def _summary_only_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a minimal payload for `output='summary'` (no regimes/series)."""
    out: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "timeframe": payload.get("timeframe"),
        "method": payload.get("method"),
        "target": payload.get("target"),
        "success": bool(payload.get("success", True)),
    }
    if "summary" in payload:
        out["summary"] = payload["summary"]
    if "reliability" in payload:
        out["reliability"] = payload["reliability"]
    if "params_used" in payload:
        out["params_used"] = payload["params_used"]
    if "threshold" in payload:
        out["threshold"] = payload["threshold"]
    if "tuning_hint" in payload:
        out["tuning_hint"] = payload["tuning_hint"]
    if "warnings" in payload:
        out["warnings"] = payload["warnings"]
    return out


__all__ = [
    "_consolidate_payload",
    "_summary_only_payload",
    "_interpret_regime_label",
    "_build_regime_descriptions",
]
