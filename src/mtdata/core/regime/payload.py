"""Payload consolidation utilities for regime detection output formatting."""
from typing import Any, Dict, List


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
        states = []
        probs = []

        if method == "bocpd":
            # For BOCPD, we define regimes by change points
            # We can create a 'regime_id' that increments at each CP
            # We also look at 'change_points' list in payload
            cps_idx = set()
            if "change_points" in payload and isinstance(payload["change_points"], list):
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
            filtered_entries.append((times[idx], state_value, prob_value))
        if filtered_entries and len(filtered_entries) != len(times):
            times = [entry[0] for entry in filtered_entries]
            states = [entry[1] for entry in filtered_entries]
            probs = [entry[2] for entry in filtered_entries]

        # Consolidate
        # Loop through
        curr_start = times[0]
        curr_state = states[0]
        curr_prob_sum = 0.0
        curr_count = 0

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
                segments.append({
                    "start": curr_start,
                    "end": times[i - 1] if i > 0 else curr_start,
                    "duration": curr_count,
                    "regime": curr_state,  # state ID or regime ID
                    "confidence": avg_prob  # average prob of being in this state/regime
                })
                # New segment
                curr_start = t
                curr_state = s
                curr_prob_sum = 0.0
                curr_count = 0

            curr_prob_sum += p
            curr_count += 1
            i += 1

        # Final segment
        if curr_count > 0:
            avg_prob = curr_prob_sum / max(1, curr_count)
            segments.append({
                "start": curr_start,
                "end": times[-1],
                "duration": curr_count,
                "regime": curr_state,
                "confidence": avg_prob
            })

        # Post-process segments for readability
        # For BOCPD, 'confidence' is avg cp_prob which is usually low except at edges.
        # Maybe we want the PEAK prob? or just drop it.
        # For HMM, 'confidence' is avg prob of that state.

        final_segments = []
        for _i, seg in enumerate(segments):
            row = {
                "start": seg["start"],
                "end": seg["end"],
                "bars": seg["duration"],
                "regime": seg["regime"]
            }
            if method != 'bocpd':
                row["avg_conf"] = round(seg["confidence"], 4)
            final_segments.append(row)

        # Restructure Payload
        # We want 'regimes' to be the MAIN table.
        # We want to hide raw series under 'series' if output='full'.

        new_payload = {
            "symbol": payload.get("symbol"),
            "timeframe": payload.get("timeframe"),
            "method": payload.get("method"),
            "success": True
        }

        if "threshold" in payload:
            new_payload["threshold"] = payload["threshold"]
        if "reliability" in payload:
            new_payload["reliability"] = payload["reliability"]
        if "tuning_hint" in payload:
            new_payload["tuning_hint"] = payload["tuning_hint"]

        # Add consolidated table
        new_payload["regimes"] = final_segments

        # Handle raw series
        if output_mode == 'full' and include_series:
            series_data = {}
            for k in ["times", "cp_prob", "state", "state_probabilities", "change_points"]:
                if k in payload:
                    series_data[k] = payload[k]
            new_payload["series"] = series_data
        elif output_mode == 'compact' and include_series:
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
    return out


__all__ = [
    "_consolidate_payload",
    "_summary_only_payload",
]
