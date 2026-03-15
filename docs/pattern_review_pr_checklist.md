# Pattern Review Follow-Up PR Checklist

## PR1: Classic Heuristic Tightening

- [x] Add a classic `min_confidence` floor after calibration.
- [x] Tighten rectangle side matching so small samples require full agreement and larger samples allow at most one outlier.
- [x] Require a minimum pole slope for flag and pennant detection to reject slow drifts.
- [x] Scale breakout tolerance from the boundary level instead of median close.
- [x] Remove the extra classic-mode forming filter in orchestration by reusing response output for signal summary.

## PR2: Elliott Stability Cleanup

- [x] Add autotune early-stop or pivot-set dedupe to avoid repeated Elliott passes with identical pivots.
- [x] Keep fallback detections labeled `Candidate` even when rules happen to validate.
- [x] Add Wave 5 projection fields alongside existing Fibonacci metrics.

## PR3: Classic Detector Coverage Expansion

- [x] Cap the H&S peak search window to avoid quadratic blowups on dense pivot sets.
- [x] Run rounding detection over multiple window sizes and keep the best fit.
- [x] Add inverted cup-and-handle detection.
- [x] Replace the diamond prior-pole hardcode with structure-aware sizing.

## PR4: Output Semantics And Enrichment

- [x] Add multi-bar candlestick span metadata (`start_time`, `end_time`, `n_bars`) where applicable.
- [x] Mark completed-pattern targets as stale or derived from an older reference point.
- [x] Move classic bias ownership into detector outputs instead of string inference.
- [x] Refine `bars_to_completion` and broaden invalidation coverage.

## Deferred Design PRs

- [x] Replace `run_patterns_detect` callable injection sprawl with a context object.
- [x] Limit default Elliott multi-timeframe scans to a smaller subset or add a default cap.
- [x] Feed regime context into pattern confidence adjustments.
