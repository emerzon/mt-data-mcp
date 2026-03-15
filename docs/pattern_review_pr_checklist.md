# Pattern Review Follow-Up PR Checklist

## PR1: Classic Heuristic Tightening

- [x] Add a classic `min_confidence` floor after calibration.
- [x] Tighten rectangle side matching so small samples require full agreement and larger samples allow at most one outlier.
- [x] Require a minimum pole slope for flag and pennant detection to reject slow drifts.
- [x] Scale breakout tolerance from the boundary level instead of median close.
- [x] Remove the extra classic-mode forming filter in orchestration by reusing response output for signal summary.

## PR2: Elliott Stability Cleanup

- [ ] Add autotune early-stop or pivot-set dedupe to avoid repeated Elliott passes with identical pivots.
- [ ] Keep fallback detections labeled `Candidate` even when rules happen to validate.
- [ ] Add Wave 5 projection fields alongside existing Fibonacci metrics.

## PR3: Classic Detector Coverage Expansion

- [ ] Cap the H&S peak search window to avoid quadratic blowups on dense pivot sets.
- [ ] Run rounding detection over multiple window sizes and keep the best fit.
- [ ] Add inverted cup-and-handle detection.
- [ ] Replace the diamond prior-pole hardcode with structure-aware sizing.

## PR4: Output Semantics And Enrichment

- [ ] Add multi-bar candlestick span metadata (`start_time`, `end_time`, `n_bars`) where applicable.
- [ ] Mark completed-pattern targets as stale or derived from an older reference point.
- [ ] Move classic bias ownership into detector outputs instead of string inference.
- [ ] Refine `bars_to_completion` and broaden invalidation coverage.

## Deferred Design PRs

- [ ] Replace `run_patterns_detect` callable injection sprawl with a context object.
- [ ] Limit default Elliott multi-timeframe scans to a smaller subset or add a default cap.
- [ ] Feed regime context into pattern confidence adjustments.
