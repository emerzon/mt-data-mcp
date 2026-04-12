# patterns/ — Pattern Detection

Chart pattern and Elliott wave detection. 12 files across `patterns/` and `classic_impl/`.

## PUBLIC API

Exported from `__init__.py`:
- `detect_classic_patterns(df, config)` → `ClassicPatternResult`
- `detect_elliott_waves(df, config)` → `ElliottWaveResult`
- Config classes: `ClassicDetectorConfig`, `ElliottWaveConfig`

## FILE MAP

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 11 | Public API exports (see above) |
| `classic.py` | — | Facade — delegates to `classic_impl/` algorithms |
| `elliott.py` | 757 | Standalone Elliott wave detection |
| `candlestick.py` | — | Candlestick pattern detection |
| `common.py` | — | Shared pattern types and helpers |

### classic_impl/ — Pattern Algorithms

| File | Purpose |
|------|---------|
| `config.py` | Detection configuration/thresholds |
| `reversal.py` | Head-and-shoulders, double tops/bottoms |
| `continuation.py` | Flags, pennants, rectangles |
| `shapes.py` | Triangles, wedges, channels |
| `trend.py` | Trendline-based patterns |
| `utils.py` | Shared geometry/math helpers |
| `__init__.py` | Subpackage init |

## CONVENTIONS

- `classic.py` is the ONLY entry point for classic patterns — never call `classic_impl/` directly.
- Optional dependency `stock-pattern` (external git dep, `patterns-ext` group) adds additional detection.
- Do NOT confuse with `utils/patterns.py` — that file has shared helpers, this package has detectors.
- Consumed by `core/patterns.py` (MCP tools) and `core/patterns_support.py` (896 lines of helpers).
