# utils/ — Cross-Cutting Utilities

Shared numerical and formatting helpers imported by `core/`, `forecast/`, `patterns/`, and `services/`. 15 files, heavy numpy/scipy/TA-Lib code.

## FILE MAP

| File | Lines | Purpose |
|------|-------|---------|
| `indicators.py` | — | 100+ technical indicators (wraps TA-Lib + pandas-ta) |
| `denoise.py` | 1285 | 10+ signal filters: wavelet, EMD, VMD, Kalman, Savgol, LOESS, etc. |
| `simplify.py` | 859 | Price series simplification/compression |
| `dimred.py` | 588 | Dimension reduction (PCA, t-SNE, UMAP wrappers) |
| `minimal_output.py` | 594 | Compact output formatting for CLI/MCP responses |
| `patterns.py` | 672 | Pattern detection helpers (NOT the `patterns/` module) |
| `formatting.py` | — | String/number formatting utilities |
| `mt5.py` | — | MT5 connection wrapper (`mt5_connection` context manager) |
| `mt5_enums.py` | — | MT5 enum definitions |
| `barriers.py` | — | Barrier calculation helpers |
| `constants.py` | — | Shared constants |
| `regime.py` | — | Regime detection helpers |
| `symbol.py` | — | Symbol normalization/lookup |
| `utils.py` | — | General-purpose helpers (`_normalize_ohlcv_arg`, etc.) |
| `__init__.py` | — | Package init (docstring only) |

## CONVENTIONS

- **`utils/patterns.py` ≠ `patterns/`**: This file has shared pattern helpers; the `patterns/` package has the actual detectors.
- **No `__init__.py` exports** — import modules directly: `from mtdata.utils.denoise import ...`
- **Heavy numerical**: Most files depend on numpy, scipy, pandas. `indicators.py` requires TA-Lib C library.
- **`denoise.py` filter methods** follow a common signature: `(series, **params) → filtered_series`. Each filter is a standalone function.
