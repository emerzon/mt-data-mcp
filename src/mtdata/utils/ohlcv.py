from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def validate_and_clean_ohlcv_frame(
    df: pd.DataFrame,
    *,
    epoch_col: str = "time",
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop malformed OHLCV rows and return user-facing warnings."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("OHLCV data must be a pandas DataFrame.")

    required_cols = [epoch_col, "open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data missing required columns: {', '.join(missing)}")

    clean = df.copy()
    warnings: List[str] = []

    numeric_cols = [epoch_col, "open", "high", "low", "close"]
    for col in numeric_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    finite_mask = np.isfinite(clean[numeric_cols].to_numpy(dtype=float)).all(axis=1)
    removed_nonfinite = int((~finite_mask).sum())
    if removed_nonfinite:
        warnings.append(
            f"Removed {removed_nonfinite} candle row(s) with non-finite time/OHLC values."
        )
        clean = clean.loc[finite_mask].copy()

    if clean.empty:
        return clean.reset_index(drop=True), warnings

    range_mask = (
        (clean["low"] <= clean["high"])
        & (clean["open"] >= clean["low"])
        & (clean["open"] <= clean["high"])
        & (clean["close"] >= clean["low"])
        & (clean["close"] <= clean["high"])
    )
    removed_range = int((~range_mask).sum())
    if removed_range:
        warnings.append(
            f"Removed {removed_range} candle row(s) with inconsistent OHLC ranges."
        )
        clean = clean.loc[range_mask].copy()

    if clean.empty:
        return clean.reset_index(drop=True), warnings

    if not bool(clean[epoch_col].is_monotonic_increasing):
        clean = clean.sort_values(epoch_col, kind="mergesort").reset_index(drop=True)
        warnings.append("Sorted candle rows by timestamp after detecting out-of-order data.")

    duplicate_mask = clean.duplicated(subset=[epoch_col], keep="last")
    removed_duplicates = int(duplicate_mask.sum())
    if removed_duplicates:
        clean = clean.loc[~duplicate_mask].reset_index(drop=True)
        warnings.append(f"Removed {removed_duplicates} duplicate candle timestamp(s).")

    return clean.reset_index(drop=True), warnings
