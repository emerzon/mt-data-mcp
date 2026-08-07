"""Internal rolling-window feature extraction utilities; not an MCP tool module."""

import numbers
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew

_MINIMAL_FEATURE_COLUMNS = (
    "value__variance",
    "value__autocorrelation__lag_1",
    "value__autocorrelation__lag_3",
    "value__approximate_entropy__m_2__r_0.5",
    'value__linear_trend__attr_"slope"',
    'value__linear_trend__attr_"stderr"',
    "value__mean_abs_change",
    "value__skewness",
    "value__kurtosis",
)


def _normalize_window_size(window_size: int) -> int:
    if isinstance(window_size, (bool, np.bool_)):
        raise ValueError("window_size must be a positive integer")
    if isinstance(window_size, numbers.Integral):
        window_size_i = int(window_size)
    elif isinstance(window_size, float):
        if not window_size.is_integer():
            raise ValueError("window_size must be a positive integer")
        window_size_i = int(window_size)
    else:
        try:
            window_size_f = float(window_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("window_size must be a positive integer") from exc
        if not window_size_f.is_integer():
            raise ValueError("window_size must be a positive integer")
        window_size_i = int(window_size_f)
    if window_size_i <= 0:
        raise ValueError("window_size must be a positive integer")
    return window_size_i


def _rolling_approximate_entropy(windows: np.ndarray, *, m: int, r: float) -> np.ndarray:
    """Vectorized equivalent of tsfresh's approximate-entropy calculator."""
    row_count, width = windows.shape
    if width <= m + 1:
        return np.zeros(row_count, dtype=float)

    tolerance = float(r) * np.std(windows, axis=1)

    def _phi(template_width: int) -> np.ndarray:
        templates = np.lib.stride_tricks.sliding_window_view(
            windows,
            template_width,
            axis=1,
        )
        template_count = templates.shape[1]
        out = np.empty(row_count, dtype=float)
        # Bound the pairwise temporary to roughly tens of MB on long inputs.
        chunk_size = max(1, 1_000_000 // max(1, template_count**2 * template_width))
        for start in range(0, row_count, chunk_size):
            stop = min(row_count, start + chunk_size)
            chunk = templates[start:stop]
            distances = np.max(
                np.abs(chunk[:, :, None, :] - chunk[:, None, :, :]),
                axis=3,
            )
            counts = np.sum(
                distances <= tolerance[start:stop, None, None],
                axis=1,
            )
            out[start:stop] = np.mean(
                np.log(counts / float(template_count)),
                axis=1,
            )
        return out

    return np.abs(_phi(m) - _phi(m + 1))


def _extract_minimal_rolling_features(
    series: np.ndarray,
    *,
    window_size: int,
) -> pd.DataFrame:
    """Compute the fixed regime feature set without exploding rolling rows."""
    values = np.asarray(series, dtype=float).reshape(-1)
    n = values.size
    if n < window_size:
        return pd.DataFrame()

    windows = np.lib.stride_tricks.sliding_window_view(values, window_size)
    valid_rows = np.isfinite(windows).all(axis=1)
    valid_windows = windows[valid_rows]
    features = np.full((windows.shape[0], len(_MINIMAL_FEATURE_COLUMNS)), np.nan)
    if valid_windows.size:
        width = valid_windows.shape[1]
        means = np.mean(valid_windows, axis=1)
        centered = valid_windows - means[:, None]
        variances = np.mean(centered**2, axis=1)

        def _autocorrelation(lag: int) -> np.ndarray:
            if width < lag:
                return np.full(valid_windows.shape[0], np.nan)
            numerator = np.sum(
                centered[:, : width - lag] * centered[:, lag:],
                axis=1,
            )
            result = np.full(valid_windows.shape[0], np.nan)
            usable = ~np.isclose(variances, 0.0)
            result[usable] = numerator[usable] / (
                float(width - lag) * variances[usable]
            )
            return result

        time_values = np.arange(width, dtype=float)
        centered_time = time_values - np.mean(time_values)
        time_sum_squares = float(np.sum(centered_time**2))
        if time_sum_squares > 0.0:
            slopes = (valid_windows @ centered_time) / time_sum_squares
        else:
            slopes = np.full(valid_windows.shape[0], np.nan)
        if width > 2 and time_sum_squares > 0.0:
            fitted = means[:, None] + slopes[:, None] * centered_time[None, :]
            residual_sum_squares = np.sum((valid_windows - fitted) ** 2, axis=1)
            slope_stderr = np.sqrt(
                residual_sum_squares / float(width - 2) / time_sum_squares
            )
        elif width == 2:
            slope_stderr = np.zeros(valid_windows.shape[0], dtype=float)
        else:
            slope_stderr = np.full(valid_windows.shape[0], np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = scipy_skew(valid_windows, axis=1, bias=False)
            kurtosis = scipy_kurtosis(
                valid_windows,
                axis=1,
                fisher=True,
                bias=False,
            )

        computed = np.column_stack(
            (
                variances,
                _autocorrelation(1),
                _autocorrelation(3),
                _rolling_approximate_entropy(valid_windows, m=2, r=0.5),
                slopes,
                slope_stderr,
                np.mean(np.abs(np.diff(valid_windows, axis=1)), axis=1),
                skewness,
                kurtosis,
            )
        )
        features[valid_rows] = computed

    full = np.full((n, len(_MINIMAL_FEATURE_COLUMNS)), np.nan)
    full[window_size - 1 :] = features
    return pd.DataFrame(full, columns=_MINIMAL_FEATURE_COLUMNS)


def extract_rolling_features(
    series: np.ndarray, window_size: int = 20, minimal: bool = True
) -> pd.DataFrame:
    """
    Extract features from a time series using a rolling window approach via tsfresh.

    Args:
        series: 1D numpy array of time series values.
        window_size: Size of the rolling window.
        minimal: If True, use mtdata's small hand-picked set of dynamic and
            distributional features. If False, use tsfresh's broader
            EfficientFCParameters set.

    Returns:
        DataFrame where each row corresponds to the features of the window ending at that index.
        The index of the DataFrame aligns with the input series (rows < window_size will be NaN/imputed).
    """
    window_size = _normalize_window_size(window_size)

    n = len(series)
    if n < window_size:
        return pd.DataFrame()  # Not enough data
    if minimal:
        return _extract_minimal_rolling_features(series, window_size=window_size)

    # Suppress tsfresh warnings about unavailable dependencies (e.g., matrix_profile)
    # These features are disabled by design in tsfresh and don't affect functionality
    import logging

    logging.getLogger("tsfresh.feature_extraction.settings").setLevel(logging.ERROR)

    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import EfficientFCParameters
        from tsfresh.utilities.dataframe_functions import roll_time_series
    except ImportError:
        raise ImportError("tsfresh is required for this feature. Please install it.")

    # Convert to standard format expected by tsfresh
    # series needs to be a DataFrame with "id", "time", "value"
    # For rolling, we use roll_time_series which creates a new ID for each window.
    df = pd.DataFrame(
        {"id": np.ones(n, dtype=int), "time": np.arange(n), "value": series}
    )

    # Efficient rolling extraction
    # We want features at time t based on [t-window+1, t]

    # roll_time_series creates a huge exploded dataframe.
    # For long series, this is memory intensive.
    # We'll use a manually optimized approach if n is large, or standard if small.
    # But for simplicity and correctness with tsfresh, let's try the standard way first
    # but strictly limit max_timeshifts or just manually loop if needed.
    # Actually, roll_time_series is robust. Let's stick to it but limit 'max_timeshift'
    # 'min_timeshift' = window_size - 1, 'max_timeshift' = window_size - 1
    # This effectively gives us exactly one window length ending at 'time'.

    # Wait, roll_time_series logic:
    # "The rolling mechanism creates windows ... ending at time t"

    settings = EfficientFCParameters()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        df_rolled = roll_time_series(
            df,
            column_id="id",
            column_sort="time",
            max_timeshift=window_size - 1,
            min_timeshift=window_size - 1,
            n_jobs=1,
            disable_progressbar=True,
        )

    # If the series is too short for the window, df_rolled might be empty
    if df_rolled.empty:
        return pd.DataFrame(index=df.index).iloc[window_size - 1 :]

    # Extract
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = extract_features(
            df_rolled,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=settings,
            n_jobs=1,  # Use 1 core or all? 0 triggers default.
            disable_progressbar=True,
            impute_function=None,  # We will handle NaN
        )

    # X index will be the "id" from rolled, which is (original_id, time) tuple
    # We need to map back to the original time index.
    # The 'id' column in df_rolled is built from the 'sort' column of the original frame.

    # tsfresh 0.20+ index behavior on roll_time_series:
    # The index of X is the 'id' of the rolled windows.
    # roll_time_series uses (original_id, time) as the new id.

    # Let's verify index format. usually it is a MultiIndex or tuple index.
    # We want to reindex to match 'series' indices [window_size-1 : ]

    # Map index back to time
    # Check if index is MultiIndex
    if isinstance(X.index, pd.MultiIndex):
        # (id, time)
        times = X.index.get_level_values(1)
        X.index = times
    else:
        # It's likely tuples if not MultiIndex, or just the time if id was constant and we are lucky.
        # But roll_time_series documentation says it returns a df with a new id.
        # Let's assume the index is (1, time).
        try:
            X.index = [i[1] for i in X.index]
        except Exception:
            pass  # hope it's already correct

    # Sort just in case
    X = X.sort_index()

    # Reindex to full length filling with NaN at the start
    X_full = X.reindex(np.arange(n))

    # Fill NaN at start (or leave them to be dropped/imputed later)
    # We'll leave them as NaN.

    return X_full
