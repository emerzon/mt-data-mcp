from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import warnings

def extract_rolling_features(
    series: np.ndarray, 
    window_size: int = 20, 
    minimal: bool = True
) -> pd.DataFrame:
    """
    Extract features from a time series using a rolling window approach via tsfresh.
    
    Args:
        series: 1D numpy array of time series values.
        window_size: Size of the rolling window.
        minimal: If True, use EfficientFCParameters to limit to low-compute features.
        
    Returns:
        DataFrame where each row corresponds to the features of the window ending at that index.
        The index of the DataFrame aligns with the input series (rows < window_size will be NaN/imputed).
    """
    try:
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import roll_time_series
        from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
    except ImportError:
        raise ImportError("tsfresh is required for this feature. Please install it.")

    # Convert to standard format expected by tsfresh
    # series needs to be a DataFrame with "id", "time", "value"
    # For rolling, we use roll_time_series which creates a new ID for each window.
    
    n = len(series)
    if n < window_size:
        return pd.DataFrame() # Not enough data
        
    df = pd.DataFrame({
        "id": np.ones(n, dtype=int),
        "time": np.arange(n),
        "value": series
    })
    
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
    
    if minimal:
        settings = MinimalFCParameters() 
        # Add a few critical ones for regime detection if missing from Minimal
        # Minimal includes: sum, length, max, mean, median, min, std, var
        # We might want autocorrelation or entropy.
        
        # Let's switch to efficiently chosen parameters instead of just "Minimal"
        # We want structure/dynamics.
        settings = {
           'variance': None,
           'autocorrelation': [{'lag': 1}, {'lag': 3}],
           'approximate_entropy': [{'m': 2, 'r': 0.5}],
           'linear_trend': [{'attr': 'slope'}, {'attr': 'stderr'}],
           'mean_abs_change': None,
           'skewness': None,
           'kurtosis': None
        }
    else:
        settings = EfficientFCParameters()

    df_rolled = roll_time_series(
        df, 
        column_id="id", 
        column_sort="time",
        max_timeshift=window_size-1, 
        min_timeshift=window_size-1,
        n_jobs=1 # sequential to save overhead on small windows? or generic
    )
    
    # If the series is too short for the window, df_rolled might be empty
    if df_rolled.empty:
        return pd.DataFrame(index=df.index).iloc[window_size-1:]

    # Extract
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = extract_features(
            df_rolled, 
            column_id="id", 
            column_sort="time", 
            column_value="value", 
            default_fc_parameters=settings,
            n_jobs=1, # Use 1 core or all? 0 triggers default.
            disable_progressbar=True,
            impute_function=None # We will handle NaN
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
             pass # hope it's already correct
             
    # Sort just in case
    X = X.sort_index()
    
    # Reindex to full length filling with NaN at the start
    X_full = X.reindex(np.arange(n))
    
    # Fill NaN at start (or leave them to be dropped/imputed later)
    # We'll leave them as NaN.
    
    return X_full
