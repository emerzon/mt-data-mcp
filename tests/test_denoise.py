import pytest
import pandas as pd
import numpy as np
from mtdata.utils.denoise import _denoise_series

def test_denoise_series_none():
    s = pd.Series([1, 2, 3, 4, 5])
    result = _denoise_series(s, method='none')
    pd.testing.assert_series_equal(s, result)

def test_denoise_series_sma():
    s = pd.Series([1, 2, 3, 4, 5])
    # SMA with window 3:
    # 1: (1+2)/2 = 1.5 (min_periods=1) -> actually rolling mean centered?
    # utils.denoise: rolling(window=window, center=True, min_periods=1).mean()
    # 1: (1+2)/2 = 1.5
    # 2: (1+2+3)/3 = 2
    # 3: (2+3+4)/3 = 3
    # 4: (3+4+5)/3 = 4
    # 5: (4+5)/2 = 4.5
    result = _denoise_series(s, method='sma', params={'window': 3})
    assert len(result) == 5
    assert result.iloc[2] == 3.0

def test_denoise_series_ema():
    s = pd.Series([1, 2, 3, 4, 5])
    result = _denoise_series(s, method='ema', params={'span': 3})
    assert len(result) == 5
    # Check it returns a series
    assert isinstance(result, pd.Series)
