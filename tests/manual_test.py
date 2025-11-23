import pandas as pd
import numpy as np
try:
    from mtdata.utils.denoise import _denoise_series
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)

def test_denoise_series_none():
    s = pd.Series([1, 2, 3, 4, 5])
    result = _denoise_series(s, method='none')
    pd.testing.assert_series_equal(s, result)
    print("test_denoise_series_none passed")

def test_denoise_series_sma():
    s = pd.Series([1, 2, 3, 4, 5])
    result = _denoise_series(s, method='sma', params={'window': 3})
    print(f"SMA result: {result.values}")
    # Expected: [?, 2.0, 3.0, 4.0, ?] depending on padding
    # utils.denoise: rolling(window=window, center=True, min_periods=1).mean()
    # 1: (1+2)/2 = 1.5
    # 2: (1+2+3)/3 = 2.0
    # 3: (2+3+4)/3 = 3.0
    # 4: (3+4+5)/3 = 4.0
    # 5: (4+5)/2 = 4.5
    assert len(result) == 5
    assert result.iloc[2] == 3.0
    print("test_denoise_series_sma passed")

def test_denoise_series_ema():
    s = pd.Series([1, 2, 3, 4, 5])
    result = _denoise_series(s, method='ema', params={'span': 3})
    assert len(result) == 5
    assert isinstance(result, pd.Series)
    print("test_denoise_series_ema passed")

if __name__ == "__main__":
    try:
        test_denoise_series_none()
        test_denoise_series_sma()
        test_denoise_series_ema()
        with open("test_results.txt", "w") as f:
            f.write("SUCCESS")
    except Exception as e:
        with open("test_results.txt", "w") as f:
            f.write(f"FAILURE: {e}")
