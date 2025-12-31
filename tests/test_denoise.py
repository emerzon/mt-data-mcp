import importlib.util
import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DENOISE_PATH = SRC / "mtdata" / "utils" / "denoise.py"
spec = importlib.util.spec_from_file_location("mtdata.utils.denoise", DENOISE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load denoise module from {DENOISE_PATH}")
denoise = importlib.util.module_from_spec(spec)
sys.modules["mtdata.utils.denoise"] = denoise
spec.loader.exec_module(denoise)
_denoise_series = denoise._denoise_series

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


def test_denoise_series_hp():
    if denoise._sps is None:
        pytest.skip("scipy sparse not available")
    s = pd.Series(np.linspace(0, 1, 30))
    result = _denoise_series(s, method='hp', params={'lamb': 1600})
    assert len(result) == len(s)
    assert np.isfinite(result.values).all()


def test_denoise_series_savgol():
    if denoise._savgol_filter is None:
        pytest.skip("savgol filter not available")
    s = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 31)))
    result = _denoise_series(s, method='savgol', params={'window': 9, 'polyorder': 2})
    assert len(result) == len(s)
    assert np.isfinite(result.values).all()


def test_denoise_series_tv_kalman():
    s = pd.Series(np.linspace(0, 1, 25) + 0.1)
    tv = _denoise_series(s, method='tv', params={'weight': 'auto', 'n_iter': 10})
    kalman = _denoise_series(s, method='kalman', params={'process_var': 'auto', 'measurement_var': 'auto'})
    assert len(tv) == len(s)
    assert len(kalman) == len(s)
    assert np.isfinite(tv.values).all()
    assert np.isfinite(kalman.values).all()


def test_denoise_series_butterworth():
    if denoise._butter is None:
        pytest.skip("scipy.signal not available")
    s = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 64)))
    result = _denoise_series(s, method='butterworth', params={'cutoff': 0.2, 'order': 3})
    assert len(result) == len(s)
    assert np.isfinite(result.values).all()


def test_denoise_series_hampel_bilateral():
    s = pd.Series([1, 1, 1, 10, 1, 1, 1], dtype=float)
    hampel = _denoise_series(s, method='hampel', params={'window': 5, 'n_sigmas': 2.0})
    bilateral = _denoise_series(s, method='bilateral', params={'sigma_s': 2.0, 'sigma_r': 3.0})
    assert len(hampel) == len(s)
    assert len(bilateral) == len(s)
    assert np.isfinite(hampel.values).all()
    assert np.isfinite(bilateral.values).all()


def test_denoise_series_wavelet_packet():
    if denoise._pywt is None:
        pytest.skip("pywt not available")
    s = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 64)))
    result = _denoise_series(s, method='wavelet_packet', params={'wavelet': 'db2', 'level': 2})
    assert len(result) == len(s)
    assert np.isfinite(result.values).all()


def test_denoise_series_ssa_l1_trend():
    s = pd.Series(np.linspace(0, 1, 30) + 0.1)
    ssa = _denoise_series(s, method='ssa', params={'window': 10, 'components': 2})
    l1 = _denoise_series(s, method='l1_trend', params={'lamb': 3.0, 'n_iter': 10})
    assert len(ssa) == len(s)
    assert len(l1) == len(s)
    assert np.isfinite(ssa.values).all()
    assert np.isfinite(l1.values).all()


def test_denoise_series_adaptive_beta():
    s = pd.Series(np.linspace(0, 1, 40))
    lms = _denoise_series(s, method='lms', params={'order': 4, 'mu': 0.05})
    rls = _denoise_series(s, method='rls', params={'order': 4, 'lambda_': 0.98, 'delta': 1.0})
    beta = _denoise_series(s, method='beta', params={'window': 7, 'beta': 1.3, 'n_iter': 5})
    assert len(lms) == len(s)
    assert len(rls) == len(s)
    assert len(beta) == len(s)
    assert np.isfinite(lms.values).all()
    assert np.isfinite(rls.values).all()
    assert np.isfinite(beta.values).all()


def test_denoise_series_vmd():
    if denoise._VMD is None:
        pytest.skip("vmdpy not available")
    s = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 64)))
    result = _denoise_series(s, method='vmd', params={'k': 3, 'alpha': 1000.0, 'drop_modes': [-1]})
    assert len(result) == len(s)
    assert np.isfinite(result.values).all()
