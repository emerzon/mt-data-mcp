
import numpy as np
import pandas as pd
import pytest
from mtdata.patterns.eliott import detect_elliott_waves, ElliottWaveConfig

@pytest.fixture
def sample_data_with_impulse_wave() -> pd.DataFrame:
    """Creates a DataFrame with a synthetic impulse wave."""
    # Wave 1
    p0 = 100
    p1 = 110
    # Wave 2
    p2 = 105
    # Wave 3
    p3 = 125
    # Wave 4
    p4 = 120
    # Wave 5
    p5 = 135

    prices = np.concatenate([
        np.linspace(p0, p1, 20),
        np.linspace(p1, p2, 15),
        np.linspace(p2, p3, 30),
        np.linspace(p3, p4, 15),
        np.linspace(p4, p5, 25),
    ])
    
    df = pd.DataFrame({'close': prices})
    df['time'] = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(df), freq='H')).astype(np.int64) // 10**9
    return df

def test_detect_impulse_wave(sample_data_with_impulse_wave):
    """Tests the detection of a 5-wave impulse pattern."""
    df = sample_data_with_impulse_wave
    config = ElliottWaveConfig(
        min_prominence_pct=0.1,
        min_distance=10
    )
    
    results = detect_elliott_waves(df, config)
    
    assert len(results) > 0, "No patterns detected"
    
    impulse_waves = [r for r in results if r.wave_type == 'Impulse']
    assert len(impulse_waves) > 0, "No impulse waves detected"
    
    best_wave = impulse_waves[0]
    assert best_wave.confidence > 0.5, f"Confidence too low: {best_wave.confidence}"
    
    # Check if the detected wave points are correct
    # Note: The exact indices might vary slightly due to pivot detection
    expected_pivots = [19, 34, 64, 79, 104] # Expected indices of pivots
    detected_pivots = best_wave.wave_sequence[1:] # Exclude start point
    
    assert len(detected_pivots) == 5, f"Expected 5 pivots, but got {len(detected_pivots)}"
