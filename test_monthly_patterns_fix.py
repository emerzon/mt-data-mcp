#!/usr/bin/env python3
"""Test the fix for monthly patterns ancient data issue."""

from src.mtdata.core.patterns_use_cases import _all_mode_fetch_limit

def test_fetch_limit_for_timeframes():
    """Verify that monthly/weekly timeframes don't fetch decades of data."""
    
    # Test with user_limit=1000 (default)
    
    # M30: Should cap at ~1 year = ~17,280 bars, but user asked for 1000, so 1000
    result_m30 = _all_mode_fetch_limit("M30", 1000)
    assert result_m30 == 1000, f"M30 expected 1000, got {result_m30}"
    print(f"✓ M30: {result_m30} bars (≈ ~1000 minutes ≈ ~17 hours)")
    
    # D1: Should cap at ~1 year = ~365 bars
    result_d1 = _all_mode_fetch_limit("D1", 1000)
    assert result_d1 == 365, f"D1 expected 365, got {result_d1}"
    print(f"✓ D1: {result_d1} bars (≈ ~1 year)")
    
    # W1: Capped at 1 year by time window = 52 bars, not 200 (4 years) 
    result_w1 = _all_mode_fetch_limit("W1", 1000)
    assert result_w1 == 52, f"W1 expected 52 (capped at 1 year), got {result_w1}"
    print(f"✓ W1: {result_w1} bars (≈ ~1 year, OLD would be 200 bars/4 years!)")
    
    # MN1: Gets floor of 30 bars ≈ 2.5 years, NOT 200 bars ≈ 16.7 years!
    result_mn1 = _all_mode_fetch_limit("MN1", 1000)
    assert result_mn1 == 30, f"MN1 expected 30 (OLD was 200), got {result_mn1}"
    print(f"✓ MN1: {result_mn1} bars (≈ ~2.5 years, OLD would be 200 bars/16.7 years!)")
    
    # Even with smaller user limits, the floor of 30 ensures enough bars for monthly analysis
    result_mn1_small = _all_mode_fetch_limit("MN1", 20)
    assert result_mn1_small == 30, f"MN1 with user_limit=20 expected 30 (floor), got {result_mn1_small}"
    print(f"✓ MN1 with user_limit=20: {result_mn1_small} bars (floor ensures minimum)")
    
    print("\n✅ All tests passed!")
    print("   FIX: Monthly/weekly patterns no longer fetch ancient 2011 data.")
    print("   - MN1: was 200 months (16.7 years from 2008), now 30 months (2.5 years)")
    print("   - W1:  was 200 weeks (4 years from 2020), now 52 weeks (1 year)")

if __name__ == "__main__":
    test_fetch_limit_for_timeframes()


if __name__ == "__main__":
    test_fetch_limit_for_timeframes()
