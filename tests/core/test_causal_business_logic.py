from mtdata.core.causal import _compact_causal_pair_rows


def test_compact_causal_pair_rows_keeps_near_miss_test_context() -> None:
    rows = [
        {
            "effect": "EURUSD",
            "cause": "GBPUSD",
            "lag": 2,
            "p_value": 0.08,
            "p_value_raw": 0.04,
            "p_value_correction": "bonferroni",
            "significance_basis": "p_value_bonferroni_adjusted",
            "significance_threshold": 0.05,
            "samples": 120,
            "significant": False,
            "extra": "omitted",
        }
    ]

    assert _compact_causal_pair_rows(rows) == [
        {
            "effect": "EURUSD",
            "cause": "GBPUSD",
            "lag": 2,
            "p_value": 0.08,
            "p_value_raw": 0.04,
            "p_value_correction": "bonferroni",
            "significance_basis": "p_value_bonferroni_adjusted",
            "significance_threshold": 0.05,
            "significant": False,
        }
    ]
