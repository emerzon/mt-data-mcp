from __future__ import annotations

import pytest

from mtdata.forecast.methods.pretrained_helpers import process_quantile_levels


def test_process_quantile_levels_normalizes_valid_values() -> None:
    assert process_quantile_levels(["0.1", 0.5, 0.9], "chronos") == [0.1, 0.5, 0.9]
    assert process_quantile_levels([], "chronos") is None
    assert process_quantile_levels(None, "chronos") is None


@pytest.mark.parametrize("quantiles", [[0], [1], [-0.1], [1.1], [float("nan")], [float("inf")]])
def test_process_quantile_levels_rejects_levels_outside_open_unit_interval(quantiles) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        process_quantile_levels(quantiles, "timesfm")


@pytest.mark.parametrize("quantiles", ["0.5", [None], ["not-a-number"]])
def test_process_quantile_levels_rejects_malformed_values(quantiles) -> None:
    with pytest.raises(ValueError, match="timesfm quantiles"):
        process_quantile_levels(quantiles, "timesfm")
