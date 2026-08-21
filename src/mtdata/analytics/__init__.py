"""Read-only trading analytics engines built from MT5-native data.

Implementation is split by engine:

- ``engine_common`` — shared frames, windows, ticks, rates
- ``microstructure`` — ``analyze_microstructure``
- ``execution_quality`` — ``analyze_execution_quality``
- ``strategy_validate`` — ``validate_strategies``
- ``portfolio_risk`` — ``decompose_portfolio_risk``
- ``relative_strength`` — ``rank_relative_strength``

``engines`` re-exports the public functions and the helpers tests import.
"""

