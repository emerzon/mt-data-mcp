
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .schema import SimplifySpec
from .constants import SIMPLIFY_DEFAULT_METHOD, SIMPLIFY_DEFAULT_MODE, SIMPLIFY_DEFAULT_MIN_POINTS, SIMPLIFY_DEFAULT_MAX_POINTS, SIMPLIFY_DEFAULT_RATIO
from ..services.simplification import _simplify_dataframe_rows_ext as _simplify_impl
from ..utils.simplify import (
    _choose_simplify_points as _choose_simplify_points_impl,
    _select_indices_for_timeseries as _select_indices_for_timeseries_impl,
    _lttb_select_indices,
    _rdp_select_indices as _rdp_select_indices_impl,
    _pla_select_indices as _pla_select_indices_impl,
    _apca_select_indices as _apca_select_indices_impl,
    _max_line_error as _max_line_error_impl,
    _point_line_distance as _point_line_distance_impl,
    _default_target_points as _default_target_points_impl,
    _rdp_autotune_epsilon as _rdp_autotune_epsilon_impl,
    _pla_autotune_max_error as _pla_autotune_max_error_impl,
    _apca_autotune_max_error as _apca_autotune_max_error_impl
)

# Export helper functions that were previously available here
__all__ = [
    '_simplify_dataframe_rows_ext',
    '_choose_simplify_points',
    '_select_indices_for_timeseries',
    '_lttb_select_indices',
    '_rdp_select_indices',
    '_pla_select_indices',
    '_apca_select_indices'
]

def _simplify_dataframe_rows_ext(df: pd.DataFrame, headers: List[str], simplify: SimplifySpec) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Delegate to services.simplification._simplify_dataframe_rows_ext
    """
    return _simplify_impl(df, headers, simplify)

def _default_target_points(total: int) -> int:
    """Default target points when simplify requested without explicit points/ratio."""
    return _default_target_points_impl(total)


def _choose_simplify_points(total: int, spec: Dict[str, Any]) -> int:
    """Determine target number of points from a simplify spec."""
    return _choose_simplify_points_impl(total, spec)


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Delegate to utils implementation."""
    return _point_line_distance_impl(px, py, x1, y1, x2, y2)


def _rdp_select_indices(x: List[float], y: List[float], epsilon: float) -> List[int]:
    return _rdp_select_indices_impl(x, y, epsilon)


def _max_line_error(x: List[float], y: List[float], i0: int, i1: int) -> float:
    """Delegate to utils implementation.""" 
    return _max_line_error_impl(x, y, i0, i1)


def _pla_select_indices(x: List[float], y: List[float], max_error: Optional[float] = None, segments: Optional[int] = None, points: Optional[int] = None) -> List[int]:
    return _pla_select_indices_impl(x, y, max_error, segments, points)


def _apca_select_indices(y: List[float], max_error: Optional[float] = None, segments: Optional[int] = None, points: Optional[int] = None) -> List[int]:
    return _apca_select_indices_impl(y, max_error, segments, points)


def _select_indices_for_timeseries(x: List[float], y: List[float], spec: Optional[Dict[str, Any]]) -> Tuple[List[int], str, Dict[str, Any]]:
    return _select_indices_for_timeseries_impl(x, y, spec)


def _rdp_autotune_epsilon(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    return _rdp_autotune_epsilon_impl(x, y, target_points, max_iter)


def _pla_autotune_max_error(x: List[float], y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    return _pla_autotune_max_error_impl(x, y, target_points, max_iter)


def _apca_autotune_max_error(y: List[float], target_points: int, max_iter: int = 24) -> Tuple[List[int], float]:
    return _apca_autotune_max_error_impl(y, target_points, max_iter)


def _simplify_dataframe_rows(df: pd.DataFrame, headers: List[str], simplify: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Delegate to utils.simplify implementation to avoid duplication."""
    from ..utils.simplify import _simplify_dataframe_rows as _impl
    return _impl(df, headers, simplify)
