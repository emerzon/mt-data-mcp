"""Base classes and registry for denoising filters."""
from typing import Callable, Dict, Any, Optional, List
import numpy as np
import pandas as pd

# Registry of filter functions
_filter_registry: Dict[str, Callable] = {}


def register_filter(name: str, func: Optional[Callable] = None) -> Callable:
    """Register a filter function in the registry.

    Can be used as a decorator:
        @register_filter("my_filter")
        def my_filter(data, params):
            ...

    Or as a function:
        register_filter("my_filter", my_filter_func)
    """
    def decorator(f: Callable) -> Callable:
        _filter_registry[name] = f
        return f

    if func is not None:
        return decorator(func)
    return decorator


def get_filter(name: str) -> Callable:
    """Get a registered filter by name."""
    if name not in _filter_registry:
        raise ValueError(f"Unknown filter: {name}. Registered filters: {list(_filter_registry.keys())}")
    return _filter_registry[name]


def list_filters() -> List[str]:
    """List all registered filter names."""
    return list(_filter_registry.keys())


def _series_like(data: Any) -> bool:
    """Check if data is series-like (1D array or pandas Series)."""
    if isinstance(data, pd.Series):
        return True
    if isinstance(data, np.ndarray):
        return data.ndim == 1
    return False


def _get_values(data: Any) -> np.ndarray:
    """Extract numpy array values from series-like data."""
    if isinstance(data, pd.Series):
        return data.values
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


__all__ = [
    "register_filter",
    "get_filter",
    "list_filters",
    "_filter_registry",
    "_series_like",
    "_get_values",
]
