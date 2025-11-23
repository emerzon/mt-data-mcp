

from typing import Any, Dict, Optional, List
import pandas as pd

from .schema import DenoiseSpec
from .server import mcp

def _denoise_series(
    s: pd.Series,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    causality: Optional[str] = None,
) -> pd.Series:
    """Delegate to utils.denoise implementation to avoid duplication."""
    from ..utils.denoise import _denoise_series as _impl
    return _impl(s, method, params, causality or 'zero_phase')


def _apply_denoise(
    df: pd.DataFrame,
    spec: Optional[DenoiseSpec],
    default_when: str = 'post_ti',
) -> List[str]:
    """Delegate to utils.denoise implementation to avoid duplication."""
    from ..utils.denoise import _apply_denoise as _impl
    return _impl(df, spec, default_when)

def _get_denoise_methods_data_safe() -> Dict[str, Any]:
    try:
        from ..utils.denoise import get_denoise_methods_data
        return get_denoise_methods_data()
    except Exception as e:
        return {"error": f"Error listing denoise methods: {e}"}


@mcp.tool()
def denoise_list_methods() -> Dict[str, Any]:
    """List available denoise methods and their parameters."""
    return _get_denoise_methods_data_safe()
