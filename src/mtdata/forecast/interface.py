from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type
import pandas as pd
import numpy as np

@dataclass
class ForecastResult:
    forecast: np.ndarray
    ci_values: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lower, upper)
    params_used: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ForecastMethod(ABC):
    """Abstract base class for all forecasting methods."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the method."""
        ...
    
    @property
    def category(self) -> str:
        """Category of the method (e.g., 'classical', 'neural', 'ensemble')."""
        return "unknown"
        
    @property
    def required_packages(self) -> List[str]:
        """List of Python packages required for this method."""
        return []

    @property
    def supports_features(self) -> Dict[str, bool]:
        """Dictionary of supported features (price, return, volatility, ci)."""
        return {"price": True, "return": True, "volatility": False, "ci": False}

    @abstractmethod
    def forecast(
        self, 
        series: pd.Series, 
        horizon: int, 
        seasonality: int, 
        params: Dict[str, Any], 
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ForecastResult:
        """
        Generate a forecast.
        
        Args:
            series: Historical time series data.
            horizon: Number of steps to forecast.
            seasonality: Seasonality period.
            params: Method-specific parameters.
            exog_future: Future exogenous variables (optional).
            **kwargs: Additional arguments.
            
        Returns:
            ForecastResult object containing the forecast and metadata.
        """
        ...
    
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate method-specific parameters.
        
        Args:
            params: Dictionary of parameters to validate.
            
        Returns:
            List of error messages (empty if valid).
        """
        return []
