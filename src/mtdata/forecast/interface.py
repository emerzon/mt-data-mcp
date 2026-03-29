from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

@dataclass
class ForecastResult:
    forecast: np.ndarray
    ci_values: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lower, upper)
    params_used: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ForecastCapabilityDescriptor:
    capability_id: str
    method: str
    adapter_method: str
    library: str
    namespace: str
    concept: str
    display_name: str
    category: str
    description: str = ""
    available: bool = True
    requires: Tuple[str, ...] = ()
    supports: Dict[str, bool] = field(default_factory=dict)
    params: Tuple[Dict[str, Any], ...] = ()
    aliases: Tuple[str, ...] = ()
    selector_key: Optional[str] = None
    selector_value: Optional[str] = None
    selector_mode: str = "method"
    source: str = "registry"
    notes: Optional[str] = None

    def selector(self) -> Dict[str, Any]:
        selector: Dict[str, Any] = {"mode": self.selector_mode}
        if self.selector_key is not None:
            selector["key"] = self.selector_key
        if self.selector_value is not None:
            selector["value"] = self.selector_value
        if self.aliases:
            selector["aliases"] = list(self.aliases)
        return selector

    def execution(self) -> Dict[str, Any]:
        execution: Dict[str, Any] = {
            "library": self.library,
            "method": self.adapter_method,
        }
        if self.selector_key is not None and self.selector_value is not None:
            execution["params"] = {self.selector_key: self.selector_value}
        return execution

    def to_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "capability_id": self.capability_id,
            "method": self.method,
            "adapter_method": self.adapter_method,
            "library": self.library,
            "namespace": self.namespace,
            "concept": self.concept,
            "display_name": self.display_name,
            "category": self.category,
            "description": self.description,
            "available": self.available,
            "requires": list(self.requires),
            "supports": dict(self.supports),
            "params": [dict(param) for param in self.params],
            "aliases": list(self.aliases),
            "selector": self.selector(),
            "execution": self.execution(),
            "source": self.source,
        }
        if self.notes:
            record["notes"] = self.notes
        return record


@dataclass(frozen=True)
class ForecastCallContext:
    method: str
    symbol: str
    timeframe: str
    quantity: str
    horizon: int
    seasonality: int
    base_col: str
    ci_alpha: Optional[float]
    as_of: Optional[str]
    denoise_spec_used: Optional[Any]
    history_df: pd.DataFrame
    target_series: pd.Series
    exog_used: Optional[np.ndarray]
    future_exog: Optional[np.ndarray]

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

    def prepare_forecast_call(
        self,
        params: Dict[str, Any],
        call_kwargs: Dict[str, Any],
        context: ForecastCallContext,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Allow methods to request engine context without engine name checks."""
        return params, call_kwargs
