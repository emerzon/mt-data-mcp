from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    forecast: np.ndarray
    ci_values: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lower, upper)
    params_used: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrainingProgress:
    """Progress update emitted during model training via ``progress_callback``."""

    step: int
    total_steps: int
    loss: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None

    @property
    def fraction(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, max(0.0, self.step / self.total_steps))


TrainingCategory = Literal["instant", "fast", "moderate", "heavy"]
ProgressCallback = Callable[[TrainingProgress], None]


@dataclass
class TrainedModelHandle:
    """Opaque reference to a persisted trained model."""

    model_id: str
    method: str
    data_scope: str
    params_hash: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Structured return value from :meth:`ForecastMethod.train`.

    *artifact_bytes* is the method-serialized model (bytes).  The model
    store persists this opaque blob; deserialization is method-owned via
    :meth:`ForecastMethod.deserialize_artifact`.
    """

    artifact_bytes: bytes
    params_used: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    # ------------------------------------------------------------------
    # Train / predict lifecycle (opt-in for methods with heavy training)
    # ------------------------------------------------------------------

    @property
    def supports_training(self) -> bool:
        """Whether this method supports a separate train → predict lifecycle."""
        return False

    @property
    def training_category(self) -> TrainingCategory:
        """Hint for task routing: 'instant', 'fast', 'moderate', or 'heavy'."""
        return "instant"

    def train(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        progress_callback: Optional[ProgressCallback] = None,
        exog: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> TrainResult:
        """Train and return a serialized model artifact.

        The returned :class:`TrainResult` contains method-serialized bytes
        that :meth:`deserialize_artifact` can reconstruct.  Override in
        methods that support separate training.
        """
        raise NotImplementedError(
            f"{self.name} does not support separate training"
        )

    def predict_with_model(
        self,
        model: Any,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> ForecastResult:
        """Generate a forecast using a previously trained model artifact.

        *model* is the deserialized object returned by
        :meth:`deserialize_artifact`.  Default implementation ignores
        *model* and falls back to :meth:`forecast`.
        """
        return self.forecast(
            series, horizon, seasonality, params,
            exog_future=exog_future, **kwargs,
        )

    def serialize_artifact(self, artifact: Any) -> bytes:
        """Serialize a trained model artifact to bytes.

        Default uses pickle.  Override for framework-specific formats
        (e.g. ``torch.save`` for neural models).
        """
        import pickle
        return pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize_artifact(self, data: bytes) -> Any:
        """Deserialize bytes produced by :meth:`serialize_artifact`.

        Default uses pickle.  Override to match :meth:`serialize_artifact`.
        """
        import pickle
        return pickle.loads(data)  # noqa: S301

    def training_fingerprint(
        self,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        timeframe: Optional[str] = None,
        has_exog: bool = False,
    ) -> Dict[str, Any]:
        """Return a canonical dict of training-relevant attributes.

        This fingerprint is hashed to produce the ``params_hash`` used
        for model identity.  Override to include method-specific training
        parameters that affect the trained artifact.
        """
        _PREDICTION_ONLY_KEYS = frozenset({
            "ci_alpha", "quantity", "as_of",
        })
        filtered_params = {
            k: v for k, v in sorted((params or {}).items())
            if k not in _PREDICTION_ONLY_KEYS
        }
        return {
            "method": self.name,
            "horizon": int(horizon),
            "seasonality": int(seasonality),
            "timeframe": timeframe,
            "has_exog": bool(has_exog),
            "params": filtered_params,
        }

    @staticmethod
    def hash_fingerprint(fingerprint: Dict[str, Any]) -> str:
        """Compute a stable short hash from a training fingerprint dict."""
        import hashlib
        import json
        blob = json.dumps(fingerprint, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
