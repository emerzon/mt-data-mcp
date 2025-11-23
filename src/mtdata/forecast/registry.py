from typing import Dict, List, Type, Optional
from .interface import ForecastMethod

class ForecastRegistry:
    """Registry for forecasting methods."""
    
    _methods: Dict[str, Type[ForecastMethod]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a forecast method class."""
        def decorator(method_cls: Type[ForecastMethod]):
            cls._methods[name] = method_cls
            return method_cls
        return decorator
        
    @classmethod
    def get(cls, name: str) -> ForecastMethod:
        """Get an instance of a registered forecast method."""
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]()
        
    @classmethod
    def list_available(cls) -> List[str]:
        """List names of all registered methods."""
        return list(cls._methods.keys())

    @classmethod
    def get_class(cls, name: str) -> Type[ForecastMethod]:
        """Get the class of a registered forecast method."""
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]
