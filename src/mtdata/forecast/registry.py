from typing import Any, Dict, List, Type

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
        return list(cls._methods)

    @classmethod
    def get_class(cls, name: str) -> Type[ForecastMethod]:
        """Get the class of a registered forecast method."""
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]

    @classmethod
    def get_all_method_names(cls) -> List[str]:
        """Get all available forecast method names from the registered classes."""
        return sorted(cls._methods.keys())

    @classmethod
    def get_method_info(cls, name: str) -> Dict[str, Any]:
        """Return capability metadata for a single registered method."""
        inst = cls.get(name)
        return {
            "name": name,
            "category": inst.category,
            "supports_training": inst.supports_training,
            "training_category": inst.training_category,
        }

    @classmethod
    def list_trainable(cls) -> List[str]:
        """Return names of methods that support the train/predict lifecycle."""
        return [
            name for name in cls._methods
            if cls._methods[name]().supports_training
        ]
