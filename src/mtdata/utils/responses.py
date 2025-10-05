"""
Unified response formatting utilities for consistent API responses.
Enforces DRY principle across all server tools and CLI commands.
"""

from typing import Any, Optional, Dict, Callable
from functools import wraps
import traceback


def success(data: Any = None, **kwargs) -> Dict[str, Any]:
    """
    Create a standard success response.
    
    Args:
        data: Primary response data (optional)
        **kwargs: Additional fields to include in response
    
    Returns:
        Dictionary with success=True and optional data/fields
    
    Examples:
        >>> success({"price": 1.0850})
        {'success': True, 'data': {'price': 1.0850}}
        
        >>> success(count=42, message="Processed")
        {'success': True, 'count': 42, 'message': 'Processed'}
    """
    response = {"success": True}
    if data is not None:
        response["data"] = data
    response.update(kwargs)
    return response


def error(message: str, code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Create a standard error response.
    
    Args:
        message: Human-readable error description
        code: Optional error code for programmatic handling
        **kwargs: Additional context fields
    
    Returns:
        Dictionary with success=False and error details
    
    Examples:
        >>> error("Symbol not found", code="SYMBOL_NOT_FOUND")
        {'success': False, 'error': 'Symbol not found', 'error_code': 'SYMBOL_NOT_FOUND'}
    """
    response = {"success": False, "error": str(message)}
    if code:
        response["error_code"] = code
    response.update(kwargs)
    return response


def safe_execute(func: Callable, *args, include_trace: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Execute a function with automatic error handling and response formatting.
    
    Args:
        func: Function to execute
        *args: Positional arguments for func
        include_trace: Include full traceback in error response (default: False)
        **kwargs: Keyword arguments for func
    
    Returns:
        Success response with func result, or error response on exception
    
    Examples:
        >>> def risky_operation(x):
        ...     return 10 / x
        >>> safe_execute(risky_operation, 2)
        {'success': True, 'data': 5.0}
        >>> safe_execute(risky_operation, 0)
        {'success': False, 'error': 'division by zero'}
    """
    try:
        result = func(*args, **kwargs)
        # If function already returns a response dict, pass through
        if isinstance(result, dict) and "success" in result:
            return result
        return success(result)
    except Exception as e:
        err_response = error(str(e))
        if include_trace:
            err_response["traceback"] = traceback.format_exc()
        return err_response


def require_fields(*required: str) -> Callable:
    """
    Decorator to validate required fields in function arguments.
    
    Args:
        *required: Names of required keyword arguments
    
    Returns:
        Decorator function
    
    Examples:
        >>> @require_fields("symbol", "timeframe")
        ... def fetch_data(symbol=None, timeframe=None, limit=100):
        ...     return f"Fetching {symbol} {timeframe}"
        >>> fetch_data(symbol="EURUSD")
        {'success': False, 'error': 'Missing required field: timeframe'}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            missing = [field for field in required if kwargs.get(field) is None]
            if missing:
                return error(f"Missing required field: {', '.join(missing)}", code="MISSING_FIELD")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_range(field: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Callable:
    """
    Decorator to validate numeric field is within range.
    
    Args:
        field: Name of field to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
    
    Returns:
        Decorator function
    
    Examples:
        >>> @validate_range("limit", min_val=1, max_val=1000)
        ... def fetch_data(limit=100):
        ...     return f"Fetching {limit} records"
        >>> fetch_data(limit=5000)
        {'success': False, 'error': 'limit must be <= 1000'}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            value = kwargs.get(field)
            if value is not None:
                if min_val is not None and value < min_val:
                    return error(f"{field} must be >= {min_val}", code="VALUE_TOO_LOW")
                if max_val is not None and value > max_val:
                    return error(f"{field} must be <= {max_val}", code="VALUE_TOO_HIGH")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_choices(field: str, choices: list) -> Callable:
    """
    Decorator to validate field value is in allowed choices.
    
    Args:
        field: Name of field to validate
        choices: List of allowed values
    
    Returns:
        Decorator function
    
    Examples:
        >>> @validate_choices("timeframe", ["M1", "M5", "H1"])
        ... def fetch_data(timeframe="H1"):
        ...     return f"Fetching {timeframe} data"
        >>> fetch_data(timeframe="D1")
        {'success': False, 'error': "timeframe must be one of ['M1', 'M5', 'H1']"}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            value = kwargs.get(field)
            if value is not None and value not in choices:
                return error(f"{field} must be one of {choices}", code="INVALID_CHOICE")
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience aliases for common patterns
ok = success
fail = error
