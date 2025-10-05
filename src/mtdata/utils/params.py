"""
Unified parameter parsing utilities.
Handles JSON strings, key=value pairs, and mixed formats consistently.
"""

import json
from typing import Any, Dict, Optional, Union


def parse(params: Optional[Union[str, dict]], defaults: Optional[dict] = None) -> dict:
    """
    Parse parameters from string or dict with automatic type coercion.
    
    Supports:
    - JSON strings: '{"key": "value", "num": 42}'
    - Key-value pairs: 'key=value,num=42,flag=true'
    - Mixed: 'key=value, nested={"a": 1}'
    - Direct dict passthrough
    
    Args:
        params: Parameter string or dict
        defaults: Default values to merge with parsed params
    
    Returns:
        Parsed parameter dictionary
    
    Examples:
        >>> parse('alpha=0.5,beta=0.3')
        {'alpha': 0.5, 'beta': 0.3}
        
        >>> parse('{"method": "arima", "order": [1,1,1]}')
        {'method': 'arima', 'order': [1, 1, 1]}
        
        >>> parse('enabled=true,count=10', defaults={'timeout': 30})
        {'timeout': 30, 'enabled': True, 'count': 10}
    """
    if params is None or params == "":
        return defaults or {}
    
    # Direct dict passthrough
    if isinstance(params, dict):
        return {**(defaults or {}), **params}
    
    params_str = str(params).strip()
    
    # Try JSON first
    if params_str.startswith('{') or params_str.startswith('['):
        try:
            parsed = json.loads(params_str)
            return {**(defaults or {}), **parsed} if isinstance(parsed, dict) else parsed
        except json.JSONDecodeError:
            pass  # Fall through to key-value parsing
    
    # Parse key=value pairs
    result = defaults.copy() if defaults else {}
    
    for pair in params_str.split(','):
        pair = pair.strip()
        if not pair or '=' not in pair:
            continue
        
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Try nested JSON value
        if value.startswith('{') or value.startswith('['):
            try:
                result[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        
        # Auto-coerce scalar values
        result[key] = _coerce_value(value)
    
    return result


def _coerce_value(value: str) -> Any:
    """
    Automatically convert string to appropriate Python type.
    
    Args:
        value: String value to convert
    
    Returns:
        Converted value (bool, int, float, or str)
    
    Examples:
        >>> _coerce_value("true")
        True
        >>> _coerce_value("42")
        42
        >>> _coerce_value("3.14")
        3.14
        >>> _coerce_value("hello")
        'hello'
    """
    # Boolean
    if value.lower() in ('true', 'yes', 'on', '1'):
        return True
    if value.lower() in ('false', 'no', 'off', '0'):
        return False
    
    # None/null
    if value.lower() in ('none', 'null'):
        return None
    
    # Numeric
    try:
        # Integer
        if '.' not in value and 'e' not in value.lower():
            return int(value)
        # Float
        return float(value)
    except ValueError:
        pass
    
    # String (strip quotes if present)
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    return value


def extract(params: dict, *keys: str, defaults: Optional[dict] = None) -> dict:
    """
    Extract specific keys from parameter dict with optional defaults.
    
    Args:
        params: Source parameter dictionary
        *keys: Keys to extract
        defaults: Default values for missing keys
    
    Returns:
        Dictionary with only specified keys
    
    Examples:
        >>> extract({'a': 1, 'b': 2, 'c': 3}, 'a', 'c')
        {'a': 1, 'c': 3}
        
        >>> extract({'a': 1}, 'a', 'b', defaults={'b': 10})
        {'a': 1, 'b': 10}
    """
    result = {}
    defaults = defaults or {}
    
    for key in keys:
        if key in params:
            result[key] = params[key]
        elif key in defaults:
            result[key] = defaults[key]
    
    return result


def merge(*dicts: dict, overwrite: bool = True) -> dict:
    """
    Merge multiple parameter dictionaries.
    
    Args:
        *dicts: Dictionaries to merge (later ones take precedence)
        overwrite: If False, only add missing keys (default: True)
    
    Returns:
        Merged dictionary
    
    Examples:
        >>> merge({'a': 1}, {'b': 2}, {'a': 3})
        {'a': 3, 'b': 2}
        
        >>> merge({'a': 1}, {'a': 3}, overwrite=False)
        {'a': 1}
    """
    result = {}
    
    for d in dicts:
        if d is None:
            continue
        for key, value in d.items():
            if overwrite or key not in result:
                result[key] = value
    
    return result


def to_string(params: dict, format: str = 'kv') -> str:
    """
    Convert parameter dict to string representation.
    
    Args:
        params: Parameter dictionary
        format: Output format ('kv' for key=value, 'json' for JSON)
    
    Returns:
        String representation
    
    Examples:
        >>> to_string({'alpha': 0.5, 'beta': 0.3})
        'alpha=0.5,beta=0.3'
        
        >>> to_string({'method': 'arima'}, format='json')
        '{"method": "arima"}'
    """
    if format == 'json':
        return json.dumps(params)
    
    # Key-value format
    pairs = []
    for key, value in params.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif isinstance(value, bool):
            value = str(value).lower()
        pairs.append(f"{key}={value}")
    
    return ','.join(pairs)


def validate_types(params: dict, schema: Dict[str, type]) -> Optional[str]:
    """
    Validate parameter types against schema.
    
    Args:
        params: Parameters to validate
        schema: Dict mapping param names to expected types
    
    Returns:
        Error message if validation fails, None if valid
    
    Examples:
        >>> validate_types({'count': 10}, {'count': int})
        None
        
        >>> validate_types({'count': '10'}, {'count': int})
        "Parameter 'count' must be <class 'int'>, got <class 'str'>"
    """
    for key, expected_type in schema.items():
        if key in params:
            value = params[key]
            if not isinstance(value, expected_type):
                return f"Parameter '{key}' must be {expected_type}, got {type(value)}"
    return None
