"""CLI parsing utilities."""
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, is_typeddict

from pydantic import BaseModel


def _is_pydantic_model_type(value: Any) -> bool:
    """Check if value is a Pydantic model type."""
    return isinstance(value, type) and issubclass(value, BaseModel)


def _is_typed_dict_type(value: Any) -> bool:
    """Check if value is a TypedDict type."""
    try:
        if is_typeddict(value):
            return True
    except Exception:
        pass
    annotations = getattr(value, "__annotations__", None)
    return isinstance(annotations, dict) and (
        getattr(value, "__required_keys__", None) is not None
        or getattr(value, "__optional_keys__", None) is not None
    )


def _unwrap_optional_type(t: Any) -> Tuple[Any, Any]:
    """Unwrap Optional type to get base type."""
    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        if origin is type(None):
            return args[0] if args else Any, None
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            return non_none[0] if non_none else Any, origin
    return t, origin


def _is_literal_origin(origin: Any) -> bool:
    """Check if origin is Literal."""
    if origin is None:
        return False
    return getattr(origin, "__name__", None) == "Literal" or str(origin).startswith("typing.Literal")


def _literal_choices_for_cli_param(param: Dict[str, Any]) -> Optional[List[str]]:
    """Get literal choices for CLI parameter."""
    try:
        ptype = param.get("type")
        base_type, origin = _unwrap_optional_type(ptype)
    except Exception:
        return None
    if not _is_literal_origin(origin):
        return None
    choices = [str(value) for value in get_args(base_type) if value is not None]
    return choices or None


def _command_variants(command: str) -> Tuple[str, ...]:
    """Get command name variants (underscore, hyphen)."""
    text = str(command or "").strip()
    if not text:
        return ()
    variants = [text]
    hyphen = text.replace("_", "-")
    underscore = text.replace("-", "_")
    if hyphen not in variants:
        variants.append(hyphen)
    if underscore not in variants:
        variants.append(underscore)
    return tuple(variants)


def _find_command_index(argv: List[str], command: str) -> Optional[int]:
    """Find command index in argv."""
    for candidate in _command_variants(command):
        try:
            return argv.index(candidate)
        except ValueError:
            continue
    return None


def _command_aliases(command: str) -> List[str]:
    """Get command aliases."""
    text = str(command or "").strip()
    if not text or "_" not in text:
        return []
    alias = text.replace("_", "-")
    return [alias] if alias != text else []


__all__ = [
    "_is_pydantic_model_type",
    "_is_typed_dict_type",
    "_unwrap_optional_type",
    "_is_literal_origin",
    "_literal_choices_for_cli_param",
    "_command_variants",
    "_find_command_index",
    "_command_aliases",
]
