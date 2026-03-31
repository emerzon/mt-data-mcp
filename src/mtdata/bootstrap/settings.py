"""Configuration settings for MetaTrader5 MCP Server."""

import logging
import os
import warnings
from datetime import datetime
from typing import Optional

try:
    import pytz  # type: ignore
except Exception:
    pytz = None  # optional

try:
    from dateutil import tz as dateutil_tz  # type: ignore
except Exception:
    dateutil_tz = None  # optional

try:
    import tzlocal  # type: ignore
except Exception:
    tzlocal = None  # optional

_WARNED_SERVER_TZ = False
_ENV_LOADED = False
_LOGGER = logging.getLogger(__name__)


def _detect_local_client_tz():
    """Best-effort local timezone detection for unset client timezone config."""
    if tzlocal is not None:
        try:
            tz = tzlocal.get_localzone()
            if tz is not None:
                return tz
        except Exception:
            pass
    if dateutil_tz is not None:
        try:
            if os.name == "nt" and hasattr(dateutil_tz, "tzwinlocal"):
                tz = dateutil_tz.tzwinlocal()
                if tz is not None:
                    return tz
        except Exception:
            pass
        try:
            tz = dateutil_tz.tzlocal()
            if tz is not None:
                return tz
        except Exception:
            pass
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        return None


def _suppress_noisy_third_party_logs() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for logger_name, level in (
        ("numba.cuda.cudadrv.driver", logging.WARNING),
        ("torch.distributed", logging.ERROR),
        ("torch._dynamo", logging.ERROR),
        ("lightning", logging.ERROR),
        ("pytorch_lightning", logging.ERROR),
    ):
        try:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(level)
            noisy_logger.propagate = False
        except Exception:
            pass
    try:
        warnings.filterwarnings(
            "ignore",
            message=r".*Redirects are currently not supported in Windows or MacOs.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=ImportWarning,
            module=r"umap(\..*)?$",
        )
    except Exception:
        pass


_suppress_noisy_third_party_logs()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    text = str(raw).strip()
    if not text:
        _LOGGER.warning("%s is blank; using default %s.", name, default)
        return int(default)
    try:
        return int(text)
    except (TypeError, ValueError):
        _LOGGER.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return int(default)


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        _LOGGER.warning("Invalid %s=%r; ignoring configured login.", name, raw)
        return None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    text = str(raw).strip()
    if not text:
        _LOGGER.warning("%s is blank; using default %s.", name, default)
        return float(default)
    try:
        return float(text)
    except (TypeError, ValueError):
        _LOGGER.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return float(default)


def load_environment(*, force: bool = False) -> bool:
    """Load environment variables from `.env` once for application entrypoints."""
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return False

    loaded = False
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore

        env_path = find_dotenv()
        if env_path:
            loaded = bool(load_dotenv(env_path))
        else:
            loaded = bool(load_dotenv())
    except Exception:
        loaded = False

    _ENV_LOADED = True
    config_obj = globals().get("mt5_config")
    if config_obj is not None:
        try:
            config_obj.reload_from_env(warn_if_timezone_missing=True)
        except Exception as exc:
            _LOGGER.warning("Failed to reload MT5 configuration from environment: %s", exc)
    embeddings_obj = globals().get("news_embeddings_config")
    if embeddings_obj is not None:
        try:
            embeddings_obj.reload_from_env()
        except Exception as exc:
            _LOGGER.warning("Failed to reload news embeddings configuration from environment: %s", exc)
    return loaded

class MT5Config:
    """MetaTrader5 connection configuration."""

    def __init__(self, *, warn_if_timezone_missing: bool = True):
        self.login: Optional[str] = None
        self._login_value: Optional[int] = None
        self.password: Optional[str] = None
        self.server: Optional[str] = None
        self.timeout = 30
        self.server_tz_name: Optional[str] = None
        self.client_tz_name: Optional[str] = None
        self.time_offset_minutes = 0
        self.broker_time_check_enabled = False
        self.broker_time_check_ttl_seconds = 60
        self.reload_from_env(warn_if_timezone_missing=warn_if_timezone_missing)

    def reload_from_env(self, *, warn_if_timezone_missing: bool = True) -> None:
        self.login = os.getenv("MT5_LOGIN")
        self._login_value = _env_optional_int("MT5_LOGIN")
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER")
        self.timeout = _env_int("MT5_TIMEOUT", 30)
        self.server_tz_name = os.getenv("MT5_SERVER_TZ")  # e.g., "Europe/Lisbon"
        self.client_tz_name = os.getenv("CLIENT_TZ") or os.getenv("MT5_CLIENT_TZ")  # e.g., "America/New_York"
        self.time_offset_minutes = _env_int("MT5_TIME_OFFSET_MINUTES", 0)
        self.broker_time_check_enabled = _env_bool("MTDATA_BROKER_TIME_CHECK", default=False)
        ttl_raw = os.getenv("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS")
        ttl_seconds = _env_int("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS", 60)
        if ttl_seconds < 0:
            _LOGGER.warning(
                "MTDATA_BROKER_TIME_CHECK_TTL_SECONDS=%r is negative; clamping to 0.",
                ttl_raw,
            )
        self.broker_time_check_ttl_seconds = max(0, ttl_seconds)
        if warn_if_timezone_missing:
            self._warn_if_timezone_missing()

    def _warn_if_timezone_missing(self) -> None:
        """Warn once if MT5 server timezone/offset is not configured."""
        global _WARNED_SERVER_TZ
        if _WARNED_SERVER_TZ:
            return
        has_server_tz = bool(self.server_tz_name)
        has_offset_env = os.getenv("MT5_TIME_OFFSET_MINUTES") is not None
        if not has_server_tz and not has_offset_env:
            _WARNED_SERVER_TZ = True
            logging.getLogger(__name__).warning(
                "MT5_SERVER_TZ or MT5_TIME_OFFSET_MINUTES not set; "
                "server timestamps may be misaligned. Configure one of them."
            )

    def get_login(self) -> Optional[int]:
        """Get login as integer if available"""
        return self._login_value
    
    def get_password(self) -> Optional[str]:
        """Get password"""
        return self.password
    
    def get_server(self) -> Optional[str]:
        """Get server name"""
        return self.server
    
    def has_credentials(self) -> bool:
        """Check if all credentials are available"""
        return self.get_login() is not None and bool(self.password) and bool(self.server)

    def get_time_offset_seconds(self) -> int:
        """Return configured MT5 server time offset relative to UTC in seconds.

        Positive value means MT5 server time is ahead of UTC (UTC+X).
        All MT5 timestamps read from the terminal should be adjusted by subtracting this offset
        to normalize into UTC epochs.
        """
        # 1. Prefer explicit offset in minutes (if set)
        if self.time_offset_minutes != 0:
            return int(self.time_offset_minutes) * 60
            
        # 2. Derive from MT5_SERVER_TZ if available
        if self.server_tz_name and pytz:
            try:
                tz = pytz.timezone(self.server_tz_name)
                # Calculate current offset (aware of DST)
                return int(datetime.now(tz).utcoffset().total_seconds())
            except Exception:
                pass
                
        return 0

    def get_server_tz(self):
        """Return a pytz timezone for the MT5 server, if configured and pytz is available."""
        try:
            if pytz and self.server_tz_name:
                return pytz.timezone(self.server_tz_name)
        except Exception:
            return None
        return None

    def get_client_tz(self):
        """Return the client timezone from config or local autodetection."""
        try:
            if self.client_tz_name:
                if pytz:
                    return pytz.timezone(self.client_tz_name)
                return None
        except Exception:
            return None
        return _detect_local_client_tz()


class NewsEmbeddingsConfig:
    """Optional embedding reranking configuration for unified news."""

    def __init__(self) -> None:
        self.model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.top_n = 20
        self.weight = 1.0
        self.truncate_dim: Optional[int] = None
        self.cache_size = 256
        self.hf_token_env_var = "HF_TOKEN"
        self.reload_from_env()

    def reload_from_env(self) -> None:
        self.model_name = (
            os.getenv("MTDATA_NEWS_EMBEDDINGS_MODEL", "Qwen/Qwen3-Embedding-0.6B").strip()
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self.top_n = max(1, _env_int("MTDATA_NEWS_EMBEDDINGS_TOP_N", 20))
        self.weight = max(0.0, _env_float("MTDATA_NEWS_EMBEDDINGS_WEIGHT", 1.0))
        self.truncate_dim = _env_optional_int("MTDATA_NEWS_EMBEDDINGS_TRUNCATE_DIM")
        if self.truncate_dim is not None and self.truncate_dim <= 0:
            _LOGGER.warning(
                "MTDATA_NEWS_EMBEDDINGS_TRUNCATE_DIM=%r is not positive; disabling truncation.",
                self.truncate_dim,
            )
            self.truncate_dim = None
        self.cache_size = max(0, _env_int("MTDATA_NEWS_EMBEDDINGS_CACHE_SIZE", 256))
        self.hf_token_env_var = (
            os.getenv("MTDATA_NEWS_EMBEDDINGS_HF_TOKEN_ENV_VAR", "HF_TOKEN").strip() or "HF_TOKEN"
        )

# Global configuration instance. Entry points call `load_environment()` explicitly.
mt5_config = MT5Config(warn_if_timezone_missing=False)
news_embeddings_config = NewsEmbeddingsConfig()
