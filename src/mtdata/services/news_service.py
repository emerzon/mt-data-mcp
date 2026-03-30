"""MT5 local news database parser."""

import subprocess
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..core.runtime_metadata import build_runtime_timezone_meta
from ..utils.mt5 import _mt5_epoch_to_utc
from ..utils.utils import _resolve_client_tz, _use_client_tz

logger = logging.getLogger(__name__)


class MT5NewsRecord:
    """Represents a single MT5 news item."""
    
    def __init__(
        self,
        timestamp: datetime,
        subject: str,
        category: str,
        source: str,
        flags: int = 0,
        priority: int = 0
    ):
        self.timestamp = timestamp
        self.subject = subject
        self.category = category
        self.source = source
        self.flags = flags
        self.priority = priority
    
    def to_dict(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        return {
            "relative_time": _relative_time_text(self.timestamp, now=now),
            "subject": self.subject,
            "category": self.category,
            "source": self.source,
            "flags": self.flags,
            "priority": self.priority
        }


class MT5NewsParser:
    """
    Parser for MT5 news.dat binary format.
    
    Reverse-engineered layout used by current MT5 news.dat files.

    Each headline lives in an inline record with a 12-byte null prefix, followed
    by a subject field (type 3), source field (type 4), and metadata blob
    (type 5). The paired compact index record sits 0x118 bytes earlier and
    stores the Unix timestamp at bytes 20:24.
    """
    
    HEADER_SIZE = 504
    INDEX_RECORD_DELTA = 0x118
    INDEX_ENTRY_SIZE = 40
    MIN_TIMESTAMP = 1577836800  # 2020-01-01
    MAX_TIMESTAMP = 2051222400  # 2035-01-01
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.records: List[MT5NewsRecord] = []
        self.header_info: Dict[str, Any] = {}
        
    def parse(self) -> List[MT5NewsRecord]:
        """Parse the entire news.dat file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"News database not found: {self.filepath}")
            
        with open(self.filepath, 'rb') as f:
            data = f.read()
        
        # Parse header
        self._parse_header(data[:self.HEADER_SIZE])
        
        # Parse index entries and records
        self.records = self._parse_records(data)
        
        return self.records
    
    def _parse_header(self, header: bytes) -> None:
        """Parse the 504-byte header section."""
        # First 4 bytes: header size (should be 504)
        header_size = struct.unpack('<I', header[:4])[0]
        
        # Copyright string at offset 4 (UTF-16-LE, variable length, null-terminated)
        copyright_raw = header[4:132]
        try:
            copyright_str = copyright_raw.decode('utf-16-le').split('\x00')[0]
        except UnicodeDecodeError:
            copyright_str = "Unknown"
        
        # "News" identifier at offset 132
        news_id_raw = header[132:140]
        try:
            news_id = news_id_raw.decode('utf-16-le').split('\x00')[0]
        except UnicodeDecodeError:
            news_id = "Unknown"
        
        self.header_info = {
            "header_size": header_size,
            "copyright": copyright_str,
            "identifier": news_id,
        }
        
        logger.debug(f"Parsed header: {self.header_info}")
    
    def _parse_records(self, data: bytes) -> List[MT5NewsRecord]:
        """Parse all news records from the binary data."""
        records: List[MT5NewsRecord] = []
        seen_keys = set()

        for offset in range(self.HEADER_SIZE, len(data) - 80):
            record = self._parse_inline_record(data, offset)
            if not record:
                continue

            record_key = (record.timestamp, record.subject, record.source)
            if record_key in seen_keys:
                continue

            seen_keys.add(record_key)
            records.append(record)

        return records

    def _parse_inline_record(self, data: bytes, offset: int) -> Optional[MT5NewsRecord]:
        """Parse one inline headline record and its paired compact index record."""
        if data[offset:offset + 12] != b'\x00' * 12:
            return None

        try:
            subject_length = struct.unpack('<I', data[offset + 12:offset + 16])[0]
            subject_type = struct.unpack('<H', data[offset + 16:offset + 18])[0]
        except struct.error:
            return None

        if subject_type != 3 or subject_length < 2 or subject_length > 2000 or subject_length % 2 != 0:
            return None

        subject_start = offset + 18
        subject_end = subject_start + subject_length
        if subject_end + 12 > len(data):
            return None

        try:
            subject = data[subject_start:subject_end].decode('utf-16-le').rstrip('\x00')
        except UnicodeDecodeError:
            return None

        if not subject or not any(ch.isalpha() for ch in subject):
            return None

        try:
            source_length = struct.unpack('<I', data[subject_end:subject_end + 4])[0]
            source_type = struct.unpack('<H', data[subject_end + 4:subject_end + 6])[0]
        except struct.error:
            return None

        if source_type != 4 or source_length < 2 or source_length > 200 or source_length % 2 != 0:
            return None

        source_start = subject_end + 6
        source_end = source_start + source_length
        if source_end + 6 > len(data):
            return None

        try:
            source = data[source_start:source_end].decode('utf-16-le').rstrip('\x00')
        except UnicodeDecodeError:
            return None

        if not source:
            return None

        try:
            metadata_length = struct.unpack('<I', data[source_end:source_end + 4])[0]
            metadata_type = struct.unpack('<H', data[source_end + 4:source_end + 6])[0]
        except struct.error:
            return None

        if metadata_type != 5 or metadata_length > 4096:
            return None

        timestamp = self._parse_timestamp_for_record(data, offset)
        if not timestamp:
            return None

        flags = 0
        metadata_start = source_end + 6
        if metadata_length >= 4 and metadata_start + 4 <= len(data):
            flags = struct.unpack('<I', data[metadata_start:metadata_start + 4])[0]

        return MT5NewsRecord(
            timestamp=timestamp,
            subject=subject,
            category=source,
            source=source,
            flags=flags,
        )

    def _parse_timestamp_for_record(self, data: bytes, inline_offset: int) -> Optional[datetime]:
        """Read the compact index record paired with an inline headline record."""
        index_offset = inline_offset - self.INDEX_RECORD_DELTA
        if index_offset < 0 or index_offset + 24 > len(data):
            return None

        try:
            delta = struct.unpack('<H', data[index_offset + 6:index_offset + 8])[0]
            timestamp_raw = struct.unpack('<I', data[index_offset + 20:index_offset + 24])[0]
        except struct.error:
            return None

        if delta != self.INDEX_RECORD_DELTA:
            return None

        if not (self.MIN_TIMESTAMP <= timestamp_raw <= self.MAX_TIMESTAMP):
            return None

        timestamp_utc = _mt5_epoch_to_utc(float(timestamp_raw))
        return datetime.fromtimestamp(timestamp_utc, tz=timezone.utc)


def _relative_time_text(timestamp: datetime, now: Optional[datetime] = None) -> str:
    """Return a compact relative-time label like '5 minutes ago'."""
    current = now or datetime.now(timezone.utc)
    current_utc = current.astimezone(timezone.utc) if current.tzinfo else current.replace(tzinfo=timezone.utc)
    timestamp_utc = timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)

    delta_seconds = int(round((current_utc - timestamp_utc).total_seconds()))
    if abs(delta_seconds) < 60:
        return "just now" if delta_seconds >= 0 else "in less than a minute"

    future = delta_seconds < 0
    delta_seconds = abs(delta_seconds)
    units = (
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
    )
    for unit_seconds, unit_name in units:
        if delta_seconds >= unit_seconds:
            count = delta_seconds // unit_seconds
            label = f"{count} {unit_name}{'' if count == 1 else 's'}"
            return f"in {label}" if future else f"{label} ago"
    return "just now"


def _parse_news_filter_datetime(value: str, client_tz: Any = None) -> datetime:
    """Interpret ISO filter inputs in the same display timezone used for output."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    if client_tz is not None:
        try:
            aware = client_tz.localize(dt) if hasattr(client_tz, "localize") else dt.replace(tzinfo=client_tz)
            return aware.astimezone(timezone.utc)
        except Exception:
            pass
    return dt.replace(tzinfo=timezone.utc)


def _rank_news_candidates(candidates: List[Path]) -> Optional[str]:
    if not candidates:
        return None

    ranked: List[tuple[bool, float, int, str]] = []
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            continue
        ranked.append((stat.st_size > MT5NewsParser.HEADER_SIZE, stat.st_mtime, stat.st_size, str(path)))

    if not ranked:
        return None

    ranked.sort(reverse=True)
    return ranked[0][3]


def _news_candidates_from_terminal_root(terminal_root: Path) -> List[Path]:
    bases_path = terminal_root / "bases"
    if not bases_path.exists():
        return []

    candidates: List[Path] = []
    for broker_dir in bases_path.iterdir():
        if not broker_dir.is_dir():
            continue
        news_path = broker_dir / "news" / "news.dat"
        if news_path.exists():
            candidates.append(news_path)
    return candidates


def _is_mt5_terminal_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal64.exe"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if "terminal64.exe" in output.lower():
            return True
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq terminal.exe"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return "terminal.exe" in output.lower()
    except Exception:
        return False


def _terminal_root_from_mt5_api() -> Optional[Path]:
    """Ask the MT5 Python binding for the active terminal data path when available."""
    try:
        import MetaTrader5 as mt5_module  # type: ignore
    except Exception:
        return None

    mt5: Any = mt5_module

    try:
        terminal_info = mt5.terminal_info()
    except Exception:
        terminal_info = None

    if terminal_info is not None:
        data_path = getattr(terminal_info, "data_path", None)
        if data_path:
            path = Path(str(data_path))
            if path.exists():
                return path

    if not _is_mt5_terminal_running():
        return None

    initialized_here = False
    try:
        if not mt5.initialize():
            return None
        initialized_here = True
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            return None
        data_path = getattr(terminal_info, "data_path", None)
        if not data_path:
            return None
        path = Path(str(data_path))
        return path if path.exists() else None
    except Exception:
        return None
    finally:
        if initialized_here:
            try:
                mt5.shutdown()
            except Exception:
                pass


def _read_origin_path(origin_file: Path) -> Optional[str]:
    try:
        raw = origin_file.read_bytes()
    except OSError:
        return None

    for encoding in ("utf-16", "utf-8", "latin1"):
        try:
            text = raw.decode(encoding).strip().strip("\x00")
        except Exception:
            continue
        if text:
            return str(Path(text))
    return None


def _terminal_roots_from_origin(roaming_path: Path) -> List[Path]:
    roots: List[Path] = []
    for terminal_hash in roaming_path.iterdir():
        if not terminal_hash.is_dir():
            continue
        origin_file = terminal_hash / "origin.txt"
        if not origin_file.exists():
            continue
        if _read_origin_path(origin_file):
            roots.append(terminal_hash)
    return roots


def get_mt5_news(
    news_db_path: Optional[str] = None,
    limit: int = 100,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    category: Optional[str] = None,
    search_term: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch news from MT5's local news database.
    
    Parameters
    ----------
    news_db_path : str, optional
        Full path to news.dat file. If not provided, will attempt to auto-detect
        from MT5 installation directories.
    limit : int
        Maximum number of news items to return (default 100)
    from_date : str, optional
        Filter news from this date (ISO format: YYYY-MM-DD)
    to_date : str, optional
        Filter news up to this date (ISO format: YYYY-MM-DD)
    category : str, optional
        Filter by news category (e.g., "FXStreet", "DailyFX")
    search_term : str, optional
        Filter news items containing this term in the subject
        
    Returns
    -------
    dict
        Dictionary containing news items and metadata
    """
    try:
        # Auto-detect path if not provided
        if not news_db_path:
            news_db_path = _auto_detect_news_path()
            if not news_db_path:
                return {
                    "error": "Could not auto-detect MT5 news database path. Please provide news_db_path explicitly.",
                    "hint": "Look for 'news.dat' in your MT5 data folder under bases/<broker>/news/"
                }
        
        # Parse the database
        parser = MT5NewsParser(news_db_path)
        records = parser.parse()
        use_client_tz = _use_client_tz()
        client_tz = _resolve_client_tz() if use_client_tz else None
        
        # Apply filters
        filtered = records
        
        if from_date:
            from_dt = _parse_news_filter_datetime(from_date, client_tz)
            filtered = [r for r in filtered if r.timestamp >= from_dt]
        
        if to_date:
            to_dt = _parse_news_filter_datetime(to_date, client_tz)
            filtered = [r for r in filtered if r.timestamp <= to_dt]
        
        if category:
            cat_lower = category.lower()
            filtered = [r for r in filtered if cat_lower in r.category.lower()]
        
        if search_term:
            term_lower = search_term.lower()
            filtered = [r for r in filtered if term_lower in r.subject.lower()]
        
        # Sort by timestamp (newest first) and apply limit
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        filtered = filtered[:limit]
        
        # Build response
        now_utc = datetime.now(timezone.utc)
        news_list = [r.to_dict(now=now_utc) for r in filtered]
        
        # Get unique categories for reference
        all_categories = list(set(r.category for r in records if r.category))
        all_sources = list(set(r.source for r in records if r.source))

        payload = {
            "success": True,
            "count": len(news_list),
            "total_records": len(records),
            "database_path": str(news_db_path),
            "header_info": parser.header_info,
            "available_categories": all_categories[:20],  # Limit for brevity
            "available_sources": all_sources[:20],
            "news": news_list
        }

        timezone_meta_input: Dict[str, Any] = dict(payload)
        if not use_client_tz:
            timezone_meta_input["timezone"] = "UTC"
            payload["timezone"] = "UTC"
        payload["meta"] = {
            "runtime": {
                "timezone": build_runtime_timezone_meta(
                    timezone_meta_input,
                    include_local=False,
                    include_now=False,
                )
            }
        }

        return payload
        
    except FileNotFoundError as e:
        logger.error(f"News database not found: {e}")
        return {
            "error": f"News database not found: {e}",
            "hint": "Verify the path to news.dat or let the tool auto-detect from MT5 directories"
        }
    except Exception as e:
        logger.exception(f"Error parsing MT5 news database: {e}")
        return {
            "error": f"Failed to parse news database: {str(e)}",
            "hint": "The news.dat file format may have changed or the file may be corrupted"
        }


def _auto_detect_news_path() -> Optional[str]:
    """Attempt to auto-detect the MT5 news.dat file path."""
    # Common MT5 data directories
    roaming_path = Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal"
    
    if not roaming_path.exists():
        return None

    terminal_root = _terminal_root_from_mt5_api()
    if terminal_root is not None:
        matched = _rank_news_candidates(_news_candidates_from_terminal_root(terminal_root))
        if matched:
            return matched

    origin_roots = _terminal_roots_from_origin(roaming_path)
    matched = _rank_news_candidates(
        [candidate for root in origin_roots for candidate in _news_candidates_from_terminal_root(root)]
    )
    if matched:
        return matched

    generic_roots = [path for path in roaming_path.iterdir() if path.is_dir()]
    return _rank_news_candidates(
        [candidate for root in generic_roots for candidate in _news_candidates_from_terminal_root(root)]
    )


def get_news_categories(news_db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get list of all available news categories from the MT5 news database.
    
    Parameters
    ----------
    news_db_path : str, optional
        Full path to news.dat file
        
    Returns
    -------
    dict
        Dictionary containing available categories and their counts
    """
    try:
        if not news_db_path:
            news_db_path = _auto_detect_news_path()
            if not news_db_path:
                return {
                    "error": "Could not auto-detect MT5 news database path"
                }
        
        parser = MT5NewsParser(news_db_path)
        records = parser.parse()
        
        # Count by category
        from collections import Counter
        category_counts = Counter(r.category for r in records if r.category)
        source_counts = Counter(r.source for r in records if r.source)

        payload = {
            "success": True,
            "total_records": len(records),
            "categories": [
                {"name": cat, "count": count}
                for cat, count in category_counts.most_common()
            ],
            "sources": [
                {"name": src, "count": count}
                for src, count in source_counts.most_common()
            ],
            "database_path": str(news_db_path),
            "header_info": parser.header_info
        }

        use_client_tz = _use_client_tz()
        timezone_meta_input: Dict[str, Any] = dict(payload)
        if not use_client_tz:
            timezone_meta_input["timezone"] = "UTC"
            payload["timezone"] = "UTC"
        payload["meta"] = {
            "runtime": {
                "timezone": build_runtime_timezone_meta(
                    timezone_meta_input,
                    include_local=False,
                    include_now=False,
                )
            }
        }

        return payload
        
    except Exception as e:
        logger.exception(f"Error getting news categories: {e}")
        return {
            "error": f"Failed to get categories: {str(e)}"
        }
