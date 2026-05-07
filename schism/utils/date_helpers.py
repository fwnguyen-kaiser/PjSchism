"""
Timestamp normalization and conversion utilities for SCHISM.

Binance REST endpoints return milliseconds (int), but some internal
endpoints return seconds. WebSocket payloads are always milliseconds.
This module provides a single normalize_ts() entry point that handles
both cases plus timezone coercion.

Public API:
    normalize_ts(ts, source)       — int ms/s → aware datetime (UTC)
    ms_to_datetime(ms)             — int ms → aware datetime (UTC)
    datetime_to_ms(dt)             — aware datetime → int ms
    datetime_to_bar_ts(dt, freq)   — snap datetime to bar boundary
    bar_index_to_utc(idx, start, freq) — bar index → aware datetime (UTC)

All returned datetimes are UTC-aware. Never naive.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal, Union

# ── Source identifiers ────────────────────────────────────────────────────────

Source = Literal["binance_ms", "binance_s", "unix_ms", "unix_s", "auto"]

# Binance REST consistently uses ms, but a few endpoints (e.g. some
# futures/data paths) have used seconds historically.
_MS_THRESHOLD = 1_000_000_000_000   # values > this are treated as ms

# Supported bar frequencies and their duration in seconds
_FREQ_SECONDS: dict[str, int] = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1_800,
    "1h":  3_600,
    "2h":  7_200,
    "4h":  14_400,
    "6h":  21_600,
    "8h":  28_800,
    "12h": 43_200,
    "1d":  86_400,
    "3d":  259_200,
    "1w":  604_800,
}


# ── Core converters ───────────────────────────────────────────────────────────

def ms_to_datetime(ms: int) -> datetime:
    """
    Convert a UNIX millisecond timestamp to a UTC-aware datetime.

    Args:
        ms: UNIX timestamp in milliseconds.

    Returns:
        Timezone-aware datetime in UTC.

    Example:
        >>> ms_to_datetime(1700000000000)
        datetime.datetime(2023, 11, 14, 22, 13, 20, tzinfo=datetime.timezone.utc)
    """
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """
    Convert a datetime to a UNIX millisecond timestamp.

    Naive datetimes are assumed UTC.

    Args:
        dt: datetime object (aware or naive).

    Returns:
        UNIX timestamp in milliseconds (int).

    Example:
        >>> from datetime import datetime, timezone
        >>> datetime_to_ms(datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc))
        1700000000000
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def normalize_ts(ts: Union[int, float, str, datetime], source: Source = "auto") -> datetime:
    """
    Normalize any timestamp representation to a UTC-aware datetime.

    Handles:
      - int/float ms   (Binance REST standard, websocket payloads)
      - int/float s    (some older Binance endpoints)
      - ISO 8601 str   (internal logs, config files)
      - aware datetime (pass-through with UTC coercion)
      - naive datetime (assumed UTC)

    Args:
        ts:     Input timestamp — int, float, str, or datetime.
        source: Hint about the source format.
                "auto"       — infer from value magnitude (int/float) or type
                "binance_ms" — force millisecond interpretation
                "binance_s"  — force second interpretation
                "unix_ms"    — alias for binance_ms
                "unix_s"     — alias for binance_s

    Returns:
        UTC-aware datetime.

    Raises:
        TypeError:  ts is not int, float, str, or datetime.
        ValueError: str ts cannot be parsed as ISO 8601.

    Examples:
        >>> normalize_ts(1700000000000)                  # auto → ms
        datetime.datetime(2023, 11, 14, 22, 13, 20, tzinfo=...)
        >>> normalize_ts(1700000000, source="binance_s") # force seconds
        datetime.datetime(2023, 11, 14, 22, 13, 20, tzinfo=...)
        >>> normalize_ts("2023-11-14T22:13:20Z")
        datetime.datetime(2023, 11, 14, 22, 13, 20, tzinfo=...)
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    if isinstance(ts, str):
        # Handle 'Z' suffix (Python < 3.11 fromisoformat doesn't accept it)
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    if isinstance(ts, (int, float)):
        if source in ("binance_ms", "unix_ms"):
            return ms_to_datetime(int(ts))
        if source in ("binance_s", "unix_s"):
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        # Auto: use magnitude heuristic
        if ts > _MS_THRESHOLD:
            return ms_to_datetime(int(ts))
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)

    raise TypeError(
        f"normalize_ts: expected int, float, str, or datetime, got {type(ts).__name__!r}"
    )


# ── Bar boundary helpers ──────────────────────────────────────────────────────

def datetime_to_bar_ts(dt: datetime, freq: str = "4h") -> datetime:
    """
    Snap a datetime down to the nearest bar open boundary for `freq`.

    Binance bar boundaries are aligned to UNIX epoch (UTC). For example,
    4h bars open at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.

    Args:
        dt:   Input datetime (aware or naive; naive assumed UTC).
        freq: Bar frequency string, e.g. "4h", "1h", "1d".

    Returns:
        UTC-aware datetime at the bar open boundary.

    Raises:
        ValueError: freq is not a recognised frequency string.

    Example:
        >>> datetime_to_bar_ts(datetime(2024, 1, 1, 5, 37, tzinfo=timezone.utc), "4h")
        datetime.datetime(2024, 1, 1, 4, 0, tzinfo=datetime.timezone.utc)
    """
    if freq not in _FREQ_SECONDS:
        raise ValueError(
            f"datetime_to_bar_ts: unrecognised frequency {freq!r}. "
            f"Valid options: {sorted(_FREQ_SECONDS)}"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    freq_s = _FREQ_SECONDS[freq]
    epoch_s = int(dt_utc.timestamp())
    bar_epoch_s = (epoch_s // freq_s) * freq_s
    return datetime.fromtimestamp(bar_epoch_s, tz=timezone.utc)


def bar_index_to_utc(idx: int, start_ts: datetime, freq: str = "4h") -> datetime:
    """
    Map a zero-based bar index to a UTC-aware datetime.

    Used to convert model array indices back to wall-clock timestamps
    for logging, persistence, and API responses.

    Args:
        idx:      Zero-based bar index (0 = first bar in series).
        start_ts: UTC-aware datetime of bar index 0 (bar open time).
        freq:     Bar frequency, e.g. "4h".

    Returns:
        UTC-aware datetime of the bar open at index `idx`.

    Raises:
        ValueError: freq is not recognised.

    Example:
        >>> start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        >>> bar_index_to_utc(3, start, "4h")
        datetime.datetime(2024, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    if freq not in _FREQ_SECONDS:
        raise ValueError(
            f"bar_index_to_utc: unrecognised frequency {freq!r}. "
            f"Valid options: {sorted(_FREQ_SECONDS)}"
        )
    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    freq_s = _FREQ_SECONDS[freq]
    return start_ts + timedelta(seconds=idx * freq_s)


# ── Format helpers ────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


def to_iso(dt: datetime) -> str:
    """Format a datetime as ISO 8601 UTC string with 'Z' suffix."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")