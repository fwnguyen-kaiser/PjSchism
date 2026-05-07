"""
Structured JSON logging via structlog.

Three channels:
  regime_logger    — state transitions and filtered-posterior events
  refit_logger     — refit triggers, cooldown events, alignment results
  ingestion_logger — data ingestion, bar build, parquet I/O

Every log record includes:
  bar_ts   — exchange bar timestamp (ISO 8601 UTC), when provided
  channel  — one of "regime" | "refit" | "ingestion"
  level    — debug | info | warning | error | critical
  event    — machine-readable event key (snake_case)

Usage:
    from schism.utils.logger import regime_logger, refit_logger, ingestion_logger

    regime_logger.info("state_transition", bar_ts="2024-01-01T00:00:00Z",
                        from_state=2, to_state=3, confidence=0.91)

    refit_logger.warning("refit_triggered", trigger="ll_degradation",
                          delta_ll=-1.23, bar_ts="2024-01-01T00:00:00Z")

    ingestion_logger.debug("klines_fetched", symbol="BTCUSDT", count=500)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


# ── stdlib logging baseline ───────────────────────────────────────────────────

def _configure_stdlib() -> None:
    """Configure stdlib logging to be structlog-compatible."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG,
    )
    # Silence noisy third-party loggers
    for name in ("httpx", "httpcore", "asyncio", "websockets", "hpack"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── shared processors ─────────────────────────────────────────────────────────

_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True, key="wall_ts"),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
]


def _configure_structlog() -> None:
    structlog.configure(
        processors=_SHARED_PROCESSORS + [
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


# ── initialize once on import ─────────────────────────────────────────────────

_configure_stdlib()
_configure_structlog()


def _make_logger(channel: str) -> structlog.BoundLogger:
    """Return a structlog logger pre-bound with channel identifier."""
    return structlog.get_logger(channel).bind(channel=channel)


# ── public loggers ────────────────────────────────────────────────────────────

regime_logger: structlog.BoundLogger = _make_logger("regime")
"""
Log state transitions and posterior events.

Recommended event keys:
  state_transition   — St-1 → St change
  posterior_emitted  — filtered posterior computed for a bar
  regime_engine_start / regime_engine_stop
  low_confidence     — max(posterior) below threshold
"""

refit_logger: structlog.BoundLogger = _make_logger("refit")
"""
Log refit lifecycle events.

Recommended event keys:
  refit_triggered    — trigger type + reason metrics
  refit_completed    — new model_ver, delta_bic, alignment_ok
  refit_cooldown     — blocked by cooldown, bars_remaining
  backstop_override  — backstop overriding active cooldown
  alignment_warning  — Hungarian cost > delta_align (per-state)
"""

ingestion_logger: structlog.BoundLogger = _make_logger("ingestion")
"""
Log data ingestion and preprocessing events.

Recommended event keys:
  klines_fetched          — REST klines response
  oi_hist_fetched         — OI history response
  bar_built               — 4h OHLCV bar assembled
  cvd_computed            — CVD computed for a bar
  parquet_written         — file written to data store
  parquet_read            — file read from data store
  rate_limit_sleep        — backing off on rate limit
  binance_rate_limit_429  — 429 received, retrying
  binance_ip_ban          — 418 received, fatal
  ws_kline_start          — WebSocket stream opened
  ws_bar_close            — live bar close event
  ws_kline_reconnect      — WebSocket reconnect attempt
  vision_crawl_start      — vision_crawler job started
  vision_crawl_done       — vision_crawler job completed
  vision_zip_processed    — one daily zip parsed
"""
