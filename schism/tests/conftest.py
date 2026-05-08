"""
conftest.py — pytest fixtures for ingestion tests.

All fixtures are unit-test scoped (no real network, no real Redis).
BinanceClient._get is patched at the httpx layer so rate-limiter logic
still executes (tests the full stack minus actual I/O).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schism.data.ingestion.bar_builder import Bar, IngestionSource
from schism.data.ingestion.binance_client import BinanceClient
from schism.data.ingestion.data_store import DataStore
from schism.utils.date_helpers import datetime_to_ms, ms_to_datetime


# ── time helpers ──────────────────────────────────────────────────────────────

def _bar_ts(hour: int = 0, day: int = 1, month: int = 1, year: int = 2024) -> datetime:
    """Return a UTC-aware 4h-aligned bar timestamp."""
    assert hour % 4 == 0, "hour must be 4h-aligned"
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# ── synthetic data factories ──────────────────────────────────────────────────

def make_kline_row(
    open_time: datetime,
    open_: float = 50000.0,
    high: float = 51000.0,
    low: float = 49000.0,
    close: float = 50500.0,
    volume: float = 100.0,
    num_trades: int = 1000,
    taker_buy_base: float = 55.0,
) -> dict:
    """Return a kline dict in BinanceClient.get_klines() output format."""
    return {
        "open_time":       open_time,
        "open":            open_,
        "high":            high,
        "low":             low,
        "close":           close,
        "volume":          volume,
        "close_time":      open_time + timedelta(hours=4) - timedelta(milliseconds=1),
        "quote_volume":    volume * close,
        "num_trades":      num_trades,
        "taker_buy_base":  taker_buy_base,
        "taker_buy_quote": taker_buy_base * close,
    }


def make_raw_kline_list(open_time: datetime, **kwargs) -> list:
    """Return Binance REST API raw kline array (before parsing)."""
    ot_ms = _ms(open_time)
    ct_ms = _ms(open_time + timedelta(hours=4) - timedelta(milliseconds=1))
    close = kwargs.get("close", 50500.0)
    volume = kwargs.get("volume", 100.0)
    taker = kwargs.get("taker_buy_base", 55.0)
    return [
        ot_ms,                          # [0] open_time ms
        str(kwargs.get("open_", 50000.0)),  # [1] open
        str(kwargs.get("high", 51000.0)),   # [2] high
        str(kwargs.get("low", 49000.0)),    # [3] low
        str(close),                         # [4] close
        str(volume),                        # [5] volume
        ct_ms,                              # [6] close_time ms
        str(volume * close),                # [7] quote_volume
        1000,                               # [8] num_trades
        str(taker),                         # [9] taker_buy_base
        str(taker * close),                 # [10] taker_buy_quote
        "0",                                # [11] ignore
    ]


def make_bar(
    bar_ts: datetime | None = None,
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    market_type: str = "perp",
    timeframe_label: str = "4h",
    close: float = 50500.0,
    volume: float = 100.0,
    cvd: float = 10.0,
    oi: float | None = 50000.0,
    lsr_top: float | None = 1.2,
    funding_rate: float | None = 0.0001,
    best_bid: float | None = 50501.0,
    best_ask: float | None = 50502.0,
    bybit_fr: float | None = None,
    source: IngestionSource | None = None,
) -> Bar:
    """Return a synthetic Bar for use in tests."""
    ts = bar_ts or _bar_ts(0)
    return Bar(
        bar_ts         = ts,
        symbol         = symbol,
        open           = close - 500,
        high           = close + 500,
        low            = close - 1000,
        close          = close,
        volume         = volume,
        exchange       = exchange,
        market_type    = market_type,
        timeframe_label = timeframe_label,
        cvd            = cvd,
        num_trades     = 1000,
        taker_buy_base = volume * 0.55,
        quote_volume   = volume * close,
        oi             = oi,
        lsr_top        = lsr_top,
        funding_rate   = funding_rate,
        best_bid       = best_bid,
        best_ask       = best_ask,
        bybit_fr       = bybit_fr,
        source         = source,
    )


def make_bars(n: int, symbol: str = "BTCUSDT", start_hour: int = 0) -> list[Bar]:
    """Return n sequential 4h bars."""
    bars = []
    for i in range(n):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i * 4)
        bars.append(make_bar(bar_ts=ts, symbol=symbol, close=50000.0 + i * 10))
    return bars


def make_oi_response(bar_ts: datetime, oi: float = 50000.0) -> dict:
    """Return a dict matching BinanceClient.get_open_interest_hist() output format."""
    return {
        "bar_ts":            bar_ts,
        "sum_open_interest": oi,
        "sum_oi_value":      oi * 50000.0,
    }


def make_funding_response(bar_ts: datetime, rate: float = 0.0001) -> dict:
    """Return a dict matching BinanceClient.get_funding_rate() output format."""
    return {
        "funding_time":  bar_ts,
        "funding_rate":  rate,
        "symbol":        "BTCUSDT",
    }


# ── BinanceClient mock factory ────────────────────────────────────────────────

def make_mock_client(
    klines: list[dict] | None = None,
    oi_records: list[dict] | None = None,
    lsr_records: list[dict] | None = None,
    funding_records: list[dict] | None = None,
) -> MagicMock:
    """
    Return a MagicMock that mimics BinanceClient.
    All async methods return configurable data.
    """
    client = MagicMock(spec=BinanceClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__  = AsyncMock(return_value=None)

    _klines   = klines   or [make_kline_row(_bar_ts(i * 4 % 24, day=1 + i * 4 // 24)) for i in range(5)]
    _oi       = oi_records       or [make_oi_response(_bar_ts(0))]
    _lsr      = lsr_records      or [{"bar_ts": _bar_ts(0), "long_short_ratio": 1.2,
                                       "long_account": 0.55, "short_account": 0.45}]
    _funding  = funding_records  or [make_funding_response(_bar_ts(0))]

    client.get_klines                 = AsyncMock(return_value=_klines)
    client.get_open_interest_hist     = AsyncMock(return_value=_oi)
    client.get_top_lsr                = AsyncMock(return_value=_lsr)
    client.get_funding_rate           = AsyncMock(return_value=_funding)
    client.get_agg_trades             = AsyncMock(return_value=[])
    client.stream_kline_close         = AsyncMock(return_value=None)
    client.get_book_ticker_snapshot   = AsyncMock(return_value={
        "symbol":    "BTCUSDT",
        "bid_price": 50501.0,
        "ask_price": 50502.0,
        "bid_qty":   1.0,
        "ask_qty":   1.0,
        "ts":        _bar_ts(0),
    })

    return client


# ── pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def tmp_parquet(tmp_path: Path) -> Path:
    """Temporary parquet root directory."""
    return tmp_path / "parquet"


@pytest.fixture
def store(tmp_parquet: Path) -> DataStore:
    """DataStore backed by a temp directory."""
    return DataStore(tmp_parquet)


@pytest.fixture
def mock_client() -> MagicMock:
    """Default mock BinanceClient."""
    return make_mock_client()


@pytest.fixture
def sample_bars() -> list[Bar]:
    """10 sequential bars for BTCUSDT."""
    return make_bars(10)


@pytest.fixture
def sample_klines() -> list[dict]:
    """5 kline dicts matching get_klines() output format."""
    return [make_kline_row(_bar_ts(i * 4 % 24, day=1 + i * 4 // 24)) for i in range(5)]
