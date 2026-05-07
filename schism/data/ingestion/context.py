"""Runtime dependency container for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import redis.asyncio as aioredis

from schism.data.ingestion.binance_client import BinanceClient
from schism.data.ingestion.cache.funding_cache import FundingCache
from schism.data.ingestion.cache.oi_cache import OICache
from schism.data.ingestion.data_store import DataStore
from schism.data.ingestion.publishers.redis_publisher import RedisPublisher


@dataclass
class AppContext:
    client: BinanceClient
    store: DataStore
    redis: aioredis.Redis
    funding_cache: FundingCache
    oi_cache: OICache
    publisher: RedisPublisher
    symbols: list[str]
    backfill_days: int
    parquet_root: Path
    env: str

