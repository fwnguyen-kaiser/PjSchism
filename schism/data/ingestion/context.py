"""Runtime dependency container for ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncEngine

from schism.data.ingestion.binance_client import BinanceClient
from schism.data.ingestion.bybit_client import BybitClient
from schism.data.ingestion.cache.cross_fr_cache import CrossExchangeFRCache
from schism.data.ingestion.cache.funding_cache import FundingCache
from schism.data.ingestion.cache.oi_cache import OICache
from schism.data.ingestion.data_store import DataStore
from schism.data.ingestion.publishers.redis_publisher import RedisPublisher
from schism.persistence.repositories.bar_repo import BarRepository


@dataclass
class AppContext:
    client: BinanceClient
    store: DataStore
    redis: aioredis.Redis
    funding_cache: FundingCache
    oi_cache: OICache
    publisher: RedisPublisher
    bar_repo: Optional[BarRepository]
    db_engine: Optional[AsyncEngine]
    symbols: list[str]
    backfill_days: int
    parquet_root: Path
    env: str
    bybit_client: Optional[BybitClient] = None
    cross_fr_cache: Optional[CrossExchangeFRCache] = None
