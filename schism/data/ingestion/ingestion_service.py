"""
ingestion_service.py - Ingestion service entrypoint.

Entry point called by Docker:
    python -m schism.data.ingestion.ingestion_service

This module owns config, dependency wiring, lifecycle, and orchestration only.
Business logic lives in cache, publisher, service, and scheduler modules.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

import redis.asyncio as aioredis
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from schism.data.ingestion.binance_client import BinanceClient
from schism.data.ingestion.cache.funding_cache import FundingCache
from schism.data.ingestion.cache.oi_cache import OICache
from schism.data.ingestion.context import AppContext
from schism.data.ingestion.data_store import DataStore
from schism.data.ingestion.publishers.redis_publisher import RedisPublisher
from schism.data.ingestion.scheduler.jobs import register_jobs
from schism.data.ingestion.services.backfill_service import BackfillService
from schism.data.ingestion.services.live_service import LiveService
from schism.utils.logger import ingestion_logger


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


SYMBOLS: list[str] = [
    symbol.strip()
    for symbol in _env("SYMBOLS", "BTCUSDT").split(",")
    if symbol.strip()
]
BACKFILL_DAYS: int = int(_env("BACKFILL_DAYS", "180"))
REDIS_URL: str = _env("REDIS_URL", "redis://localhost:6379")
PARQUET_ROOT: Path = Path(_env("PARQUET_ROOT", "/app/data/volumes/parquet"))
API_KEY: str = _env("BINANCE_API_KEY")
API_SECRET: str = _env("BINANCE_API_SECRET")
ENV: str = _env("ENV", "dev")


async def bootstrap() -> AppContext:
    """Build and connect runtime dependencies."""
    store = DataStore(PARQUET_ROOT)
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        await redis.ping()
        ingestion_logger.info("redis_connected", url=REDIS_URL)
    except Exception as exc:
        ingestion_logger.error("redis_connect_failed", url=REDIS_URL, error=str(exc))
        if ENV == "prod":
            sys.exit(1)

    client = BinanceClient(api_key=API_KEY, api_secret=API_SECRET)
    await client.__aenter__()

    publisher = RedisPublisher(redis)
    return AppContext(
        client=client,
        store=store,
        redis=redis,
        funding_cache=FundingCache(),
        oi_cache=OICache(),
        publisher=publisher,
        symbols=SYMBOLS,
        backfill_days=BACKFILL_DAYS,
        parquet_root=PARQUET_ROOT,
        env=ENV,
    )


async def run_backfill(ctx: AppContext) -> None:
    """Run startup backfill and warm live caches."""
    backfill = BackfillService(ctx)
    await asyncio.gather(
        *(backfill.run(symbol) for symbol in ctx.symbols),
        return_exceptions=True,
    )
    await ctx.funding_cache.refresh(ctx.client, ctx.symbols)
    await ctx.oi_cache.refresh(ctx.client, ctx.symbols)
    ingestion_logger.info("backfill_complete", symbols=ctx.symbols)


def start_scheduler(ctx: AppContext) -> AsyncIOScheduler:
    """Register and start background ingestion jobs."""
    scheduler = AsyncIOScheduler(timezone="UTC")
    register_jobs(scheduler, ctx)
    scheduler.start()
    ingestion_logger.info("scheduler_started")
    return scheduler


async def run_live(ctx: AppContext) -> tuple[list[asyncio.Task], asyncio.Event]:
    """Start one live loop task per symbol and wire shutdown signals."""
    live = LiveService(ctx)
    live_tasks = [
        asyncio.create_task(live.run(symbol), name=f"live_{symbol}")
        for symbol in ctx.symbols
    ]

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        ingestion_logger.info("ingestion_service_shutdown_signal")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _signal_handler())

    ingestion_logger.info("live_loops_started", symbols=ctx.symbols)
    return live_tasks, shutdown_event


async def main() -> None:
    ingestion_logger.info(
        "ingestion_service_start",
        symbols=SYMBOLS,
        backfill_days=BACKFILL_DAYS,
        parquet_root=str(PARQUET_ROOT),
        env=ENV,
    )

    ctx = await bootstrap()
    scheduler: AsyncIOScheduler | None = None
    live_tasks: list[asyncio.Task] = []
    try:
        await run_backfill(ctx)
        scheduler = start_scheduler(ctx)
        live_tasks, shutdown_event = await run_live(ctx)
        await shutdown_event.wait()
    finally:
        ingestion_logger.info("ingestion_service_stopping")
        for task in live_tasks:
            task.cancel()
        if live_tasks:
            await asyncio.gather(*live_tasks, return_exceptions=True)
        if scheduler is not None:
            scheduler.shutdown(wait=False)
        await ctx.client.__aexit__(None, None, None)
        await ctx.redis.aclose()
        ingestion_logger.info("ingestion_service_stopped")


if __name__ == "__main__":
    asyncio.run(main())
