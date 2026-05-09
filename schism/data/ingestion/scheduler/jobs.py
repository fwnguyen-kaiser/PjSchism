"""Scheduler job registration for ingestion."""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from schism.data.ingestion.context import AppContext
from schism.data.ingestion.services.backfill_service import BackfillService
from schism.utils.logger import ingestion_logger


async def daily_vision_refresh(ctx: AppContext) -> None:
    """Re-crawl yesterday's vision zip, which is often published with 1-day lag."""
    backfill = BackfillService(ctx)
    for symbol in ctx.symbols:
        ingestion_logger.info("scheduled_vision_refresh", symbol=symbol)
        await backfill.run_vision(symbol, days=2)


async def cross_fr_refresh(ctx: AppContext) -> None:
    if ctx.bybit_client is not None and ctx.cross_fr_cache is not None:
        await ctx.cross_fr_cache.refresh(ctx.bybit_client, ctx.symbols)


def register_jobs(scheduler: AsyncIOScheduler, ctx: AppContext) -> None:
    scheduler.add_job(
        daily_vision_refresh,
        trigger="cron",
        hour=1,
        minute=0,
        args=[ctx],
        id="vision_refresh",
        name="Daily vision crawl refresh",
        max_instances=1,
    )
    scheduler.add_job(
        ctx.funding_cache.refresh,
        trigger="interval",
        hours=1,
        args=[ctx.client, ctx.symbols],
        id="funding_refresh",
        name="Hourly funding rate refresh",
        max_instances=1,
    )
    scheduler.add_job(
        cross_fr_refresh,
        trigger="interval",
        hours=8,
        args=[ctx],
        id="cross_fr_refresh",
        name="8-hourly Bybit funding rate refresh",
        max_instances=1,
    )
