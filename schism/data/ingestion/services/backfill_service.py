"""Backfill workflows for ingestion."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from schism.data.ingestion.bar_builder import Bar
from schism.data.ingestion.context import AppContext
from schism.data.ingestion.vision_crawler import VisionCrawler
from schism.utils.date_helpers import datetime_to_bar_ts, to_iso, utc_now
from schism.utils.exceptions import DataMissingError
from schism.utils.logger import ingestion_logger


class BackfillService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    async def run(self, symbol: str) -> None:
        """Orchestrate vision + kline backfill for one symbol."""
        await self.run_vision(symbol)
        await self.run_klines(symbol)

    async def run_vision(self, symbol: str, days: int | None = None) -> None:
        """
        Crawl binance.vision for OI/LSR history and write to parquet.
        Only fetches data older than what's already cached.
        """
        backfill_days = days if days is not None else self.ctx.backfill_days
        start = utc_now() - timedelta(days=backfill_days)
        end = utc_now()

        ingestion_logger.info(
            "backfill_vision_start",
            symbol=symbol,
            start=to_iso(start),
            days=backfill_days,
        )

        crawler = VisionCrawler(symbol=symbol, out_dir=self.ctx.parquet_root)
        try:
            records = await crawler.fetch_range(start=start, end=end)
            if records:
                await self.ctx.store.write_vision_metrics(records, symbol)
                ingestion_logger.info(
                    "backfill_vision_done",
                    symbol=symbol,
                    records=len(records),
                )
            else:
                ingestion_logger.warning("backfill_vision_empty", symbol=symbol)
        except Exception as exc:
            ingestion_logger.error(
                "backfill_vision_failed",
                symbol=symbol,
                error=str(exc),
            )

    async def run_klines(self, symbol: str) -> None:
        """
        Fetch historical 4h klines + funding via REST and write to parquet.
        Paginates in 1000-bar chunks (Binance max per request).
        """
        end = datetime_to_bar_ts(utc_now(), "4h")
        start = end - timedelta(days=self.ctx.backfill_days)

        ingestion_logger.info(
            "backfill_klines_start",
            symbol=symbol,
            start=to_iso(start),
            days=self.ctx.backfill_days,
        )

        all_bars: list[Bar] = []
        total_bars = 0
        chunk_start = start
        funding_records: list[dict] = []
        try:
            funding_records = await self.ctx.client.get_funding_rate(
                symbol=symbol,
                start_time=start - timedelta(hours=8),
                end_time=end,
                limit=1000,
            )
            funding_records.sort(key=lambda row: row["funding_time"])
        except Exception as exc:
            ingestion_logger.warning(
                "backfill_funding_fetch_failed",
                symbol=symbol,
                start=to_iso(start),
                end=to_iso(end),
                error=str(exc),
            )

        funding_idx = 0
        current_funding_rate: Optional[float] = None

        while chunk_start < end:
            try:
                klines = await self.ctx.client.get_klines(
                    symbol=symbol,
                    interval="4h",
                    start_time=chunk_start,
                    end_time=end,
                    limit=1000,
                )
            except DataMissingError as exc:
                ingestion_logger.error(
                    "backfill_klines_fetch_failed",
                    symbol=symbol,
                    chunk_start=to_iso(chunk_start),
                    error=str(exc),
                )
                break

            if not klines:
                break

            for kline in klines:
                if kline["open_time"] >= end:
                    continue
                while (
                    funding_idx < len(funding_records)
                    and funding_records[funding_idx]["funding_time"] <= kline["open_time"]
                ):
                    current_funding_rate = funding_records[funding_idx]["funding_rate"]
                    funding_idx += 1

                bar = Bar(
                    bar_ts=kline["open_time"],
                    symbol=symbol,
                    open=kline["open"],
                    high=kline["high"],
                    low=kline["low"],
                    close=kline["close"],
                    volume=kline["volume"],
                    cvd=0.0,
                    num_trades=kline.get("num_trades", 0),
                    taker_buy_base=kline.get("taker_buy_base", 0.0),
                    quote_volume=kline.get("quote_volume", 0.0),
                    funding_rate=current_funding_rate,
                )
                all_bars.append(bar)
                total_bars += 1

            last_ts = klines[-1]["open_time"]
            chunk_start = last_ts + timedelta(hours=4)

            if len(all_bars) >= 5000:
                await self.ctx.store.write_bars(all_bars)
                ingestion_logger.info(
                    "backfill_klines_chunk_written",
                    symbol=symbol,
                    bars=len(all_bars),
                    up_to=to_iso(last_ts),
                )
                all_bars = []

            if len(klines) < 1000:
                break

        if all_bars:
            await self.ctx.store.write_bars(all_bars)

        ingestion_logger.info(
            "backfill_klines_done",
            symbol=symbol,
            total_bars=total_bars,
        )
