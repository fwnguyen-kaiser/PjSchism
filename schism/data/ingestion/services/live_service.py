"""Live ingestion loop service."""

from __future__ import annotations

from schism.data.ingestion.bar_builder import Bar, IngestionSource
from schism.data.ingestion.context import AppContext
from schism.utils.logger import ingestion_logger


class LiveService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    async def run(self, symbol: str) -> None:
        """Run the live ingestion loop for one symbol."""
        ingestion_logger.info("live_loop_start", symbol=symbol)
        await self.ctx.client.stream_kline_close(
            symbol=symbol,
            on_bar_close=lambda bar: self._on_bar_close(symbol, bar),
            interval="4h",
        )

    async def _on_bar_close(self, symbol: str, bar: Bar) -> None:
        """Attach metrics, persist the bar, refresh snapshots, and publish."""
        bar.funding_rate = self.ctx.funding_cache.get(symbol)
        bar.oi = self.ctx.oi_cache.get_oi(symbol)
        bar.lsr_top = self.ctx.oi_cache.get_lsr(symbol)
        if self.ctx.cross_fr_cache is not None:
            bar.bybit_fr = self.ctx.cross_fr_cache.get(symbol)
        if bar.source is None:
            bar.source = IngestionSource.BINANCE_WS

        # L1 snapshot at bar close — non-fatal if the REST call fails.
        try:
            snap = await self.ctx.client.get_book_ticker_snapshot(symbol)
            bar.best_bid = snap["bid_price"]
            bar.best_ask = snap["ask_price"]
        except Exception as exc:
            ingestion_logger.warning(
                "book_ticker_snapshot_failed",
                symbol=symbol,
                bar_ts=bar.bar_ts.isoformat(),
                error=str(exc),
            )

        ingestion_logger.info(
            "live_bar_complete",
            symbol=symbol,
            bar_ts=bar.bar_ts.isoformat(),
            close=bar.close,
            volume=bar.volume,
            cvd=round(bar.cvd, 4),
            oi=bar.oi,
            funding_rate=bar.funding_rate,
            best_bid=bar.best_bid,
            best_ask=bar.best_ask,
            bybit_fr=bar.bybit_fr,
        )

        try:
            await self.ctx.store.write_bars([bar])
        except Exception as exc:
            ingestion_logger.error(
                "live_bar_parquet_failed",
                symbol=symbol,
                bar_ts=bar.bar_ts.isoformat(),
                error=str(exc),
            )
        else:
            await self._write_bar_to_db(symbol, bar)

        await self.ctx.oi_cache.refresh(self.ctx.client, [symbol])
        if self.ctx.bybit_client is not None and self.ctx.cross_fr_cache is not None:
            await self.ctx.cross_fr_cache.refresh(self.ctx.bybit_client, [symbol])
        await self.ctx.publisher.publish(bar)

    async def _write_bar_to_db(self, symbol: str, bar: Bar) -> None:
        if self.ctx.bar_repo is None:
            return
        try:
            await self.ctx.bar_repo.upsert_bars([bar])
            ingestion_logger.info(
                "timescale_bar_written",
                symbol=symbol,
                bar_ts=bar.bar_ts.isoformat(),
            )
        except Exception as exc:
            ingestion_logger.error(
                "timescale_bar_write_failed",
                symbol=symbol,
                bar_ts=bar.bar_ts.isoformat(),
                error=str(exc),
            )
