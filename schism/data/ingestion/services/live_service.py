"""Live ingestion loop service."""

from __future__ import annotations

from schism.data.ingestion.bar_builder import Bar
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

        ingestion_logger.info(
            "live_bar_complete",
            symbol=symbol,
            bar_ts=bar.bar_ts.isoformat(),
            close=bar.close,
            volume=bar.volume,
            cvd=round(bar.cvd, 4),
            oi=bar.oi,
            funding_rate=bar.funding_rate,
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

        await self.ctx.oi_cache.refresh(self.ctx.client, [symbol])
        await self.ctx.publisher.publish(bar)

