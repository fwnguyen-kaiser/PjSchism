"""Redis Stream publisher for completed bars."""

from __future__ import annotations

import redis.asyncio as aioredis

from schism.data.ingestion.bar_builder import Bar
from schism.utils.logger import ingestion_logger


class RedisPublisher:
    STREAM_KEY = "schism:bars:{symbol}"
    STREAM_MAXLEN = 2000

    def __init__(self, redis: aioredis.Redis) -> None:
        self.redis = redis

    async def publish(self, bar: Bar) -> None:
        """
        Publish a completed Bar to the Redis Stream for the model service.

        Redis publish failures are non-fatal because parquet is the source of
        truth for ingestion.
        """
        key = self.STREAM_KEY.format(symbol=bar.symbol)
        fields = {
            "bar_ts": bar.bar_ts.isoformat(),
            "symbol": bar.symbol,
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": str(bar.volume),
            "cvd": str(bar.cvd),
            "oi": str(bar.oi) if bar.oi is not None else "",
            "lsr_top": str(bar.lsr_top) if bar.lsr_top is not None else "",
            "funding_rate": str(bar.funding_rate) if bar.funding_rate is not None else "",
        }
        try:
            await self.redis.xadd(
                key,
                fields,
                maxlen=self.STREAM_MAXLEN,
                approximate=True,
            )
            ingestion_logger.debug(
                "redis_xadd",
                symbol=bar.symbol,
                bar_ts=bar.bar_ts.isoformat(),
                stream=key,
            )
        except Exception as exc:
            ingestion_logger.warning(
                "redis_xadd_failed",
                symbol=bar.symbol,
                bar_ts=bar.bar_ts.isoformat(),
                error=str(exc),
            )

