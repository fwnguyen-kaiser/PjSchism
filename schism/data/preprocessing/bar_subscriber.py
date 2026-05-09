"""
Redis Stream consumer: receives bar_completed events from the ingestion service
and triggers FeatureEngine.compute_and_store for each new bar.

Stream key   : schism:bars:{symbol}  (published by RedisPublisher)
Consumer group: schism:feature_engine
Delivery      : at-least-once (XACK after successful processing)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import redis.asyncio as aioredis

from schism.data.preprocessing.feature_engine import FeatureEngine
from schism.persistence.repositories.bar_repo import BarRepository
from schism.utils.logger import ingestion_logger

_GROUP = "schism:feature_engine"
_CONSUMER = "fe_worker"
_BLOCK_MS = 5_000   # 5 s long-poll
_BATCH = 10         # messages per xreadgroup call


class BarSubscriber:
    """
    Consumes completed bars from Redis Streams and computes feature vectors.

    One BarSubscriber instance covers all configured symbols. Each received
    bar event triggers compute_and_store for that bar's timestamp; the engine
    fetches the necessary lookback window from TimescaleDB and writes results
    to feature_vectors.
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        feature_engine: FeatureEngine,
        bar_repo: BarRepository,
        symbols: list[str],
    ) -> None:
        self._redis = redis
        self._engine = feature_engine
        self._bar_repo = bar_repo
        self._symbols = symbols
        self._stream_keys = {
            f"schism:bars:{sym}": sym for sym in symbols
        }

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Create consumer groups (idempotent) then run the consume loop."""
        await self._ensure_groups()
        await self._consume_loop()

    async def _ensure_groups(self) -> None:
        for stream in self._stream_keys:
            try:
                # MKSTREAM creates the stream if it doesn't exist yet
                await self._redis.xgroup_create(stream, _GROUP, id="$", mkstream=True)
                ingestion_logger.info("redis_group_created", stream=stream, group=_GROUP)
            except aioredis.ResponseError as exc:
                if "BUSYGROUP" in str(exc):
                    pass  # group already exists — expected on restart
                else:
                    raise

    # ── consume loop ──────────────────────────────────────────────────────────

    async def _consume_loop(self) -> None:
        ingestion_logger.info(
            "bar_subscriber_start",
            symbols=self._symbols,
            group=_GROUP,
        )
        stream_ids = {key: ">" for key in self._stream_keys}

        while True:
            try:
                results = await self._redis.xreadgroup(
                    groupname=_GROUP,
                    consumername=_CONSUMER,
                    streams=stream_ids,
                    count=_BATCH,
                    block=_BLOCK_MS,
                )
            except asyncio.CancelledError:
                ingestion_logger.info("bar_subscriber_stopped")
                return
            except Exception as exc:
                ingestion_logger.error("bar_subscriber_xread_error", error=str(exc))
                await asyncio.sleep(2)
                continue

            if not results:
                continue  # timeout, no new messages

            for stream_key_bytes, messages in results:
                stream_key = (
                    stream_key_bytes.decode()
                    if isinstance(stream_key_bytes, bytes)
                    else stream_key_bytes
                )
                for msg_id, fields in messages:
                    await self._handle(stream_key, msg_id, fields)

    # ── message handler ───────────────────────────────────────────────────────

    async def _handle(
        self,
        stream_key: str,
        msg_id: bytes | str,
        fields: dict,
    ) -> None:
        raw_ts = fields.get(b"bar_ts") or fields.get("bar_ts", "")
        symbol = fields.get(b"symbol") or fields.get("symbol", "")
        market_type = fields.get(b"market_type") or fields.get("market_type", b"perp")
        exchange = fields.get(b"exchange") or fields.get("exchange", b"binance")

        if isinstance(raw_ts, bytes):
            raw_ts = raw_ts.decode()
        if isinstance(symbol, bytes):
            symbol = symbol.decode()
        if isinstance(market_type, bytes):
            market_type = market_type.decode()
        if isinstance(exchange, bytes):
            exchange = exchange.decode()

        if not raw_ts or not symbol:
            ingestion_logger.warning(
                "bar_subscriber_bad_message",
                stream=stream_key,
                msg_id=str(msg_id),
            )
            await self._ack(stream_key, msg_id)
            return

        try:
            bar_ts = datetime.fromisoformat(raw_ts)
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.replace(tzinfo=timezone.utc)
        except ValueError:
            ingestion_logger.warning(
                "bar_subscriber_bad_ts",
                raw_ts=raw_ts,
                stream=stream_key,
            )
            await self._ack(stream_key, msg_id)
            return

        try:
            instrument_id, timeframe_id = await self._bar_repo.resolve_ids(
                exchange=exchange,
                symbol=symbol,
                market_type=market_type,
                timeframe_label="4h",
            )
            written = await self._engine.compute_and_store(
                instrument_id=instrument_id,
                timeframe_id=timeframe_id,
                from_ts=bar_ts,
                to_ts=bar_ts,
                market_type=market_type,
            )
            ingestion_logger.info(
                "bar_subscriber_processed",
                symbol=symbol,
                bar_ts=raw_ts,
                rows_written=written,
            )
        except Exception as exc:
            ingestion_logger.error(
                "bar_subscriber_process_failed",
                symbol=symbol,
                bar_ts=raw_ts,
                error=str(exc),
            )
            # ACK anyway — dead-letter handling is out of scope for now
            # A persistent failure here means the bar will be reprocessed
            # on next startup via backfill reconciliation.

        await self._ack(stream_key, msg_id)

    async def _ack(self, stream_key: str, msg_id: bytes | str) -> None:
        try:
            await self._redis.xack(stream_key, _GROUP, msg_id)
        except Exception as exc:
            ingestion_logger.warning(
                "bar_subscriber_ack_failed",
                stream=stream_key,
                msg_id=str(msg_id),
                error=str(exc),
            )
