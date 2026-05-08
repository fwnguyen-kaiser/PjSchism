"""Repository for TimescaleDB OHLCV bar persistence."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from schism.data.ingestion.bar_builder import Bar
from schism.persistence.db import session_scope


@dataclass(frozen=True)
class _InstrumentKey:
    exchange: str
    symbol: str
    market_type: str


_UPSERT_INSTRUMENT_SQL = text(
    """
    INSERT INTO instruments (exchange, symbol, market_type)
    VALUES (:exchange, :symbol, :market_type)
    ON CONFLICT (exchange, symbol, market_type) DO NOTHING
    """
)

_SELECT_INSTRUMENT_SQL = text(
    """
    SELECT instrument_id
    FROM instruments
    WHERE exchange = :exchange
      AND symbol = :symbol
      AND market_type = :market_type
    """
)

_UPSERT_TIMEFRAME_SQL = text(
    """
    INSERT INTO timeframes_metadata (timeframe_id, code, label, duration_seconds, is_primary)
    VALUES (:timeframe_id, :code, :label, :duration_seconds, :is_primary)
    ON CONFLICT (timeframe_id) DO UPDATE SET
        code = EXCLUDED.code,
        label = EXCLUDED.label,
        duration_seconds = EXCLUDED.duration_seconds,
        is_primary = EXCLUDED.is_primary
    """
)

_SELECT_TIMEFRAME_SQL = text(
    """
    SELECT timeframe_id
    FROM timeframes_metadata
    WHERE label = :label
    """
)

_PATCH_OI_LSR_SQL = text(
    """
    UPDATE ohlcv_bars
    SET oi = :oi, lsr_top = :lsr_top
    WHERE instrument_id = :instrument_id
      AND timeframe_id  = :timeframe_id
      AND bar_ts        = :bar_ts
    """
)

_UPSERT_BARS_SQL = text(
    """
    INSERT INTO ohlcv_bars (
        bar_ts,
        instrument_id,
        timeframe_id,
        open,
        high,
        low,
        close,
        volume,
        cvd,
        oi,
        lsr_top,
        funding_rate,
        best_bid,
        best_ask,
        bybit_fr,
        num_trades,
        taker_buy_base,
        quote_volume,
        source
    )
    VALUES (
        :bar_ts,
        :instrument_id,
        :timeframe_id,
        :open,
        :high,
        :low,
        :close,
        :volume,
        :cvd,
        :oi,
        :lsr_top,
        :funding_rate,
        :best_bid,
        :best_ask,
        :bybit_fr,
        :num_trades,
        :taker_buy_base,
        :quote_volume,
        :source
    )
    ON CONFLICT (instrument_id, timeframe_id, bar_ts) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        cvd = EXCLUDED.cvd,
        oi = EXCLUDED.oi,
        lsr_top = EXCLUDED.lsr_top,
        funding_rate = EXCLUDED.funding_rate,
        best_bid = EXCLUDED.best_bid,
        best_ask = EXCLUDED.best_ask,
        bybit_fr = EXCLUDED.bybit_fr,
        num_trades = EXCLUDED.num_trades,
        taker_buy_base = EXCLUDED.taker_buy_base,
        quote_volume = EXCLUDED.quote_volume,
        source = EXCLUDED.source
    """
)


class BarRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory
        self._instrument_cache: dict[_InstrumentKey, int] = {}
        self._timeframe_cache: dict[str, int] = {}

    async def upsert_bars(self, bars: list[Bar]) -> None:
        if not bars:
            return
        async with session_scope(self.session_factory) as session:
            rows = []
            for bar in bars:
                instrument_id = await self._resolve_instrument_id(session, bar)
                timeframe_id = await self._resolve_timeframe_id(session, bar.timeframe_label)
                row = bar.to_dict()
                row["instrument_id"] = instrument_id
                row["timeframe_id"] = timeframe_id
                rows.append(row)
            await session.execute(_UPSERT_BARS_SQL, rows)

    async def resolve_ids(
        self,
        exchange: str,
        symbol: str,
        market_type: str,
        timeframe_label: str,
    ) -> tuple[int, int]:
        """Return (instrument_id, timeframe_id), upserting metadata rows if needed."""
        async with session_scope(self.session_factory) as session:
            instrument_id = await self._resolve_instrument_id_by_key(
                session, exchange, symbol, market_type
            )
            timeframe_id = await self._resolve_timeframe_id(session, timeframe_label)
        return instrument_id, timeframe_id

    async def patch_oi_lsr(
        self,
        instrument_id: int,
        timeframe_id: int,
        rows: list[dict],
    ) -> None:
        """Batch-update oi and lsr_top on existing ohlcv_bars rows."""
        if not rows:
            return
        async with session_scope(self.session_factory) as session:
            params = [
                {
                    "instrument_id": instrument_id,
                    "timeframe_id": timeframe_id,
                    "bar_ts": row["bar_ts"],
                    "oi": row["oi"],
                    "lsr_top": row.get("lsr_top"),
                }
                for row in rows
            ]
            await session.execute(_PATCH_OI_LSR_SQL, params)

    async def _resolve_instrument_id(self, session: AsyncSession, bar: Bar) -> int:
        return await self._resolve_instrument_id_by_key(
            session, bar.exchange, bar.symbol, bar.market_type
        )

    async def _resolve_instrument_id_by_key(
        self, session: AsyncSession, exchange: str, symbol: str, market_type: str
    ) -> int:
        key = _InstrumentKey(
            exchange=exchange.lower(),
            symbol=symbol.upper(),
            market_type=market_type.lower(),
        )
        cached = self._instrument_cache.get(key)
        if cached is not None:
            return cached
        await session.execute(
            _UPSERT_INSTRUMENT_SQL,
            {
                "exchange": key.exchange,
                "symbol": key.symbol,
                "market_type": key.market_type,
            },
        )
        result = await session.execute(
            _SELECT_INSTRUMENT_SQL,
            {
                "exchange": key.exchange,
                "symbol": key.symbol,
                "market_type": key.market_type,
            },
        )
        instrument_id = result.scalar_one()
        self._instrument_cache[key] = int(instrument_id)
        return int(instrument_id)

    async def _resolve_timeframe_id(self, session: AsyncSession, label: str) -> int:
        normalized = label.lower()
        cached = self._timeframe_cache.get(normalized)
        if cached is not None:
            return cached

        timeframe_seed = {
            "15m": (1, "PT15M", 900, False),
            "1h": (2, "PT1H", 3600, False),
            "4h": (3, "PT4H", 14400, True),
            "1d": (4, "P1D", 86400, False),
        }.get(normalized)
        if timeframe_seed is not None:
            timeframe_id, code, duration_seconds, is_primary = timeframe_seed
            await session.execute(
                _UPSERT_TIMEFRAME_SQL,
                {
                    "timeframe_id": timeframe_id,
                    "code": code,
                    "label": normalized,
                    "duration_seconds": duration_seconds,
                    "is_primary": is_primary,
                },
            )
        result = await session.execute(_SELECT_TIMEFRAME_SQL, {"label": normalized})
        timeframe_id = result.scalar_one()
        self._timeframe_cache[normalized] = int(timeframe_id)
        return int(timeframe_id)

