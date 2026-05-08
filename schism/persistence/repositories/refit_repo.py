"""refit_log queries: insert refit entries, query for /refit/log endpoint."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_RESOLVE_INSTRUMENT_SQL = text("""
    SELECT instrument_id FROM instruments
    WHERE exchange = :exchange AND symbol = :symbol AND market_type = :market_type
""")

_RESOLVE_TIMEFRAME_SQL = text("""
    SELECT timeframe_id FROM timeframes_metadata WHERE label = :label
""")

_LOG_SQL = text("""
    SELECT refit_id, refit_ts, instrument_id, timeframe_id, trigger,
           delta_bic, alignment_ok, drift_alert, dim_used, model_ver,
           cooldown_end_ts, notes
    FROM refit_log
    WHERE instrument_id = :instrument_id AND timeframe_id = :timeframe_id
    ORDER BY refit_ts DESC
    LIMIT :limit
""")

_INSERT_SQL = text("""
    INSERT INTO refit_log
        (instrument_id, timeframe_id, trigger, delta_bic, alignment_ok,
         drift_alert, dim_used, model_ver, cooldown_end_ts, notes)
    VALUES
        (:instrument_id, :timeframe_id, :trigger, :delta_bic, :alignment_ok,
         :drift_alert, :dim_used, :model_ver, :cooldown_end_ts, :notes)
    RETURNING refit_id, refit_ts
""")


async def resolve_instrument_id(
    session: AsyncSession,
    exchange: str,
    symbol: str,
    market_type: str,
) -> int | None:
    result = await session.execute(
        _RESOLVE_INSTRUMENT_SQL,
        {"exchange": exchange.lower(), "symbol": symbol.upper(), "market_type": market_type.lower()},
    )
    row = result.fetchone()
    return int(row[0]) if row else None


async def resolve_timeframe_id(session: AsyncSession, label: str) -> int | None:
    result = await session.execute(_RESOLVE_TIMEFRAME_SQL, {"label": label.lower()})
    row = result.fetchone()
    return int(row[0]) if row else None


async def get_log(
    session: AsyncSession,
    instrument_id: int,
    timeframe_id: int,
    limit: int = 100,
) -> list[dict]:
    result = await session.execute(
        _LOG_SQL,
        {"instrument_id": instrument_id, "timeframe_id": timeframe_id, "limit": limit},
    )
    return [
        {
            "refit_id": row.refit_id,
            "refit_ts": row.refit_ts,
            "instrument_id": row.instrument_id,
            "timeframe_id": row.timeframe_id,
            "trigger": row.trigger,
            "delta_bic": row.delta_bic,
            "alignment_ok": row.alignment_ok,
            "drift_alert": row.drift_alert,
            "dim_used": row.dim_used,
            "model_ver": row.model_ver,
            "cooldown_end_ts": row.cooldown_end_ts,
            "notes": row.notes,
        }
        for row in result.fetchall()
    ]


async def insert(
    session: AsyncSession,
    instrument_id: int,
    timeframe_id: int,
    trigger: str,
    *,
    delta_bic: float | None = None,
    alignment_ok: bool | None = None,
    drift_alert: bool = False,
    dim_used: int | None = None,
    model_ver: str | None = None,
    cooldown_end_ts: datetime | None = None,
    notes: str | None = None,
) -> tuple[int, datetime]:
    result = await session.execute(
        _INSERT_SQL,
        {
            "instrument_id": instrument_id,
            "timeframe_id": timeframe_id,
            "trigger": trigger,
            "delta_bic": delta_bic,
            "alignment_ok": alignment_ok,
            "drift_alert": drift_alert,
            "dim_used": dim_used,
            "model_ver": model_ver,
            "cooldown_end_ts": cooldown_end_ts,
            "notes": notes,
        },
    )
    row = result.fetchone()
    return int(row.refit_id), row.refit_ts
