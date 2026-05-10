"""
GET /regime/current  — latest state + posterior + features snapshot.
GET /regime/history  — bars with regime overlay within a time window.
GET /regime/stats    — per-state frequency, mean sojourn, health.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from schism.api.dependencies import get_session
from schism.api.schemas import BarWithPosterior, BarWithRegime, RegimeSnapshot, RegimeStats
from schism.persistence.repositories import state_repo

router = APIRouter()

Session = Annotated[AsyncSession, Depends(get_session)]


async def _resolve(
    session: AsyncSession,
    exchange: str,
    symbol: str,
    market_type: str,
    timeframe: str,
) -> tuple[int, int]:
    instrument_id = await state_repo.resolve_instrument_id(session, exchange, symbol, market_type)
    if instrument_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"instrument not found: {exchange}/{symbol}/{market_type}",
        )
    timeframe_id = await state_repo.resolve_timeframe_id(session, timeframe)
    if timeframe_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"timeframe not found: {timeframe}",
        )
    return instrument_id, timeframe_id


@router.get("/current", response_model=RegimeSnapshot)
async def current(
    session: Session,
    exchange: str = Query(default="binance"),
    symbol: str = Query(default="BTCUSDT"),
    market_type: str = Query(default="perp"),
    timeframe: str = Query(default="4h"),
) -> RegimeSnapshot:
    instrument_id, timeframe_id = await _resolve(session, exchange, symbol, market_type, timeframe)
    data = await state_repo.get_current(session, instrument_id, timeframe_id)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no regime state available yet",
        )
    return RegimeSnapshot(**data)


@router.get("/history", response_model=list[BarWithRegime])
async def history(
    session: Session,
    exchange: str = Query(default="binance"),
    symbol: str = Query(default="BTCUSDT"),
    market_type: str = Query(default="perp"),
    timeframe: str = Query(default="4h"),
    from_ts: datetime = Query(default=None),
    to_ts: datetime = Query(default=None),
) -> list[BarWithRegime]:
    instrument_id, timeframe_id = await _resolve(session, exchange, symbol, market_type, timeframe)

    now = datetime.now(tz=timezone.utc)
    resolved_to = to_ts or now
    resolved_from = from_ts or resolved_to.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    rows = await state_repo.get_history(session, instrument_id, timeframe_id, resolved_from, resolved_to)
    return [BarWithRegime(**r) for r in rows]


@router.get("/posteriors", response_model=list[BarWithPosterior])
async def posteriors(
    session: Session,
    exchange: str = Query(default="binance"),
    symbol: str = Query(default="BTCUSDT"),
    market_type: str = Query(default="perp"),
    timeframe: str = Query(default="4h"),
    from_ts: datetime = Query(default=None),
    to_ts: datetime = Query(default=None),
) -> list[BarWithPosterior]:
    instrument_id, timeframe_id = await _resolve(session, exchange, symbol, market_type, timeframe)

    now = datetime.now(tz=timezone.utc)
    resolved_to = to_ts or now
    resolved_from = from_ts or resolved_to.replace(hour=0, minute=0, second=0, microsecond=0)

    rows = await state_repo.get_posteriors(session, instrument_id, timeframe_id, resolved_from, resolved_to)
    return [BarWithPosterior(**r) for r in rows]


@router.get("/stats", response_model=list[RegimeStats])
async def stats(
    session: Session,
    exchange: str = Query(default="binance"),
    symbol: str = Query(default="BTCUSDT"),
    market_type: str = Query(default="perp"),
    timeframe: str = Query(default="4h"),
) -> list[RegimeStats]:
    instrument_id, timeframe_id = await _resolve(session, exchange, symbol, market_type, timeframe)
    rows = await state_repo.get_stats(session, instrument_id, timeframe_id)
    return [RegimeStats(**r) for r in rows]
