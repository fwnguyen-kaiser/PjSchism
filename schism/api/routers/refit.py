"""
GET /refit/log — refit history with trigger, delta_bic, alignment result.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from schism.api.dependencies import get_session
from schism.api.schemas import RefitLogEntry
from schism.persistence.repositories import refit_repo

router = APIRouter()

Session = Annotated[AsyncSession, Depends(get_session)]


@router.get("/log", response_model=list[RefitLogEntry])
async def log(
    session: Session,
    exchange: str = Query(default="binance"),
    symbol: str = Query(default="BTCUSDT"),
    market_type: str = Query(default="perp"),
    timeframe: str = Query(default="4h"),
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[RefitLogEntry]:
    instrument_id = await refit_repo.resolve_instrument_id(session, exchange, symbol, market_type)
    if instrument_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"instrument not found: {exchange}/{symbol}/{market_type}",
        )
    timeframe_id = await refit_repo.resolve_timeframe_id(session, timeframe)
    if timeframe_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"timeframe not found: {timeframe}",
        )
    rows = await refit_repo.get_log(session, instrument_id, timeframe_id, limit=limit)
    return [RefitLogEntry(**r) for r in rows]
