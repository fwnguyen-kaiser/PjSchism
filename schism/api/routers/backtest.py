"""
GET /backtest/results?run_id= — equity curve + regime bars + metrics.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/results")
async def results(run_id: str | None = None) -> dict:
    return {"status": "not_implemented", "run_id": run_id}
