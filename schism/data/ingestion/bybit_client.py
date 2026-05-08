"""
Lightweight Bybit V5 public client.

Single responsibility: funding rate history for the cross-exchange FR spread
(Ut component 3: FR_t^Bnb - FR_t^Bybit). No auth required — public endpoint only.

Bybit linear perp symbols match Binance naming (BTCUSDT, ETHUSDT, …).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx

from schism.utils.logger import ingestion_logger

_BYBIT_BASE = "https://api.bybit.com"
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class BybitClient:
    """
    Async Bybit V5 public client — funding rate only.

    Usage:
        client = BybitClient()
        records = await client.get_funding_rate("BTCUSDT", limit=200)
    """

    def __init__(self, timeout: httpx.Timeout = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch funding rate history for a linear perp symbol.

        Bybit returns results newest-first; this method reverses to ascending order.

        Returns:
            List of {funding_time: datetime, funding_rate: float}, oldest first.

        Raises:
            RuntimeError: Bybit retCode != 0 or HTTP error.
        """
        params: dict = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time is not None:
            params["endTime"] = int(end_time.timestamp() * 1000)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                resp = await client.get(
                    f"{_BYBIT_BASE}/v5/market/funding/history",
                    params=params,
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Bybit HTTP error: {exc}") from exc

        data = resp.json()
        if data.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}"
            )

        records = []
        for row in reversed(data["result"]["list"]):  # newest-first → oldest-first
            records.append({
                "funding_time": datetime.fromtimestamp(
                    int(row["fundingRateTimestamp"]) / 1000, tz=timezone.utc
                ),
                "funding_rate": float(row["fundingRate"]),
            })

        ingestion_logger.debug(
            "bybit_funding_fetched",
            symbol=symbol,
            count=len(records),
        )
        return records
