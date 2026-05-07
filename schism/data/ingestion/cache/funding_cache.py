"""Funding rate cache for live ingestion."""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from schism.data.ingestion.binance_client import BinanceClient
from schism.utils.date_helpers import utc_now
from schism.utils.logger import ingestion_logger


class FundingCache:
    """
    In-memory cache of the latest funding rate per symbol.

    The snapshot is replaced instead of mutated in place so readers always see
    a consistent view while refreshes are in flight.
    """

    def __init__(self) -> None:
        self._snapshot: dict[str, float] = {}

    def update(self, symbol: str, rate: float) -> None:
        self._snapshot = {**self._snapshot, symbol: rate}

    def get(self, symbol: str) -> Optional[float]:
        return self._snapshot.get(symbol)

    async def refresh(self, client: BinanceClient, symbols: list[str]) -> None:
        """Fetch latest funding rate for each symbol via REST."""
        for symbol in symbols:
            try:
                end = utc_now()
                start = end - timedelta(hours=8)
                records = await client.get_funding_rate(
                    symbol, start_time=start, end_time=end, limit=1
                )
                if records:
                    rate = records[-1]["funding_rate"]
                    self.update(symbol, rate)
                    ingestion_logger.debug(
                        "funding_cache_updated",
                        symbol=symbol,
                        funding_rate=rate,
                    )
            except Exception as exc:
                ingestion_logger.warning(
                    "funding_cache_refresh_failed",
                    symbol=symbol,
                    error=str(exc),
                )

