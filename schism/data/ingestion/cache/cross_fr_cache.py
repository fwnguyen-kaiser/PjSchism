"""Cross-exchange funding rate cache for Ut component 3 (FR_t^Bnb - FR_t^Bybit)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from schism.utils.logger import ingestion_logger

if TYPE_CHECKING:
    from schism.data.ingestion.bybit_client import BybitClient


class CrossExchangeFRCache:
    """
    In-memory cache of the latest Bybit funding rate per symbol.

    The feature engine computes u3 = bnb_fr - bybit_fr; this cache
    supplies the bybit_fr half. Snapshot replaced atomically on refresh.
    """

    def __init__(self) -> None:
        self._snapshot: dict[str, float] = {}

    def update(self, symbol: str, bybit_fr: float) -> None:
        self._snapshot = {**self._snapshot, symbol: bybit_fr}

    def get(self, symbol: str) -> Optional[float]:
        return self._snapshot.get(symbol)

    async def refresh(self, client: "BybitClient", symbols: list[str]) -> None:
        """Fetch latest Bybit funding rate for each symbol via REST."""
        for symbol in symbols:
            try:
                records = await client.get_funding_rate(symbol, limit=1)
                if records:
                    rate = records[-1]["funding_rate"]
                    self.update(symbol, rate)
                    ingestion_logger.debug(
                        "cross_fr_cache_updated",
                        symbol=symbol,
                        bybit_fr=rate,
                    )
            except Exception as exc:
                ingestion_logger.warning(
                    "cross_fr_cache_refresh_failed",
                    symbol=symbol,
                    error=str(exc),
                )
