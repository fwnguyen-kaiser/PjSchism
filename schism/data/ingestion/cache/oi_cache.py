"""Open interest and long/short ratio cache for live ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from schism.data.ingestion.binance_client import BinanceClient
from schism.utils.logger import ingestion_logger


@dataclass(frozen=True)
class OISnapshot:
    oi: float
    lsr: float


class OICache:
    """
    In-memory cache of the latest OI and LSR snapshot per symbol.

    OI and LSR are stored together so a reader never observes one updated
    without the other.
    """

    def __init__(self) -> None:
        self._snapshot: dict[str, OISnapshot] = {}

    def update(self, symbol: str, oi: float, lsr: float) -> None:
        self._snapshot = {**self._snapshot, symbol: OISnapshot(oi=oi, lsr=lsr)}

    def get_oi(self, symbol: str) -> Optional[float]:
        snapshot = self._snapshot.get(symbol)
        return snapshot.oi if snapshot is not None else None

    def get_lsr(self, symbol: str) -> Optional[float]:
        snapshot = self._snapshot.get(symbol)
        return snapshot.lsr if snapshot is not None else None

    async def refresh(self, client: BinanceClient, symbols: list[str]) -> None:
        """Fetch latest OI and LSR for each symbol via REST."""
        for symbol in symbols:
            try:
                oi_records = await client.get_open_interest_hist(
                    symbol, period="4h", limit=1
                )
                if oi_records:
                    oi_val = oi_records[-1]["sum_open_interest"]
                else:
                    cached_oi = self.get_oi(symbol)
                    oi_val = cached_oi if cached_oi is not None else 0.0

                lsr_records = await client.get_top_lsr(
                    symbol, period="4h", limit=1
                )
                lsr_val = (
                    lsr_records[-1]["long_short_ratio"]
                    if lsr_records
                    else self.get_lsr(symbol)
                )
                if lsr_val is None:
                    lsr_val = 1.0

                self.update(symbol, oi_val, lsr_val)
                ingestion_logger.debug(
                    "oi_cache_updated",
                    symbol=symbol,
                    oi=oi_val,
                    lsr=lsr_val,
                )
            except Exception as exc:
                ingestion_logger.warning(
                    "oi_cache_refresh_failed",
                    symbol=symbol,
                    error=str(exc),
                )
