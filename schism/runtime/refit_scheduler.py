"""
Enforce 5-day cooldown between refits. Backstop overrides cooldown.
"""

from __future__ import annotations

from schism.utils.logger import ingestion_logger

_LOG = ingestion_logger


class RefitScheduler:
    """
    Gate that prevents refitting more often than once per cooldown_bars.

    Parameters
    ----------
    cooldown_bars : minimum bars between refits (default 30 = 5 days × 6 4H bars)
    """

    def __init__(self, cooldown_bars: int = 30) -> None:
        self.cooldown_bars = cooldown_bars
        self._bars_since_last: int = cooldown_bars  # start ready to refit

    # ── Public ───────────────────────────────────────────────────────────────

    @property
    def bars_since_last(self) -> int:
        return self._bars_since_last

    def tick(self) -> None:
        """Advance the internal bar counter by one."""
        self._bars_since_last += 1

    def can_refit(self, backstop_override: bool = False) -> bool:
        """
        Return True if a refit is permitted.
        Backstop flag bypasses the cooldown gate.
        """
        if backstop_override:
            return True
        return self._bars_since_last >= self.cooldown_bars

    def record_refit(self) -> None:
        """Reset the cooldown counter after a refit completes."""
        _LOG.info("refit_scheduler_record", bars_since_last=self._bars_since_last)
        self._bars_since_last = 0
