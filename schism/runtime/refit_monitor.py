"""
Track delta_LL rolling stats, RVratio consecutive bars, backstop countdown.

Refit is triggered when ANY of:
  (a) ΔLL_t < μ_{ΔLL} − 2σ_{ΔLL}  (30-day rolling window, default 180 4H bars)
  (b) RV_ratio > 1.8 for 12 consecutive bars
  (c) bars_since_refit ≥ backstop_bars  (90-day backstop, default 540 4H bars)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from schism.utils.logger import ingestion_logger

_LOG = ingestion_logger


class RefitMonitor:
    """
    Stateful monitor that decides whether a model refit is warranted.

    Parameters
    ----------
    ll_window      : rolling window length for ΔLL mean/std (bars)
    rv_threshold   : RV_ratio threshold for consecutive-bar trigger
    rv_consec      : number of consecutive bars above rv_threshold required
    backstop_bars  : hard refit trigger regardless of cooldown (bars)
    """

    def __init__(
        self,
        ll_window: int = 180,
        rv_threshold: float = 1.8,
        rv_consec: int = 12,
        backstop_bars: int = 540,
    ) -> None:
        self.ll_window = ll_window
        self.rv_threshold = rv_threshold
        self.rv_consec = rv_consec
        self.backstop_bars = backstop_bars

        self._ll_deltas: deque[float] = deque(maxlen=ll_window)
        self._rv_streak: int = 0

    # ── Public ───────────────────────────────────────────────────────────────

    def update(self, delta_ll: float, rv_ratio: float, bars_since_refit: int) -> bool:
        """
        Record one bar and return True if any refit trigger fires.

        Parameters
        ----------
        delta_ll        : incremental log-likelihood for this bar
        rv_ratio        : f7_rv_ratio observation for this bar
        bars_since_refit: bars elapsed since the last model refit
        """
        self._ll_deltas.append(delta_ll)

        if rv_ratio > self.rv_threshold:
            self._rv_streak += 1
        else:
            self._rv_streak = 0

        fired = (
            self._ll_triggered()
            or self._rv_triggered()
            or self._backstop_triggered(bars_since_refit)
        )
        if fired:
            _LOG.info(
                "refit_monitor_trigger",
                ll_trigger=self._ll_triggered(),
                rv_trigger=self._rv_triggered(),
                backstop_trigger=self._backstop_triggered(bars_since_refit),
                rv_streak=self._rv_streak,
                bars_since_refit=bars_since_refit,
                ll_window_len=len(self._ll_deltas),
            )
        return fired

    def backstop_triggered(self, bars_since_refit: int) -> bool:
        """True if the hard backstop has been reached (overrides cooldown)."""
        return self._backstop_triggered(bars_since_refit)

    def reset(self) -> None:
        """Clear rolling window and streak counters after a refit."""
        self._ll_deltas.clear()
        self._rv_streak = 0

    # ── Private ──────────────────────────────────────────────────────────────

    def _ll_triggered(self) -> bool:
        if len(self._ll_deltas) < self.ll_window:
            return False
        arr = np.array(self._ll_deltas)
        mu = float(arr.mean())
        sigma = float(arr.std()) + 1e-10
        return bool(self._ll_deltas[-1] < mu - 2.0 * sigma)

    def _rv_triggered(self) -> bool:
        return self._rv_streak >= self.rv_consec

    def _backstop_triggered(self, bars_since_refit: int) -> bool:
        return bars_since_refit >= self.backstop_bars
