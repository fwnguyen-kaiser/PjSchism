"""
RollingZScore: fit on train window, transform online. Interaction-last order per V1.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-8  # spec §4, ε = 10^-8


class RollingZScore:
    """
    Online rolling Z-score normalizer.

    Maintains a fixed-length sliding buffer. Each `update` call appends the new
    value, evicts the oldest if the window is full, then returns the Z-score of
    that value relative to the current buffer.

    Edge cases:
      - len(buf) < 2  → nan  (insufficient history, not a valid Z-score)
      - std < _EPS    → 0.0  (constant series; point equals mean by definition)

    Usage (online / live mode):
        zs = RollingZScore(window=360)
        zs.fit(historical_array)   # prime buffer with lookback data
        z = zs.update(new_value)

    Usage (batch / backfill):
        z_arr = RollingZScore.batch_transform(raw_array, window=360)
    """

    def __init__(self, window: int = 360) -> None:
        self.window = window
        self._buf: list[float] = []

    def fit(self, values: np.ndarray) -> None:
        """Prime the buffer with the last `window` entries of historical data."""
        arr = np.asarray(values, dtype=float)
        self._buf = list(arr[-self.window :])

    def update(self, x: float) -> float:
        """
        Append x to the buffer and return its Z-score over the current window.

        Returns nan if fewer than 2 values are in the buffer.
        Returns 0.0 if the series is constant (std < _EPS).
        """
        self._buf.append(float(x))
        if len(self._buf) > self.window:
            self._buf.pop(0)
        if len(self._buf) < 2:
            return float("nan")
        arr = np.array(self._buf, dtype=float)
        mean = arr.mean()
        std = arr.std(ddof=1)
        if std < _EPS:
            return 0.0
        return float((x - mean) / std)

    def reset(self) -> None:
        self._buf = []

    @staticmethod
    def batch_transform(values: np.ndarray, window: int = 360) -> np.ndarray:
        """
        Vectorised rolling Z-score for a full array (backfill use).

        Each element at index i is Z-scored relative to values[max(0, i-window+1):i+1].
        Returns nan for index 0 (single sample) and 0.0 for constant sub-windows.
        """
        s = pd.Series(np.asarray(values, dtype=float))
        roll = s.rolling(window, min_periods=2)
        mean = roll.mean()
        std = roll.std(ddof=1)

        # Avoid division by zero: where std is known zero, result is 0.0;
        # where std is nan (min_periods not met), result propagates as nan.
        zero_std = std.notna() & (std < _EPS)
        result = (s - mean) / std.where(~zero_std, other=1.0)
        result = result.where(~zero_std, other=0.0)
        return result.to_numpy(dtype=float)
