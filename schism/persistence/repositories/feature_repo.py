"""Repository for fetching raw bars and upserting computed feature vectors."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from schism.persistence.db import session_scope

_FETCH_FEATURES_SQL = text(
    """
    SELECT
        bar_ts,
        f1_cvd_vol, f2_oi_chg, f3_norm_ret, f4_liq_sq, f5_spread,
        f6_illiq, f7_rv_ratio, f8_vol_shock, f9_flow_liq, f10_flow_pos,
        u1_ewma_fr, u2_delta_fr, u3_fr_spread, u4_delta_lsr
    FROM feature_vectors
    WHERE instrument_id = :instrument_id
      AND timeframe_id  = :timeframe_id
      AND bar_ts BETWEEN :from_ts AND :to_ts
    ORDER BY bar_ts ASC
    """
)

_O_FEATURE_COLS = [
    "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq", "f5_spread",
    "f6_illiq", "f7_rv_ratio", "f8_vol_shock", "f9_flow_liq", "f10_flow_pos",
]
_U_FEATURE_COLS = ["u1_ewma_fr", "u2_delta_fr", "u3_fr_spread", "u4_delta_lsr"]

_FETCH_BARS_SQL = text(
    """
    SELECT
        bar_ts,
        open, high, low, close,
        volume, cvd,
        oi, lsr_top,
        funding_rate,
        best_bid, best_ask,
        bybit_fr
    FROM ohlcv_bars
    WHERE instrument_id = :instrument_id
      AND timeframe_id  = :timeframe_id
      AND bar_ts BETWEEN :from_ts AND :to_ts
    ORDER BY bar_ts ASC
    """
)

_UPSERT_FEATURES_SQL = text(
    """
    INSERT INTO feature_vectors (
        bar_ts, instrument_id, timeframe_id,
        f1_cvd_vol, f2_oi_chg, f3_norm_ret, f4_liq_sq, f5_spread,
        f6_illiq, f7_rv_ratio, f8_vol_shock, f9_flow_liq, f10_flow_pos,
        u1_ewma_fr, u2_delta_fr, u3_fr_spread, u4_delta_lsr,
        dim_used
    )
    VALUES (
        :bar_ts, :instrument_id, :timeframe_id,
        :f1_cvd_vol, :f2_oi_chg, :f3_norm_ret, :f4_liq_sq, :f5_spread,
        :f6_illiq, :f7_rv_ratio, :f8_vol_shock, :f9_flow_liq, :f10_flow_pos,
        :u1_ewma_fr, :u2_delta_fr, :u3_fr_spread, :u4_delta_lsr,
        :dim_used
    )
    ON CONFLICT (instrument_id, timeframe_id, bar_ts) DO UPDATE SET
        f1_cvd_vol   = EXCLUDED.f1_cvd_vol,
        f2_oi_chg    = EXCLUDED.f2_oi_chg,
        f3_norm_ret  = EXCLUDED.f3_norm_ret,
        f4_liq_sq    = EXCLUDED.f4_liq_sq,
        f5_spread    = EXCLUDED.f5_spread,
        f6_illiq     = EXCLUDED.f6_illiq,
        f7_rv_ratio  = EXCLUDED.f7_rv_ratio,
        f8_vol_shock = EXCLUDED.f8_vol_shock,
        f9_flow_liq  = EXCLUDED.f9_flow_liq,
        f10_flow_pos = EXCLUDED.f10_flow_pos,
        u1_ewma_fr   = EXCLUDED.u1_ewma_fr,
        u2_delta_fr  = EXCLUDED.u2_delta_fr,
        u3_fr_spread = EXCLUDED.u3_fr_spread,
        u4_delta_lsr = EXCLUDED.u4_delta_lsr,
        dim_used     = EXCLUDED.dim_used
    """
)

# f5_spread is nullable (migration 005 dropped NOT NULL — historical bars have no bid/ask)
_O_REQUIRED_COLS = [
    "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq",
    "f6_illiq", "f7_rv_ratio", "f8_vol_shock", "f9_flow_liq", "f10_flow_pos",
]
_O_NULLABLE_COLS = ["f5_spread"]
_U_COLS = ["u1_ewma_fr", "u2_delta_fr", "u3_fr_spread", "u4_delta_lsr"]


def _to_float(v: object) -> float | None:
    """Convert a value to float, returning None for NaN/None (SQL NULL)."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


class FeatureRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def fetch_bars(
        self,
        instrument_id: int,
        timeframe_id: int,
        from_ts: datetime,
        to_ts: datetime,
    ) -> pd.DataFrame:
        """Return raw ohlcv_bars rows in [from_ts, to_ts] as a DataFrame."""
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                _FETCH_BARS_SQL,
                {
                    "instrument_id": instrument_id,
                    "timeframe_id": timeframe_id,
                    "from_ts": from_ts,
                    "to_ts": to_ts,
                },
            )
            rows = result.fetchall()
            cols = list(result.keys())
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    async def fetch_features(
        self,
        instrument_id: int,
        timeframe_id: int,
        from_ts: datetime,
        to_ts: datetime,
    ) -> pd.DataFrame:
        """Return feature_vectors rows in [from_ts, to_ts] as a DataFrame."""
        async with session_scope(self.session_factory) as session:
            result = await session.execute(
                _FETCH_FEATURES_SQL,
                {
                    "instrument_id": instrument_id,
                    "timeframe_id": timeframe_id,
                    "from_ts": from_ts,
                    "to_ts": to_ts,
                },
            )
            rows = result.fetchall()
            cols = list(result.keys())
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    async def upsert_features(
        self,
        instrument_id: int,
        timeframe_id: int,
        features: pd.DataFrame,
    ) -> int:
        """
        Batch-upsert feature rows to feature_vectors.

        Rows where any O_t feature is NaN are silently skipped (cannot satisfy
        the NOT NULL constraint on f1–f10). U_t NaN values are stored as NULL.

        Returns the number of rows actually written.
        """
        if features.empty:
            return 0

        params: list[dict] = []
        for row in features.to_dict("records"):
            required_vals = {c: _to_float(row.get(c)) for c in _O_REQUIRED_COLS}
            if any(v is None for v in required_vals.values()):
                continue  # skip: NOT NULL constraint on f1–f4, f6–f10
            params.append(
                {
                    "bar_ts": row["bar_ts"],
                    "instrument_id": instrument_id,
                    "timeframe_id": timeframe_id,
                    **required_vals,
                    **{c: _to_float(row.get(c)) for c in _O_NULLABLE_COLS},
                    **{c: _to_float(row.get(c)) for c in _U_COLS},
                    "dim_used": int(row["dim_used"]) if row.get("dim_used") is not None else None,
                }
            )

        if not params:
            return 0

        async with session_scope(self.session_factory) as session:
            await session.execute(_UPSERT_FEATURES_SQL, params)
        return len(params)
