"""
Build Ot (dim 10, Eq.5) and Ut (dim 4, Eq.2). Interaction terms formed from raw inputs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from schism.data.preprocessing.vif_checker import check_vif
from schism.data.preprocessing.zscore import RollingZScore
from schism.persistence.repositories.feature_repo import FeatureRepository
from schism.utils.logger import ingestion_logger

_O_COLS = [
    "f1_cvd_vol",
    "f2_oi_chg",
    "f3_norm_ret",
    "f4_liq_sq",
    "f5_spread",
    "f6_illiq",
    "f7_rv_ratio",
    "f8_vol_shock",
    "f9_flow_liq",
    "f10_flow_pos",
]
_U_COLS = ["u1_ewma_fr", "u2_delta_fr", "u3_fr_spread", "u4_delta_lsr"]

_BAR_HOURS = 4  # 4h timeframe


class FeatureEngine:
    """
    Fetch ohlcv_bars from TimescaleDB, compute O_t (dim 10) and U_t (dim 4),
    winsorise, Z-score, observe VIF, and upsert to feature_vectors.

    VIF check is purely diagnostic — it annotates dim_used but never mutates
    feature values. Feature exclusion decisions are made outside the pipeline.

    Config keys read from feature_cfg:
        winsorize_pct       [lo, hi] percentile clip (default [1, 99])
        alpha_vol_ewma      EWMA alpha for volume (F8 denominator)
        alpha_fr_ewma       EWMA alpha for funding rate (U1)
        rv24h_warmup_bars   bars required for RV_24h (default 6)
        rv7d_warmup_bars    bars required for RV_7d  (default 42)

    Config keys read from validation_cfg:
        zscore_window_bars  rolling Z-score window (default 360)
        vif_threshold       F8 alert level (default 5)
        rho_threshold       pairwise |corr| alert level (default 0.85)
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        feature_cfg: dict | None = None,
        validation_cfg: dict | None = None,
    ) -> None:
        self._repo = FeatureRepository(session_factory)

        fc = feature_cfg or {}
        vc = validation_cfg or {}

        self._winsorize_lo: float = (fc.get("winsorize_pct") or [1, 99])[0]
        self._winsorize_hi: float = (fc.get("winsorize_pct") or [1, 99])[1]
        self._alpha_vol: float = fc.get("alpha_vol_ewma", 0.1)
        self._alpha_fr: float = fc.get("alpha_fr_ewma", 0.3)
        self._rv24h_bars: int = fc.get("rv24h_warmup_bars", 6)
        self._rv7d_bars: int = fc.get("rv7d_warmup_bars", 42)
        self._eps: float = fc.get("epsilon", 1e-8)  # spec §4, ε = 10^-8

        self._zscore_window: int = vc.get("zscore_window_bars", 360)
        self._vif_threshold: float = vc.get("vif_threshold", 5.0)
        self._rho_threshold: float = vc.get("rho_threshold", 0.85)

    # ── public entry point ────────────────────────────────────────────────────

    async def compute_and_store(
        self,
        instrument_id: int,
        timeframe_id: int,
        from_ts: datetime,
        to_ts: datetime,
        market_type: str = "perp",
    ) -> int:
        """
        Compute O_t and U_t for every bar in [from_ts, to_ts] and upsert to
        feature_vectors. Returns the number of rows written.

        Extra lookback bars are fetched before from_ts to warm up RV, EWMA,
        and the rolling Z-score window; those lookback rows are not stored.
        """
        lookback_bars = self._rv7d_bars + self._zscore_window + 1
        fetch_from = from_ts - timedelta(hours=_BAR_HOURS * lookback_bars)

        bars = await self._repo.fetch_bars(instrument_id, timeframe_id, fetch_from, to_ts)

        if bars.empty:
            ingestion_logger.warning(
                "feature_engine_no_bars",
                instrument_id=instrument_id,
                timeframe_id=timeframe_id,
                from_ts=from_ts.isoformat(),
                to_ts=to_ts.isoformat(),
            )
            return 0

        if len(bars) < self._rv7d_bars + 2:
            ingestion_logger.warning(
                "feature_engine_insufficient_bars",
                instrument_id=instrument_id,
                rows=len(bars),
                required=self._rv7d_bars + 2,
            )
            return 0

        raw = self._compute_raw(bars, market_type)
        winsorized = self._winsorize(raw)
        features = self._apply_zscore(winsorized)

        # VIF: observe and annotate only — no column mutation
        f8_exceeds, vif_dict = check_vif(
            features[_O_COLS].dropna(),
            vif_threshold=self._vif_threshold,
            rho_threshold=self._rho_threshold,
        )
        features["dim_used"] = 9 if f8_exceeds else 10

        # Restrict to the requested window (lookback rows excluded)
        mask = (features["bar_ts"] >= from_ts) & (features["bar_ts"] <= to_ts)
        out = features[mask].copy()

        written = await self._repo.upsert_features(instrument_id, timeframe_id, out)

        ingestion_logger.info(
            "feature_engine_done",
            instrument_id=instrument_id,
            timeframe_id=timeframe_id,
            rows_in_window=int(mask.sum()),
            rows_written=written,
            dim_used=9 if f8_exceeds else 10,
            f8_vif=round(vif_dict.get("f8_vol_shock", float("nan")), 4),
        )
        return written

    # ── raw feature computation ───────────────────────────────────────────────

    def _compute_raw(self, bars: pd.DataFrame, market_type: str) -> pd.DataFrame:
        df = bars.sort_values("bar_ts").reset_index(drop=True)

        log_ret = np.log(df["close"] / df["close"].shift(1))

        # Realised variance — rolling sum of squared log returns
        rv24h = (log_ret ** 2).rolling(self._rv24h_bars, min_periods=1).sum()
        rv7d = (log_ret ** 2).rolling(self._rv7d_bars, min_periods=1).sum()

        delta_oi = df["oi"] - df["oi"].shift(1)
        oi_prev = df["oi"].shift(1).replace(0.0, np.nan)
        vol_safe = df["volume"].replace(0.0, np.nan)

        # ── O_t raw ──────────────────────────────────────────────────────────

        # F1: ΔCVD_t / Vol_t  (spec §4, Eq.5 row 1)
        f1_raw = df["cvd"] / vol_safe

        # F2: ΔOI_t / OI_{t-1}  (row 2)
        f2_raw = delta_oi / oi_prev

        # F3: asinh(R_t / √RV_24h)  (row 3)
        f3_raw = np.arcsinh(log_ret / np.sqrt(rv24h.clip(lower=self._eps)))

        # F4: asinh(-min(0, ΔOI_t) × sgn(R_t))  (row 4)
        f4_raw = np.arcsinh(-np.minimum(0.0, delta_oi) * np.sign(log_ret))

        # F5: best_ask - best_bid  (row 5)
        f5_raw = df["best_ask"] - df["best_bid"]

        # F6: ILLIQ_t = log(|R_t| / (Vol_t + ε))  (spec §4, Eq.4)
        f6_raw = np.log(log_ret.abs() / (df["volume"] + self._eps))

        # F7: RV_24h / RV_7d  (row 7)
        f7_raw = rv24h / rv7d.clip(lower=self._eps)

        # F8: log(Vol_t / EWMA(Vol_t))  (row 8)
        ewma_vol = df["volume"].ewm(alpha=self._alpha_vol, adjust=False).mean()
        f8_raw = np.log(df["volume"] / ewma_vol.clip(lower=self._eps))

        # F9, F10: interaction terms formed from raw (un-Z-scored) inputs
        # Z-scoring applied to the products, not the individual factors (spec V1.4)
        f9_raw = f1_raw * f6_raw   # flow × liquidity
        f10_raw = f1_raw * f2_raw  # flow × %ΔOI

        # ── U_t raw (perp only) ───────────────────────────────────────────────

        if market_type == "perp":
            fr = df["funding_rate"]
            u1_raw = fr.ewm(alpha=self._alpha_fr, adjust=False).mean()
            u2_raw = fr - fr.shift(1)
            u3_raw = df["funding_rate"] - df["bybit_fr"]
            lsr = df["lsr_top"].replace(0.0, np.nan)
            u4_raw = np.log(lsr) - np.log(lsr.shift(1))
        else:
            _nan = pd.Series(np.nan, index=df.index)
            u1_raw = u2_raw = u3_raw = u4_raw = _nan

        # ── assemble ─────────────────────────────────────────────────────────

        out = df[["bar_ts"]].copy()
        for col, series in [
            ("f1_cvd_vol", f1_raw),
            ("f2_oi_chg", f2_raw),
            ("f3_norm_ret", f3_raw),
            ("f4_liq_sq", f4_raw),
            ("f5_spread", f5_raw),
            ("f6_illiq", f6_raw),
            ("f7_rv_ratio", f7_raw),
            ("f8_vol_shock", f8_raw),
            ("f9_flow_liq", f9_raw),
            ("f10_flow_pos", f10_raw),
            ("u1_ewma_fr", u1_raw),
            ("u2_delta_fr", u2_raw),
            ("u3_fr_spread", u3_raw),
            ("u4_delta_lsr", u4_raw),
        ]:
            out[col] = series.values
        return out

    # ── winsorisation ─────────────────────────────────────────────────────────

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip each feature to [p_lo, p_hi] computed over non-NaN values in that column."""
        out = df.copy()
        for col in _O_COLS + _U_COLS:
            s = out[col].dropna()
            if s.empty:
                continue
            p_lo = float(np.percentile(s, self._winsorize_lo))
            p_hi = float(np.percentile(s, self._winsorize_hi))
            out[col] = out[col].clip(lower=p_lo, upper=p_hi)
        return out

    # ── rolling Z-score ───────────────────────────────────────────────────────

    def _apply_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling Z-score to all feature columns.

        Interaction terms (f9, f10) are formed from the raw un-Z-scored inputs
        and then Z-scored as products — the order of computation in _compute_raw
        already satisfies this requirement. Z-scoring is applied uniformly here.
        """
        out = df.copy()
        w = self._zscore_window
        for col in _O_COLS + _U_COLS:
            out[col] = RollingZScore.batch_transform(out[col].to_numpy(), window=w)
        return out
