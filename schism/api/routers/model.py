"""
GET /model/params — IOHMM learned parameters for visualization.

Loads model_latest.pkl and returns emission means/stds, baseline transition
matrix (U=0), initial distribution, and EM convergence curve.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from scipy.special import softmax
from scipy.stats import percentileofscore
from sqlalchemy.ext.asyncio import AsyncSession

from schism.api.dependencies import get_session
from schism.api.schemas import FeatureSignature, ModelParams, StateProfile
from schism.models.iohmm import IOHMM
from schism.persistence.repositories import feature_repo, state_repo

router = APIRouter()

Session = Annotated[AsyncSession, Depends(get_session)]

_O_FEATURE_NAMES = [
    "cvd_vol", "oi_chg", "norm_ret", "liq_sq", "spread",
    "illiq", "rv_ratio", "vol_shock", "flow_liq", "flow_pos",
]
_U_FEATURE_NAMES = ["ewma_fr", "delta_fr", "fr_spread", "delta_lsr"]
# DB column names matching feature_repo order (f1..f10, u1..u4)
_DB_O_COLS = [
    "f1_cvd_vol", "f2_oi_chg", "f3_norm_ret", "f4_liq_sq", "f5_spread",
    "f6_illiq", "f7_rv_ratio", "f8_vol_shock", "f9_flow_liq", "f10_flow_pos",
]
_DB_U_COLS = ["u1_ewma_fr", "u2_delta_fr", "u3_fr_spread", "u4_delta_lsr"]

_DEFAULT_MODEL_PATH = Path("model_latest.pkl")


def _load_model() -> IOHMM:
    path = Path(os.environ.get("MODEL_PATH", str(_DEFAULT_MODEL_PATH)))
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"model file not found: {path}",
        )
    return IOHMM.load(path)


@router.get("/params", response_model=ModelParams)
async def params() -> ModelParams:
    model = _load_model()

    if not model._fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model not yet fitted",
        )

    K, D = model.K, model.D
    feature_names = _O_FEATURE_NAMES[:D]

    # Emission means (K, D)
    emission_means = model.mu.tolist()

    # Emission stds: sqrt of diagonal of each state covariance (K, D)
    emission_stds = [
        np.sqrt(np.diag(model.sigma[k])).tolist() for k in range(K)
    ]

    # Baseline transition matrix: softmax(alpha[i]) for each row i, U=0
    # alpha shape: (K, K), logits for P(S_t=j | S_{t-1}=i, U=0)
    transition_matrix = softmax(model.alpha, axis=1).tolist()

    pi = model.pi.tolist() if model.pi is not None else [1.0 / K] * K

    return ModelParams(
        n_states=K,
        n_obs=D,
        feature_names=feature_names,
        state_labels=list(model.labels[:K]),
        emission_means=emission_means,
        emission_stds=emission_stds,
        transition_matrix=transition_matrix,
        pi=pi,
        model_ver=model.model_ver or "",
        ll_history=[float(x) for x in model.ll_history],
    )


@router.get("/state-profiles", response_model=list[StateProfile])
async def state_profiles(
    session: Session,
    exchange: str = Query(default="binance"),
    symbol: str = Query(default="BTCUSDT"),
    market_type: str = Query(default="perp"),
    timeframe: str = Query(default="4h"),
) -> list[StateProfile]:
    import traceback
    try:
        return await _compute_state_profiles(session, exchange, symbol, market_type, timeframe)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        ) from exc


async def _compute_state_profiles(
    session: AsyncSession,
    exchange: str,
    symbol: str,
    market_type: str,
    timeframe: str,
) -> list[StateProfile]:
    model = _load_model()
    if not model._fitted:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="model not fitted")

    K, D = model.K, model.D
    mu: np.ndarray = model.mu  # (K, D) — observation features only

    # Resolve instrument for DB query
    instrument_id = await state_repo.resolve_instrument_id(session, exchange, symbol, market_type)
    timeframe_id = await state_repo.resolve_timeframe_id(session, timeframe)
    if instrument_id is None or timeframe_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="instrument or timeframe not found")

    # Historical feature distributions from DB
    dists = await feature_repo.get_feature_distributions(session, instrument_id, timeframe_id)

    # Build name → (emission_mean_per_state, db_col) mapping
    # O features: model.mu columns 0..D-1 map to _O_FEATURE_NAMES[:D]
    # U features: model has no mu for exogenous — skip (beta handles them)
    o_names = _O_FEATURE_NAMES[:D]
    o_db_cols = _DB_O_COLS[:D]

    # Option A: z-score of mu[k, f] relative to cross-state mean/std
    cross_mean = mu.mean(axis=0)        # (D,)
    cross_std = mu.std(axis=0)          # (D,)
    cross_std = np.where(cross_std < 1e-10, 1.0, cross_std)
    zscores = (mu - cross_mean) / cross_std   # (K, D)

    profiles: list[StateProfile] = []
    for k in range(K):
        sigs: list[FeatureSignature] = []
        for f_idx, (name, db_col) in enumerate(zip(o_names, o_db_cols)):
            em = float(mu[k, f_idx])
            z = float(zscores[k, f_idx])
            direction = "high" if z > 0.3 else ("low" if z < -0.3 else "neutral")

            # Option B: percentile of emission mean in historical distribution
            hist = dists.get(db_col)
            if hist is not None and len(hist) > 0:
                pct = float(percentileofscore(hist, em, kind="rank"))
            else:
                # Fallback: derive from z-score via normal CDF approximation
                from scipy.stats import norm
                pct = float(norm.cdf(z) * 100)

            sigs.append(FeatureSignature(
                feature=name,
                emission_mean=round(em, 6),
                zscore=round(z, 3),
                direction=direction,
                percentile=round(pct, 1),
            ))

        # Sort by abs(zscore) descending so most distinctive features come first
        sigs.sort(key=lambda s: abs(s.zscore), reverse=True)

        profiles.append(StateProfile(
            state=k,
            label=model.labels[k] if k < len(model.labels) else f"state_{k}",
            signatures=sigs,
        ))

    return profiles
