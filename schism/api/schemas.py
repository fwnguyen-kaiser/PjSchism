"""Pydantic v2 response models for all endpoints. Shared across routers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RegimeSnapshot(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    bar_ts: datetime
    instrument_id: int
    timeframe_id: int
    state: int
    label: str
    confidence: float | None
    posterior: list[float] | None
    model_ver: str | None
    forecast_t1: list[float] | None = None
    forecast_t2: list[float] | None = None


class BarWithRegime(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    bar_ts: datetime
    close: float | None
    volume: float | None
    state: int
    label: str
    confidence: float | None


class RegimeStats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    state: int
    label: str
    frequency_pct: float
    mean_sojourn_bars: float


class RefitLogEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    refit_id: int
    refit_ts: datetime
    instrument_id: int
    timeframe_id: int
    trigger: str
    delta_bic: float | None
    alignment_ok: bool | None
    drift_alert: bool
    dim_used: int | None
    model_ver: str | None
    cooldown_end_ts: datetime | None
    notes: str | None


class BarWithPosterior(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    bar_ts: datetime
    close: float | None
    volume: float | None
    state: int
    label: str
    confidence: float | None
    posterior: list[float] | None
    forecast_t1: list[float] | None
    forecast_t2: list[float] | None


class FeatureSignature(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    feature: str
    emission_mean: float
    zscore: float          # deviation from cross-state mean in σ units
    direction: str         # "high" | "low" | "neutral"
    percentile: float      # where emission_mean falls in historical dist (0–100)


class StateProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    state: int
    label: str
    signatures: list[FeatureSignature]   # all features, sorted by abs(zscore) desc


class ModelParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    n_states: int
    n_obs: int
    feature_names: list[str]
    state_labels: list[str]
    # (K, D) emission means — row = state, col = feature
    emission_means: list[list[float]]
    # (K, D) emission std devs (sqrt of diagonal Σ)
    emission_stds: list[list[float]]
    # (K, K) baseline transition matrix P(j|i, U=0) — row = from, col = to
    transition_matrix: list[list[float]]
    # (K,) initial state distribution
    pi: list[float]
    model_ver: str
    # EM convergence curve (log-likelihood per iteration)
    ll_history: list[float]
